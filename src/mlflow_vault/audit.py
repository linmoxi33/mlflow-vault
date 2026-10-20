"""Audit: the risk report, computed from the store in front of you.

This exists because "we have backups" is almost always false in a specific way.
Instead of asserting risk, ``audit`` measures it: does the store have a db, is it
in WAL mode, do the artifact URIs resolve, are they absolute (i.e. will they
break on move), is there an existing backup, does that backup verify.

Every check prints the evidence it used, so a reviewer can disagree with the
conclusion rather than having to trust it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import db as dbmod
from . import store as storemod
from .backup import load_manifest
from .secrets import classify

RISK_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "ok": 0}


@dataclass
class Risk:
    level: str
    code: str
    title: str
    evidence: str
    consequence: str
    mitigation: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "title": self.title,
            "evidence": self.evidence,
            "consequence": self.consequence,
            "mitigation": self.mitigation,
        }


@dataclass
class AuditReport:
    root: str
    kind: str
    inventory: Dict[str, object] = field(default_factory=dict)
    risks: List[Risk] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def worst(self) -> str:
        if not self.risks:
            return "ok"
        return max((r.level for r in self.risks), key=lambda lvl: RISK_ORDER[lvl])

    @property
    def score(self) -> int:
        """0-100, higher is safer. Deliberately crude and explainable."""
        penalty = {"critical": 40, "high": 22, "medium": 10, "low": 4}
        return max(0, 100 - sum(penalty.get(r.level, 0) for r in self.risks))

    def as_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "kind": self.kind,
            "worst": self.worst,
            "score": self.score,
            "inventory": self.inventory,
            "stats": self.stats,
            "risks": [r.as_dict() for r in self.risks],
        }


def audit(root: str, backup_dir: Optional[str] = None) -> AuditReport:
    layout = storemod.detect(root)
    rep = AuditReport(root=layout.root, kind=layout.kind)
    rep.inventory = storemod.inventory(layout.db_path, layout)

    local = layout.local_uris()
    remote = layout.remote_uris()
    base = layout.artifact_base()

    rep.stats = {
        "runs": len(layout.runs),
        "local_artifact_runs": len(local),
        "remote_artifact_runs": len([r for r in layout.runs if r.artifact_uri in set(remote)]),
        "artifact_base": base,
        "db": layout.db_path,
        "mlruns": layout.mlruns_dir,
    }

    # ---- R1: no database at all -------------------------------------------
    if layout.db_path is None:
        rep.risks.append(
            Risk(
                "high",
                "no-tracking-db",
                "Store has no tracking database",
                "only an mlruns/ directory was found",
                "runs, params, metrics and the experiment registry exist only as loose files; "
                "any tool that expects a db cannot read them",
                "archive the whole mlruns/ tree (mlflow-vault backup handles file stores)",
            )
        )

    # ---- R2: WAL sidecars exist -------------------------------------------
    if layout.db_path:
        wal = dbmod.wal_report(layout.db_path)
        # A 0-byte -wal alongside a 32 KB -shm is what a *cleanly closed* WAL-mode
        # database looks like: the shm is a shared-memory index, not data. Flagging
        # that as critical is the fastest way to train users to ignore the report.
        pending = {k: v for k, v in wal.items() if k.startswith("-") and int(v) > 0}
        if pending.get("-wal", 0) > 0:
            rep.risks.append(
                Risk(
                    "critical",
                    "wal-not-checkpointed",
                    "Live WAL file holds unmerged transactions",
                    f"{layout.db_path}-wal is {pending['-wal']} bytes",
                    "copying only mlflow.db loses every transaction still in the -wal file; "
                    "the restored store looks fine but is missing the newest runs",
                    "snapshot with the SQLite backup API (mlflow-vault does), or checkpoint first",
                )
            )
        elif wal.get("sidecars_present"):
            rep.stats["wal_note"] = (
                "WAL-mode store, but the -wal is empty (cleanly closed) — a plain copy is "
                "safe *now*, and unsafe the moment a writer is active"
            )
        # integrity
        try:
            con = dbmod.connect(layout.db_path, readonly=True)
            verdict = con.execute("PRAGMA integrity_check").fetchone()[0]
            con.close()
        except Exception as exc:
            verdict = f"unreadable: {exc}"
        if verdict != "ok":
            rep.risks.append(
                Risk(
                    "critical",
                    "db-damaged",
                    "Tracking database fails integrity_check",
                    str(verdict),
                    "reads may already be returning partial data",
                    "restore from a verified backup; do not write to this file again",
                )
            )

    # ---- R3: absolute artifact URIs --------------------------------------
    if local:
        absolute = []
        for u in local:
            p = storemod.local_path(u) or ""
            if p and (os.path.isabs(p) or (len(p) > 2 and p[1] == ":")):
                absolute.append(u)
        if absolute:
            rep.risks.append(
                Risk(
                    "high",
                    "absolute-artifact-uris",
                    "artifact_uri values are absolute filesystem paths",
                    f"{len(absolute)}/{len(local)} local runs, e.g. {absolute[0]}",
                    "the db moves fine but every artifact link breaks the moment the store is "
                    "copied to another directory, machine, or container — and MLflow reports no error",
                    "mlflow-vault restore rewrites the base prefix; verify dangling=0 afterwards",
                )
            )

    # ---- R4: dangling artifact directories -------------------------------
    dangling = [u for u in local if not os.path.isdir(storemod.local_path(u) or "")]
    if dangling:
        rep.risks.append(
            Risk(
                "critical",
                "dangling-artifacts",
                "Runs reference artifact directories that do not exist",
                f"{len(dangling)}/{len(local)} local runs, e.g. {dangling[0]}",
                "artifacts are already gone; metrics remain, so the loss is easy to miss",
                "restore artifacts from a backup that included them, then verify",
            )
        )

    # ---- R5: remote artifacts --------------------------------------------
    if remote:
        rep.risks.append(
            Risk(
                "medium",
                "remote-artifacts",
                "Artifacts live in remote object storage",
                f"{len(remote)} distinct URIs, e.g. {remote[0]}",
                "a database-only backup produces a store whose artifacts are all unreachable "
                "if the bucket or credentials go away",
                "enable versioning + lifecycle on the bucket; record the bucket in the runbook",
            )
        )

    # ---- R6: secrets in params/tags --------------------------------------
    if layout.db_path:
        secret_hits = _count_secrets(layout.db_path)
        if secret_hits:
            rep.risks.append(
                Risk(
                    "high",
                    "credentials-in-metadata",
                    "Credentials stored in run params/tags",
                    f"{secret_hits} key/value pairs look like secrets",
                    "every copy of the db — including backups, shared disks and CI artefacts — "
                    "is a copy of those credentials",
                    "rotate them, move to env/secret manager, and ship shared copies with "
                    "`mlflow-vault redact`",
                )
            )

    # ---- R7: is there a backup, and does it verify? ----------------------
    if backup_dir:
        from .verify import verify

        v = verify(backup_dir, deep=True)
        # "Has artifact bytes" is checked independently of whether the package
        # verifies: a db-only package fails verification too, and reporting only
        # "backup does not verify" hides the actual, actionable cause.
        summary = load_manifest(backup_dir).get("summary") or {}
        if local and not summary.get("has_artifacts"):
            rep.risks.append(
                Risk(
                    "critical",
                    "backup-db-only",
                    "Backup contains the database but no artifact bytes",
                    f"local runs in store: {len(local)}, packaged artifact files: 0",
                    "restoring it yields runs with metrics and zero artifacts — MLflow shows "
                    "no error, only empty artifact lists",
                    "re-run backup with the artifact tree reachable, then verify again",
                )
            )
        if not v.ok:
            rep.risks.append(
                Risk(
                    "critical",
                    "backup-unverified",
                    "Existing backup does not verify",
                    "; ".join(f"[{f.code}] {f.path or f.message}" for f in v.errors[:3]),
                    "an unverified backup is a guess; the restore drill would fail",
                    "re-run backup, then verify again",
                )
            )
        else:
            rep.risks.append(
                Risk(
                    "ok",
                    "backup-verified",
                    "Backup verifies",
                    f"{v.checked_files} files hashed, rows match",
                    "none",
                    "keep the drill on a schedule; re-verify after every backup",
                )
            )
    else:
        rep.risks.append(
            Risk(
                "critical",
                "no-backup-evidence",
                "No backup was supplied for inspection",
                "audit ran without --backup",
                "the store's only copy is the one being read right now",
                "run `mlflow-vault backup` and then `mlflow-vault verify`",
            )
        )
    return rep


def _count_secrets(db_path: str) -> int:
    con = dbmod.connect(db_path, readonly=True)
    try:
        tables = dbmod.tables(db_path)
        n = 0
        for table in ("params", "tags"):
            if table not in tables:
                continue
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            if "key" not in cols or "value" not in cols:
                continue
            for key, value in con.execute(f"SELECT key, value FROM {table}"):
                if classify(str(key), "" if value is None else str(value)):
                    n += 1
        return n
    finally:
        con.close()


def summarize(rep: AuditReport) -> str:
    lines = [
        f"store    {rep.root}  ({rep.kind})",
        f"risk     worst={rep.worst.upper()}  score={rep.score}/100",
        f"content  {rep.inventory.get('experiments', 0)} experiments, "
        f"{rep.inventory.get('runs', 0)} runs, "
        f"{rep.inventory.get('metrics', 0)} metrics, {rep.inventory.get('params', 0)} params",
        "",
    ]
    for risk in sorted(rep.risks, key=lambda r: -RISK_ORDER[r.level]):
        lines.append(f"[{risk.level.upper():8}] {risk.title}")
        lines.append(f"           evidence: {risk.evidence}")
        if risk.level != "ok":
            lines.append(f"           impact:   {risk.consequence}")
            lines.append(f"           fix:      {risk.mitigation}")
    return "\n".join(lines)
