"""Restore: put the store back somewhere it actually works.

The hard part is not copying files, it is the **absolute paths**. MLflow stores
``runs.artifact_uri`` and ``experiments.artifact_location`` as literal strings, so
a package taken from ``/data/mlruns`` and unpacked into ``/restore`` yields a
database that is byte-perfect, opens without error, and lists every run — while
every artifact lookup resolves to a directory that no longer exists. MLflow does
not warn; you get empty artifact lists and ``No such file or directory`` deep in
a notebook.

So restore does three things, in this order:

1. **Plan first.** Compute the rewrite map and show it before touching disk.
2. **Copy, then rewrite.** Artifacts land under the new root; then ``artifact_uri``
   is rewritten *only* for URIs that were inside the recorded base. Runs whose
   artifacts live elsewhere (a hand-set custom directory, or S3) keep their URI.
3. **Prove equivalence.** Recompute the logical digest with ``artifact_uri``
   excluded and compare against the manifest. If the rows differ, restore failed
   loudly instead of quietly producing a plausible-looking store.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import db as dbmod
from . import store as storemod
from .backup import load_manifest
from .hashing import sha256_file

MANIFEST_RUNS = "_manifest_runs"


@dataclass
class RestorePlan:
    backup_dir: str
    target_root: str
    db_source: Optional[str]
    db_target: Optional[str]
    artifact_base_old: Optional[str]
    artifact_base_new: Optional[str]
    file_store: bool = False
    rewrites: Dict[str, str] = field(default_factory=dict)  # run_id -> new uri
    kept: Dict[str, str] = field(default_factory=dict)  # run_id -> untouched uri
    external: List[str] = field(default_factory=list)
    origin_uris: Dict[str, str] = field(default_factory=dict)

    @property
    def rewrite_count(self) -> int:
        return len(self.rewrites)

    def as_dict(self) -> Dict[str, object]:
        return {
            "backup": self.backup_dir,
            "target": self.target_root,
            "db": {"from": self.db_source, "to": self.db_target},
            "artifact_base": {"from": self.artifact_base_old, "to": self.artifact_base_new},
            "file_store": self.file_store,
            "rewrite_count": self.rewrite_count,
            "kept_count": len(self.kept),
            "external_count": len(self.external),
            "sample": [
                {"run_id": k, "from": self.origin_uris.get(k, ""), "to": v}
                for k, v in list(self.rewrites.items())[:5]
            ],
        }


@dataclass
class RestoreResult:
    target_root: str
    plan: RestorePlan
    db_rows_matched: bool
    verified_rows: Optional[int]
    files_copied: int
    files_skipped: int
    bytes_copied: int
    dry_run: bool = False
    warnings: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.db_rows_matched and not self.problems

    def as_dict(self) -> Dict[str, object]:
        return {
            "target": self.target_root,
            "ok": self.ok,
            "dry_run": self.dry_run,
            "db_rows_matched": self.db_rows_matched,
            "verified_rows": self.verified_rows,
            "files_copied": self.files_copied,
            "files_skipped": self.files_skipped,
            "bytes_copied": self.bytes_copied,
            "rewrite_count": self.plan.rewrite_count,
            "kept_count": len(self.plan.kept),
            "warnings": self.warnings,
            "problems": self.problems,
        }


def plan(backup_dir: str, target_root: str) -> RestorePlan:
    backup_dir = os.path.abspath(backup_dir)
    target_root = os.path.abspath(target_root)
    manifest = load_manifest(backup_dir)

    store_meta = manifest.get("store") or {}
    db_info = manifest.get("db") or {}
    file_store = store_meta.get("kind") == "file"

    db_source = None
    if db_info.get("present") and db_info.get("archive_path"):
        db_source = os.path.join(backup_dir, str(db_info["archive_path"]).replace("/", os.sep))
    db_target = os.path.join(target_root, os.path.basename(db_source)) if db_source else None

    old_base = store_meta.get("artifact_base")
    base_rel = store_meta.get("artifact_base_rel")
    # Preserve the *shape* of the tree, not just the leaf: if the base was
    # ``<root>/mlruns/1`` (one experiment), the restored base must be
    # ``<target>/mlruns/1`` — dropping the ``1`` would rewrite every URI to a
    # directory that holds the wrong run's artifacts.
    if base_rel:
        new_base = os.path.join(target_root, str(base_rel).replace("/", os.sep))
    else:
        new_base = os.path.join(target_root, "mlruns")

    p = RestorePlan(
        backup_dir=backup_dir,
        target_root=target_root,
        db_source=db_source,
        db_target=db_target,
        artifact_base_old=old_base,
        artifact_base_new=new_base,
        file_store=file_store,
    )

    for run_id, meta in sorted((manifest.get("runs") or {}).items()):
        uri = str(meta.get("artifact_uri") or "")
        if not uri:
            continue
        p.origin_uris[run_id] = uri
        if storemod.is_remote(uri):
            p.external.append(uri)
            continue
        rel = meta.get("rel_base")
        if rel is None and old_base:
            rel = storemod.split_uri(uri, old_base)
        if rel is None:
            # Outside the recorded base: a hand-set custom path. Leave the string
            # alone rather than silently redirecting someone's deliberate choice.
            p.kept[run_id] = uri
            continue
        p.rewrites[run_id] = storemod.join_uri(new_base, str(rel))
    return p


def restore(
    backup_dir: str,
    target_root: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    overwrite_artifacts: bool = True,
) -> RestoreResult:
    p = plan(backup_dir, target_root)
    manifest = load_manifest(p.backup_dir)
    result = RestoreResult(
        target_root=p.target_root,
        plan=p,
        db_rows_matched=False,
        verified_rows=None,
        files_copied=0,
        files_skipped=0,
        bytes_copied=0,
        dry_run=dry_run,
    )

    if dry_run:
        result.db_rows_matched = True
        return result

    if p.db_target and os.path.exists(p.db_target) and not force:
        raise FileExistsError(
            f"{p.db_target} already exists (pass force=True to overwrite; "
            "restoring into a fresh directory is the safe default)"
        )
    os.makedirs(p.target_root, exist_ok=True)

    # ---- 1. artifacts first: the db must never point at an empty directory ----
    prefix = str((manifest.get("store") or {}).get("artifact_base_rel") or "artifacts")
    for entry in manifest.get("files") or []:
        if entry.get("role") != "artifact":
            continue
        rel = str(entry["path"])
        src = os.path.join(p.backup_dir, rel.replace("/", os.sep))
        if not os.path.isfile(src):
            result.problems.append(f"missing in package: {rel}")
            continue
        dst = os.path.join(p.target_root, _target_rel(rel, prefix).replace("/", os.sep))
        if os.path.exists(dst) and not overwrite_artifacts:
            result.files_skipped += 1
            continue
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != entry["sha256"]:
            result.problems.append(f"copy corrupted: {rel}")
            continue
        result.files_copied += 1
        result.bytes_copied += int(entry["size"])

    # ---- 2. database ---------------------------------------------------------
    if p.db_source and p.db_target:
        if not os.path.isfile(p.db_source):
            result.problems.append(f"database snapshot missing: {p.db_source}")
        else:
            shutil.copyfile(p.db_source, p.db_target)
            if sha256_file(p.db_target) != sha256_file(p.db_source):
                result.problems.append("database copy corrupted")
            elif p.rewrites:
                changed = dbmod.rewrite_artifact_uris(p.db_target, p.rewrites)
                if p.artifact_base_old and p.artifact_base_new:
                    dbmod.rewrite_experiment_locations(
                        p.db_target, p.artifact_base_old, p.artifact_base_new
                    )
                if changed == 0:
                    result.warnings.append(
                        "planned artifact_uri rewrites but the runs table changed 0 rows"
                    )
            if os.path.isfile(p.db_target):
                got = dbmod.digest_without_relocatable(p.db_target)
                want = (manifest.get("db") or {}).get("logical") or {}
                result.verified_rows = got.get("__total_rows__")
                result.db_rows_matched = _logical_equal(got, want, result.problems)
    elif p.file_store:
        result.db_rows_matched = _file_store_intact(p.target_root, manifest, result.problems)

    _write_restore_report(result, manifest)
    return result


def _target_rel(rel: str, prefix: str) -> str:
    """Where a packaged artifact path lands inside the restored root.

    Artifacts were archived at their *base-relative* path, so a package whose
    ``artifact_base_rel`` was ``mlruns/1`` stores files under ``mlruns/1/...``
    and must unpack back to exactly that — dropping the ``1`` would put every
    run's weights in the wrong experiment directory.

    Only the *flattened* case (base lived outside the store root, archived under
    ``artifacts/``) needs rewriting: those go to ``mlruns/<tail>``.
    """
    if prefix and prefix != "artifacts" and rel.startswith(prefix + "/"):
        return rel
    if rel.startswith("artifacts/"):
        return "mlruns/" + rel[len("artifacts/") :]
    return rel


def _logical_equal(got: Dict[str, object], want: Dict[str, object], problems: List[str]) -> bool:
    ok = True
    for table, info in want.items():
        if table == "__total_rows__":
            if got.get("__total_rows__") != info:
                problems.append(f"row total {got.get('__total_rows__')} != {info}")
                ok = False
            continue
        now = got.get(table)
        if now is None:
            problems.append(f"table {table} missing after restore")
            ok = False
            continue
        if now["digest"] != info["digest"]:  # type: ignore[index]
            problems.append(
                f"table {table} differs after restore "
                f"({now['rows']} rows vs {info['rows']})"  # type: ignore[index]
            )
            ok = False
    return ok


def _file_store_intact(root: str, manifest: Dict[str, object], problems: List[str]) -> bool:
    """File stores have no db; equivalence means every packaged file landed."""
    ok = True
    for entry in manifest.get("files") or []:
        if entry.get("role") != "artifact":
            continue
        rel = str(entry["path"])
        dst = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(dst):
            problems.append(f"file store entry missing after restore: {rel}")
            ok = False
        elif sha256_file(dst) != entry["sha256"]:
            problems.append(f"file store entry differs after restore: {rel}")
            ok = False
    return ok


def _write_restore_report(result: RestoreResult, manifest: Dict[str, object]) -> None:
    path = os.path.join(result.target_root, "RESTORE-REPORT.json")
    payload = {
        "tool": "mlflow-vault",
        "backup_created_utc": manifest.get("created_utc"),
        "source_backup": result.plan.backup_dir,
        **result.as_dict(),
        "plan": result.plan.as_dict(),
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        fh.write("\n")


def audit_restored(result: RestoreResult) -> Dict[str, object]:
    """Post-restore honesty check: are the URIs actually resolvable now?"""
    target = result.target_root.replace("\\", "/")
    out: Dict[str, object] = {"rows_match": result.db_rows_matched}
    db_path = result.plan.db_target
    if db_path and os.path.isfile(db_path):
        con = dbmod.connect(db_path, readonly=True)
        try:
            uris = [str(r[0] or "") for r in con.execute("SELECT artifact_uri FROM runs")]
        finally:
            con.close()
        local = [u for u in uris if not storemod.is_remote(u)]
        out["artifact_uris_total"] = len(uris)
        out["artifact_uris_inside_target"] = sum(
            1 for u in local if u.replace("\\", "/").startswith(target)
        )
        out["dangling"] = [u for u in local if not os.path.isdir(u)][:20]
        out["dangling_count"] = sum(1 for u in local if not os.path.isdir(u))
    out["mlruns_present"] = os.path.isdir(os.path.join(result.target_root, "mlruns"))
    return out
