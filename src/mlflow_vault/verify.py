"""Verification: prove the package is complete, not merely present.

Three independent questions, because they fail independently:

1. **Bytes** — does every file match the digest recorded at backup time?
2. **Structure** — is anything the manifest promised *missing*, and did anything
   appear that the manifest never recorded?
3. **Meaning** — does the database still contain the same rows? A store can pass
   (1) and (2) and still be useless if the db snapshot was taken before the runs
   were written.

A fourth check answers the question that actually causes data loss in practice:
**does every run's artifact directory exist?** MLflow will happily serve a store
whose artifact directories are gone; it just returns empty listings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from . import db as dbmod
from . import store as storemod
from .backup import load_manifest
from .hashing import sha256_file

SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    path: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"severity": self.severity, "code": self.code, "message": self.message}
        if self.path:
            d["path"] = self.path
        return d


@dataclass
class VerifyReport:
    backup_dir: str
    findings: List[Finding] = field(default_factory=list)
    checked_files: int = 0
    checked_bytes: int = 0
    logical: Dict[str, object] = field(default_factory=dict)
    artifact_coverage: Dict[str, object] = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str, path: Optional[str] = None) -> None:
        self.findings.append(Finding(severity, code, message, path))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, object]:
        return {
            "backup": self.backup_dir,
            "ok": self.ok,
            "checked_files": self.checked_files,
            "checked_bytes": self.checked_bytes,
            "errors": [f.as_dict() for f in self.errors],
            "warnings": [f.as_dict() for f in self.warnings],
            "artifact_coverage": self.artifact_coverage,
            "logical_rows": self.logical.get("__total_rows__"),
        }

    def exit_code(self) -> int:
        return 0 if self.ok else 1


def verify(backup_dir: str, *, deep: bool = True, allow_missing_ledger: bool = False) -> VerifyReport:
    backup_dir = os.path.abspath(backup_dir)
    manifest = load_manifest(backup_dir)
    rep = VerifyReport(backup_dir=backup_dir)

    files = manifest.get("files", [])
    recorded: Dict[str, Dict[str, object]] = {str(f["path"]): f for f in files}

    # ---- 1. bytes --------------------------------------------------------
    for rel, entry in sorted(recorded.items()):
        abs_p = os.path.join(backup_dir, rel.replace("/", os.sep))
        if not os.path.isfile(abs_p):
            if entry.get("role") == "secrets-ledger" and allow_missing_ledger:
                rep.add("warning", "ledger-missing-allowed", "secrets ledger absent (allowed)", rel)
                continue
            rep.add("error", "file-missing", "file named in manifest is gone", rel)
            continue
        actual_size = os.path.getsize(abs_p)
        if actual_size != int(entry["size"]):
            rep.add(
                "error",
                "size-mismatch",
                f"size {actual_size} != recorded {entry['size']}",
                rel,
            )
            continue
        if deep:
            actual = sha256_file(abs_p)
            if actual != entry["sha256"]:
                rep.add(
                    "error",
                    "hash-mismatch",
                    f"sha256 {actual[:12]}… != recorded {str(entry['sha256'])[:12]}…",
                    rel,
                )
                continue
        rep.checked_files += 1
        rep.checked_bytes += actual_size

    # ---- 2. structure ----------------------------------------------------
    extra: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(backup_dir):
        dirnames.sort()
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), backup_dir).replace("\\", "/")
            if rel in ("manifest.json", "RESTORE.md", ".gitignore"):
                continue
            if rel not in recorded:
                extra.add(rel)
    for rel in sorted(extra):
        rep.add("warning", "unexpected-file", "file present but not in manifest", rel)

    # ---- 3. meaning ------------------------------------------------------
    db_info = manifest.get("db") or {}
    snap_rel = db_info.get("archive_path") if isinstance(db_info, dict) else None
    snap_abs = os.path.join(backup_dir, str(snap_rel).replace("/", os.sep)) if snap_rel else None
    if db_info.get("present") and snap_abs and os.path.isfile(snap_abs):
        try:
            rep.logical = dbmod.digest_without_relocatable(snap_abs)
        except Exception as exc:
            rep.add("error", "db-unreadable", f"database snapshot cannot be read: {exc}", snap_rel)
        else:
            recorded_logical = db_info.get("logical") or {}
            for table, info in recorded_logical.items():
                if table == "__total_rows__":
                    continue
                now = rep.logical.get(table)
                if now is None:
                    rep.add("error", "table-lost", f"table {table} disappeared", snap_rel)
                    continue
                if now["digest"] != info["digest"]:
                    rep.add(
                        "error",
                        "rows-changed",
                        f"table {table}: {now['rows']} rows / digest {str(now['digest'])[:12]}… "
                        f"!= {info['rows']} rows / {str(info['digest'])[:12]}…",
                        snap_rel,
                    )
            rec_total = recorded_logical.get("__total_rows__")
            if rec_total is not None and rep.logical.get("__total_rows__") != rec_total:
                rep.add(
                    "error",
                    "rowcount-mismatch",
                    f"total rows {rep.logical.get('__total_rows__')} != recorded {rec_total}",
                    snap_rel,
                )
            integrity = dbmod.connect(snap_abs, readonly=True)
            try:
                verdict = integrity.execute("PRAGMA integrity_check").fetchone()
            except Exception as exc:
                rep.add("error", "integrity-check-failed", str(exc), snap_rel)
                verdict = None
            finally:
                integrity.close()
            if verdict and verdict[0] != "ok":
                rep.add("error", "integrity-check", f"SQLite says {verdict[0]!r}", snap_rel)

    # ---- 4. does every run still have artifacts? -------------------------
    rep.artifact_coverage = _coverage(backup_dir, manifest, snap_abs, rep)
    return rep


def _coverage(
    backup_dir: str,
    manifest: Dict[str, object],
    snap_abs: Optional[str],
    rep: VerifyReport,
) -> Dict[str, object]:
    """Re-derive each run's artifact directory from the *manifest*, then check it.

    Deriving from the manifest rather than from "did we copy some files" is what
    catches the classic half-backup: db archived, artifact tree silently skipped.

    Two failure shapes, both fatal and both silent in MLflow:

    * the directory is absent from the package at all;
    * the directory exists but **holds no files for that run** — which is what
      happens when folders are recreated (or cloned from a scaffold) without the
      payload. Testing ``isdir`` alone would call that a success.
    """
    runs = manifest.get("runs") or {}
    prefix = str((manifest.get("store") or {}).get("artifact_base_rel") or "artifacts")
    packaged = [str(f["path"]) for f in (manifest.get("files") or []) if f.get("role") == "artifact"]

    local = remote = missing = empty = 0
    missing_runs: List[str] = []
    empty_runs: List[str] = []

    for run_id, meta in sorted(runs.items()):
        if str(meta.get("artifact_locality") or "local") == "remote":
            remote += 1
            continue
        local += 1
        rel_base = meta.get("rel_base")
        want = f"{prefix}/{rel_base}".rstrip("/") if rel_base else "artifacts"
        # Present *on disk*, not merely listed in the manifest: a package whose
        # manifest claims files that were then deleted would otherwise pass this
        # check (the byte-level loop does catch it, but coverage must not contradict it).
        under = [
            p
            for p in packaged
            if (p == want or p.startswith(want + "/"))
            and os.path.isfile(os.path.join(backup_dir, p.replace("/", os.sep)))
        ]

        if not under:
            dir_exists = any(
                os.path.isdir(os.path.join(backup_dir, c.replace("/", os.sep)))
                for c in {want, want.split("/")[0] if want else want}
            )
            if dir_exists:
                empty += 1
                empty_runs.append(run_id)
                rep.add(
                    "error",
                    "artifact-empty",
                    f"run's artifact directory exists but contains no files for it "
                    f"(expected under {want}/)",
                    run_id,
                )
            else:
                missing += 1
                missing_runs.append(run_id)
                rep.add(
                    "error",
                    "artifact-dir-missing",
                    f"run's artifact directory is not in the package (expected near {want}/)",
                    run_id,
                )

    if remote:
        rep.add(
            "info",
            "remote-artifacts",
            f"{remote} run(s) point at remote storage and were not downloaded",
        )
    return {
        "local_runs": local,
        "remote_runs": remote,
        "missing_dirs": missing,
        "empty_dirs": empty,
        "missing_run_ids": missing_runs[:20],
        "empty_run_ids": empty_runs[:20],
        "packaged_artifact_files": len(packaged),
    }


def summarize(rep: VerifyReport) -> str:
    lines = [
        f"backup   {rep.backup_dir}",
        f"result   {'OK' if rep.ok else 'FAILED'}",
        f"files    {rep.checked_files} verified ({rep.checked_bytes} bytes)",
        f"rows     {rep.logical.get('__total_rows__', 'n/a')} across "
        f"{max(len(rep.logical) - 1, 0)} tables",
        "artifacts local={local_runs} remote={remote_runs} missing={missing_dirs}".format(
            **rep.artifact_coverage
        )
        if rep.artifact_coverage
        else "artifacts n/a",
    ]
    for f in rep.errors:
        lines.append(f"  ERROR  [{f.code}] {f.message}" + (f"  <- {f.path}" if f.path else ""))
    for f in rep.warnings:
        lines.append(f"  WARN   [{f.code}] {f.message}" + (f"  <- {f.path}" if f.path else ""))
    return "\n".join(lines)


def compare(backup_a: str, backup_b: str) -> Dict[str, object]:
    """Diff two packages: same store, different time — what changed?"""
    a, b = load_manifest(backup_a), load_manifest(backup_b)
    fa = {str(f["path"]): f for f in a.get("files", [])}
    fb = {str(f["path"]): f for f in b.get("files", [])}
    return {
        "root_hash_equal": a.get("root_hash") == b.get("root_hash"),
        "only_in_a": sorted(set(fa) - set(fb)),
        "only_in_b": sorted(set(fb) - set(fa)),
        "changed": sorted(
            p for p in set(fa) & set(fb) if fa[p]["sha256"] != fb[p]["sha256"]
        ),
    }


def dump_json(rep: VerifyReport) -> str:
    return json.dumps(rep.as_dict(), indent=2, ensure_ascii=False, sort_keys=True)


__all__ = ["Finding", "VerifyReport", "compare", "dump_json", "storemod", "summarize", "verify"]
