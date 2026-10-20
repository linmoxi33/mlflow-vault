"""Backup: turn a store into a self-describing, hash-verified package.

The package is deliberately boring and inspectable — no tar, no compression, no
custom container. Anyone can open it with ``ls`` and ``sha256sum``:

    <backup>/
      manifest.json          every file + digest + the store's logical state
      secrets.json           where the secrets are (fingerprints only)
      db/mlflow.db           consistency snapshot (``VACUUM INTO`` semantics)
      artifacts/...          byte-for-byte copy of every run's artifacts
      mlruns/...             file-store layout, when the store is a file store
      RESTORE.md             the recovery procedure, in the package itself

A backup nobody can restore is not a backup, so the recovery instructions travel
with the data instead of living in a wiki.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

from . import __version__
from . import db as dbmod
from . import store as storemod
from .hashing import merkle_root, sha256_file
from .secrets import SecretsLedger, classify, write_gitignore

MANIFEST_SCHEMA = 1


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _walk(base: str) -> List[Tuple[str, str]]:
    """(absolute, posix-relative) for every file under ``base``, sorted."""
    out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for name in sorted(filenames):
            abs_p = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_p, base).replace("\\", "/")
            out.append((abs_p, rel))
    return out


class BackupResult:
    def __init__(self, manifest: Dict[str, object], path: str) -> None:
        self.manifest = manifest
        self.path = path

    @property
    def file_count(self) -> int:
        return len(self.manifest["files"])  # type: ignore[arg-type]

    @property
    def total_bytes(self) -> int:
        return int(self.manifest["summary"]["total_bytes"])  # type: ignore[index]

    @property
    def secret_count(self) -> int:
        return int(self.manifest["secrets"]["count"])  # type: ignore[index]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"<BackupResult {self.path} files={self.file_count} bytes={self.total_bytes}>"


def backup(
    root: str,
    out_dir: str,
    *,
    overwrite: bool = False,
    include_secrets: bool = True,
    sqlite_timeout: float = 30.0,
) -> BackupResult:
    """Create a backup package of the store at ``root`` into ``out_dir``."""
    layout = storemod.detect(root)
    out_dir = os.path.abspath(out_dir)

    if os.path.exists(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists and is not empty (pass overwrite=True to replace)"
        )
    os.makedirs(out_dir, exist_ok=True)

    files: List[Dict[str, object]] = []
    runs_meta: Dict[str, Dict[str, object]] = {}
    warnings: List[str] = []
    ledger = SecretsLedger()

    # ---- phase 1: database -------------------------------------------------
    db_info: Dict[str, object] = {"present": False}
    if layout.db_path:
        snap_rel = "db/" + os.path.basename(layout.db_path)
        snap_abs = os.path.join(out_dir, snap_rel)
        if dbmod.busy_probe(layout.db_path, timeout=min(sqlite_timeout, 5.0)):
            warnings.append(
                "store was locked by a live writer during backup; the snapshot is consistent "
                "but may not include the very last transactions"
            )
        info = dbmod.snapshot(layout.db_path, snap_abs)
        logical = dbmod.digest_without_relocatable(snap_abs)
        db_info = {
            "present": True,
            "source": os.path.relpath(layout.db_path, layout.root).replace("\\", "/"),
            "archive_path": snap_rel,
            "sha256": sha256_file(snap_abs),
            "size": os.path.getsize(snap_abs),
            "snapshot": info,
            "wal": dbmod.wal_report(layout.db_path),
            "logical": logical,
            "tables": dbmod.tables(snap_abs),
        }
        files.append(
            {
                "path": snap_rel,
                "sha256": db_info["sha256"],
                "size": db_info["size"],
                "role": "db-snapshot",
                "origin": db_info["source"],
            }
        )
        if info.get("wal_was_meaningful"):
            warnings.append(
                "source DB had a non-empty -wal file; a plain file copy would have lost "
                "committed transactions — snapshot used the SQLite backup API instead"
            )
        if info.get("integrity_check") != "ok":
            warnings.append(
                f"snapshot integrity_check returned {info.get('integrity_check')!r} — "
                "source store may already be damaged"
            )

        # secrets inside params/tags
        if include_secrets:
            _scan_db_secrets(snap_abs, ledger)
            if len(ledger):
                db_info["secrets_found"] = len(ledger)

    # ---- phase 2: artifact trees ------------------------------------------
    base = layout.artifact_base()
    file_store = layout.db_path is None and layout.mlruns_dir is not None
    roots: List[Tuple[str, str]] = []  # (absolute dir, archive prefix)

    if file_store and layout.mlruns_dir:
        roots.append((layout.mlruns_dir, "mlruns"))
        base_rel = os.path.relpath(layout.mlruns_dir, layout.root).replace("\\", "/")
    elif base:
        base_rel = _rel_or_none(base, layout.root)
        if base_rel is None:
            warnings.append(
                f"artifact base {base!r} lives outside the store root; it is archived under "
                "artifacts/ and will be relocated on restore"
            )
            roots.append((base, "artifacts"))
            base_rel = None
        else:
            roots.append((base, base_rel))
    else:
        base_rel = None
        if not layout.runs:
            warnings.append("no runs found — nothing to archive beyond the database")

    seen_prefixes: set[str] = set()
    for abs_dir, prefix in roots:
        if not os.path.isdir(abs_dir):
            warnings.append(f"artifact directory declared but missing: {abs_dir}")
            continue
        if os.path.abspath(abs_dir) in seen_prefixes:  # pragma: no cover
            continue
        seen_prefixes.add(os.path.abspath(abs_dir))
        for abs_p, rel in _walk(abs_dir):
            arc = f"{prefix}/{rel}"
            dst = os.path.join(out_dir, arc.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            size, digest = _copy_verified(abs_p, dst)
            files.append(
                {
                    "path": arc,
                    "sha256": digest,
                    "size": size,
                    "role": "artifact",
                    "origin": os.path.relpath(abs_p, layout.root).replace("\\", "/")
                    if _inside(abs_p, layout.root)
                    else abs_p,
                }
            )

    # ---- phase 3: run index + external artifacts --------------------------
    for run in layout.runs:
        entry: Dict[str, object] = {
            "experiment_id": run.experiment_id,
            "artifact_uri": run.artifact_uri,
            "status": run.status,
            "lifecycle_stage": run.lifecycle_stage,
        }
        if run.artifact_uri and storemod.is_remote(run.artifact_uri):
            entry["artifact_locality"] = "remote"
            warnings.append(
                f"run {run.run_id} points at remote storage ({run.artifact_uri}); "
                "its artifacts are NOT in this backup"
            )
        else:
            rel = storemod.split_uri(run.artifact_uri, base) if base else None
            entry["artifact_locality"] = "local"
            entry["rel_base"] = rel
            if base is None:
                entry["archive_dir"] = "artifacts"
        runs_meta[run.run_id] = entry

    # ---- phase 4: secrets ledger + guardrails -----------------------------
    secrets_rel = "secrets.json"
    if include_secrets and len(ledger):
        ledger.write(os.path.join(out_dir, secrets_rel))
        files.append(
            {
                "path": secrets_rel,
                "sha256": sha256_file(os.path.join(out_dir, secrets_rel)),
                "size": os.path.getsize(os.path.join(out_dir, secrets_rel)),
                "role": "secrets-ledger",
                "origin": "(generated)",
            }
        )
    elif include_secrets:
        ledger.write(os.path.join(out_dir, secrets_rel))
        files.append(
            {
                "path": secrets_rel,
                "sha256": sha256_file(os.path.join(out_dir, secrets_rel)),
                "size": os.path.getsize(os.path.join(out_dir, secrets_rel)),
                "role": "secrets-ledger",
                "origin": "(generated, empty)",
            }
        )
    write_gitignore(out_dir)

    # ---- phase 5: manifest --------------------------------------------------
    has_local = any(f["role"] == "artifact" for f in files)
    manifest: Dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "tool": {"name": "mlflow-vault", "version": __version__},
        "created_utc": _now(),
        "store": {
            "kind": layout.kind,
            "root": layout.root,
            "db_present": bool(layout.db_path),
            "mlruns_present": bool(layout.mlruns_dir),
            "artifact_base": base,
            "artifact_base_rel": base_rel,
            "notes": layout.notes,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "db": db_info,
        "runs": runs_meta,
        "experiments": layout.experiments,
        "inventory": storemod.inventory(layout.db_path, layout),
        "files": sorted(files, key=lambda f: str(f["path"])),
        "summary": {
            "file_count": len(files),
            "artifact_count": sum(1 for f in files if f["role"] == "artifact"),
            "total_bytes": sum(int(f["size"]) for f in files),
            "has_artifacts": has_local,
        },
        "secrets": {
            "policy": _secrets_policy(include_secrets),
            "count": len(ledger),
            "ledger": secrets_rel if include_secrets else None,
            "values_stored_in_manifest": False,
        },
        "warnings": sorted(set(warnings)),
        "limits": _limits(layout),
    }
    manifest["root_hash"] = merkle_root(files)  # type: ignore[arg-type]

    _write_manifest(out_dir, manifest)
    _write_restore_doc(out_dir, manifest)
    return BackupResult(manifest, out_dir)


# ---- helpers ---------------------------------------------------------------


def _secrets_policy(include: bool) -> Dict[str, str]:
    return {
        "manifest": "fingerprints only (sha256 + length + location), never cleartext",
        "artifacts": "copied as-is — a backup that edits payloads is not a backup",
        "params_and_tags": (
            "copied as-is in the db snapshot; `mlflow-vault redact <backup>` rewrites them "
            "in place for sharing"
            if include
            else "ledger disabled for this backup"
        ),
        "storage_hygiene": "package contains a .gitignore that refuses to commit itself",
    }


def _limits(layout: storemod.StoreLayout) -> List[str]:
    out = []
    if layout.remote_uris():
        out.append(
            "remote artifact stores are recorded but not downloaded; back them up with the "
            "storage provider's own tool or a bucket sync"
        )
    out.append(
        "restoring into an existing store requires --force and replaces the tracking db; "
        "artifacts are merged, never deleted"
    )
    return out


def _inside(path: str, root: str) -> bool:
    try:
        return not os.path.relpath(os.path.abspath(path), os.path.abspath(root)).startswith("..")
    except ValueError:  # pragma: no cover - different drives on Windows
        return False


def _rel_or_none(path: str, root: str) -> Optional[str]:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")
    return None if rel.startswith("..") else rel


def _copy_verified(src: str, dst: str) -> Tuple[int, str]:
    """Copy and prove the destination hash equals the source hash."""
    import shutil

    shutil.copyfile(src, dst)
    src_digest = sha256_file(src)
    dst_digest = sha256_file(dst)
    if src_digest != dst_digest:
        raise OSError(f"corrupted copy: {src} -> {dst}")
    return os.path.getsize(dst), dst_digest


def _scan_db_secrets(db_path: str, ledger: SecretsLedger) -> None:
    con = dbmod.connect(db_path, readonly=True)
    try:
        for table, key_col in (("params", "key"), ("tags", "key")):
            if table not in dbmod.tables(db_path):
                continue
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            value_col = "value" if "value" in cols else None
            if not value_col:
                continue
            owner = "run_uuid" if "run_uuid" in cols else ("run_id" if "run_id" in cols else None)
            sql = f"SELECT {owner + ', ' if owner else ''}{key_col}, {value_col} FROM {table}"
            for row in con.execute(sql):
                if owner:
                    run_id, key, value = str(row[0]), str(row[1]), str(row[2])
                    where = f"{table}/{run_id}"
                else:  # pragma: no cover
                    key, value = str(row[0]), str(row[1])
                    where = table
                reasons = classify(key, value)
                if reasons:
                    ledger.add(where, key, value, reasons)
    finally:
        con.close()


def _write_manifest(out_dir: str, manifest: Dict[str, object]) -> str:
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        fh.write("\n")
    return path


def _write_restore_doc(out_dir: str, manifest: Dict[str, object]) -> str:
    store = manifest["store"]
    base = store.get("artifact_base_rel") or "(package root)/artifacts"
    body = f"""# Restoring this backup

Created {manifest['created_utc']} by mlflow-vault {__version__}.
Store kind: **{store['kind']}**. Root at backup time: `{store['root']}`.

## 1. Check it before you trust it

```bash
mlflow-vault verify .            # exit 0 means every byte matches the manifest
```

`verify` re-hashes every file and re-reads the database. If it reports a
mismatch it also names the exact file, so you can re-fetch just that one.

## 2. Restore into a new directory first

```bash
mlflow-vault restore . /path/to/new-store
```

Never restore over the live store on the first attempt: restoring into a fresh
directory is what proves the backup is complete, and it cannot destroy the
original while you find out.

## 3. Why the paths get rewritten

`artifact_uri` in MLflow is stored as an **absolute path**. This package was
taken from `{store['root']}`, so every run currently points inside that
directory. `restore` rewrites each one to the new root and recomputes the
logical digest with `artifact_uri` excluded — that is how it can claim
"nothing was lost" even though the strings changed.

Artifact base relative to the store: `{base}`

## 4. Then start MLflow against the new root

```bash
mlflow server --backend-store-uri sqlite:///new-store/mlflow.db \\
              --default-artifact-root new-store/mlruns
```

## If your artifacts lived in S3/GCS

They were **not** downloaded (see `warnings` in `manifest.json`). Restore the
database, keep `artifact_uri` untouched for those runs, and back the bucket up
with the provider's own tooling.
"""
    path = os.path.join(out_dir, "RESTORE.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return path


def load_manifest(backup_dir: str) -> Dict[str, object]:
    path = os.path.join(backup_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{backup_dir} has no manifest.json — not an mlflow-vault backup")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema {data.get('schema')!r}")
    return data
