"""Redaction: turn a backup into something you can actually hand to someone.

Two different needs get confused constantly:

* **Restore-grade backup** — must contain the original bytes. If a param says
  ``sk-live-…``, the backup keeps it, because a backup that edits payloads is not
  a backup. Secrets are handled by *access control* (0600 ledger, .gitignore,
  not syncing it to a shared drive).
* **Share-grade artefact** — needed to attach a store to a bug report, send it to
  a vendor, or archive it off-site. Here the secrets must be gone, but the
  *evidence* that they were present must remain, or verification becomes
  meaningless ("the hash changed, is the data corrupt?").

``redact`` produces the second kind: values are replaced in place with
``<redacted:sha256:xxxxxxxxxxxx>``. The package still verifies — because the
manifest is recomputed to match the redacted bytes, and the *original* digests are
preserved under ``redaction.original_digests`` so you can prove later that the
only difference was the redaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

from . import db as dbmod
from .backup import _write_manifest, _write_restore_doc, load_manifest
from .hashing import merkle_root, sha256_file
from .secrets import classify, mask


def redact_backup(
    backup_dir: str,
    out_dir: Optional[str] = None,
    *,
    in_place: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Rewrite params/tags in the packaged db, replacing sensitive values."""
    backup_dir = os.path.abspath(backup_dir)
    manifest = load_manifest(backup_dir)

    db_info = manifest.get("db") or {}
    if not db_info.get("present"):
        raise ValueError("this package has no database snapshot; nothing to redact")

    snap_rel = str(db_info["archive_path"])

    if dry_run:
        # Scan the original, not a copy: a dry run must leave no directory behind.
        hits = _scan(os.path.join(backup_dir, snap_rel.replace("/", os.sep)))
        return {
            "backup": backup_dir,
            "work_dir": None,
            "candidates": len(hits),
            "redacted": 0,
            "by_table": {},
            "dry_run": True,
            "sample": [{"where": h[0], "key": h[1], "to": mask(h[2])} for h in hits[:5]],
        }

    if in_place:
        work_dir = backup_dir
    else:
        import shutil

        work_dir = os.path.abspath(out_dir or (backup_dir.rstrip(os.sep) + "-redacted"))
        if os.path.exists(work_dir):
            raise FileExistsError(f"{work_dir} exists; choose another --out")
        shutil.copytree(backup_dir, work_dir)

    snap = os.path.join(work_dir, snap_rel.replace("/", os.sep))

    hits = _scan(snap)
    report: Dict[str, object] = {
        "backup": backup_dir,
        "work_dir": work_dir,
        "candidates": len(hits),
        "redacted": 0,
        "by_table": {},
        "dry_run": False,
        "sample": [{"where": h[0], "key": h[1], "to": mask(h[2])} for h in hits[:5]],
    }
    if not hits:
        # Still purge: an untagged secret can hide in freed pages from earlier runs.
        _secure_purge(snap)
        _refresh_manifest(work_dir, manifest, snap_rel)
        report["manifest_updated"] = True
        report["plaintext_remaining"] = 0
        return report

    by_table: Dict[str, int] = {}
    con = dbmod.connect(snap)
    try:
        # The id column differs across MLflow versions and across stores migrated
        # from older ones: ``run_uuid`` in older schemas, ``run_id`` in newer ones,
        # and some stores carry both. Hardcoding either name makes redaction crash
        # with "no such column" on half the real-world stores.
        id_columns = {
            table: _id_column(con, table) for table in ("params", "tags")
        }
        for table, run_id, key, value in hits:
            id_col = id_columns.get(table)
            if not id_col:  # pragma: no cover - table vanished between scan and write
                continue
            cur = con.execute(
                f"UPDATE {table} SET value = ? WHERE {id_col} = ? AND key = ?",
                (mask(value), run_id, key),
            )
            if cur.rowcount and cur.rowcount > 0:
                report["redacted"] = int(report["redacted"]) + cur.rowcount
                by_table[table] = by_table.get(table, 0) + cur.rowcount
        con.commit()
    finally:
        con.close()
    report["by_table"] = by_table

    # 🔴 An UPDATE does not remove the old bytes. SQLite leaves the previous cell
    # payload in the page's free space (and in the freelist), so the "redacted"
    # database still contains every plaintext secret verbatim — `grep sk-live` finds
    # them. Rewriting the value only changes what a query returns.
    #
    # Whether that happens depends on ``secure_delete``, whose default is a
    # build-time choice (SQLITE_SECURE_DELETE). It cannot be assumed in either
    # direction, so _secure_purge sets it explicitly and also VACUUMs to clear
    # pages that were already free before this connection opened.
    report["purge"] = _secure_purge(snap)

    leaked = _scan_raw(snap, [v for _, _, _, v in hits])
    report["plaintext_remaining"] = len(leaked)
    report["raw_scan"] = (
        "clean: no original secret survives in the file"
        if not leaked
        else f"{len(leaked)} original value(s) still byte-scanable — redaction INCOMPLETE"
    )

    _refresh_manifest(work_dir, manifest, snap_rel)
    report["manifest_updated"] = True
    return report


def _secure_purge(snap: str) -> Dict[str, object]:
    """Erase freed cell contents, then rebuild the file.

    🔴 An UPDATE does not remove the old bytes: SQLite leaves the previous cell
    payload in the page's free space, so the "redacted" database still contains
    every plaintext secret verbatim — ``grep sk-live`` finds them. Rewriting the
    value only changes what a query returns.

    Whether that happens depends on ``secure_delete``, and **its default is a
    build-time choice**: some SQLite builds ship with ``SQLITE_SECURE_DELETE``
    compiled in (default ON), most distro/CPython builds do not. Measured on the
    same data: with ``secure_delete=OFF`` all 6 secrets survived a bare UPDATE;
    with ``ON``, zero. So the default cannot be relied on in either direction —
    it has to be set explicitly, which is what this does.

    ``VACUUM`` is still required on top of ``secure_delete``: secure_delete zeroes
    cells as they are freed, but pages that were *already* on the freelist before
    this connection opened keep their old images.
    """
    con = dbmod.connect(snap)
    try:
        default_mode = con.execute("PRAGMA secure_delete").fetchone()[0]
        con.execute("PRAGMA secure_delete=ON")
        effective = con.execute("PRAGMA secure_delete").fetchone()[0]
        con.commit()
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()
    # The VACUUM may leave a journal behind; the package must stay one file.
    for suffix in ("-wal", "-shm", "-journal"):
        p = snap + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:  # pragma: no cover - locked on Windows
                pass
    return {
        "secure_delete_default": int(default_mode),
        "secure_delete_effective": int(effective),
        "vacuum": True,
    }


def _scan_raw(snap: str, originals: List[str]) -> List[str]:
    """Grep the raw bytes, because a query-level check cannot see freed pages."""
    if not originals:
        return []
    with open(snap, "rb") as fh:
        blob = fh.read()
    found: List[str] = []
    for value in originals:
        needle = value.encode("utf-8", "ignore")
        if len(needle) >= 6 and needle in blob:
            found.append(value[:4] + "…")  # never echo the secret itself
    return found


def _id_column(con, table: str) -> Optional[str]:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    for candidate in ("run_uuid", "run_id"):
        if candidate in cols:
            return candidate
    return None


def _scan(snap: str) -> List[Tuple[str, str, str, str]]:
    out: List[Tuple[str, str, str, str]] = []
    con = dbmod.connect(snap, readonly=True)
    try:
        tables = dbmod.tables(snap)
        for table in ("params", "tags"):
            if table not in tables:
                continue
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            id_col = "run_id" if "run_id" in cols else ("run_uuid" if "run_uuid" in cols else None)
            if not id_col or "key" not in cols or "value" not in cols:
                continue
            for run_id, key, value in con.execute(
                f"SELECT {id_col}, key, value FROM {table}"
            ):
                value = "" if value is None else str(value)
                if classify(str(key), value):
                    out.append((table, str(run_id), str(key), value))
    finally:
        con.close()
    return out


def _refresh_manifest(work_dir: str, manifest: Dict[str, object], snap_rel: str) -> None:
    """Recompute every digest so the redacted package verifies as a package.

    Two subtleties that each cost a failed verification:

    * the db's size changes (VACUUM rebuilds it), so size must be refreshed too;
    * ``secrets.json`` gets a ``redaction`` note appended *after* the ledger was
      hashed at backup time. Its digest therefore has to be recomputed here, or the
      redacted package fails its own verify on a file the tool itself edited.
    """
    original = {
        "snapshot_sha256": (manifest.get("db") or {}).get("sha256"),
        "logical": (manifest.get("db") or {}).get("logical"),
        "root_hash": manifest.get("root_hash"),
    }
    files = manifest.get("files") or []
    snap_abs = os.path.join(work_dir, snap_rel.replace("/", os.sep))

    # Stamp the ledger note first, so the digest computed below covers final bytes.
    ledger = os.path.join(work_dir, "secrets.json")
    if os.path.isfile(ledger):
        with open(ledger, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["redaction"] = "values in the db have been replaced; this ledger is historical"
        with open(ledger, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")

    for entry in files:
        path = str(entry["path"])
        abs_p = os.path.join(work_dir, path.replace("/", os.sep))
        if path == snap_rel:
            entry["redacted"] = True
        if os.path.isfile(abs_p):
            entry["sha256"] = sha256_file(abs_p)
            entry["size"] = os.path.getsize(abs_p)

    db_info = manifest.setdefault("db", {})
    db_info["sha256"] = sha256_file(snap_abs)  # type: ignore[index]
    db_info["size"] = os.path.getsize(snap_abs)  # type: ignore[index]
    db_info["logical"] = dbmod.digest_without_relocatable(snap_abs)  # type: ignore[index]
    manifest["files"] = files
    manifest["root_hash"] = merkle_root(files)  # type: ignore[arg-type]
    manifest["redaction"] = {
        "applied": True,
        "method": "values replaced with <redacted:sha256:prefix12>, then secure_delete + VACUUM",
        "original_digests": original,
        "note": (
            "original digests are kept only so a reviewer can prove the sole difference "
            "between the two packages is redaction — they contain no plaintext"
        ),
    }
    manifest["secrets"] = {
        "policy": {
            "params_and_tags": "redacted in place and VACUUM-ed; originals are not recoverable",
            "artifacts": "NOT redacted — check artifacts/ for embedded credentials manually",
        },
        "count": int((manifest.get("secrets") or {}).get("count", 0)),
        "values_stored_in_manifest": False,
    }
    _write_manifest(work_dir, manifest)
    _write_restore_doc(work_dir, manifest)


def _scan_raw(snap: str, originals: List[str]) -> List[str]:
    """Grep the raw bytes, because a query-level check cannot see freed pages."""
    if not originals:
        return []
    with open(snap, "rb") as fh:
        blob = fh.read()
    found: List[str] = []
    for value in originals:
        needle = value.encode("utf-8", "ignore")
        if len(needle) >= 6 and needle in blob:
            found.append(value[:4] + "…")  # never echo the secret itself
    return found


def ledger_of(backup_dir: str) -> Dict[str, object]:
    path = os.path.join(backup_dir, "secrets.json")
    if not os.path.isfile(path):
        return {"count": 0, "entries": [], "note": "no ledger in this package"}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["ledger_of", "redact_backup", "sqlite3"]
