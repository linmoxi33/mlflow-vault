"""SQLite snapshotting, logical digests and URI rewriting.

Why not ``cp mlflow.db``: MLflow runs SQLite in WAL mode, so the newest commits
live in ``mlflow.db-wal``. Copying only ``mlflow.db`` from a live store silently
drops them, and copying *while a writer is active* can capture a torn page image
that opens fine but reports ``database disk image is malformed`` later. The
supported answer is the online backup API, which yields a single consistent file
with the WAL already folded in.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, Iterable, List, Optional, Sequence

from .hashing import digest_of

# Tables MLflow may create; anything else found is still digested.
KNOWN_TABLES = (
    "experiments",
    "runs",
    "metrics",
    "latest_metrics",
    "params",
    "tags",
    "registered_models",
    "registered_model_tags",
    "model_versions",
    "model_version_tags",
    "datasets",
    "inputs",
    "input_tags",
    "dataset_inputs",
    "logged_models",
    "logged_model_tags",
    "trace_info",
    "trace_tags",
    "experiment_tags",
    "run_aliases",
    "model_version_aliases",
)

# Columns whose value legitimately changes when a store is relocated.
RELOCATABLE_COLUMNS = {
    "runs": ("artifact_uri",),
    "experiments": ("artifact_location",),
}

WAL_SUFFIXES = ("-wal", "-shm", "-journal")


def connect(db_path: str, readonly: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    p = db_path.replace("\\", "/")
    if readonly:
        if p.startswith("/"):
            uri = f"file:{p}?mode=ro"
        else:  # Windows drive path needs the third slash
            uri = f"file:///{p}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=timeout)
    else:
        con = sqlite3.connect(db_path, timeout=timeout)
    con.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return con


def snapshot(src_db: str, dst_db: str) -> Dict[str, object]:
    """Consistent copy of a live SQLite database, WAL included.

    The snapshot is deliberately converted to ``journal_mode=DELETE``: the backup
    API copies the source's journal mode, so a WAL-mode store yields a WAL-mode
    snapshot whose *committed* pages can live in a ``-wal`` sidecar. A backup that
    only consists of the ``.db`` file would then be subtly incomplete, and anyone
    who later copies just that one file re-introduces the very bug this tool
    exists to catch. One self-contained file is worth more than a faithful mode.
    """
    os.makedirs(os.path.dirname(dst_db) or ".", exist_ok=True)
    for stale in (dst_db, *(dst_db + sfx for sfx in WAL_SUFFIXES)):
        if os.path.exists(stale):
            os.remove(stale)
    src = connect(src_db, readonly=True)
    try:
        dst = sqlite3.connect(dst_db)
        try:
            src.backup(dst)  # page-level, transactional, WAL-aware
            dst.commit()
            dst.execute("PRAGMA journal_mode=DELETE")  # fold everything into the file
            dst.commit()
            ok = dst.execute("PRAGMA integrity_check").fetchone()
        finally:
            dst.close()
    finally:
        src.close()

    # journal_mode=DELETE leaves nothing behind; clean up if the engine disagreed.
    for stale in tuple(dst_db + s for s in WAL_SUFFIXES):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:  # pragma: no cover - locked by another reader
                pass

    wal = [s for s in WAL_SUFFIXES if os.path.exists(src_db + s)]
    main_size = os.path.getsize(src_db)
    wal_size = os.path.getsize(src_db + "-wal") if os.path.exists(src_db + "-wal") else 0
    return {
        "source_size": main_size,
        "source_wal_size": wal_size,
        "source_sidecars": wal,
        "source_journal_mode": _journal_mode(src_db),
        "snapshot_size": os.path.getsize(dst_db),
        "snapshot_journal_mode": _journal_mode(dst_db),
        "integrity_check": (ok[0] if ok else "unknown"),
        # If the WAL was bigger than the DB, a naive `cp` would have lost most of it.
        "wal_was_meaningful": wal_size > 0,
    }


def _journal_mode(db_path: str) -> str:
    con = connect(db_path, readonly=True)
    try:
        return str(con.execute("PRAGMA journal_mode").fetchone()[0])
    except sqlite3.DatabaseError:  # pragma: no cover
        return "unknown"
    finally:
        con.close()


def tables(db_path: str) -> List[str]:
    con = connect(db_path, readonly=True)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return sorted(r[0] for r in rows)
    finally:
        con.close()


def _rows(con: sqlite3.Connection, table: str, order_by: Sequence[str]) -> List[list]:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    order = [c for c in order_by if c in cols]
    sql = f"SELECT * FROM {table}"
    if order:
        sql += " ORDER BY " + ", ".join(order)
    return [list(r) for r in con.execute(sql)], cols  # type: ignore[return-value]


def logical_digest(
    db_path: str,
    exclude: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict[str, object]:
    """Per-table digest of row *content*, independent of file layout.

    ``exclude`` drops columns from the digest — this is what makes
    "database survived the move" checkable even though ``artifact_uri`` had to
    be rewritten. Without it every restore would look like data corruption.
    """
    exclude = {k: set(v) for k, v in (exclude or {}).items()}
    con = connect(db_path, readonly=True)
    try:
        out: Dict[str, object] = {}
        total_rows = 0
        for table in tables(db_path):
            rows, cols = _rows(con, table, order_by=("run_uuid", "run_id", "key", "step", "timestamp"))
            drop = {c for c in exclude.get(table, ()) if c in cols}
            keep_idx = [i for i, c in enumerate(cols) if c not in drop]
            trimmed = [[row[i] for i in keep_idx] for row in rows]
            out[table] = {
                "rows": len(trimmed),
                "columns": [cols[i] for i in keep_idx],
                "dropped_columns": sorted(drop),
                "digest": digest_of(trimmed),
            }
            total_rows += len(trimmed)
        out["__total_rows__"] = total_rows
        return out
    finally:
        con.close()


def digest_without_relocatable(db_path: str) -> Dict[str, object]:
    return logical_digest(db_path, exclude=RELOCATABLE_COLUMNS)


def rewrite_artifact_uris(db_path: str, mapping: Dict[str, str]) -> int:
    """Point each run at its new artifact location. Returns rows touched."""
    if not mapping:
        return 0
    con = connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)")]
        if "artifact_uri" not in cols:
            return 0
        id_col = "run_uuid" if "run_uuid" in cols else "run_id"
        changed = 0
        for run_id, uri in mapping.items():
            cur = con.execute(
                f"UPDATE runs SET artifact_uri = ? WHERE {id_col} = ?", (uri, run_id)
            )
            changed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        # Experiments can carry an absolute artifact_location too; repoint the ones
        # that still equal the old base so new runs do not land in a dead directory.
        return changed
    finally:
        con.commit()
        con.close()


def _norm_local(value: str) -> str:
    """Normalise a local path/URI for comparison: drop ``file:``, unify slashes.

    Experiment ``artifact_location`` is stored *with* the ``file:`` scheme while
    the manifest's ``artifact_base`` is stored without it. Comparing the raw
    strings therefore never matches, and the experiment locations silently keep
    pointing at the old machine after a restore.
    """
    v = value.replace("\\", "/")
    if v[:5].lower() == "file:":
        v = v[5:]
        while v.startswith("//"):
            v = v[1:]
        if len(v) > 2 and v[0] == "/" and v[2] == ":":
            v = v[1:]
    return v.rstrip("/")


def rewrite_experiment_locations(db_path: str, old_base: str, new_base: str) -> int:
    """Repoint experiments whose artifact_location sits under the old base.

    The new value is written *with* a ``file:`` scheme when the original had one.
    Experiments created by the MLflow SDK carry ``file:E:/…``; writing a bare path
    would still resolve, but it makes the column inconsistent with every other row
    and confuses later prefix matching.
    """
    con = connect(db_path)
    try:
        if "experiments" not in tables(db_path):
            return 0
        ob, nb = _norm_local(old_base), _norm_local(new_base)
        if not ob:
            return 0
        changed = 0
        for eid, loc in con.execute("SELECT experiment_id, artifact_location FROM experiments"):
            if not loc:
                continue
            raw = str(loc)
            had_scheme = raw[:5].lower() == "file:"
            norm = _norm_local(raw)
            if norm == ob or norm.startswith(ob + "/"):
                tail = norm[len(ob) :]
                new_value = ("file:" + nb + tail) if had_scheme else (nb + tail)
                con.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                    (new_value, eid),
                )
                changed += 1
        return changed
    finally:
        con.commit()
        con.close()


def wal_report(db_path: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for suf in WAL_SUFFIXES:
        p = db_path + suf
        if os.path.exists(p):
            out[suf] = os.path.getsize(p)
    out["sidecars_present"] = sorted(out)
    return out


def busy_probe(db_path: str, timeout: float = 1.0) -> bool:
    """True if the store is currently being written to (a live writer blocks us)."""
    con = connect(db_path, timeout=timeout)
    try:
        t0 = time.time()
        try:
            con.execute("BEGIN IMMEDIATE")  # fails fast when another writer holds the lock
            con.execute("ROLLBACK")
            return False
        except sqlite3.OperationalError:
            return time.time() - t0 >= 0  # locked, by definition
    finally:
        con.close()
