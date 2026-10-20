""" Discovery and inspection of an MLflow tracking store.

Two storage shapes are supported, because they fail differently:

``file store``   ``mlruns/<exp>/<run>/{artifacts,metrics,params,tags}`` plus
                 ``mlruns/<exp>/meta.yaml`` — no database at all.

``sql store``    a SQLite ``mlflow.db`` holding experiments/runs/metrics/params,
                 with each run's ``artifact_uri`` pointing *somewhere else*.

The second shape is where data quietly disappears: the database is the thing
people back up, yet it is also the only piece that stays healthy while every
artifact it references goes missing.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REMOTE_SCHEMES = (
    "s3://",
    "gs://",
    "gcs://",
    "wasbs://",
    "wasb://",
    "abfs://",
    "adl://",
    "hdfs://",
    "ftp://",
    "sftp://",
    "http://",
    "https://",
    "mlflow-artifacts:",
)

DB_CANDIDATES = ("mlflow.db", "mlflow.sqlite", "mlflow.sqlite3", "db.sqlite", "tracking.db")


def _posix(path: str) -> str:
    """Normalise a URI/path to forward slashes so Windows and POSIX agree.

    MLflow writes local artifact URIs as ``file:E:/x/mlruns/1`` — a *single* slash,
    not the RFC form ``file:///``. Treating that as a remote scheme, or forgetting
    to strip it, makes the artifact tree look absent and produces a database-only
    backup that verifies clean. Both bugs were found by running the drill.
    """
    p = path.replace("\\", "/")
    if p[:5].lower() == "file:":
        p = p[5:]
        while p.startswith("//"):
            p = p[1:]
        # file:///C:/x -> /C:/x -> C:/x   |   file:///home/x -> /home/x
        if len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
    return p.rstrip("/")


def is_remote(uri: str) -> bool:
    if not uri:
        return False
    if _posix(uri).lower().startswith("file:"):
        return False
    return any(_posix(uri).lower().startswith(s) for s in REMOTE_SCHEMES)


def local_path(uri: str) -> Optional[str]:
    """Filesystem path behind a local artifact URI, or None when it is remote.

    Every existence check must go through this. Testing ``os.path.isdir("file:E:/x")``
    returns False on every platform, which turns a perfectly healthy store into a
    "3/3 artifact directories are gone" critical finding — a false alarm of exactly
    the kind that teaches people to ignore the report.
    """
    if not uri or is_remote(uri):
        return None
    p = _posix(uri)
    return p.replace("/", os.sep) if os.sep != "/" else p


@dataclass
class RunRef:
    run_id: str
    experiment_id: str
    artifact_uri: str
    status: str = ""
    lifecycle_stage: str = ""


@dataclass
class StoreLayout:
    """What we found on disk, and what we are therefore able to protect."""

    root: str
    db_path: Optional[str] = None
    mlruns_dir: Optional[str] = None
    runs: List[RunRef] = field(default_factory=list)
    experiments: Dict[str, str] = field(default_factory=dict)  # id -> artifact_location
    notes: List[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.db_path and self.mlruns_dir:
            return "sql+file"
        if self.db_path:
            return "sql"
        return "file"

    # ---- artifact topology -------------------------------------------------

    def local_uris(self) -> List[str]:
        return [r.artifact_uri for r in self.runs if r.artifact_uri and not is_remote(r.artifact_uri)]

    def remote_uris(self) -> List[str]:
        return sorted({r.artifact_uri for r in self.runs if r.artifact_uri and is_remote(r.artifact_uri)})

    def artifact_base(self) -> Optional[str]:
        """Longest common ancestor of every *local* artifact URI.

        This is the string we must rewrite on restore. It is computed from actual
        run data rather than assumed to be ``<root>/mlruns``, because users point
        experiments at arbitrary directories all the time.
        """
        uris = [_posix(u) for u in self.local_uris()]
        if not uris:
            return None
        if self.mlruns_dir:
            mlr = _posix(self.mlruns_dir)
            if all(u == mlr or u.startswith(mlr + "/") for u in uris):
                return mlr
        parts = [u.split("/") for u in uris]
        prefix: List[str] = []
        for column in zip(*parts):
            if len(set(column)) == 1:
                prefix.append(column[0])
            else:
                break
        base = "/".join(prefix)
        return base.rstrip("/") or None


def detect(root: str) -> StoreLayout:
    """Locate the tracking DB and/or the ``mlruns`` tree under ``root``."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no such store root: {root}")

    layout = StoreLayout(root=root)

    # A path pointing straight at the db is a legitimate target too.
    if os.path.isfile(root):
        layout.db_path, layout.root = root, os.path.dirname(root)
        root = layout.root

    for name in DB_CANDIDATES:
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            layout.db_path = cand
            break
    if layout.db_path is None:
        for name in os.listdir(root):
            if name.endswith((".db", ".sqlite", ".sqlite3")) and os.path.isfile(
                os.path.join(root, name)
            ):
                layout.db_path = os.path.join(root, name)
                break

    mlruns = os.path.join(root, "mlruns")
    if os.path.isdir(mlruns):
        layout.mlruns_dir = mlruns
    elif os.path.isdir(os.path.join(root, "artifacts")):
        layout.mlruns_dir = None

    if layout.db_path is None and layout.mlruns_dir is None:
        raise ValueError(
            f"{root} does not look like an MLflow store "
            "(no tracking db and no mlruns/ directory)"
        )

    if layout.db_path:
        layout.runs, layout.experiments = _read_sql(layout)
    elif layout.mlruns_dir:
        layout.runs, layout.experiments = _read_file_store(layout)
        layout.notes.append("file store: experiments are directories, no database present")
    return layout


def _connect_ro(db_path: str) -> sqlite3.Connection:
    uri = "file:" + _posix(db_path) + "?mode=ro"
    if not _posix(db_path).startswith("/"):
        uri = "file:///" + _posix(db_path) + "?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:  # pragma: no cover - odd path shapes
        return sqlite3.connect(db_path)


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
    except sqlite3.DatabaseError:
        return []


def _read_sql(layout: StoreLayout) -> Tuple[List[RunRef], Dict[str, str]]:
    assert layout.db_path
    con = _connect_ro(layout.db_path)
    try:
        cols = _table_columns(con, "runs")
        if not cols:
            layout.notes.append("tracking db has no `runs` table — empty or uninitialised store")
            return [], {}
        id_col = "run_uuid" if "run_uuid" in cols else "run_id"
        uri_col = "artifact_uri" if "artifact_uri" in cols else None
        if uri_col is None:
            layout.notes.append("runs table has no artifact_uri column")
        exp_col = "experiment_id" if "experiment_id" in cols else None
        status_col = "status" if "status" in cols else None
        stage_col = "lifecycle_stage" if "lifecycle_stage" in cols else None

        select = [id_col] + ([exp_col] if exp_col else []) + ([uri_col] if uri_col else [])
        for extra in (status_col, stage_col):
            if extra:
                select.append(extra)
        rows = con.execute(f"SELECT {', '.join(select)} FROM runs").fetchall()

        # Map column name -> value once, rather than tracking a positional cursor:
        # the column set varies by MLflow version, and an index that drifts by one
        # silently attributes the status to the artifact_uri.
        runs: List[RunRef] = []
        for row in rows:
            rec = dict(zip(select, row))
            runs.append(
                RunRef(
                    run_id=str(rec[id_col]),
                    experiment_id=str(rec[exp_col]) if exp_col else "0",
                    artifact_uri=str(rec[uri_col] or "") if uri_col else "",
                    status=str(rec[status_col] or "") if status_col else "",
                    lifecycle_stage=str(rec[stage_col] or "") if stage_col else "",
                )
            )

        exps: Dict[str, str] = {}
        if _table_columns(con, "experiments"):
            for eid, loc in con.execute("SELECT experiment_id, artifact_location FROM experiments"):
                exps[str(eid)] = str(loc or "")
        return runs, exps
    finally:
        con.close()


def _read_file_store(layout: StoreLayout) -> Tuple[List[RunRef], Dict[str, str]]:
    """Walk ``mlruns/<exp>/<run>/`` — every run's artifacts live beside its metrics."""
    assert layout.mlruns_dir
    runs: List[RunRef] = []
    exps: Dict[str, str] = {}
    for exp in sorted(os.listdir(layout.mlruns_dir)):
        exp_dir = os.path.join(layout.mlruns_dir, exp)
        if not os.path.isdir(exp_dir):
            continue
        exps[exp] = exp_dir
        for run in sorted(os.listdir(exp_dir)):
            run_dir = os.path.join(exp_dir, run)
            if not os.path.isdir(run_dir) or run == "meta.yaml":
                continue
            runs.append(
                RunRef(run_id=run, experiment_id=exp, artifact_uri=os.path.join(run_dir, "artifacts"))
            )
    return runs, exps


# ---- URI rewriting ---------------------------------------------------------


def split_uri(uri: str, base: str) -> Optional[str]:
    """Return the base-relative POSIX path of ``uri``, or None if outside ``base``."""
    u, b = _posix(uri), _posix(base)
    if u == b:
        return ""
    if u.startswith(b + "/"):
        return u[len(b) + 1 :]
    return None


def join_uri(base: str, rel: str) -> str:
    if not rel:
        return base
    return _posix(base) + "/" + rel


def inventory(db_path: Optional[str], layout: StoreLayout) -> Dict[str, object]:
    """Counts used by ``verify`` to prove *semantic* completeness, not just bytes."""
    inv: Dict[str, object] = {
        "experiments": len(layout.experiments),
        "runs": len(layout.runs),
    }
    if db_path and os.path.isfile(db_path):
        con = _connect_ro(db_path)
        try:
            for table in ("metrics", "params", "tags", "registered_models", "model_versions"):
                if _table_columns(con, table):
                    inv[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            con.close()
    return inv
