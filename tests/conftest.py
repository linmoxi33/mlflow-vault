"""Synthetic MLflow stores, built without the MLflow SDK.

The test suite must run on a bare interpreter, so fixtures speak MLflow's on-disk
format directly. Kept deliberately close to what the SDK actually writes — that
similarity is the whole point of the fixtures.
"""

from __future__ import annotations

import json
import os
import sqlite3

# The subset of MLflow's schema the tool touches. Created with plain SQL so the
# tests do not depend on alembic migrations or the mlflow package.
SCHEMA = """
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    artifact_location TEXT,
    lifecycle_stage TEXT,
    creation_time INTEGER,
    last_update_time INTEGER
);
CREATE TABLE runs (
    run_uuid TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    name TEXT,
    source_type TEXT,
    source_name TEXT,
    entry_point_name TEXT,
    user_id TEXT,
    status TEXT,
    start_time INTEGER,
    end_time INTEGER,
    source_version TEXT,
    lifecycle_stage TEXT,
    artifact_uri TEXT,
    run_id TEXT
);
CREATE TABLE metrics (
    run_uuid TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL,
    timestamp INTEGER,
    step INTEGER,
    is_nan INTEGER DEFAULT 0,
    PRIMARY KEY (run_uuid, key, timestamp, step)
);
CREATE TABLE params (
    run_uuid TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (run_uuid, key)
);
CREATE TABLE tags (
    run_uuid TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (run_uuid, key)
);
CREATE TABLE registered_models (
    name TEXT PRIMARY KEY,
    creation_time INTEGER,
    last_updated_time INTEGER,
    description TEXT
);
CREATE TABLE model_versions (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    creation_time INTEGER,
    last_updated_time INTEGER,
    description TEXT,
    user_id TEXT,
    current_stage TEXT,
    source TEXT,
    run_id TEXT,
    status TEXT,
    status_message TEXT,
    run_link TEXT,
    storage_location TEXT,
    PRIMARY KEY (name, version)
);
"""


def write_file(path: str, data: bytes) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def make_store(
    root: str,
    *,
    runs: int = 2,
    wal: bool = False,
    secrets: bool = True,
    remote_runs: int = 0,
    artifact_root_inside: bool = True,
    omit_artifacts: bool = False,
) -> dict:
    """Build a store and return a description of what was put in it."""
    os.makedirs(root, exist_ok=True)
    db_path = os.path.join(root, "mlflow.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    art_root = os.path.join(root, "mlruns") if artifact_root_inside else os.path.join(
        os.path.dirname(root), "external-artifacts"
    )
    os.makedirs(art_root, exist_ok=True)

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
        ("1", "demo", "file:" + art_root.replace("\\", "/"), "active", 1, 2),
    )
    con.execute(
        "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
        ("2", "Default", "file:" + art_root.replace("\\", "/"), "active", 1, 2),
    )

    run_ids = []
    for i in range(runs):
        rid = f"{i:032x}"
        run_ids.append(rid)
        uri = (
            f"s3://bucket/runs/{rid}"
            if i >= runs - remote_runs
            else "file:" + os.path.join(art_root, rid, "artifacts").replace("\\", "/")
        )
        con.execute(
            "INSERT INTO runs (run_uuid, experiment_id, name, status, lifecycle_stage, "
            "artifact_uri, run_id) VALUES (?,?,?,?,?,?,?)",
            (rid, "1", f"run-{i}", "FINISHED", "active", uri, rid),
        )
        con.execute(
            "INSERT INTO params VALUES (?,?,?)", (rid, "lr", str(0.01 * (i + 1)))
        )
        if secrets:
            con.execute(
                "INSERT INTO params VALUES (?,?,?)", (rid, "api_key", f"sk-live-{i}" + "A" * 20)
            )
            con.execute(
                "INSERT INTO tags VALUES (?,?,?)", (rid, "db_password", f"hunter2-{i}")
            )
        con.execute("INSERT INTO tags VALUES (?,?,?)", (rid, "model_family", "resnet"))
        for step in range(3):
            con.execute(
                "INSERT INTO metrics VALUES (?,?,?,?,?,0)",
                (rid, "loss", 1.0 / (step + 1), 1000 + step, step),
            )
        if not omit_artifacts and i < runs - remote_runs:
            base = os.path.join(art_root, rid, "artifacts")
            write_file(os.path.join(base, "metrics", f"run{i}.csv"), b"epoch,loss\n0,1.0\n")
            write_file(os.path.join(base, "weights", "model.bin"), bytes(range(256)) * 4)
    con.commit()
    con.close()
    return {
        "root": root,
        "db": db_path,
        "artifact_root": art_root,
        "run_ids": run_ids,
        "wal": wal,
    }


def make_file_store(root: str, runs: int = 2) -> dict:
    """An ``mlruns/``-only store: no tracking database at all."""
    mlruns = os.path.join(root, "mlruns")
    os.makedirs(mlruns, exist_ok=True)
    write_file(os.path.join(mlruns, "1", "meta.yaml"), b"name: demo\n")
    run_ids = []
    for i in range(runs):
        rid = f"fr{i:030x}"
        run_ids.append(rid)
        base = os.path.join(mlruns, "1", rid)
        write_file(os.path.join(base, "meta.yaml"), f"run_id: {rid}\n".encode())
        write_file(os.path.join(base, "params", "lr"), b"0.01")
        write_file(os.path.join(base, "metrics", "loss"), b"1700000000 1.0 0\n")
        write_file(os.path.join(base, "artifacts", "out.txt"), b"artifact payload\n")
    return {"root": root, "mlruns": mlruns, "run_ids": run_ids}


def manifest_of(backup_dir: str) -> dict:
    with open(os.path.join(backup_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)
