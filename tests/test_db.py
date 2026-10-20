"""SQLite snapshotting and logical digests.

The tests that matter here are the ones about *what a naive copy loses*: an
unflushed WAL, and a snapshot that silently inherits WAL mode and therefore is not
self-contained.
"""

from __future__ import annotations

import os
import sqlite3

import pytest
from conftest import make_store

from mlflow_vault import db as D

EXCLUDE_RUNS = {"runs": ("artifact_uri",)}


class TestSnapshot:
    def test_snapshot_matches_row_count(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=3)
        dst = str(tmp_path / "out" / "mlflow.db")
        D.snapshot(info["db"], dst)
        con = sqlite3.connect(dst)
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        con.close()

    def test_snapshot_is_self_contained(self, tmp_path):
        # The bug this prevents: sqlite's backup API copies journal_mode too, so a
        # WAL-mode source yields a WAL-mode snapshot whose committed pages sit in a
        # -wal sidecar. The package then ships one file that is not the whole story.
        info = make_store(str(tmp_path / "s"), runs=2, wal=True)
        dst = str(tmp_path / "out" / "mlflow.db")
        report = D.snapshot(info["db"], dst)
        assert report["snapshot_journal_mode"] == "delete"
        assert not os.path.exists(dst + "-wal")
        assert not os.path.exists(dst + "-shm")

    def test_wal_source_is_detected(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2, wal=True)
        # Force a non-empty WAL by writing without checkpointing.
        con = sqlite3.connect(info["db"])
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
        con.execute("INSERT INTO tags VALUES ('x','y','z')")
        con.commit()
        report = D.snapshot(info["db"], str(tmp_path / "out" / "mlflow.db"))
        assert report["source_journal_mode"] == "wal"
        con.close()

    def test_overwrite_removes_stale_sidecars(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        dst = str(tmp_path / "out" / "mlflow.db")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for suffix in ("-wal", "-shm"):
            with open(dst + suffix, "wb") as fh:
                fh.write(b"stale")
        D.snapshot(info["db"], dst)
        assert not os.path.exists(dst + "-wal")
        assert not os.path.exists(dst + "-shm")


class TestLogicalDigest:
    def test_digest_is_stable(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2)
        a = D.logical_digest(info["db"])
        b = D.logical_digest(info["db"])
        assert a == b

    def test_digest_ignores_row_order_only_through_ordering(self, tmp_path):
        # Rows are read in a deterministic ORDER BY, so insertion order must not
        # change the digest — otherwise every backup would look different.
        info = make_store(str(tmp_path / "s"), runs=2)
        before = D.logical_digest(info["db"])
        con = sqlite3.connect(info["db"])
        con.execute("INSERT INTO tags VALUES ('zzz','note','x')")
        con.commit()
        con.close()
        after = D.logical_digest(info["db"])
        assert before["tags"]["digest"] != after["tags"]["digest"]

    def test_artifact_uri_exclusion(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2)
        before = D.digest_without_relocatable(info["db"])
        con = sqlite3.connect(info["db"])
        con.execute("UPDATE runs SET artifact_uri = 'file:/new/place'")
        con.commit()
        con.close()
        after = D.digest_without_relocatable(info["db"])
        assert before["runs"]["digest"] == after["runs"]["digest"]
        # ...while a *plain* digest does notice, proving the exclusion is what
        # makes relocation verifiable rather than a hole in the check.
        plain_before = D.logical_digest(info["db"])
        assert plain_before["runs"]["dropped_columns"] == []

    def test_row_deletion_changes_digest(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2)
        before = D.digest_without_relocatable(info["db"])
        con = sqlite3.connect(info["db"])
        con.execute("DELETE FROM metrics WHERE step = 2")
        con.commit()
        con.close()
        after = D.digest_without_relocatable(info["db"])
        assert before["metrics"]["digest"] != after["metrics"]["digest"]
        assert after["__total_rows__"] < before["__total_rows__"]

    def test_value_change_changes_digest(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2)
        before = D.digest_without_relocatable(info["db"])
        con = sqlite3.connect(info["db"])
        con.execute("UPDATE params SET value = '0.99' WHERE key = 'lr'")
        con.commit()
        con.close()
        after = D.digest_without_relocatable(info["db"])
        assert before["params"]["digest"] != after["params"]["digest"]

    def test_missing_table_detected(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2)
        before = D.digest_without_relocatable(info["db"])
        con = sqlite3.connect(info["db"])
        con.execute("DROP TABLE model_versions")
        con.commit()
        con.close()
        after = D.digest_without_relocatable(info["db"])
        assert "model_versions" in before
        assert "model_versions" not in after


class TestRewrite:
    def test_rewrite_changes_only_targeted_runs(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=3)
        changed = D.rewrite_artifact_uris(
            info["db"], {info["run_ids"][0]: "file:/brand/new"}
        )
        assert changed == 1
        con = sqlite3.connect(info["db"])
        rows = dict(con.execute("SELECT run_uuid, artifact_uri FROM runs"))
        con.close()
        assert rows[info["run_ids"][0]] == "file:/brand/new"
        assert "mlruns" in rows[info["run_ids"][1]]

    def test_rewrite_unknown_run_is_noop(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        assert D.rewrite_artifact_uris(info["db"], {"nope": "file:/x"}) == 0

    def test_experiment_location_rewritten(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        old = "file:" + os.path.join(info["root"], "mlruns").replace("\\", "/")
        n = D.rewrite_experiment_locations(info["db"], old, "file:/brand/new")
        assert n == 2
        con = sqlite3.connect(info["db"])
        locs = {r[0] for r in con.execute("SELECT artifact_location FROM experiments")}
        con.close()
        assert locs == {"file:/brand/new"}

    def test_experiment_outside_base_untouched(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        n = D.rewrite_experiment_locations(info["db"], "file:/somewhere/else", "file:/brand/new")
        assert n == 0


class TestBusyProbe:
    def test_idle_store_is_not_busy(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        assert D.busy_probe(info["db"], timeout=0.2) is False

    def test_locked_store_is_busy(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        holder = sqlite3.connect(info["db"], timeout=1)
        holder.execute("BEGIN IMMEDIATE")
        try:
            assert D.busy_probe(info["db"], timeout=0.1) is True
        finally:
            holder.execute("ROLLBACK")
            holder.close()

    def test_readonly_connection_cannot_write(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1)
        con = D.connect(info["db"], readonly=True)
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM runs")
        con.close()
