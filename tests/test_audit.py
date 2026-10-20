"""The risk audit: every finding must be earned by evidence.

An audit that cries wolf gets ignored, so these tests pin down both directions:
each real hazard is reported, and the benign look-alikes are not.
"""

from __future__ import annotations

import os
import sqlite3

import pytest
from conftest import make_file_store, make_store

from mlflow_vault import backup as B
from mlflow_vault.audit import audit


def codes(rep) -> set[str]:
    return {r.code for r in rep.risks}


@pytest.fixture()
def store(tmp_path):
    return make_store(str(tmp_path / "store"), runs=2)


class TestNoBackup:
    def test_missing_backup_is_critical(self, store):
        rep = audit(store["root"])
        assert "no-backup-evidence" in codes(rep)
        assert rep.worst == "critical"

    def test_score_drops(self, store):
        assert audit(store["root"]).score < 60


class TestWal:
    def test_empty_wal_is_not_critical(self, store):
        # A cleanly-closed WAL-mode db leaves a 0-byte -wal plus a 32KB -shm.
        # Calling that critical is how a risk report loses its audience.
        con = sqlite3.connect(store["db"])
        con.execute("PRAGMA journal_mode=WAL")
        con.commit()
        con.close()
        rep = audit(store["root"])
        assert "wal-not-checkpointed" not in codes(rep)
        assert "wal_note" in rep.stats

    def test_nonempty_wal_is_critical(self, store):
        con = sqlite3.connect(store["db"])
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
        con.execute("INSERT INTO tags VALUES ('a','b','c')")
        con.commit()
        assert os.path.getsize(store["db"] + "-wal") > 0
        rep = audit(store["root"])
        assert "wal-not-checkpointed" in codes(rep)
        con.close()

    def test_delete_mode_store_has_no_wal_finding(self, store):
        rep = audit(store["root"])
        assert "wal-not-checkpointed" not in codes(rep)


class TestArtifacts:
    def test_dangling_artifacts_critical(self, store):
        import shutil

        shutil.rmtree(store["artifact_root"])
        rep = audit(store["root"])
        assert "dangling-artifacts" in codes(rep)

    def test_healthy_artifacts_not_flagged(self, store):
        rep = audit(store["root"])
        assert "dangling-artifacts" not in codes(rep)

    def test_absolute_uris_flagged(self, store):
        rep = audit(store["root"])
        assert "absolute-artifact-uris" in codes(rep)

    def test_remote_artifacts_medium(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, remote_runs=1)
        rep = audit(info["root"])
        assert "remote-artifacts" in codes(rep)
        assert {r.level for r in rep.risks if r.code == "remote-artifacts"} == {"medium"}


class TestSecrets:
    def test_credentials_flagged(self, store):
        rep = audit(store["root"])
        assert "credentials-in-metadata" in codes(rep)

    def test_clean_store_not_flagged(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, secrets=False)
        rep = audit(info["root"])
        assert "credentials-in-metadata" not in codes(rep)

    def test_finding_never_echoes_values(self, store):
        rep = audit(store["root"])
        blob = str(rep.as_dict())
        assert "sk-live" not in blob
        assert "hunter2" not in blob


class TestFileStoreAudit:
    def test_no_db_is_high(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"))
        rep = audit(info["root"])
        assert rep.kind == "file"
        assert "no-tracking-db" in codes(rep)


class TestBackupInteraction:
    def test_verified_backup_adds_ok_finding(self, store, tmp_path):
        res = B.backup(store["root"], str(tmp_path / "pkg"))
        rep = audit(store["root"], backup_dir=res.path)
        assert "backup-verified" in codes(rep)

    def test_unverified_backup_is_critical(self, store, tmp_path):
        res = B.backup(store["root"], str(tmp_path / "pkg"))
        os.remove(os.path.join(res.path, "db", "mlflow.db"))
        rep = audit(store["root"], backup_dir=res.path)
        assert "backup-unverified" in codes(rep)

    def test_db_only_backup_reported_separately(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        bare = tmp_path / "bare"
        bare.mkdir()
        import shutil

        shutil.copyfile(info["db"], str(bare / "mlflow.db"))
        con = sqlite3.connect(str(bare / "mlflow.db"))
        con.execute("UPDATE runs SET artifact_uri = 'file:" + str(bare / "gone").replace("\\", "/") + "'")
        con.commit()
        con.close()
        res = B.backup(str(bare), str(tmp_path / "pkg"))
        rep = audit(info["root"], backup_dir=res.path)
        # Both the cause and the symptom are reported; the cause comes first.
        assert "backup-db-only" in codes(rep)
        assert "backup-unverified" in codes(rep)


class TestScoring:
    def test_worst_level_wins(self, store):
        rep = audit(store["root"])
        assert rep.worst in {"critical", "high", "medium", "low", "ok"}

    def test_score_bounded(self, store):
        assert 0 <= audit(store["root"]).score <= 100

    def test_clean_store_scores_higher(self, tmp_path):
        dirty = make_store(str(tmp_path / "dirty"), runs=2, secrets=True)
        clean = make_store(str(tmp_path / "clean"), runs=2, secrets=False)
        a = audit(dirty["root"])
        b = audit(clean["root"])
        assert b.score > a.score

    def test_report_serialisable(self, store):
        import json

        json.dumps(audit(store["root"]).as_dict())

    def test_evidence_present_on_every_finding(self, store):
        for r in audit(store["root"]).risks:
            assert r.evidence, r.code
            assert r.mitigation, r.code
