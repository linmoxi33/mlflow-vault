"""Verification: does it prove completeness, or only presence?

The distinction is the whole project. A verifier that returns "OK" for a
database-only package has exactly the failure mode it was hired to prevent, so
several tests here are written as "this package is *broken* and must be caught".
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
from conftest import make_file_store, make_store

from mlflow_vault import backup as B
from mlflow_vault import verify as V


@pytest.fixture()
def pkg(tmp_path):
    info = make_store(str(tmp_path / "store"), runs=3)
    res = B.backup(info["root"], str(tmp_path / "pkg"))
    return {"backup": res.path, "store": info, "res": res}


def codes(rep: V.VerifyReport) -> set[str]:
    return {f.code for f in rep.errors}


class TestCleanPackage:
    def test_clean_package_passes(self, pkg):
        rep = V.verify(pkg["backup"])
        assert rep.ok, [f.as_dict() for f in rep.errors]
        assert rep.checked_files == len(pkg["res"].manifest["files"])

    def test_exit_code_zero(self, pkg):
        assert V.verify(pkg["backup"]).exit_code() == 0

    def test_bytes_counted(self, pkg):
        rep = V.verify(pkg["backup"])
        assert rep.checked_bytes == pkg["res"].manifest["summary"]["total_bytes"]

    def test_artifact_coverage_complete(self, pkg):
        rep = V.verify(pkg["backup"])
        assert rep.artifact_coverage["local_runs"] == 3
        assert rep.artifact_coverage["missing_dirs"] == 0

    def test_logical_rows_match(self, pkg):
        rep = V.verify(pkg["backup"])
        assert rep.logical["__total_rows__"] == pkg["res"].manifest["db"]["logical"]["__total_rows__"]


class TestByteLevelDamage:
    def _clone(self, src, tmp_path, name="broken"):
        dst = os.path.join(str(tmp_path), name)
        shutil.copytree(src, dst)
        return dst

    def test_flipped_byte_is_caught(self, pkg, tmp_path):
        bad = self._clone(pkg["backup"], tmp_path)
        art = next(f for f in pkg["res"].manifest["files"] if f["role"] == "artifact")
        path = os.path.join(bad, art["path"].replace("/", os.sep))
        with open(path, "r+b") as fh:
            first = fh.read(1)
            fh.seek(0)
            fh.write(bytes([first[0] ^ 0xFF]))
        rep = V.verify(bad)
        assert not rep.ok
        assert "hash-mismatch" in codes(rep)
        assert any(f.path == art["path"] for f in rep.errors)

    def test_deleted_file_is_caught_and_named(self, pkg, tmp_path):
        bad = self._clone(pkg["backup"], tmp_path)
        art = next(f for f in pkg["res"].manifest["files"] if f["role"] == "artifact")
        os.remove(os.path.join(bad, art["path"].replace("/", os.sep)))
        rep = V.verify(bad)
        assert "file-missing" in codes(rep)
        assert any(f.path == art["path"] for f in rep.errors)

    def test_truncated_db_is_caught(self, pkg, tmp_path):
        bad = self._clone(pkg["backup"], tmp_path)
        db = os.path.join(bad, "db", "mlflow.db")
        size = os.path.getsize(db)
        with open(db, "r+b") as fh:
            fh.truncate(size // 2)
        rep = V.verify(bad)
        assert not rep.ok
        # Either the hash or the size check fires first; both are acceptable.
        assert codes(rep) & {"hash-mismatch", "size-mismatch", "db-unreadable", "integrity-check"}

    def test_size_only_mode_still_catches_truncation(self, pkg, tmp_path):
        bad = self._clone(pkg["backup"], tmp_path)
        db = os.path.join(bad, "db", "mlflow.db")
        with open(db, "r+b") as fh:
            fh.truncate(os.path.getsize(db) // 2)
        rep = V.verify(bad, deep=False)
        assert "size-mismatch" in codes(rep)

    def test_size_only_mode_misses_byte_flip(self, pkg, tmp_path):
        # Documents the trade-off honestly: --shallow is fast and weaker.
        bad = self._clone(pkg["backup"], tmp_path)
        entry = next(f for f in pkg["res"].manifest["files"] if f["role"] == "artifact")
        path = os.path.join(bad, entry["path"].replace("/", os.sep))
        with open(path, "r+b") as fh:
            fh.seek(0)
            first = fh.read(1)
            fh.seek(0)
            fh.write(bytes([first[0] ^ 0xFF]))
        assert V.verify(bad, deep=False).ok is True
        assert V.verify(bad, deep=True).ok is False


class TestStructuralDamage:
    def test_unexpected_file_warns(self, pkg, tmp_path):
        bad = os.path.join(str(tmp_path), "broken")
        shutil.copytree(pkg["backup"], bad)
        with open(os.path.join(bad, "extra.bin"), "wb") as fh:
            fh.write(b"x")
        rep = V.verify(bad)
        assert rep.ok  # not an error: presence of an extra file does not corrupt data
        assert any(f.code == "unexpected-file" for f in rep.warnings)

    def test_manifest_absent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            V.verify(str(tmp_path))

    def test_missing_db_snapshot_is_caught(self, pkg, tmp_path):
        bad = os.path.join(str(tmp_path), "broken")
        shutil.copytree(pkg["backup"], bad)
        os.remove(os.path.join(bad, "db", "mlflow.db"))
        rep = V.verify(bad)
        assert "file-missing" in codes(rep)


class TestSemanticDamage:
    """The failures that pass a byte check because the manifest was rebuilt."""

    def test_row_deleted_before_backup_is_caught(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        assert V.verify(res.path).ok

        import sqlite3

        db = os.path.join(res.path, "db", "mlflow.db")
        con = sqlite3.connect(db)
        con.execute("DELETE FROM metrics WHERE step = 2")
        con.commit()
        con.close()
        # Manifest now describes the damaged db, so only the logical comparison
        # (recorded digest vs fresh digest) can notice.
        rep = V.verify(res.path)
        assert "rows-changed" in codes(rep)
        assert "rowcount-mismatch" in codes(rep)

    def test_table_dropped_is_caught(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        import sqlite3

        db = os.path.join(res.path, "db", "mlflow.db")
        con = sqlite3.connect(db)
        con.execute("DROP TABLE params")
        con.commit()
        con.close()
        assert "table-lost" in codes(V.verify(res.path))

    def test_artifact_dir_removed_is_caught(self, tmp_path):
        """The signature failure: db intact, artifact tree gone."""
        info = make_store(str(tmp_path / "store"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        art_dirs = [
            p
            for p in (
                os.path.join(res.path, f["path"].replace("/", os.sep))
                for f in res.manifest["files"]
                if f["role"] == "artifact"
            )
        ]
        # Delete one run's whole artifact directory from the package.
        run_id = info["run_ids"][0]
        for p in list(art_dirs):
            if run_id in p:
                os.remove(p)
        rep = V.verify(res.path)
        assert not rep.ok
        # Deleting the files but leaving the directory behind is the subtler shape:
        # coverage must still refuse to call it complete.
        assert codes(rep) & {"artifact-dir-missing", "artifact-empty"}
        assert rep.artifact_coverage["missing_dirs"] + rep.artifact_coverage["empty_dirs"] == 1


class TestDbOnlyPackageIsRejected:
    """The realistic shape: someone carries mlflow.db to a new machine.

    On the old machine the absolute ``artifact_uri`` still resolves, so nothing
    looks wrong. Move the file alone and every artifact link dangles while the
    database stays perfectly healthy — which is why the check has to be about
    *coverage*, not about whether the db opens.
    """

    def _db_only_copy(self, info, tmp_path):
        root = tmp_path / "new-machine"
        root.mkdir()
        shutil.copyfile(info["db"], str(root / "mlflow.db"))
        import sqlite3

        con = sqlite3.connect(str(root / "mlflow.db"))
        con.execute(
            "UPDATE runs SET artifact_uri = ? ",
            ("file:" + str(root / "mlruns" / "vanished" / "artifacts").replace("\\", "/"),),
        )
        con.execute(
            "UPDATE experiments SET artifact_location = ?",
            ("file:" + str(root / "mlruns").replace("\\", "/"),),
        )
        con.commit()
        con.close()
        return root

    def test_package_without_artifacts_reports_zero_coverage(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        root = self._db_only_copy(info, tmp_path)
        res = B.backup(str(root), str(tmp_path / "pkg"))
        rep = V.verify(res.path)
        assert res.manifest["summary"]["has_artifacts"] is False
        assert rep.artifact_coverage["packaged_artifact_files"] == 0
        assert rep.artifact_coverage["missing_dirs"] == 2
        assert "artifact-dir-missing" in codes(rep)

    def test_audit_flags_db_only_backup(self, tmp_path):
        from mlflow_vault.audit import audit

        info = make_store(str(tmp_path / "store"), runs=2)
        root = self._db_only_copy(info, tmp_path)
        res = B.backup(str(root), str(tmp_path / "pkg"))
        rep = audit(info["root"], backup_dir=res.path)
        assert "backup-db-only" in {r.code for r in rep.risks}


class TestFileStoreVerify:
    def test_file_store_roundtrip(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        assert V.verify(res.path).ok

    def test_file_store_damage_caught(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        art = next(f for f in res.manifest["files"] if f["role"] == "artifact")
        with open(os.path.join(res.path, art["path"].replace("/", os.sep)), "ab") as fh:
            fh.write(b"x")
        assert V.verify(res.path).ok is False


class TestReporting:
    def test_summary_text(self, pkg):
        text = V.summarize(V.verify(pkg["backup"]))
        assert "OK" in text
        assert "files" in text

    def test_json_dump(self, pkg):
        rep = V.verify(pkg["backup"])
        data = json.loads(V.dump_json(rep))
        assert data["ok"] is True
        assert data["checked_files"] == rep.checked_files

    def test_errors_sorted_before_warnings_in_summary(self, pkg, tmp_path):
        bad = os.path.join(str(tmp_path), "broken")
        shutil.copytree(pkg["backup"], bad)
        open(os.path.join(bad, "extra.txt"), "wb").write(b"x")
        os.remove(os.path.join(bad, "db", "mlflow.db"))
        text = V.summarize(V.verify(bad))
        assert text.index("ERROR") < text.index("WARN")


class TestCompare:
    def test_identical_packages(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        a = B.backup(info["root"], str(tmp_path / "a"))
        b = B.backup(info["root"], str(tmp_path / "b"))
        diff = V.compare(a.path, b.path)
        assert diff["root_hash_equal"] is True
        assert diff["changed"] == []

    def test_changed_artifact_detected(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        a = B.backup(info["root"], str(tmp_path / "a"))
        path = os.path.join(info["artifact_root"], info["run_ids"][0], "artifacts", "weights", "model.bin")
        with open(path, "ab") as fh:
            fh.write(b"more")
        b = B.backup(info["root"], str(tmp_path / "b"))
        diff = V.compare(a.path, b.path)
        assert diff["root_hash_equal"] is False
        assert diff["changed"]
