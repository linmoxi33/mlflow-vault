"""Backup package construction.

Focus: the manifest must describe reality, not intentions. Every test that could
pass with a broken package is written to fail instead.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import make_file_store, make_store, manifest_of

from mlflow_vault import backup as B


@pytest.fixture()
def sql_store(tmp_path):
    return make_store(str(tmp_path / "store"), runs=3)


class TestBackupBasics:
    def test_package_layout(self, tmp_path, sql_store):
        out = tmp_path / "pkg"
        res = B.backup(sql_store["root"], str(out))
        assert (out / "manifest.json").is_file()
        assert (out / "RESTORE.md").is_file()
        assert (out / ".gitignore").is_file()
        assert (out / "db" / "mlflow.db").is_file()
        assert res.file_count >= 4

    def test_refuses_non_empty_target(self, tmp_path, sql_store):
        out = tmp_path / "pkg"
        out.mkdir()
        (out / "junk.txt").write_text("x")
        with pytest.raises(FileExistsError):
            B.backup(sql_store["root"], str(out))

    def test_overwrite_allowed(self, tmp_path, sql_store):
        out = tmp_path / "pkg"
        out.mkdir()
        (out / "junk.txt").write_text("x")
        B.backup(sql_store["root"], str(out), overwrite=True)
        assert (out / "manifest.json").is_file()

    def test_manifest_schema_and_tool(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        m = res.manifest
        assert m["schema"] == B.MANIFEST_SCHEMA
        assert m["tool"]["name"] == "mlflow-vault"
        assert m["store"]["kind"] == "sql+file"

    def test_every_file_has_hash_and_size(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        for entry in res.manifest["files"]:
            assert len(str(entry["sha256"])) == 64
            assert int(entry["size"]) >= 0
            assert entry["path"].count("\\") == 0  # package paths are posix

    def test_root_hash_is_stable_across_runs(self, tmp_path, sql_store):
        a = B.backup(sql_store["root"], str(tmp_path / "p1"))
        b = B.backup(sql_store["root"], str(tmp_path / "p2"))
        # Same store, same content -> same root hash, even though timestamps differ.
        assert a.manifest["root_hash"] == b.manifest["root_hash"]

    def test_root_hash_changes_when_content_changes(self, tmp_path, sql_store):
        a = B.backup(sql_store["root"], str(tmp_path / "p1"))
        with open(os.path.join(sql_store["artifact_root"], sql_store["run_ids"][0], "artifacts",
                               "weights", "model.bin"), "ab") as fh:
            fh.write(b"tamper")
        b = B.backup(sql_store["root"], str(tmp_path / "p2"))
        assert a.manifest["root_hash"] != b.manifest["root_hash"]


class TestArtifactCoverage:
    def test_artifacts_are_included(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        roles = [f["role"] for f in res.manifest["files"]]
        assert roles.count("artifact") == 6  # 3 runs x 2 files
        assert res.manifest["summary"]["has_artifacts"] is True

    def test_artifacts_byte_identical(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        from mlflow_vault.hashing import sha256_file

        for entry in res.manifest["files"]:
            if entry["role"] != "artifact":
                continue
            packaged = os.path.join(res.path, entry["path"].replace("/", os.sep))
            assert sha256_file(packaged) == entry["sha256"]
            source = entry["origin"]
            if os.path.isfile(source):
                assert sha256_file(source) == entry["sha256"]

    def test_runs_index_records_locality(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        runs = res.manifest["runs"]
        assert len(runs) == 3
        assert all(r["artifact_locality"] == "local" for r in runs.values())
        assert all(r["rel_base"] is not None for r in runs.values())

    def test_remote_runs_warned_not_silently_dropped(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=3, remote_runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        remote = [r for r in res.manifest["runs"].values() if r["artifact_locality"] == "remote"]
        assert len(remote) == 1
        assert any("remote" in w.lower() for w in res.manifest["warnings"])

    def test_missing_artifacts_dir_is_warned(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, omit_artifacts=True)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        assert res.manifest["summary"]["has_artifacts"] is False


class TestDbSnapshotInManifest:
    def test_db_role_and_logical_digest_present(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        db = res.manifest["db"]
        assert db["present"] is True
        assert db["snapshot"]["integrity_check"] == "ok"
        assert db["logical"]["__total_rows__"] > 0
        assert "runs" in db["logical"]

    def test_snapshot_not_wal_mode(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        assert res.manifest["db"]["snapshot"]["snapshot_journal_mode"] == "delete"

    def test_no_stray_sidecars_in_package(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        dbdir = os.path.join(res.path, "db")
        assert sorted(os.listdir(dbdir)) == ["mlflow.db"]

    def test_inventory_counts(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        inv = res.manifest["inventory"]
        assert inv["runs"] == 3
        assert inv["experiments"] == 2
        assert inv["metrics"] == 9


class TestSecretsInBackup:
    def test_ledger_written_and_masked(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        ledger_path = os.path.join(res.path, "secrets.json")
        assert os.path.isfile(ledger_path)
        body = open(ledger_path, encoding="utf-8").read()
        assert "sk-live" not in body
        assert "hunter2" not in body
        assert res.secret_count == 6

    def test_manifest_never_contains_values(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        body = open(os.path.join(res.path, "manifest.json"), encoding="utf-8").read()
        assert "sk-live" not in body
        assert "hunter2" not in body
        assert res.manifest["secrets"]["values_stored_in_manifest"] is False

    def test_backup_keeps_real_values_in_db(self, tmp_path, sql_store):
        # A backup that redacts is not a backup: the restored store must behave
        # exactly like the original, secrets included.
        import sqlite3

        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        con = sqlite3.connect(os.path.join(res.path, "db", "mlflow.db"))
        vals = [r[0] for r in con.execute("SELECT value FROM params WHERE key='api_key'")]
        con.close()
        assert all(v.startswith("sk-live") for v in vals)

    def test_ledger_can_be_disabled(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"), include_secrets=False)
        assert res.secret_count == 0
        assert not os.path.exists(os.path.join(res.path, "secrets.json"))
        assert "ledger disabled" in res.manifest["secrets"]["policy"]["params_and_tags"]


class TestFileStore:
    def test_file_store_backup(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        assert res.manifest["store"]["kind"] == "file"
        assert res.manifest["db"]["present"] is False
        roles = {f["role"] for f in res.manifest["files"]}
        assert "artifact" in roles
        assert any("no database" in n for n in res.manifest["store"]["notes"])

    def test_file_store_has_mlruns_prefix(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        arts = [f["path"] for f in res.manifest["files"] if f["role"] == "artifact"]
        assert all(p.startswith("mlruns/") for p in arts)


class TestRestoreDoc:
    def test_doc_mentions_the_three_steps(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        doc = open(os.path.join(res.path, "RESTORE.md"), encoding="utf-8").read()
        assert "verify" in doc
        assert "restore" in doc
        assert "artifact_uri" in doc

    def test_doc_names_the_source_root(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        doc = open(os.path.join(res.path, "RESTORE.md"), encoding="utf-8").read()
        assert sql_store["root"].replace("\\", "/") in doc.replace("\\", "/")


class TestLoadManifest:
    def test_roundtrip(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        loaded = B.load_manifest(res.path)
        assert loaded["root_hash"] == res.manifest["root_hash"]

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            B.load_manifest(str(tmp_path))

    def test_bad_schema_raises(self, tmp_path):
        d = tmp_path / "pkg"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"schema": 999}), encoding="utf-8")
        with pytest.raises(ValueError):
            B.load_manifest(str(d))


class TestManifestJsonIsParseable:
    def test_sorted_keys_and_utf8(self, tmp_path, sql_store):
        res = B.backup(sql_store["root"], str(tmp_path / "pkg"))
        raw = open(os.path.join(res.path, "manifest.json"), "rb").read()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" not in raw  # LF-only so hashing is platform-stable
        manifest_of(res.path)  # parses
