"""Redaction: a shareable package must be verifiable *and* free of secrets."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from conftest import make_store

from mlflow_vault import backup as B
from mlflow_vault import redact as RD
from mlflow_vault import verify as V


@pytest.fixture()
def pkg(tmp_path):
    info = make_store(str(tmp_path / "store"), runs=3, secrets=True)
    res = B.backup(info["root"], str(tmp_path / "pkg"))
    return {"store": info, "backup": res.path}


def values(db: str, table: str, key: str):
    con = sqlite3.connect(db)
    rows = [r[0] for r in con.execute(f"SELECT value FROM {table} WHERE key = ?", (key,))]
    con.close()
    return rows


class TestRedactProducesCleanCopy:
    def test_secrets_gone(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        res = RD.redact_backup(pkg["backup"], out)
        assert res["redacted"] == 6
        db = os.path.join(out, "db", "mlflow.db")
        assert all(v.startswith("<redacted:sha256:") for v in values(db, "params", "api_key"))
        assert all(v.startswith("<redacted:sha256:") for v in values(db, "tags", "db_password"))

    def test_no_plaintext_anywhere_in_package(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        for dirpath, _, filenames in os.walk(out):
            for name in filenames:
                p = os.path.join(dirpath, name)
                with open(p, "rb") as fh:
                    blob = fh.read()
                assert b"sk-live" not in blob, p
                assert b"hunter2" not in blob, p

    def test_innocuous_values_preserved(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        db = os.path.join(out, "db", "mlflow.db")
        assert values(db, "params", "lr") == ["0.01", "0.02", "0.03"]
        assert values(db, "tags", "model_family") == ["resnet", "resnet", "resnet"]

    def test_original_untouched(self, pkg):
        before = open(os.path.join(pkg["backup"], "db", "mlflow.db"), "rb").read()
        RD.redact_backup(pkg["backup"], os.path.join(os.path.dirname(pkg["backup"]), "shared"))
        after = open(os.path.join(pkg["backup"], "db", "mlflow.db"), "rb").read()
        assert before == after

    def test_rows_otherwise_intact(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        db = os.path.join(out, "db", "mlflow.db")
        con = sqlite3.connect(db)
        runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        metrics = con.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        con.close()
        assert (runs, metrics) == (3, 9)


class TestRedactedPackageStillVerifies:
    """The point of a *provable* redaction: it must not look like corruption."""

    def test_verify_passes_after_redaction(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        rep = V.verify(out)
        assert rep.ok, [f.as_dict() for f in rep.errors]

    def test_manifest_records_original_digests(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        m = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
        original = m["redaction"]["original_digests"]
        assert len(original["snapshot_sha256"]) == 64
        assert original["snapshot_sha256"] != m["db"]["sha256"]
        assert original["root_hash"] != m["root_hash"]

    def test_redaction_marker_on_file_entry(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        m = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
        db_entry = next(f for f in m["files"] if f["role"] == "db-snapshot")
        assert db_entry.get("redacted") is True

    def test_artifacts_not_redacted_and_said_so(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        RD.redact_backup(pkg["backup"], out)
        m = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
        assert "NOT redacted" in m["secrets"]["policy"]["artifacts"]


class TestRedactModes:
    def test_dry_run_reports_only(self, pkg, tmp_path):
        out = tmp_path / "shared"
        res = RD.redact_backup(pkg["backup"], str(out), dry_run=True)
        assert res["candidates"] == 6
        assert res["redacted"] == 0
        # A dry run must not leave a half-built copy that later fails to verify.
        assert not out.exists()

    def test_in_place(self, pkg):
        RD.redact_backup(pkg["backup"], in_place=True)
        db = os.path.join(pkg["backup"], "db", "mlflow.db")
        assert all(v.startswith("<redacted") for v in values(db, "params", "api_key"))
        assert V.verify(pkg["backup"]).ok

    def test_refuses_existing_out(self, pkg, tmp_path):
        out = tmp_path / "shared"
        out.mkdir()
        with pytest.raises(FileExistsError):
            RD.redact_backup(pkg["backup"], str(out))

    def test_nothing_to_redact(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, secrets=False)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        out = str(tmp_path / "shared")
        report = RD.redact_backup(res.path, out)
        assert report["candidates"] == 0
        assert report["redacted"] == 0

    def test_masks_are_deterministic_across_packages(self, pkg, tmp_path):
        a = str(tmp_path / "a")
        b = str(tmp_path / "b")
        RD.redact_backup(pkg["backup"], a)
        RD.redact_backup(pkg["backup"], b)
        assert values(os.path.join(a, "db", "mlflow.db"), "params", "api_key") == values(
            os.path.join(b, "db", "mlflow.db"), "params", "api_key"
        )


class TestSecurePurgeIsLoadBearing:
    """A query-level check is not enough, and this proves it by construction.

    ``UPDATE params SET value='<redacted…>'`` changes what a query returns while
    leaving the previous cell bytes in the page's free space — *if* ``secure_delete``
    is off. And that default is a **build-time** choice: SQLite can be compiled with
    ``SQLITE_SECURE_DELETE``, and some distributions do exactly that.

    This cost a CI failure to learn: the original version of this test asserted the
    residue unconditionally and passed on Windows (default off) while failing on the
    Ubuntu runner (default on). The lesson is not "the test was too strict" — it is
    that **the default must never be relied on**, which is why ``_secure_purge`` sets
    ``secure_delete`` explicitly instead of trusting the build.
    """

    def _bare_update(self, snap, secure_delete: str):
        """Apply the mask with a bare UPDATE and an explicit secure_delete mode."""
        from mlflow_vault import db as dbmod

        hits = RD._scan(snap)
        con = dbmod.connect(snap)
        con.execute(f"PRAGMA secure_delete={secure_delete}")
        for table, run_id, key, _value in hits:
            id_col = RD._id_column(con, table)
            con.execute(
                f"UPDATE {table} SET value = ? WHERE {id_col} = ? AND key = ?",
                ("<redacted>", run_id, key),
            )
        con.commit()
        con.close()
        return hits

    def test_update_alone_leaves_plaintext_when_secure_delete_is_off(self, pkg):
        snap = os.path.join(pkg["backup"], "db", "mlflow.db")
        hits = self._bare_update(snap, "OFF")
        originals = [v for _, _, _, v in hits]
        assert RD._scan_raw(snap, originals), (
            "with secure_delete=OFF the old cell bytes must still be in the file; "
            "if this fails the premise of the purge has changed"
        )

    def test_update_alone_is_clean_when_secure_delete_is_on(self, pkg):
        # The other half of the same fact, and the reason the mode is forced.
        snap = os.path.join(pkg["backup"], "db", "mlflow.db")
        hits = self._bare_update(snap, "ON")
        assert RD._scan_raw(snap, [v for _, _, _, v in hits]) == []

    def test_purge_forces_the_mode_regardless_of_default(self, pkg):
        snap = os.path.join(pkg["backup"], "db", "mlflow.db")
        hits = self._bare_update(snap, "OFF")
        info = RD._secure_purge(snap)
        assert info["secure_delete_effective"] == 1
        assert RD._scan_raw(snap, [v for _, _, _, v in hits]) == []

    def test_real_redaction_reports_zero_remaining(self, pkg, tmp_path):
        out = str(tmp_path / "shared")
        report = RD.redact_backup(pkg["backup"], out)
        assert report["plaintext_remaining"] == 0
        assert "clean" in report["raw_scan"]
        assert report["purge"]["vacuum"] is True

    def test_purge_leaves_no_sidecars(self, pkg):
        snap = os.path.join(pkg["backup"], "db", "mlflow.db")
        RD._secure_purge(snap)
        for suffix in ("-wal", "-shm", "-journal"):
            assert not os.path.exists(snap + suffix)

    def test_purged_db_still_queryable(self, pkg):
        from mlflow_vault import db as dbmod

        snap = os.path.join(pkg["backup"], "db", "mlflow.db")
        RD._secure_purge(snap)
        con = dbmod.connect(snap, readonly=True)
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
        con.close()


class TestSecretsCommand:
    def test_ledger_listing(self, pkg):
        led = RD.ledger_of(pkg["backup"])
        assert led["count"] == 6
        assert all("sha256" in e for e in led["entries"])
        assert "sk-live" not in json.dumps(led)

    def test_missing_ledger(self, tmp_path):
        assert RD.ledger_of(str(tmp_path))["count"] == 0
