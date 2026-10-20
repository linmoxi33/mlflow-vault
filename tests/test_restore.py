"""Restore: the part that either works on another machine or does not work.

The dominant failure is *silent*: the database opens, every run is listed, and
artifacts resolve to a dead absolute path. So the assertions here are about
whether the restored store is usable, not about whether files landed.
"""

from __future__ import annotations

import os
import sqlite3

import pytest
from conftest import make_file_store, make_store

from mlflow_vault import backup as B
from mlflow_vault import restore as R
from mlflow_vault import store as S


@pytest.fixture()
def pkg(tmp_path):
    info = make_store(str(tmp_path / "store"), runs=3)
    res = B.backup(info["root"], str(tmp_path / "pkg"))
    return {"store": info, "backup": res.path, "res": res}


class TestPlan:
    def test_plan_rewrites_every_local_run(self, pkg, tmp_path):
        p = R.plan(pkg["backup"], str(tmp_path / "restored"))
        assert p.rewrite_count == 3
        assert p.external == []

    def test_plan_keeps_base_leaf(self, pkg, tmp_path):
        # A store whose base is <root>/mlruns must restore to <target>/mlruns, not
        # to <target> — otherwise every URI points one level too high.
        p = R.plan(pkg["backup"], str(tmp_path / "restored"))
        assert p.artifact_base_new.replace("\\", "/").endswith("/mlruns")

    def test_plan_targets_new_root(self, pkg, tmp_path):
        p = R.plan(pkg["backup"], str(tmp_path / "restored"))
        target = str(tmp_path / "restored").replace("\\", "/")
        for uri in p.rewrites.values():
            assert uri.replace("\\", "/").startswith(target)

    def test_plan_leaves_remote_alone(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, remote_runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        p = R.plan(res.path, str(tmp_path / "restored"))
        assert len(p.external) == 1
        assert p.external[0].startswith("s3://")
        assert p.rewrite_count == 1

    def test_plan_is_readonly(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.plan(pkg["backup"], str(target))
        assert not target.exists()

    def test_dry_run_reports_without_writing(self, pkg, tmp_path):
        target = tmp_path / "restored"
        res = R.restore(pkg["backup"], str(target), dry_run=True)
        assert res.dry_run is True
        assert res.plan.rewrite_count == 3
        assert not target.exists()


class TestRestoreCore:
    def test_files_and_rows_restored(self, pkg, tmp_path):
        target = tmp_path / "restored"
        res = R.restore(pkg["backup"], str(target))
        assert res.ok, res.problems
        assert res.files_copied == 6
        assert res.db_rows_matched is True
        assert (target / "mlflow.db").is_file()

    def test_artifact_trees_land_under_target(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.restore(pkg["backup"], str(target))
        for run_id in pkg["store"]["run_ids"]:
            base = target / "mlruns" / run_id / "artifacts"
            assert (base / "weights" / "model.bin").is_file()
            assert (base / "metrics" / f"run{pkg['store']['run_ids'].index(run_id)}.csv").is_file()

    def test_artifact_bytes_identical(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.restore(pkg["backup"], str(target))
        from mlflow_vault.hashing import sha256_file

        for entry in pkg["res"].manifest["files"]:
            if entry["role"] != "artifact":
                continue
            dst = os.path.join(str(target), entry["path"].replace("/", os.sep))
            assert sha256_file(dst) == entry["sha256"]

    def test_uris_point_inside_target(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.restore(pkg["backup"], str(target))
        con = sqlite3.connect(str(target / "mlflow.db"))
        uris = [r[0] for r in con.execute("SELECT artifact_uri FROM runs")]
        con.close()
        for uri in uris:
            assert S.local_path(uri).replace("\\", "/").startswith(
                str(target).replace("\\", "/")
            )

    def test_experiment_locations_rewritten(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.restore(pkg["backup"], str(target))
        con = sqlite3.connect(str(target / "mlflow.db"))
        locs = {r[0] for r in con.execute("SELECT artifact_location FROM experiments")}
        con.close()
        for loc in locs:
            assert S.local_path(loc).replace("\\", "/").startswith(str(target).replace("\\", "/"))

    def test_report_written(self, pkg, tmp_path):
        target = tmp_path / "restored"
        R.restore(pkg["backup"], str(target))
        assert (target / "RESTORE-REPORT.json").is_file()

    def test_refuses_existing_db(self, pkg, tmp_path):
        target = tmp_path / "restored"
        target.mkdir()
        (target / "mlflow.db").write_bytes(b"x")
        with pytest.raises(FileExistsError):
            R.restore(pkg["backup"], str(target))

    def test_force_overwrites(self, pkg, tmp_path):
        target = tmp_path / "restored"
        target.mkdir()
        (target / "mlflow.db").write_bytes(b"x")
        res = R.restore(pkg["backup"], str(target), force=True)
        assert res.ok

    def test_original_store_untouched(self, pkg, tmp_path):
        from mlflow_vault.hashing import sha256_file

        before = sha256_file(pkg["store"]["db"])
        R.restore(pkg["backup"], str(tmp_path / "restored"))
        assert sha256_file(pkg["store"]["db"]) == before


class TestPostRestoreAudit:
    def test_no_dangling(self, pkg, tmp_path):
        res = R.restore(pkg["backup"], str(tmp_path / "restored"))
        post = R.audit_restored(res)
        assert post["dangling_count"] == 0
        assert post["artifact_uris_inside_target"] == post["artifact_uris_total"]

    def test_dangling_detected_when_artifacts_dropped(self, pkg, tmp_path):
        target = tmp_path / "restored"
        res = R.restore(pkg["backup"], str(target))
        import shutil

        shutil.rmtree(str(target / "mlruns" / pkg["store"]["run_ids"][0]))
        post = R.audit_restored(res)
        assert post["dangling_count"] == 1


class TestFileStoreRestore:
    def test_file_store_roundtrip(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        out = R.restore(res.path, str(tmp_path / "restored"))
        assert out.ok, out.problems
        assert (tmp_path / "restored" / "mlruns" / "1" / info["run_ids"][0] / "artifacts" / "out.txt").is_file()

    def test_file_store_bytes_match(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"), runs=1)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        R.restore(res.path, str(tmp_path / "restored"))
        src = os.path.join(info["mlruns"], "1", info["run_ids"][0], "artifacts", "out.txt")
        dst = os.path.join(str(tmp_path / "restored"), "mlruns", "1", info["run_ids"][0],
                           "artifacts", "out.txt")
        assert open(src, "rb").read() == open(dst, "rb").read()


class TestRestoreDetectsLoss:
    def test_tampered_package_fails_restore(self, pkg, tmp_path):
        import shutil

        bad = str(tmp_path / "bad")
        shutil.copytree(pkg["backup"], bad)
        art = next(f for f in pkg["res"].manifest["files"] if f["role"] == "artifact")
        with open(os.path.join(bad, art["path"].replace("/", os.sep)), "ab") as fh:
            fh.write(b"x")
        res = R.restore(bad, str(tmp_path / "restored"))
        assert res.ok is False
        assert any("corrupted" in p for p in res.problems)

    def test_missing_file_fails_restore(self, pkg, tmp_path):
        import shutil

        bad = str(tmp_path / "bad")
        shutil.copytree(pkg["backup"], bad)
        art = next(f for f in pkg["res"].manifest["files"] if f["role"] == "artifact")
        os.remove(os.path.join(bad, art["path"].replace("/", os.sep)))
        res = R.restore(bad, str(tmp_path / "restored"))
        assert res.ok is False
        assert any("missing in package" in p for p in res.problems)

    def test_row_mismatch_surfaces(self, pkg, tmp_path):
        import shutil

        bad = str(tmp_path / "bad")
        shutil.copytree(pkg["backup"], bad)
        db = os.path.join(bad, "db", "mlflow.db")
        con = sqlite3.connect(db)
        con.execute("DELETE FROM params")
        con.commit()
        con.close()
        res = R.restore(bad, str(tmp_path / "restored"))
        assert res.db_rows_matched is False
        assert any("differs" in p or "row total" in p for p in res.problems)


class TestNestedBaseShape:
    def test_whole_mlruns_tree_is_chosen_when_possible(self, tmp_path):
        """Archive the enclosing ``mlruns/`` rather than the deepest common prefix.

        If every run lives under ``mlruns/``, taking ``mlruns`` as the base means
        the package captures the experiment ``meta.yaml`` files and anything else
        sitting beside the run directories. Narrowing to ``mlruns/1`` would save a
        few bytes and quietly leave that context out of the backup.
        """
        root = str(tmp_path / "store")
        info = make_store(root, runs=2)
        con = sqlite3.connect(info["db"])
        con.execute(
            "UPDATE experiments SET artifact_location = ?",
            ("file:" + os.path.join(root, "mlruns", "1").replace("\\", "/"),),
        )
        con.execute(
            "UPDATE runs SET artifact_uri = ?",
            ("file:" + os.path.join(root, "mlruns", "1", "gone", "artifacts").replace("\\", "/"),),
        )
        con.commit()
        con.close()
        os.makedirs(os.path.join(root, "mlruns", "1", "gone", "artifacts"), exist_ok=True)
        with open(os.path.join(root, "mlruns", "1", "gone", "artifacts", "x.txt"), "wb") as fh:
            fh.write(b"payload")

        res = B.backup(root, str(tmp_path / "pkg"))
        assert res.manifest["store"]["artifact_base_rel"] == "mlruns"
        out = R.restore(res.path, str(tmp_path / "restored"))
        assert out.ok, out.problems
        assert (tmp_path / "restored" / "mlruns" / "1" / "gone" / "artifacts" / "x.txt").is_file()

    def test_nested_external_base_keeps_relative_shape(self, tmp_path):
        """A base outside the store root must round-trip with consistent paths.

        Two things have to hold, and only these two matter:

        * the base-relative *shape* survives, so ``<base>/<run>/artifacts`` becomes
          ``<target>/mlruns/<run>/artifacts`` rather than collapsing to one level;
        * the rewritten URIs resolve to files that actually exist.

        The base's own trailing directory names (``deep/exp`` here) are not part of
        the per-run path and are intentionally dropped — keeping them would invent
        directories on restore that never existed in the package.
        """
        root = str(tmp_path / "store")
        info = make_store(root, runs=2, artifact_root_inside=False)
        external = info["artifact_root"]
        os.makedirs(os.path.join(external, "deep", "exp"), exist_ok=True)
        con = sqlite3.connect(info["db"])
        for rid in info["run_ids"]:
            con.execute(
                "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                (
                    "file:" + os.path.join(external, "deep", "exp", rid, "artifacts").replace("\\", "/"),
                    rid,
                ),
            )
            d = os.path.join(external, "deep", "exp", rid, "artifacts")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "w.bin"), "wb") as fh:
                fh.write(b"weights")
        con.commit()
        con.close()

        res = B.backup(root, str(tmp_path / "pkg"))
        assert res.manifest["store"]["artifact_base_rel"] is None  # archived under artifacts/
        out = R.restore(res.path, str(tmp_path / "restored"))
        assert out.ok, out.problems
        assert out.plan.rewrite_count == 2

        post = R.audit_restored(out)
        assert post["dangling_count"] == 0
        for uri in out.plan.rewrites.values():
            p = S.local_path(uri)
            assert os.path.isdir(p or ""), uri
            assert os.path.isfile(os.path.join(p or "", "w.bin"))
