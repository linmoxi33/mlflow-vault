"""URI handling and store discovery.

Every test here is a regression test for a bug that the drill produced when the
tool was first run against a real MLflow store.
"""

from __future__ import annotations

import os

import pytest
from conftest import make_file_store, make_store

from mlflow_vault import store as S


class TestPosix:
    def test_single_slash_file_uri_is_stripped(self):
        # Real MLflow writes file:E:/x, not file:///E:/x. Treating the single-slash
        # form as opaque made the artifact tree invisible -> db-only backup that
        # verified clean. This is the regression test for that bug.
        assert S._posix("file:E:/data/mlruns/1") == "E:/data/mlruns/1"

    def test_triple_slash_file_uri_keeps_windows_drive(self):
        assert S._posix("file:///E:/data/mlruns") == "E:/data/mlruns"

    def test_triple_slash_posix_path(self):
        assert S._posix("file:///data/mlruns") == "/data/mlruns"

    def test_backslashes_normalised(self):
        assert S._posix("E:\\data\\mlruns\\1") == "E:/data/mlruns/1"

    def test_trailing_slash_removed(self):
        assert S._posix("E:/data/mlruns/") == "E:/data/mlruns"


class TestRemote:
    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/path",
            "gs://bucket/path",
            "wasbs://c@a.blob.core.windows.net/x",
            "hdfs://nn/path",
            "mlflow-artifacts:/1/abc/artifacts",
            "https://example.com/x",
        ],
    )
    def test_remote_schemes(self, uri):
        assert S.is_remote(uri) is True

    @pytest.mark.parametrize("uri", ["file:E:/x", "E:/x", "/x", "", "file:///x"])
    def test_local_paths(self, uri):
        assert S.is_remote(uri) is False

    def test_file_scheme_is_not_remote_even_with_slashes(self):
        # A naive prefix check against REMOTE_SCHEMES misses this because "file://"
        # is not in the list — but the guard must also not treat "file:" as remote.
        assert S.is_remote("file:///tmp/mlruns") is False


class TestLocalPath:
    def test_strips_scheme(self):
        assert S.local_path("file:E:/a/b").replace("\\", "/") == "E:/a/b"

    def test_remote_returns_none(self):
        assert S.local_path("s3://b/k") is None

    def test_existing_dir_resolves(self, tmp_path):
        d = tmp_path / "art"
        d.mkdir()
        assert os.path.isdir(S.local_path("file:" + str(d).replace("\\", "/")) or "")


class TestDetect:
    def test_sql_store(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2)
        layout = S.detect(info["root"])
        assert layout.kind == "sql+file"
        assert len(layout.runs) == 2
        assert layout.db_path.endswith("mlflow.db")

    def test_file_store(self, tmp_path):
        info = make_file_store(str(tmp_path / "fs"))
        layout = S.detect(info["root"])
        assert layout.kind == "file"
        assert layout.db_path is None
        assert len(layout.runs) == 2

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            S.detect(str(tmp_path / "nope"))

    def test_not_a_store_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        (d / "readme.txt").write_text("hi")
        with pytest.raises(ValueError):
            S.detect(str(d))

    def test_experiments_are_read(self, tmp_path):
        info = make_store(str(tmp_path / "store"))
        layout = S.detect(info["root"])
        assert set(layout.experiments) == {"1", "2"}


class TestArtifactBase:
    def test_inside_root_is_rooted_at_mlruns(self, tmp_path):
        info = make_store(str(tmp_path / "store"), artifact_root_inside=True)
        layout = S.detect(info["root"])
        assert layout.artifact_base() == os.path.join(info["root"], "mlruns").replace("\\", "/")

    def test_outside_root_is_still_found(self, tmp_path):
        info = make_store(str(tmp_path / "store"), artifact_root_inside=False)
        layout = S.detect(info["root"])
        base = layout.artifact_base()
        assert base and "external-artifacts" in base

    def test_remote_only_yields_none(self, tmp_path):
        info = make_store(str(tmp_path / "store"), runs=2, remote_runs=2)
        layout = S.detect(info["root"])
        assert layout.local_uris() == []
        assert layout.artifact_base() is None


class TestSplitJoin:
    def test_roundtrip(self):
        base = "E:/data/mlruns"
        uri = "file:E:/data/mlruns/1/abc/artifacts"
        rel = S.split_uri(uri, base)
        assert rel == "1/abc/artifacts"
        assert S.join_uri("E:/new/mlruns", rel) == "E:/new/mlruns/1/abc/artifacts"

    def test_outside_base_returns_none(self):
        assert S.split_uri("E:/other/x", "E:/data/mlruns") is None

    def test_prefix_lookalike_is_not_inside(self):
        # "mlruns2" must not be treated as being under "mlruns".
        assert S.split_uri("E:/data/mlruns2/x", "E:/data/mlruns") is None

    def test_base_itself_is_empty_rel(self):
        assert S.split_uri("E:/data/mlruns", "E:/data/mlruns") == ""
        assert S.join_uri("E:/new", "") == "E:/new"
