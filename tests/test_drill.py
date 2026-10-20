"""The drill's own logic, tested without needing the MLflow SDK installed.

The SDK-dependent steps (build/readback) are exercised by the real drill run and
skipped here; what matters for the unit suite is that the damage helpers and the
ledger of steps behave correctly, because a drill that cannot fail is worthless.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import make_store

from mlflow_vault import drill as D


class TestStepBookkeeping:
    def test_empty_report_fails(self):
        rep = D.DrillReport(workdir="x")
        assert rep.ok is False  # no steps is not success

    def test_partial_report_fails(self):
        rep = D.DrillReport(workdir="x")
        rep.add(D.Step("build-store", True, "ok"))
        assert rep.ok is False  # 1 of 7 steps

    def test_all_steps_pass(self):
        rep = D.DrillReport(workdir="x")
        for name in D.STEP_NAMES:
            rep.add(D.Step(name, True, "ok"))
        assert rep.ok is True

    def test_one_failure_fails_the_drill(self):
        rep = D.DrillReport(workdir="x")
        for i, name in enumerate(D.STEP_NAMES):
            rep.add(D.Step(name, i != 3, "ok" if i != 3 else "boom"))
        assert rep.ok is False

    def test_summary_marks_each_step(self):
        rep = D.DrillReport(workdir="x")
        for name in D.STEP_NAMES:
            rep.add(D.Step(name, True, "ok"))
        text = rep.summarize()
        assert text.count("[PASS]") == len(D.STEP_NAMES)
        assert "DRILL PASSED" in text

    def test_serialisable(self):
        rep = D.DrillReport(workdir="x")
        for name in D.STEP_NAMES:
            rep.add(D.Step(name, True, "ok"))
        json.dumps(rep.as_dict())


class TestDamageHelpers:
    def _file(self, tmp_path, content=b"0123456789abcdef"):
        p = tmp_path / "f.bin"
        p.write_bytes(content)
        return str(p)

    def test_flip_changes_first_byte_only(self, tmp_path):
        p = self._file(tmp_path)
        before = open(p, "rb").read()
        D._apply_damage(p, "flip")
        after = open(p, "rb").read()
        assert len(before) == len(after)
        assert before[0] != after[0]
        assert before[1:] == after[1:]

    def test_flip_is_detected_by_hash(self, tmp_path):
        from mlflow_vault.hashing import sha256_file

        p = self._file(tmp_path)
        before = sha256_file(p)
        D._apply_damage(p, "flip")
        assert sha256_file(p) != before

    def test_delete_removes_file(self, tmp_path):
        p = self._file(tmp_path)
        D._apply_damage(p, "delete")
        assert not os.path.exists(p)

    def test_truncate_halves_and_floors_at_512(self, tmp_path):
        p = self._file(tmp_path, b"x" * 4096)
        D._apply_damage(p, "truncate")
        assert os.path.getsize(p) == 2048

    def test_truncate_small_file_keeps_512(self, tmp_path):
        p = self._file(tmp_path, b"x" * 100)
        D._apply_damage(p, "truncate")
        # Truncating below the 512-byte floor would make the file unopenable as a db
        # rather than merely damaged; the floor keeps the failure mode realistic.
        assert os.path.getsize(p) == 512


class TestTamperCaseSelection:
    def test_cases_target_real_files(self, tmp_path):
        from mlflow_vault import backup as B

        info = make_store(str(tmp_path / "store"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        cases = D._tamper_cases(res.path, str(tmp_path))
        names = [c[0] for c in cases]
        assert "artifact-byte-flip" in names
        assert "db-truncated" in names
        for _, path, how in cases:
            assert os.path.exists(path), path
            assert how in {"flip", "delete", "truncate"}

    def test_each_case_is_caught_by_verify(self, tmp_path):
        import shutil

        from mlflow_vault import backup as B
        from mlflow_vault import verify as V

        info = make_store(str(tmp_path / "store"), runs=2)
        res = B.backup(info["root"], str(tmp_path / "pkg"))
        for name, path, how in D._tamper_cases(res.path, str(tmp_path)):
            broken = str(tmp_path / f"t-{name}")
            shutil.copytree(res.path, broken)
            D._apply_damage(path.replace(res.path, broken, 1), how)
            rep = V.verify(broken)
            assert not rep.ok, f"{name} was not detected"


class TestReadbackGuard:
    def test_missing_db_raises_assertion(self, tmp_path):
        with pytest.raises(AssertionError):
            D.readback_via_sdk(str(tmp_path), expect_runs=1)


class TestDumpHelpers:
    def test_dump_json_is_lf_and_sorted(self, tmp_path):
        p = D._dump(str(tmp_path), "x.json", {"b": 1, "a": 2})
        raw = open(p, "rb").read()
        assert b"\r\n" not in raw
        assert raw.index(b'"a"') < raw.index(b'"b"')

    def test_dump_text_appends_newline(self, tmp_path):
        p = D._dump_text(str(tmp_path), "x.txt", "hello")
        assert open(p, encoding="utf-8").read() == "hello\n"
