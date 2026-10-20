"""CLI behaviour: exit codes, encoding, and the contract a script depends on."""

from __future__ import annotations

import json
import os

import pytest
from conftest import make_store

from mlflow_vault.cli import main


@pytest.fixture()
def store(tmp_path):
    return make_store(str(tmp_path / "store"), runs=2)


class TestVersionAndHelp:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "mlflow-vault" in capsys.readouterr().out

    def test_help_lists_all_commands(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        for cmd in ("audit", "backup", "verify", "restore", "redact", "secrets", "compare", "drill"):
            assert cmd in out

    def test_no_args_is_an_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0


class TestAuditCommand:
    def test_human_output(self, store, capsys):
        assert main(["audit", store["root"]]) == 0
        out = capsys.readouterr().out
        assert "risk" in out
        assert "score" in out

    def test_json_output(self, store, capsys):
        assert main(["audit", store["root"], "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["kind"] == "sql+file"
        assert isinstance(data["risks"], list)

    def test_fail_on_threshold(self, store):
        # The fixture store has absolute URIs and credentials -> high.
        assert main(["audit", store["root"], "--fail-on", "high"]) == 2

    def test_fail_on_critical_without_backup(self, tmp_path):
        # No --backup given -> "no-backup-evidence" is critical by design, so even
        # the strictest threshold trips. This documents that the gate is honest
        # about the absence of a backup instead of assuming one exists.
        info = make_store(str(tmp_path / "s"), runs=1, secrets=False)
        assert main(["audit", info["root"], "--fail-on", "critical"]) == 2

    def test_fail_on_passes_when_a_clean_backup_exists(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=1, secrets=False)
        pkg = str(tmp_path / "pkg")
        assert main(["backup", info["root"], pkg]) == 0
        # Now everything measured is at most "high" (absolute URIs), so a critical
        # threshold legitimately passes.
        assert main(["audit", info["root"], "--backup", pkg, "--fail-on", "critical"]) == 0

    def test_missing_store_exit_2(self, tmp_path):
        assert main(["audit", str(tmp_path / "nope")]) == 2


class TestBackupCommand:
    def test_creates_package(self, store, tmp_path, capsys):
        out = str(tmp_path / "pkg")
        assert main(["backup", store["root"], out]) == 0
        text = capsys.readouterr().out
        assert "root hash" in text
        assert os.path.isfile(os.path.join(out, "manifest.json"))

    def test_json(self, store, tmp_path, capsys):
        out = str(tmp_path / "pkg")
        assert main(["backup", store["root"], out, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["files"] > 0
        assert len(data["root_hash"]) == 64

    def test_refuse_existing(self, store, tmp_path, capsys):
        out = tmp_path / "pkg"
        out.mkdir()
        (out / "x").write_text("y")
        assert main(["backup", store["root"], str(out)]) == 2
        assert "error" in capsys.readouterr().out.lower()

    def test_overwrite_flag(self, store, tmp_path):
        out = tmp_path / "pkg"
        out.mkdir()
        (out / "x").write_text("y")
        assert main(["backup", store["root"], str(out), "--overwrite"]) == 0

    def test_no_ledger_flag(self, store, tmp_path, capsys):
        out = str(tmp_path / "pkg")
        assert main(["backup", store["root"], out, "--no-secrets-ledger", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["secrets"] == 0


class TestVerifyCommand:
    def test_ok_exit_zero(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["verify", pkg]) == 0
        assert "OK" in capsys.readouterr().out

    def test_failure_exit_one(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        os.remove(os.path.join(pkg, "db", "mlflow.db"))
        capsys.readouterr()
        assert main(["verify", pkg]) == 1
        assert "FAILED" in capsys.readouterr().out

    def test_json(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["verify", pkg, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_shallow(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["verify", pkg, "--shallow"]) == 0
        capsys.readouterr()


class TestRestoreCommand:
    def test_restore_ok(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        target = str(tmp_path / "restored")
        capsys.readouterr()
        assert main(["restore", pkg, target]) == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "dangling  0" in out

    def test_dry_run_prints_plan_without_writing(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        target = tmp_path / "restored"
        capsys.readouterr()
        assert main(["restore", pkg, str(target), "--dry-run"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["rewrite_count"] == 2
        assert not target.exists()

    def test_existing_target_needs_force(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        target = tmp_path / "restored"
        main(["restore", pkg, str(target)])
        capsys.readouterr()
        assert main(["restore", pkg, str(target)]) == 2
        assert main(["restore", pkg, str(target), "--force"]) == 0

    def test_json(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["restore", pkg, str(tmp_path / "r"), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["post"]["dangling_count"] == 0


class TestRedactCommand:
    def test_redact(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        out = str(tmp_path / "shared")
        capsys.readouterr()
        assert main(["redact", pkg, "--out", out]) == 0
        assert "<redacted" not in capsys.readouterr().out.split("candidates")[0]
        assert os.path.isfile(os.path.join(out, "manifest.json"))

    def test_redact_then_verify(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        out = str(tmp_path / "shared")
        main(["redact", pkg, "--out", out])
        capsys.readouterr()
        assert main(["verify", out]) == 0

    def test_dry_run(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["redact", pkg, "--dry-run", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["candidates"] == 4
        assert data["redacted"] == 0


class TestSecretsCommand:
    def test_lists_fingerprints_only(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["secrets", pkg]) == 0
        out = capsys.readouterr().out
        assert "sha256:" in out
        assert "sk-live" not in out
        assert "hunter2" not in out

    def test_json(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["secrets", pkg, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["count"] == 4


class TestCompareCommand:
    def test_same_package(self, store, tmp_path, capsys):
        pkg = str(tmp_path / "pkg")
        main(["backup", store["root"], pkg])
        capsys.readouterr()
        assert main(["compare", pkg, pkg]) == 0
        assert "identical root hash: True" in capsys.readouterr().out

    def test_json(self, store, tmp_path, capsys):
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        main(["backup", store["root"], a])
        main(["backup", store["root"], b])
        capsys.readouterr()
        assert main(["compare", a, b, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["root_hash_equal"] is True


class TestEncoding:
    def test_chinese_path_does_not_crash(self, tmp_path, capsys):
        root = str(tmp_path / "存储" / "实验")
        info = make_store(root, runs=1)
        assert main(["audit", info["root"]]) == 0
        assert "risk" in capsys.readouterr().out

    def test_chinese_in_output_is_encodable(self, tmp_path, capsys):
        # Windows consoles default to cp936/cp1252; the CLI reconfigures to utf-8.
        root = str(tmp_path / "中文目录")
        info = make_store(root, runs=1)
        out = str(tmp_path / "备份包")
        assert main(["backup", info["root"], out]) == 0
        capsys.readouterr()


class TestErrorHandling:
    def test_bad_backup_dir(self, tmp_path, capsys):
        assert main(["verify", str(tmp_path / "nope")]) == 2
        assert "error" in capsys.readouterr().out.lower()

    def test_restore_from_non_package(self, tmp_path, capsys):
        d = tmp_path / "junk"
        d.mkdir()
        assert main(["restore", str(d), str(tmp_path / "r")]) == 2
