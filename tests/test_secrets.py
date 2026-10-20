"""Sensitive-value detection and the redaction policy.

Detection is deliberately conservative in the "looks like a key" direction and
strict in the "looks like a key *shape*" direction: a false positive costs a
masked param, a false negative costs a leaked credential.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import make_store

from mlflow_vault import secrets as SEC


class TestKeyDetection:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "apiKey",
            "API-KEY",
            "apikey",
            "openai_api_key",
            "secret",
            "client_secret",
            "password",
            "passwd",
            "pwd",
            "token",
            "hf_token",
            "gh_token",
            "access_key",
            "aws_access_key_id",
            "aws_secret_access_key",
            "private_key",
            "connection_string",
            "conn_str",
            "authorization",
            "Bearer",
        ],
    )
    def test_sensitive_keys(self, key):
        assert SEC.is_sensitive_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "lr",
            "batch_size",
            "model_family",
            "hidden_units",
            "note",
            "description",
            "dataset",
            "run_name",
            "epoch",
            "loss",
            "author",
        ],
    )
    def test_innocuous_keys(self, key):
        assert SEC.is_sensitive_key(key) is False

    def test_empty_key(self):
        assert SEC.is_sensitive_key("") is False

    def test_key_with_inner_separators_is_a_match(self):
        # "my_api_key_value" really is a credential name; the separators are
        # conventions, not evidence to the contrary.
        assert SEC.is_sensitive_key("my_api_key_value") is True
        assert SEC.is_sensitive_key("prod.api_key.v2") is True
        assert SEC.is_sensitive_key("MY-API-KEY") is True

    def test_substring_is_not_a_match(self):
        # "tokenizer" contains "token" but is not a credential.
        assert SEC.is_sensitive_key("tokenizer") is False
        assert SEC.is_sensitive_key("secretary") is False


class TestValueShapes:
    @pytest.mark.parametrize(
        "value,shape",
        [
            ("sk-" + "a" * 48, "openai"),
            ("sk-ant-" + "a" * 40, "anthropic"),
            ("ghp_" + "a" * 36, "github"),
            ("AKIA" + "A" * 16, "aws"),
            ("xoxb-" + "1" * 20, "slack"),
            ("AIza" + "a" * 35, "google"),
            ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "jwt"),
            ("-----BEGIN RSA PRIVATE KEY-----", "pem"),
            ("postgres://user:pass@host/db", "uri_credentials"),
        ],
    )
    def test_shapes_detected(self, value, shape):
        assert shape in SEC.scan_value(value)

    @pytest.mark.parametrize(
        "value",
        ["0.01", "resnet", "a normal sentence about tokens", "", "12345", "file:/tmp/x"],
    )
    def test_plain_values_clean(self, value):
        assert SEC.scan_value(value) == []

    def test_non_string_value(self):
        assert SEC.scan_value(None) == []  # type: ignore[arg-type]


class TestClassify:
    def test_key_reason(self):
        assert SEC.classify("api_key", "abc") == ["sensitive_key"]

    def test_value_reason_only(self):
        reasons = SEC.classify("note", "sk-" + "b" * 40)
        assert reasons == ["secret_shape:openai"]

    def test_both(self):
        reasons = SEC.classify("api_key", "sk-" + "b" * 40)
        assert "sensitive_key" in reasons and "secret_shape:openai" in reasons

    def test_clean(self):
        assert SEC.classify("lr", "0.01") == []


class TestMasking:
    def test_mask_keeps_no_plaintext(self):
        secret = "sk-live-super-secret-value"
        masked = SEC.mask(secret)
        assert secret not in masked
        assert masked.startswith("<redacted:sha256:")
        assert masked.endswith(">")

    def test_mask_is_deterministic(self):
        assert SEC.mask("abc") == SEC.mask("abc")

    def test_mask_differs_for_different_values(self):
        assert SEC.mask("abc") != SEC.mask("abd")

    def test_mask_length_hint_is_twelve(self):
        inner = SEC.mask("abc").split(":")[-1].rstrip(">")
        assert len(inner) == 12

    def test_fingerprint_matches_sha256(self):
        import hashlib

        assert SEC.fingerprint("x") == hashlib.sha256(b"x").hexdigest()


class TestPolicySwitch:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv(SEC.REDACTED_ENV, raising=False)
        assert SEC.redact_enabled() is True

    def test_disabled(self, monkeypatch):
        monkeypatch.setenv(SEC.REDACTED_ENV, "none")
        assert SEC.redact_enabled() is False
        value, changed, _reasons = SEC.redact_if_sensitive("api_key", "sk-live-abc")
        assert changed is False and value == "sk-live-abc"

    def test_single_level(self, monkeypatch):
        monkeypatch.setenv(SEC.REDACTED_ENV, "sensitive_key")
        masked_value, changed, _ = SEC.redact_if_sensitive("api_key", "plain")
        assert changed is True
        assert masked_value.startswith("<redacted")
        # A shape-only secret is left alone when only key detection is enabled.
        v2, c2, _ = SEC.redact_if_sensitive("note", "sk-" + "c" * 40)
        assert c2 is False and v2.startswith("sk-")

    def test_unknown_value_untouched(self, monkeypatch):
        monkeypatch.delenv(SEC.REDACTED_ENV, raising=False)
        value, changed, _ = SEC.redact_if_sensitive("lr", "0.01")
        assert changed is False and value == "0.01"


class TestLedger:
    def test_ledger_stores_no_plaintext(self, tmp_path):
        led = SEC.SecretsLedger()
        led.add("params/run1", "api_key", "sk-live-abcdef", ["sensitive_key"])
        path = tmp_path / "secrets.json"
        led.write(str(path))
        body = path.read_text(encoding="utf-8")
        assert "sk-live-abcdef" not in body
        data = json.loads(body)
        assert data["count"] == 1
        assert data["entries"][0]["key"] == "api_key"
        assert len(data["entries"][0]["sha256"]) == 64
        assert data["entries"][0]["length"] == len("sk-live-abcdef")

    def test_ledger_permissions_posix(self, tmp_path):
        if os.name == "nt":  # Windows does not honour chmod the same way
            pytest.skip("POSIX mode bits not meaningful on Windows")
        led = SEC.SecretsLedger()
        led.add("t/r", "token", "v", ["sensitive_key"])
        path = tmp_path / "secrets.json"
        led.write(str(path))
        assert oct(os.stat(path).st_mode)[-3:] == "600"

    def test_ledger_sorted_and_deduped_by_position(self, tmp_path):
        led = SEC.SecretsLedger()
        led.add("b", "k", "v", ["sensitive_key"])
        led.add("a", "k", "v", ["sensitive_key"])
        path = tmp_path / "s.json"
        led.write(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert [e["where"] for e in data["entries"]] == ["a", "b"]


class TestGitignore:
    def test_blocks_everything_but_itself(self, tmp_path):
        path = SEC.write_gitignore(str(tmp_path))
        body = open(path, encoding="utf-8").read()
        assert "*" in body
        assert "!.gitignore" in body


class TestScanRealStore:
    def test_finds_secrets_in_params_and_tags(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2, secrets=True)
        led = SEC.SecretsLedger()
        from mlflow_vault.backup import _scan_db_secrets

        _scan_db_secrets(info["db"], led)
        keys = {e["key"] for e in led.entries}
        assert "api_key" in keys
        assert "db_password" in keys
        assert len(led) == 4  # 2 runs x (1 param + 1 tag)
        # and no plaintext anywhere in the serialised ledger
        dumped = json.dumps(led.to_dict())
        assert "sk-live" not in dumped
        assert "hunter2" not in dumped

    def test_store_without_secrets_yields_empty_ledger(self, tmp_path):
        info = make_store(str(tmp_path / "s"), runs=2, secrets=False)
        led = SEC.SecretsLedger()
        from mlflow_vault.backup import _scan_db_secrets

        _scan_db_secrets(info["db"], led)
        assert len(led) == 0
