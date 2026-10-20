"""Sensitive-value handling.

MLflow params and tags are free-form key/value text, and in practice people put
API keys, DB passwords and SAS tokens in them. A backup is a *new copy of those
secrets on a new piece of media*, so it deserves an explicit policy rather than a
default that quietly duplicates them.

Policy implemented here:

* Every value that *looks* sensitive is recorded in the manifest as a **hash
  only** — never in cleartext — so a manifest can be shipped to a colleague, a
  ticket, or CI logs to prove "the secret was backed up" without revealing it.
* ``backup`` keeps original values (a backup that redacts is not a backup) but
  writes ``secrets.json`` alongside, mode 0600, listing *where* each secret lives.
* ``redact`` produces a shareable package where values are replaced in place with
  ``<redacted:sha256:xxxxxxxxxxxx>`` — still verifiable, no longer usable.
* ``.gitignore`` inside the export directory refuses to let a backup be committed.
"""

from __future__ import annotations

import json
import os
import re
import stat
from typing import Dict, Iterable, List, Tuple

from .hashing import sha256_text

# Key fragments that mark a value as sensitive. Matched case-insensitively
# against the *whole* key, allowing separators (api_key, apiKey, API-KEY).
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[._\-])(?:api[._\-]?key|apikey|secret|token|passwd|password|pwd|credential|"
    r"credentials|access[._\-]?key|private[._\-]?key|sas[._\-]?token|auth|bearer|"
    r"session[._\-]?key|client[._\-]?secret|conn(?:ection)?[._\-]?str(?:ing)?)(?:$|[._\-])",
    re.IGNORECASE,
)

# Whole-key names that carry no separator but are still secrets.
SENSITIVE_EXACT = {
    "apikey",
    "api_key",
    "apikeys",
    "authorization",
    "bearer",
    "password",
    "passwd",
    "pwd",
    "secret",
    "credentials",
    "accesskey",
    "access_key",
    "secretkey",
    "secret_key",
    "privatekey",
    "private_key",
    "connectionstring",
    "connection_string",
    "hf_token",
    "gh_token",
    "github_token",
    "openai_api_key",
    "anthropic_api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
}

# Value shapes that are secrets regardless of the key (e.g. tag name "note").
SECRET_VALUE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("pem", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    (
        "uri_credentials",
        re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]{1,64}:[^/\s@]{1,128}@", re.IGNORECASE),
    ),
)

MASK = "<redacted:sha256:{}>"
REDACTED_ENV = "MLFLOW_VAULT_REDACT_LEVELS"


def is_sensitive_key(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return False
    if k.lower() in SENSITIVE_EXACT:
        return True
    return bool(SENSITIVE_KEY_PATTERN.search(k))


def scan_value(value: str) -> List[str]:
    """Return the names of secret *shapes* found inside a free-form value."""
    if not isinstance(value, str):
        return []
    return [name for name, pat in SECRET_VALUE_PATTERNS if pat.search(value)]


def classify(key: str, value: str) -> List[str]:
    """Why this pair is sensitive — empty list means it is not."""
    reasons: List[str] = []
    if is_sensitive_key(key):
        reasons.append("sensitive_key")
    for shape in scan_value(value):
        reasons.append(f"secret_shape:{shape}")
    return reasons


def fingerprint(value: str) -> str:
    return sha256_text(value)


def mask(value: str) -> str:
    return MASK.format(fingerprint(value)[:12])


def redaction_levels() -> List[str]:
    """Which detection families are active; ``none`` disables redaction entirely."""
    raw = os.environ.get(REDACTED_ENV, "").strip()
    if not raw:
        return ["sensitive_key"] + [f"secret_shape:{n}" for n, _ in SECRET_VALUE_PATTERNS]
    if raw.lower() in ("none", "0", "off"):
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def redact_enabled() -> bool:
    return bool(redaction_levels())


def redact_if_sensitive(key: str, value: str) -> Tuple[str, bool, List[str]]:
    reasons = classify(key, value)
    if not reasons or not redact_enabled():
        return value, False, reasons
    active = set(redaction_levels())
    if not (active & set(reasons)):
        return value, False, reasons
    return mask(value), True, reasons


class SecretsLedger:
    """Where the secrets are, without saying what they are.

    Stored as a sidecar so the manifest itself stays safe to share, while an
    operator who legitimately needs to rotate credentials knows which runs to look at.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, object]] = []

    def add(self, where: str, key: str, value: str, reasons: Iterable[str]) -> None:
        self.entries.append(
            {
                "where": where,
                "key": key,
                "sha256": fingerprint(value),
                "length": len(value),
                "reasons": sorted(set(reasons)),
            }
        )

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict[str, object]:
        return {
            "note": (
                "Values are NOT stored here — only a SHA-256 fingerprint, length and location, "
                "so credentials can be located and rotated without this file leaking them."
            ),
            "count": len(self.entries),
            "entries": sorted(self.entries, key=lambda e: (str(e["where"]), str(e["key"]))),
        }

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        try:  # best effort: 0600 where the platform honours POSIX modes
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # pragma: no cover - Windows
            pass


GITIGNORE_BODY = """# Written by mlflow-vault: a backup must never be committed by accident.
*
!.gitignore
"""


def write_gitignore(directory: str) -> str:
    path = os.path.join(directory, ".gitignore")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(GITIGNORE_BODY)
    return path
