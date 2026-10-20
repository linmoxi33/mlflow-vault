"""Streaming hashing helpers.

Everything is content-addressed by SHA-256. We hash **while writing** when we can
(so the recorded hash describes the bytes that actually landed on disk, not the
bytes we hoped to copy), and we hash in chunks so a 2 GB checkpoint does not
turn into a 2 GB memory spike.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable

CHUNK = 1024 * 1024


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic serialisation used for every digest in the manifest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest_of(obj: Any) -> str:
    """Digest of any JSON-able object, via canonical serialisation."""
    return sha256_text(canonical_json(obj))


def copy_hashing(src: str, dst: str) -> Dict[str, Any]:
    """Copy ``src`` to ``dst`` and return the hash/size of what was written.

    Hashing happens on the bytes read *and* re-read from the destination, so a
    truncated write or a short read cannot be recorded as success.
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    h = hashlib.sha256()
    size = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for chunk in iter(lambda: fi.read(CHUNK), b""):
            h.update(chunk)
            size += len(chunk)
            fo.write(chunk)
    written = sha256_file(dst)
    if written != h.hexdigest():  # pragma: no cover - disk-level corruption
        raise OSError(f"copy of {src} was corrupted in flight ({h.hexdigest()} != {written})")
    return {"sha256": written, "size": size}


def merkle_root(entries: Iterable[Dict[str, Any]]) -> str:
    """Roll per-file digests into one digest, order-independent by path.

    Used for the manifest's own ``root_hash`` so two backups of the same store can
    be compared with a single string, and so a single missing file changes the root.
    """
    items = sorted(
        ({"path": e["path"], "sha256": e["sha256"], "size": e["size"]} for e in entries),
        key=lambda e: e["path"],
    )
    return digest_of(items)
