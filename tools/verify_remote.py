"""Verify a pushed tree against the local working copy, byte for byte.

Blob hashes are not a reliable comparison here: ``.gitattributes`` declares
``* text=auto eol=lf``, so the blob GitHub stores may be normalised while the local
file still has CRLF. The only honest check is to download each file and compare
bytes (and to report which files differ in line endings, so a normalisation is not
mistaken for corruption).

Usage:
    python tools/verify_remote.py --repo linmoxi33/mlflow-vault --branch main
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Tuple

SKIP_DIRS = {
    ".git",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "_drill",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".whl", ".egg-info"}


def gh_api(path: str) -> dict:
    proc = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GET {path} failed: {proc.stderr.strip()[:400]}")
    return json.loads(proc.stdout)


def local_files(root: pathlib.Path) -> List[Tuple[str, pathlib.Path]]:
    out: List[Tuple[str, pathlib.Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES or path.name.endswith(".egg-info"):
            continue
        out.append((rel.as_posix(), path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    tree = gh_api(f"repos/{args.repo}/git/trees/{args.branch}?recursive=1")
    remote = {
        item["path"]: item["sha"]
        for item in tree["tree"]
        if item["type"] == "blob"
    }
    local = local_files(root)

    missing_remote: List[str] = []
    extra_remote: List[str] = []
    mismatched: List[Tuple[str, str]] = []
    eol_only: List[str] = []
    ok = 0

    local_paths = {rel for rel, _ in local}
    for rel, _ in local:
        if rel not in remote:
            missing_remote.append(rel)
    for rel in remote:
        if rel not in local_paths:
            extra_remote.append(rel)

    for rel, path in local:
        if rel not in remote:
            continue
        blob = gh_api(f"repos/{args.repo}/git/blobs/{remote[rel]}")
        data = base64.b64decode(blob["content"])
        disk = path.read_bytes()
        if data == disk:
            ok += 1
            continue
        # Same content, different line endings? That is the .gitattributes doing its
        # job, not a bad upload — but it must be named, not silently forgiven.
        if data.replace(b"\r\n", b"\n") == disk.replace(b"\r\n", b"\n"):
            eol_only.append(rel)
            ok += 1
            continue
        mismatched.append(
            (rel, f"remote {hashlib.sha256(data).hexdigest()[:12]} != "
                  f"local {hashlib.sha256(disk).hexdigest()[:12]}")
        )

    print(f"repo     {args.repo}@{args.branch}")
    print(f"remote   {len(remote)} blobs")
    print(f"local    {len(local)} files")
    print(f"verified {ok} byte-identical" + (f" ({len(eol_only)} eol-normalised)" if eol_only else ""))
    for rel in eol_only:
        print(f"  eol  {rel}")
    for rel in missing_remote:
        print(f"  MISSING-REMOTE  {rel}")
    for rel in extra_remote:
        print(f"  EXTRA-REMOTE    {rel}")
    for rel, why in mismatched:
        print(f"  MISMATCH        {rel}: {why}")

    bad = bool(missing_remote or extra_remote or mismatched)
    print("\nresult   " + ("FAILED" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
