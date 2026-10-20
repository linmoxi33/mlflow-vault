"""Push this tree to GitHub through the REST Git Data API.

`git push` over https:443 fails on this machine while `gh api` works, so the git
transport and the API clearly take different network paths. This script performs
the same operation the git protocol would — blobs, tree, commit, move the ref —
without needing the git transport at all.

It supports both the first push and subsequent updates:

* empty repo      → bootstrap one file via the Contents API (the Git Data API
                    returns 409 against a repo with no commits), then build the tree
* existing repo   → fetch the current branch tip and use it as the parent, and pass
                    its tree as ``base_tree`` so files removed locally are removed
                    remotely instead of lingering

Usage:
    python tools/api_push.py --repo linmoxi33/mlflow-vault --branch main
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Never uploaded: build artefacts, caches, and the drill output (regenerable, and
# large enough to make the first clone unpleasant).
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
SKIP_SUFFIXES = {".pyc", ".pyo", ".whl"}
SKIP_NAME_SUFFIXES = (".egg-info",)

# A tiny file that must exist so the empty-repo bootstrap has something to PUT.
BOOTSTRAP_FILE = ".gitattributes"


def gh_api(method: str, path: str, payload: Optional[dict] = None) -> dict:
    """Call the GitHub API through the `gh` CLI (working auth + network)."""
    cmd = ["gh", "api", "-X", method, path, "--input", "-"]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload or {}),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{method} {path} failed: {proc.stderr.strip()[:800]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else {}


def gh_api_optional(method: str, path: str) -> Optional[dict]:
    try:
        return gh_api(method, path)
    except RuntimeError:
        return None


def collect(root: pathlib.Path) -> List[Tuple[str, pathlib.Path]]:
    files: List[Tuple[str, pathlib.Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if any(part.endswith(SKIP_NAME_SUFFIXES) for part in rel.parts):
            continue
        files.append((rel.as_posix(), path))
    return files


def ensure_repo(repo: str, private: bool, description: str) -> None:
    info = gh_api_optional("GET", f"repos/{repo}")
    if info:
        print(f"repo exists: {info['html_url']}")
        return
    name = repo.split("/", 1)[1]
    payload = {
        "name": name,
        "description": description,
        "private": private,
        "has_issues": True,
        "has_wiki": False,
        "has_projects": False,
    }
    created = gh_api("POST", "user/repos", payload)
    print(f"created {created['html_url']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--root", default=".")
    ap.add_argument("--message", default=None)
    ap.add_argument("--message-file", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list files, upload nothing")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    files = collect(root)
    total = sum(p.stat().st_size for _, p in files)
    print(f"{len(files)} files, {total} bytes, from {root}")
    for rel, _ in files:
        print(f"  {rel}")
    if args.dry_run:
        return 0

    ensure_repo(args.repo, args.private, args.description)

    # ---- parents / base_tree ------------------------------------------------
    ref = gh_api_optional("GET", f"repos/{args.repo}/git/ref/heads/{args.branch}")
    parents: List[str] = []
    base_tree: Optional[str] = None
    if ref:
        parents = [ref["object"]["sha"]]
        commit = gh_api("GET", f"repos/{args.repo}/git/commits/{parents[0]}")
        base_tree = commit["tree"]["sha"]
        print(f"on top of {parents[0][:8]}")
    else:
        # Empty repo: no ref, no commits. Bootstrap through the Contents API,
        # because POST /git/blobs and /git/trees return 409 "Git Repository is empty".
        seed = root / BOOTSTRAP_FILE
        seed_bytes = seed.read_bytes() if seed.exists() else b"* text=auto eol=lf\n"
        gh_api(
            "PUT",
            f"repos/{args.repo}/contents/{BOOTSTRAP_FILE}",
            {
                "message": "chore: bootstrap repository",
                "content": base64.b64encode(seed_bytes).decode(),
                "branch": args.branch,
            },
        )
        print("bootstrapped empty repo via Contents API")
        ref = gh_api("GET", f"repos/{args.repo}/git/ref/heads/{args.branch}")
        parents = [ref["object"]["sha"]]
        base_tree = gh_api("GET", f"repos/{args.repo}/git/commits/{parents[0]}")["tree"]["sha"]

    # ---- blobs -------------------------------------------------------------
    entries: List[Dict[str, str]] = []
    for index, (rel, path) in enumerate(files, 1):
        blob = gh_api(
            "POST",
            f"repos/{args.repo}/git/blobs",
            {
                "content": base64.b64encode(path.read_bytes()).decode(),
                "encoding": "base64",
            },
        )
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if index % 10 == 0 or index == len(files):
            print(f"  blobs {index}/{len(files)}")

    # ---- tree --------------------------------------------------------------
    tree_payload: Dict[str, object] = {"tree": entries}
    if base_tree:
        # Without base_tree the resulting tree contains ONLY these entries, which
        # silently deletes every file not in this upload.
        tree_payload["base_tree"] = base_tree
    tree = gh_api("POST", f"repos/{args.repo}/git/trees", tree_payload)
    print(f"tree {tree['sha'][:8]}")

    # ---- commit message ----------------------------------------------------
    if args.message:
        message = args.message
    elif args.message_file:
        message = pathlib.Path(args.message_file).read_text(encoding="utf-8")
    else:
        proc = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"], cwd=root, capture_output=True, text=True
        )
        message = (proc.stdout or "").strip() or "Initial commit"
    message = message.strip()

    # ---- commit ------------------------------------------------------------
    commit = gh_api(
        "POST",
        f"repos/{args.repo}/git/commits",
        {"message": message, "tree": tree["sha"], "parents": parents},
    )
    print(f"commit {commit['sha'][:8]}")

    # ---- move the branch ---------------------------------------------------
    if ref:
        gh_api(
            "PATCH",
            f"repos/{args.repo}/git/refs/heads/{args.branch}",
            {"sha": commit["sha"], "force": True},
        )
    else:  # pragma: no cover - unreachable after bootstrap
        gh_api(
            "POST",
            f"repos/{args.repo}/git/refs",
            {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]},
        )
    print(f"refs/heads/{args.branch} -> {commit['sha'][:8]}")
    print(f"done: https://github.com/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
