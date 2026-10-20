"""Command line entry point.

Encoding is handled explicitly: Chinese/arrow output on a Windows console with a
cp936 or cp1252 default raises ``UnicodeEncodeError`` and takes the whole command
down with it. Per the project rule, reconfigure the stream rather than trusting
the locale, and fall back to backslash-escaped output if that fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__


def _reconfigure() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover - very old/odd streams
            pass


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - belt and braces
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc, "replace"))


def _emit(obj: object) -> None:
    _safe_print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlflow-vault",
        description=(
            "Integrity-first backup, verification and restore for MLflow tracking stores. "
            "Protects the database AND the artifacts it points at."
        ),
        epilog=(
            "typical run:  mlflow-vault audit ./mlruns  &&  "
            "mlflow-vault backup ./mlruns ./backup-$(date +%F)  &&  "
            "mlflow-vault verify ./backup-$(date +%F)  &&  "
            "mlflow-vault restore ./backup-$(date +%F) ./restored"
        ),
    )
    p.add_argument("--version", action="version", version=f"mlflow-vault {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="measure data-loss risk in a store")
    a.add_argument("store", help="store root (contains mlflow.db and/or mlruns/)")
    a.add_argument("--backup", help="also verify an existing backup package")
    a.add_argument("--json", action="store_true", help="machine-readable output")
    a.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium", "low"],
        help="exit 2 when the worst risk is at least this level",
    )

    b = sub.add_parser("backup", help="create a verifiable backup package")
    b.add_argument("store")
    b.add_argument("out", help="package directory (created)")
    b.add_argument("--overwrite", action="store_true", help="replace a non-empty out dir")
    b.add_argument("--no-secrets-ledger", action="store_true", help="skip secrets.json")
    b.add_argument("--json", action="store_true")

    v = sub.add_parser("verify", help="prove a package matches its manifest")
    v.add_argument("backup")
    v.add_argument("--shallow", action="store_true", help="size check only, skip hashing")
    v.add_argument("--json", action="store_true")

    r = sub.add_parser("restore", help="restore a package to a new location")
    r.add_argument("backup")
    r.add_argument("target")
    r.add_argument("--force", action="store_true", help="overwrite an existing tracking db")
    r.add_argument("--dry-run", action="store_true", help="print the URI rewrite plan only")
    r.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="do not overwrite artifact files that already exist",
    )
    r.add_argument("--json", action="store_true")

    d = sub.add_parser("redact", help="produce a shareable copy with secrets masked")
    d.add_argument("backup")
    d.add_argument("--out", help="output directory (default: <backup>-redacted)")
    d.add_argument("--in-place", action="store_true", help="rewrite the package itself")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--json", action="store_true")

    s = sub.add_parser("secrets", help="list where credentials live (fingerprints only)")
    s.add_argument("backup")
    s.add_argument("--json", action="store_true")

    c = sub.add_parser("compare", help="diff two packages of the same store")
    c.add_argument("backup_a")
    c.add_argument("backup_b")
    c.add_argument("--json", action="store_true")

    dr = sub.add_parser("drill", help="run a full backup+restore rehearsal in a temp dir")
    dr.add_argument("--workdir", help="where to run (default ./_drill)")
    dr.add_argument("--runs", type=int, default=3, help="how many demo runs to create")
    dr.add_argument("--json", action="store_true")
    dr.add_argument(
        "--require-mlflow",
        action="store_true",
        help="fail instead of skipping when the mlflow SDK is unavailable",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    _reconfigure()
    args = build_parser().parse_args(argv)

    try:
        return _dispatch(args)
    except FileNotFoundError as exc:
        _safe_print(f"error: {exc}")
        return 2
    except FileExistsError as exc:
        _safe_print(f"error: {exc}")
        return 2
    except ValueError as exc:
        _safe_print(f"error: {exc}")
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    from .audit import audit
    from .audit import summarize as audit_summary
    from .backup import backup
    from .redact import ledger_of, redact_backup
    from .restore import plan as restore_plan
    from .restore import restore
    from .verify import compare, dump_json, verify
    from .verify import summarize as verify_summary

    cmd = args.command

    if cmd == "audit":
        rep = audit(args.store, backup_dir=args.backup)
        if args.json:
            _emit(rep.as_dict())
        else:
            _safe_print(audit_summary(rep))
        if args.fail_on:
            from .audit import RISK_ORDER

            if RISK_ORDER[rep.worst] >= RISK_ORDER[args.fail_on]:
                return 2
        return 0

    if cmd == "backup":
        res = backup(args.store, args.out, overwrite=args.overwrite,
                     include_secrets=not args.no_secrets_ledger)
        payload = {
            "package": res.path,
            "files": res.file_count,
            "bytes": res.total_bytes,
            "secrets": res.secret_count,
            "root_hash": res.manifest["root_hash"],
            "warnings": res.manifest["warnings"],
        }
        if args.json:
            _emit(payload)
        else:
            _safe_print(f"package   {res.path}")
            _safe_print(f"files     {res.file_count} ({res.total_bytes} bytes)")
            _safe_print(f"root hash {res.manifest['root_hash']}")
            _safe_print(f"secrets   {res.secret_count} located (fingerprints only)")
            for w in res.manifest["warnings"]:  # type: ignore[union-attr]
                _safe_print(f"warning   {w}")
            _safe_print("")
            _safe_print("next: mlflow-vault verify " + args.out)
        return 0

    if cmd == "verify":
        rep = verify(args.backup, deep=not args.shallow)
        _safe_print(dump_json(rep) if args.json else verify_summary(rep))
        return rep.exit_code()

    if cmd == "restore":
        if args.dry_run:
            p = restore_plan(args.backup, args.target)
            _safe_print(json.dumps(p.as_dict(), indent=2, ensure_ascii=False, default=str))
            return 0
        res = restore(
            args.backup,
            args.target,
            force=args.force,
            overwrite_artifacts=not args.keep_artifacts,
        )
        from .restore import audit_restored

        post = audit_restored(res)
        if args.json:
            _emit({**res.as_dict(), "post": post})
        else:
            _safe_print(f"target    {res.target_root}")
            _safe_print(f"copied    {res.files_copied} artifacts ({res.bytes_copied} bytes)")
            _safe_print(f"rewritten {res.plan.rewrite_count} artifact_uri values")
            _safe_print(f"rows      {res.verified_rows} verified, match={res.db_rows_matched}")
            _safe_print(f"dangling  {post.get('dangling_count')} artifact dirs unresolved")
            for w in res.warnings:
                _safe_print(f"warning   {w}")
            for pr in res.problems:
                _safe_print(f"problem   {pr}")
            _safe_print("")
            _safe_print("result    " + ("OK" if res.ok else "FAILED"))
        return 0 if res.ok else 1

    if cmd == "redact":
        res = redact_backup(args.backup, args.out, in_place=args.in_place, dry_run=args.dry_run)
        if args.json:
            _emit(res)
        else:
            _safe_print(f"candidates {res['candidates']}")
            _safe_print(f"redacted   {res['redacted']}  {res['by_table'] or ''}")
            if not args.dry_run and res["redacted"]:
                _safe_print(f"package    {res['work_dir']}")
                _safe_print("note: artifacts are NOT redacted — inspect them for embedded credentials")
        return 0

    if cmd == "secrets":
        led = ledger_of(args.backup)
        if args.json:
            _emit(led)
        else:
            _safe_print(f"{led.get('count', 0)} sensitive entries")
            for e in led.get("entries", [])[:40]:
                _safe_print(
                    f"  {e['where']:40} {e['key']:24} sha256:{str(e['sha256'])[:12]} "
                    f"len={e['length']}"
                )
            _safe_print("")
            _safe_print(str(led.get("note", "")))
        return 0

    if cmd == "compare":
        diff = compare(args.backup_a, args.backup_b)
        if args.json:
            _emit(diff)
        else:
            _safe_print(f"identical root hash: {diff['root_hash_equal']}")
            _safe_print(f"only in A: {len(diff['only_in_a'])}  only in B: {len(diff['only_in_b'])} "
                        f"changed: {len(diff['changed'])}")
            for p in (diff["only_in_b"] or [])[:10]:
                _safe_print(f"  + {p}")
            for p in (diff["changed"] or [])[:10]:
                _safe_print(f"  ~ {p}")
        return 0

    if cmd == "drill":
        try:
            from .drill import run_drill
        except ImportError as exc:  # pragma: no cover
            _safe_print(f"error: drill unavailable: {exc}")
            return 2
        if args.require_mlflow:
            try:
                import mlflow  # noqa: F401
            except ImportError:
                _safe_print("error: --require-mlflow given but the mlflow SDK is not installed")
                return 2
        rep = run_drill(args.workdir, runs=args.runs)
        if args.json:
            _emit(rep.as_dict())
        else:
            _safe_print(rep.summarize())
        return 0 if rep.ok else 1

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
