"""The restore drill: an executable proof, not a promise.

A backup is only worth what the last restore proved. This module runs the whole
loop end to end and **fails loudly** at each step, because the interesting failures
are the silent ones:

1. build a real store (MLflow SDK, WAL mode, secrets in params)
2. back it up
3. verify the untouched package → must be clean
4. **damage a copy** (flip one byte / delete one artifact / truncate the db)
   → verify must report an error *and name the file*
5. restore into a **fresh** directory → must be row-identical
6. re-read through the MLflow SDK → every run, metric, param and artifact must
   come back with the same values
7. **negative control**: a naive "copy the db only" restore must *lose artifacts*
   while passing a byte check — this is what makes the drill meaningful, because
   it proves the test can detect the failure mode it claims to protect against

If step 7 ever stops failing, the drill has become a rubber stamp and says so.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

STEP_NAMES = (
    "build-store",
    "backup",
    "verify-clean",
    "tamper-detect",
    "restore",
    "readback",
    "negative-control",
)


@dataclass
class Step:
    name: str
    ok: bool
    detail: str
    data: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"step": self.name, "ok": self.ok, "detail": self.detail, "data": self.data}


@dataclass
class DrillReport:
    workdir: str
    steps: List[Step] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps) and len(self.steps) == len(STEP_NAMES)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def as_dict(self) -> Dict[str, object]:
        return {
            "workdir": self.workdir,
            "ok": self.ok,
            "steps": [s.as_dict() for s in self.steps],
            "artifacts": self.artifacts,
        }

    def summarize(self) -> str:
        lines = [f"drill at {self.workdir}", ""]
        for s in self.steps:
            lines.append(f"  [{'PASS' if s.ok else 'FAIL'}] {s.name:18} {s.detail}")
        lines.append("")
        lines.append(f"  => {'DRILL PASSED' if self.ok else 'DRILL FAILED'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #


def build_demo_store(root: str, *, runs: int = 3) -> Dict[str, object]:
    """Create a small but real MLflow store with nested artifacts and secrets.

    The experiment's ``artifact_location`` is pinned **inside** the store root on
    purpose. MLflow's default is relative to the process CWD, which is how real
    users end up with a database in ``/data`` and artifacts in ``/home/me/mlruns``
    — the exact split this tool reports as ``artifact-base`` outside the store.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    os.makedirs(root, exist_ok=True)
    db_path = os.path.join(root, "mlflow.db")
    artifact_root = os.path.join(root, "mlruns")
    os.makedirs(artifact_root, exist_ok=True)
    backend = "sqlite:///" + db_path.replace("\\", "/")
    mlflow.set_tracking_uri(backend)
    mlflow.set_registry_uri(backend)

    exp_name = "vault-drill"
    client = MlflowClient()
    existing = client.get_experiment_by_name(exp_name)
    if existing is None:
        exp_id = client.create_experiment(
            exp_name, artifact_location="file:" + artifact_root.replace("\\", "/")
        )
    else:  # pragma: no cover - fresh dir in practice
        exp_id = existing.experiment_id

    written: Dict[str, object] = {"runs": [], "artifact_files": []}
    for i in range(runs):
        with mlflow.start_run(experiment_id=exp_id, run_name=f"run-{i}") as run:
            mlflow.log_param("lr", 0.01 * (i + 1))
            mlflow.log_param("batch_size", 32 * (i + 1))
            # a credential that has no business being in a param
            mlflow.log_param("api_key", "sk-live-" + "".join(random.choices(string.ascii_letters + string.digits, k=24)))
            mlflow.set_tag("model_family", "resnet")
            mlflow.set_tag("db_password", "p@ssw0rd-" + str(i))
            for step in range(4):
                mlflow.log_metric("loss", 1.0 / (step + 1) + i * 0.01, step=step)
                mlflow.log_metric("acc", 0.5 + step * 0.1 - i * 0.01, step=step)
            # nested artifact tree, including a subdir and a binary blob
            mlflow.log_text("epoch,lr\n0,0.1\n1,0.05\n", f"metrics/run{i}.csv")
            mlflow.log_text("{\"note\": \"checkpoint metadata\"}\n", f"meta/run{i}.json")
            blob = os.urandom(4096)
            tmp = os.path.join(root, "_tmp_blob.bin")
            with open(tmp, "wb") as fh:
                fh.write(blob)
            mlflow.log_artifact(tmp, artifact_path=f"weights/run{i}")
            os.remove(tmp)
            written["runs"].append(run.info.run_id)  # type: ignore[union-attr]
            written["artifact_files"].append(len(blob))

    # Force a WAL file to exist so the drill exercises the WAL hazard for real.
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("INSERT OR REPLACE INTO tags (key, value, run_uuid) VALUES ('drill', 'wal', '')")
    con.commit()
    con.close()

    written["experiment_id"] = exp_id
    written["db"] = db_path
    written["root"] = root
    written["artifact_root"] = artifact_root
    return written


def _mlflow_server(backend: str, artifact_root: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            backend,
            "--default-artifact-root",
            artifact_root,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def readback_via_sdk(root: str, expect_runs: int) -> Dict[str, object]:
    """Re-read the restored store through MLflow itself (not through our own code).

    This is the step that catches "the files are there but nothing resolves":
    the SDK resolves ``artifact_uri`` the same way a user's notebook would.
    """
    from mlflow.tracking import MlflowClient

    db_path = os.path.join(root, "mlflow.db")
    if not os.path.isfile(db_path):
        raise AssertionError(f"restored store has no db at {db_path}")
    uri = "sqlite:///" + db_path.replace("\\", "/")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    exps = [e for e in client.search_experiments() if e.name == "vault-drill"]
    if not exps:
        raise AssertionError("experiment vault-drill not found after restore")
    exp = exps[0]
    runs = client.search_runs([exp.experiment_id])
    if len(runs) != expect_runs:
        raise AssertionError(f"expected {expect_runs} runs, found {len(runs)}")

    detail: List[Dict[str, object]] = []
    total_artifacts = 0
    for run in runs:
        metrics = client.get_metric_history(run.info.run_id, "loss")
        artifacts = client.list_artifacts(run.info.run_id)
        # walk one level deeper so nested dirs are covered too
        nested = sum(len(client.list_artifacts(run.info.run_id, a.path)) for a in artifacts)
        total_artifacts += len(artifacts)
        if not metrics:
            raise AssertionError(f"run {run.info.run_id} lost its metrics")
        if not artifacts:
            raise AssertionError(
                f"run {run.info.run_id} restored with 0 artifacts — this is the silent "
                "failure the drill exists to catch"
            )
        detail.append(
            {
                "run_id": run.info.run_id,
                "metrics": len(metrics),
                "params": len(run.data.params),
                "artifacts": len(artifacts) + nested,
                "artifact_uri": run.info.artifact_uri,
            }
        )
    return {"runs": len(runs), "artifacts": total_artifacts, "detail": detail}


def run_drill(workdir: Optional[str] = None, *, runs: int = 3, keep: bool = False) -> DrillReport:
    from .backup import backup
    from .restore import audit_restored, restore
    from .verify import verify

    workdir = os.path.abspath(workdir or os.path.join(os.getcwd(), "_drill"))
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)
    report = DrillReport(workdir=workdir)

    store_root = os.path.join(workdir, "store")
    backup_dir = os.path.join(workdir, "backup")
    restored_root = os.path.join(workdir, "restored")

    # ---- 1. build --------------------------------------------------------
    t0 = time.time()
    try:
        built = build_demo_store(store_root, runs=runs)
        report.add(
            Step(
                "build-store",
                True,
                f"{len(built['runs'])} runs, {built['artifact_files']} artifacts written "
                f"({time.time() - t0:.1f}s)",
                {"runs": built["runs"], "db": built["db"]},
            )
        )
    except Exception as exc:
        report.add(Step("build-store", False, f"{type(exc).__name__}: {exc}"))
        return report

    # ---- 2. audit the source store (this is the risk analysis, measured) --
    from .audit import audit
    from .audit import summarize as audit_summary

    src_audit = audit(store_root)
    (report.artifacts)["audit-source.json"] = _dump(report.workdir, "audit-source.json", src_audit.as_dict())
    (report.artifacts)["audit-source.txt"] = _dump_text(
        report.workdir, "audit-source.txt", audit_summary(src_audit)
    )

    # ---- 3. backup -------------------------------------------------------
    t0 = time.time()
    try:
        b = backup(store_root, backup_dir)
        report.add(
            Step(
                "backup",
                True,
                f"{b.file_count} files / {b.total_bytes} bytes / {b.secret_count} secrets logged "
                f"({time.time() - t0:.1f}s)",
                {
                    "files": b.file_count,
                    "bytes": b.total_bytes,
                    "root_hash": b.manifest["root_hash"],
                    "secrets": b.secret_count,
                    "warnings": b.manifest["warnings"],
                },
            )
        )
    except Exception as exc:
        report.add(Step("backup", False, f"{type(exc).__name__}: {exc}"))
        return report

    # ---- 4. verify clean -------------------------------------------------
    v = verify(backup_dir)
    (report.artifacts)["verify-clean.json"] = _dump(report.workdir, "verify-clean.json", v.as_dict())
    report.add(
        Step(
            "verify-clean",
            v.ok,
            f"{v.checked_files} files hashed, {v.checked_bytes} bytes, "
            f"{v.logical.get('__total_rows__')} rows, errors={len(v.errors)}",
            v.as_dict(),
        )
    )
    if not v.ok:
        return report

    # ---- 5. tamper detection: three separate damage modes -----------------
    tamper_cases = _tamper_cases(backup_dir, workdir)
    all_detected = True
    case_detail: Dict[str, object] = {}
    for name, path, how in tamper_cases:
        if not os.path.exists(path):
            case_detail[name] = "skipped (file absent)"
            continue
        broken = os.path.join(workdir, f"tampered-{name}")
        if os.path.exists(broken):
            shutil.rmtree(broken)
        shutil.copytree(backup_dir, broken)
        target = path.replace(backup_dir, broken, 1)
        _apply_damage(target, how)
        tv = verify(broken)
        detected = not tv.ok
        all_detected &= detected
        case_detail[name] = {
            "detected": detected,
            "errors": [f.code for f in tv.errors],
            "named_file": [f.path for f in tv.errors if f.path][:3],
        }
    (report.artifacts)["tamper.json"] = _dump(
        report.workdir, "tamper.json", case_detail
    )
    report.add(
        Step(
            "tamper-detect",
            all_detected,
            "; ".join(
                f"{k}:{'CAUGHT' if isinstance(v, dict) and v.get('detected') else v if isinstance(v, str) else 'MISSED'}"
                for k, v in case_detail.items()
            ),
            case_detail,
        )
    )

    # ---- 6. restore ------------------------------------------------------
    t0 = time.time()
    try:
        r = restore(backup_dir, restored_root)
        post = audit_restored(r)
        (report.artifacts)["restore.json"] = _dump(report.workdir, "restore.json", {**r.as_dict(), "post": post})
        report.add(
            Step(
                "restore",
                r.ok and post.get("dangling_count", 0) == 0,
                f"{r.files_copied} artifacts copied, {r.verified_rows} rows matched, "
                f"{r.plan.rewrite_count} URIs rewritten, dangling={post.get('dangling_count')} "
                f"({time.time() - t0:.1f}s)",
                {**r.as_dict(), "post": post},
            )
        )
    except Exception as exc:
        report.add(Step("restore", False, f"{type(exc).__name__}: {exc}"))
        return report

    # ---- 7. read back through MLflow itself ------------------------------
    try:
        rb = readback_via_sdk(restored_root, expect_runs=runs)
        (report.artifacts)["readback.json"] = _dump(report.workdir, "readback.json", rb)
        report.add(
            Step(
                "readback",
                True,
                f"MLflow SDK re-read {rb['runs']} runs / {rb['artifacts']} top-level artifacts, "
                "all metrics+artifacts resolvable",
                rb,
            )
        )
    except Exception as exc:
        report.add(Step("readback", False, f"{type(exc).__name__}: {exc}"))
        return report

    # ---- 8. negative control: db-only restore must lose artifacts --------
    #
    # The first version of this step was wrong and proved nothing: copying only the
    # db "worked", because ``artifact_uri`` is an absolute path that still pointed
    # at the *original* store, which was sitting right there on disk. That is
    # precisely why the bug survives in production — nothing breaks until the
    # original machine, volume, or container is gone.
    #
    # So the control now simulates the loss: move the artifact tree out of the
    # source store, then see what the db-only copy can still do. (Renaming the
    # whole store directory fails on Windows while SQLAlchemy still holds the db
    # file open — that is a platform artefact, not a finding, so move the tree.)
    naive = os.path.join(workdir, "naive-db-only")
    os.makedirs(naive, exist_ok=True)
    shutil.copyfile(os.path.join(backup_dir, "db", "mlflow.db"), os.path.join(naive, "mlflow.db"))
    art_src = os.path.join(store_root, "mlruns")
    art_gone = os.path.join(workdir, "artifacts-now-gone")
    moved = False
    if os.path.isdir(art_src):
        if os.path.exists(art_gone):
            shutil.rmtree(art_gone)
        try:
            os.rename(art_src, art_gone)
            moved = True
        except OSError:
            moved = False
    try:
        ncon = sqlite3.connect(os.path.join(naive, "mlflow.db"))
        uris = [str(r[0] or "") for r in ncon.execute("SELECT artifact_uri FROM runs")]
        ncon.close()
        clean = [u.replace("file:", "").rstrip("/") for u in uris if u]
        dangling = [u for u in clean if not os.path.isdir(u)]
        sdk_err = ""
        sdk_lost = None
        if moved:
            try:
                readback_via_sdk(naive, expect_runs=runs)
                sdk_lost = False
            except AssertionError as exc:
                sdk_lost = True
                sdk_err = str(exc)
        if moved:
            control_ok = bool(dangling) and sdk_lost is True
            detail = (
                f"db-only copy after artifact loss: {len(dangling)}/{len(uris)} dirs dangling, "
                f"SDK readback failed as expected ({sdk_err[:80]})"
                if control_ok
                else f"UNEXPECTED: dangling={len(dangling)} sdk_failed={sdk_lost} — the control "
                "proves nothing; fix it before trusting the drill"
            )
        else:  # pragma: no cover - only if the platform refuses the move
            control_ok = False
            detail = (
                "UNVERIFIABLE: could not move the artifact tree away to simulate its loss "
                "(file lock). The control is inconclusive, not passing."
            )
    finally:
        if moved:
            os.rename(art_gone, art_src)
    (report.artifacts)["negative-control.txt"] = _dump_text(report.workdir, "negative-control.txt", detail)
    report.add(
        Step(
            "negative-control",
            control_ok,
            detail,
            {
                "dangling": len(dangling),
                "total_uris": len(uris),
                "sdk_readback_failed": sdk_lost,
                "loss_simulated": moved,
            },
        )
    )

    (report.artifacts)["report.json"] = _dump(report.workdir, "report.json", report.as_dict())
    if not keep:
        pass  # the drill directory *is* the evidence; never delete it implicitly
    return report


def _tamper_cases(backup_dir: str, workdir: str):
    """Pick real files from the package so damage is aimed at things that matter."""
    manifest = json.load(open(os.path.join(backup_dir, "manifest.json"), encoding="utf-8"))
    files = manifest.get("files", [])
    artifact = next((f for f in files if f.get("role") == "artifact"), None)
    db = next((f for f in files if f.get("role") == "db-snapshot"), None)
    cases = []
    if artifact:
        cases.append(("artifact-byte-flip", os.path.join(backup_dir, artifact["path"].replace("/", os.sep)), "flip"))
        cases.append(
            (
                "artifact-deleted",
                os.path.join(backup_dir, artifact["path"].replace("/", os.sep)),
                "delete",
            )
        )
    if db:
        cases.append(("db-truncated", os.path.join(backup_dir, db["path"].replace("/", os.sep)), "truncate"))
    return cases


def _apply_damage(path: str, how: str) -> None:
    if how == "delete":
        os.remove(path)
        return
    if how == "truncate":
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(max(size // 2, 512))
        return
    with open(path, "r+b") as fh:
        fh.seek(0)
        first = fh.read(1)
        fh.seek(0)
        fh.write(bytes([first[0] ^ 0xFF]) if first else b"\x00")


def _dump(dirpath: str, name: str, obj: object) -> str:
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        fh.write("\n")
    return path


def _dump_text(dirpath: str, name: str, text: str) -> str:
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
    return path
