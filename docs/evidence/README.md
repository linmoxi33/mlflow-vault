# Evidence from the development drill

Kept in the repository because a claim like "the drill caught three bugs" is worth
nothing without the artefacts. Everything here was produced by real runs, not
hand-written.

## Committed evidence files

| File | Produced by | What it shows |
|---|---|---|
| `report.json` | `mlflow-vault drill --runs 4` | all seven steps, `ok: true`, with per-step data |
| `verify-clean.json` | the drill's `verify-clean` step | 14 files hashed, 85 rows, 0 errors |
| `tamper.json` | the drill's `tamper-detect` step | 3 damage modes, each caught and attributed to the exact file |
| `negative-control.txt` | the drill's `negative-control` step | db-only copy → 4/4 dirs dangling, SDK readback fails |

Note the `db-truncated` entry: `size-mismatch` fires before the database is even
opened, and `db-unreadable` confirms the file cannot be parsed. Two independent
signals for one fault is intentional — a size check alone can be fooled by a
same-length overwrite, and a parse attempt alone would miss a file that is merely
truncated at a page boundary.

## `first-run-artifacts-outside-store/`

The first drill run failed at `verify-clean` with:

```
artifacts local=3 remote=0 missing=3
ERROR [artifact-dir-missing] run's artifact directory is not in the package
```

Root cause: MLflow's default `artifact_location` is **relative to the process CWD**,
so `build_demo_store()` wrote its artifacts to `<cwd>/mlruns/1/…` — outside the
store root — while the database sat inside it. The backup faithfully recorded
`artifact_base = file:…/mlflow-vault/mlruns/1` and archived nothing, and the verifier
correctly refused to call the package complete.

This is the real-world shape of the bug: **the store root and the artifact root
drifted apart without anyone deciding that.** The directory is preserved here as
the physical evidence; the fixture now pins the experiment's artifact location
inside the store root.

Note the second bug it exposed: MLflow writes local artifact URIs as `file:E:/…`
(a *single* slash, not RFC `file:///`). Treating that as an opaque/remote string made
the artifact tree invisible in the first place. `store._posix()` now strips the
scheme explicitly, and `TestPosix::test_single_slash_file_uri_is_stripped` locks it.

## Regenerating

```bash
pip install -e ".[dev,drill]"
mlflow-vault drill --workdir _drill --runs 4
```

`_drill/` is gitignored (it is regenerable and large). The interesting files it
writes are:

| File | Contents |
|---|---|
| `report.json` | all seven steps with their data |
| `verify-clean.json` | the clean-package verification |
| `tamper.json` | per-damage-mode detection result and the file each error named |
| `readback.json` | what the MLflow SDK re-read after restore |
| `negative-control.txt` | proof that a db-only copy loses artifacts |
| `audit-source.txt` | the risk report for the freshly built store |
