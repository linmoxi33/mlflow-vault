# Data-loss risk analysis: an MLflow tracking store

This document is the reasoning behind the tool. Every claim below was measured on
a real store during development, and the ones that surprised me are marked.

**Scope.** An MLflow tracking server is chosen as the subject because it is a
common shape: a project that stores *important results* (experiments, metrics,
model versions) in a place that feels durable (a database) while most of the actual
payload lives somewhere else, coupled by a string. The same analysis applies to any
tool with a `*_uri` / `*_path` column pointing at a sibling directory — DVC, Label
Studio, Airflow's log store, a JupyterHub workspace.

---

## 1. Where the data actually lives

| Piece | Storage | Rebuildable? |
|---|---|---|
| Experiments, runs, params, metrics, tags | `mlflow.db` (SQLite by default) | **No** |
| Registered models, versions, aliases | same DB | **No** |
| Artifact files (models, plots, datasets, checkpoints) | a directory, or S3/GCS | **No** |
| The link between them | `runs.artifact_uri`, `experiments.artifact_location` | **No** — it is a literal string |

Two stores, one weak link. That is the whole risk surface.

## 2. The five loss scenarios, ranked by how quietly they happen

### 2.1 Copying `mlflow.db` alone — the default mistake

The database is a file; it is obviously the thing to copy. It is also only half the
store. What makes this dangerous rather than merely wrong is the failure signature:

* the restored store **opens**;
* `mlflow.search_runs()` **returns every run** with the right metrics and params;
* `client.list_artifacts(run_id)` returns **`[]`** — an empty list, not an error;
* nothing in the UI says "these artifacts are missing".

A user looking at a metric table concludes the restore worked. They find out
otherwise when they try to load a model — possibly weeks later, possibly after the
original machine is gone.

**Measured:** with 4 runs × 3 artifact files, a db-only copy reproduced 4/4 runs
and 0/12 artifacts, with no exception raised at any point.

### 2.2 Absolute paths — the backup that only works on the machine it came from

`runs.artifact_uri` is written by the MLflow SDK as an absolute path, e.g.

```
file:E:/work/project/mlruns/8f3c…/artifacts
```

That string is data. It is copied faithfully by any backup tool. So a package can
be **byte-perfect and still be unusable**: restore it to `/srv/store` and every run
now points at `E:/work/project/...`, which does not exist. MLflow does not check;
`list_artifacts` just returns nothing.

This is scenario 2.1 arriving by a different route, and it is why `restore` has to
rewrite URIs rather than merely copy files.

**Measured:** the drill's negative control was initially *wrong* because of this.
The db-only copy "worked" — the original store was still on disk, so the absolute
paths resolved. The control only became meaningful once the artifact tree was moved
away first. That is precisely why the bug survives in production: nothing breaks
until the source is gone.

### 2.3 WAL mode — losing the newest commits

SQLite in WAL mode keeps recent transactions in `mlflow.db-wal` until a checkpoint.
Copying `mlflow.db` with `cp`/`shutil.copy` therefore loses them. Worse, copying a
*live* database can capture a torn page image that opens fine and later reports
`database disk image is malformed`.

Two distinct hazards, one wrong fix (`cp`):

| Hazard | Symptom | Correct handling |
|---|---|---|
| `-wal` not included | newest runs/params missing, DB looks fine | SQLite online backup API |
| torn page from a live copy | opens now, corrupt later | same API (transactional) |
| snapshot inherits WAL mode | the "single file" is not self-contained | convert to `journal_mode=DELETE` |

**Measured:** the backup API copies the source's `journal_mode`, so a WAL-mode
source produced a WAL-mode snapshot whose committed pages sat in a `-wal` sidecar
next to the packaged file. Anyone later copying just that one file re-created
scenario 2.3. The snapshot is now explicitly converted to `journal_mode=delete` and
the test asserts no sidecars remain.

### 2.4 Credentials in params and tags

Params and tags are free text. In practice they contain `api_key`, `db_password`,
`hf_token`, SAS tokens, and connection strings with embedded passwords. A backup
multiplies those: every copy on every disk, every CI artifact, every shared drive.

This is a *confidentiality* failure caused by a *durability* practice — the two
concerns collide precisely at backup time, which is why the policy has to be
explicit rather than left to a default.

### 2.5 Silent divergence — the backup that was never checked

The most common state in the wild is not "no backup" but "a backup nobody has ever
restored". Files get excluded by a `.gitignore` or a sync rule; a directory gets
renamed; a disk fills mid-copy. Without a verification step these are invisible.

**Measured:** three damage modes were injected into a package and each was caught
*and attributed to the exact file* — a single flipped byte in an artifact, a deleted
artifact, and a truncated database. A verifier that only checks presence would have
passed all three.

---

## 3. Why the obvious fixes are insufficient

| Obvious fix | Why it fails |
|---|---|
| `cp mlflow.db backup/` | drops the WAL; not transactional; ignores artifacts |
| `tar czf backup.tgz mlruns/` | includes artifacts but no integrity record — you cannot tell a good archive from a truncated one |
| `rsync -a` | fast, resumable, no manifest, no logical check; silent partial syncs look like successes |
| `git` for the store | artifacts are binary and huge; `.gitignore` rules quietly drop files |
| "MLflow has a backup command" | it does not; there is no `mlflow backup`, `mlflow verify` or `mlflow restore` |
| Database dump (`sqlite3 .dump`) | text dump, slow, loses types/indexes fidelity, still no artifacts |

None of these produce an artefact that can answer *"is this restorable, and how do
you know?"*.

---

## 4. What "complete" has to mean

A backup is complete when three independent properties hold. Each needs its own
check because each fails on its own:

1. **Byte integrity** — every file is what it was when the backup was taken.
   → per-file SHA-256 + a `root_hash` over all of them.
2. **Structural integrity** — nothing promised is missing; nothing unexpected
   appeared. → manifest ↔ filesystem both ways.
3. **Semantic integrity** — the database still contains the same *rows*, and every
   run's artifact directory still holds files.

Point 3 is the one that catches the expensive failures, and it needs a subtlety:
after a restore the `artifact_uri` strings are *supposed* to differ. So the logical
digest is computed with `artifact_uri` **excluded**, and the manifest records both
the digest and which columns were excluded. Without that exclusion every restore
would look like corruption; without recording it, the exclusion would be a hole
nobody could audit.

---

## 5. Decisions taken, and what was rejected

| Decision | Rationale |
|---|---|
| Package is a plain directory, not an archive | `ls` + `sha256sum` must work without this tool; a backup you need special software to inspect is a liability |
| `verify` exits non-zero | it is meant to be a cron/CI gate, not a report you read |
| Secrets ledger stores hashes, not values | the ledger's job is "where do I rotate", not "what was it" |
| Backup keeps original secret values | a backup that edits payloads is not a backup; redaction is a separate, explicit act |
| `redact` forces `secure_delete` + `VACUUM` | an UPDATE leaves old bytes in free pages when `secure_delete` is off (6/6 still findable); the default is a build-time choice, so it is set explicitly |
| Restore defaults to a fresh directory | restoring into a fresh dir *is* the test; refusing to overwrite prevents the drill from destroying the evidence |
| Remote artifacts are recorded, not downloaded | downloading an S3 bucket is the provider's job; pretending to do it would be worse than saying so |
| `drill` includes a negative control | a test that cannot fail proves nothing; the control keeps the drill honest |

**Rejected:** encrypting the package. It sounds responsible, but a lost key converts
a recoverable backup into permanent data loss — the worst possible trade for a tool
whose entire purpose is durability. Encryption belongs at the volume/object-store
layer where key recovery is somebody's actual job. The `redact` path covers the
"share this safely" need without that risk.

**Rejected:** automatic scheduled backups inside the tool. Scheduling is the
environment's job (cron, systemd timer, CI). A CLI with honest exit codes composes;
a daemon does not.

---

## 6. Residual risk after using this tool

Stated plainly, because a risk analysis that ends with "risk eliminated" is not one:

* **Off-site copies are still your problem.** A verified backup on the same disk as
  the store survives corruption, not disk failure.
* **The package is not encrypted.** Use an encrypted volume or bucket policy.
* **Artifact payloads are not scanned** for embedded credentials.
* **Remote artifact stores** need their own backup and versioning.
* **Concurrent writers.** The snapshot is consistent, but a run writing *during* the
  backup may be only partially captured; the manifest records a warning when the
  store was locked at snapshot time.
* **Restore is not atomic.** A crash mid-restore leaves a partial target; the
  `RESTORE-REPORT.json` is written last, so its absence means "did not finish".

---

## 7. What the drill actually demonstrated

Run on this repository (`mlflow-vault drill --runs 4`, MLflow 3.17.1, Windows,
Python 3.13):

| Claim | Evidence |
|---|---|
| 4 runs, 12 artifacts, 85 DB rows survive a round trip | `restore`: rows matched, 0 dangling |
| The MLflow SDK can read the restored store | `readback`: 4 runs, 12 artifacts, metrics + params intact |
| Damage is detected and located | `tamper-detect`: 3/3 modes caught, file named |
| A db-only copy loses artifacts | `negative-control`: 4/4 dirs dangling, SDK readback failed |
| Redaction leaves no plaintext | raw byte scan of the whole package: 0 hits |
| A redacted package still verifies | `verify` on the redacted copy: OK, 14 files, 85 rows |

The interesting result is the last pair. Redaction that breaks verification would
be useless (you could not tell redaction from corruption); redaction that passes
verification while leaving plaintext in free pages would be dangerous. Both
properties are asserted by tests.

The `secure_delete` story is the clearest example of why the drill-and-CI loop
matters more than reasoning. The first version of that test asserted, as a fact of
nature, that a bare `UPDATE` leaves plaintext behind. It passed on Windows and
failed on the Ubuntu CI runner — because `secure_delete`'s default is a **build-time
option**, not a universal. The correct conclusion was not "the test was too strict"
but "the default must never be relied on": `redact` now sets the pragma explicitly,
and the test pins both halves (`OFF` → residue, `ON` → none) so a future SQLite
build cannot quietly change what the tool's safety claim means.
