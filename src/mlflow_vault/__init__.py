"""mlflow-vault — integrity-first backup, verification and restore for MLflow.

The package is intentionally dependency-free at runtime: it speaks to MLflow's
on-disk formats (SQLite tracking DB + artifact trees / ``mlruns`` file store)
directly, so it can still rescue a store when MLflow itself will not import.
"""

__version__ = "0.1.0"

MANIFEST_NAME = "manifest.json"
BLOB_DIR = "blobs"

__all__ = ["BLOB_DIR", "MANIFEST_NAME", "__version__"]
