import os
from pathlib import Path

STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/app/storage")


def ensure_storage_dirs():
    Path(STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
    Path(f"{STORAGE_ROOT}/uploads").mkdir(parents=True, exist_ok=True)
    Path(f"{STORAGE_ROOT}/results").mkdir(parents=True, exist_ok=True)
    Path(f"{STORAGE_ROOT}/reports").mkdir(parents=True, exist_ok=True)


def get_uploads_dir() -> str:
    return f"{STORAGE_ROOT}/uploads"


def get_reports_dir() -> str:
    return f"{STORAGE_ROOT}/reports"
