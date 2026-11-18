"""Central place for repository paths. Override the data root with PATHSNAP_DATA."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "Student folder"))
CACHE_DIR = REPO_ROOT / "data_cache"
CONFIG_DIR = REPO_ROOT / "configs"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"


def partition_dir(part: str) -> Path:
    """Directory holding the trip folders of a development partition."""
    if part not in ("train", "student_test"):
        raise ValueError(f"development partitions are 'train' and 'student_test', got {part!r}")
    return DATA_ROOT / part


def list_trips(part: str) -> list[str]:
    return sorted(p.name for p in partition_dir(part).iterdir() if p.is_dir())


def trip_dir(part: str, trip_id: str) -> Path:
    return partition_dir(part) / trip_id
