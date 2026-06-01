"""Shared pytest fixtures: tmp DBs, Chroma client, env config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# HERMETIC REWARD CONFIG: aiutopia.env.reward rebinds its public constants from
# load_reward_config() AT IMPORT, resolving an on-disk overlay (default:
# <repo>/config/rewards.json — exactly where the GUI's PUT /api/rewards writes).
# Pin the overlay path to a guaranteed-absent file BEFORE any test imports reward,
# so the suite always sees the literal defaults regardless of a stray config left
# on disk by a developer using the GUI. Set at module import (collection time),
# before reward.py is first imported by a test module.
os.environ.setdefault(
    "AIUTOPIA_REWARD_CONFIG",
    str(Path(__file__).resolve().parent / "_nonexistent_rewards_overlay.json"),
)

from aiutopia.common.config import Paths
from aiutopia.common.logging import setup_logging


@pytest.fixture(autouse=True, scope="session")
def _logging_once() -> None:
    setup_logging("DEBUG")


@pytest.fixture
def aiutopia_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated AIUTOPIA_ROOT per test."""
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    p = Paths.from_env()
    p.ensure()
    return tmp_path


@pytest.fixture
def chroma_dir(aiutopia_root: Path) -> Path:
    return aiutopia_root / "chroma"


@pytest.fixture
def identity_db_path(aiutopia_root: Path) -> Path:
    return aiutopia_root / "identity.db"


@pytest.fixture
def planner_state_db_path(aiutopia_root: Path) -> Path:
    return aiutopia_root / "planner_state.db"
