"""Shared pytest fixtures: tmp DBs, Chroma client, env config."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

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
