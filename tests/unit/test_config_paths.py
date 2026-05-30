"""Guard the AIUTOPIA_DATA_DIR corruption-safety layout (commit bf89c97).

The volatile WAL-mode SQLite stores (identity.db, planner_state.db, Chroma) must
relocate under AIUTOPIA_DATA_DIR when set (local disk, off a file-sync client that
would corrupt -wal/-shm sidecars), while repo assets + bulk artifacts (weights,
runs, goal_templates) stay under AIUTOPIA_ROOT. Unset => data_root == root.
Structural assertions only (paths are .resolve()'d → absolute/platform-specific)."""

from aiutopia.common.config import Paths


def test_data_dir_unset_data_root_equals_root(monkeypatch):
    monkeypatch.setenv("AIUTOPIA_ROOT", "/tmp/aiutest_root")
    monkeypatch.delenv("AIUTOPIA_DATA_DIR", raising=False)
    p = Paths.from_env()
    assert p.data_root == p.root
    assert p.identity_db == p.root / "identity.db"
    assert p.planner_state_db == p.root / "planner_state.db"
    assert p.chroma_dir == p.root / "chroma"
    assert p.weights_dir == p.root / "weights"


def test_data_dir_set_relocates_only_volatile_stores(monkeypatch):
    monkeypatch.setenv("AIUTOPIA_ROOT", "/tmp/aiutest_root/myroot")
    monkeypatch.setenv("AIUTOPIA_DATA_DIR", "/tmp/aiu_localdisk")
    p = Paths.from_env()
    # volatile WAL stores relocate under data_root = <DATA_DIR>/<root.name>
    assert p.data_root != p.root
    assert p.data_root.name == p.root.name == "myroot"  # per-worker suffix preserved
    assert p.data_root.parent.name == "aiu_localdisk"
    assert p.identity_db.parent == p.data_root
    assert p.planner_state_db.parent == p.data_root
    assert p.chroma_dir.parent == p.data_root
    # repo assets + bulk artifacts STAY under root (not relocated)
    assert p.weights_dir == p.root / "weights"
    assert p.runs_dir == p.root / "runs"
    assert p.goal_templates == p.root / "goal_templates"


def test_data_dir_preserves_per_worker_suffix(monkeypatch):
    # env/wrapper suffixes AIUTOPIA_ROOT as ..._w{N}; data_root must carry it so
    # concurrent EnvRunners keep isolated stores (root.name == _w3 dir).
    monkeypatch.setenv("AIUTOPIA_ROOT", "/tmp/aiutest_root_w3")
    monkeypatch.setenv("AIUTOPIA_DATA_DIR", "/tmp/aiu_localdisk")
    p = Paths.from_env()
    assert p.data_root.name == "aiutest_root_w3"
    assert p.identity_db.parent.name == "aiutest_root_w3"
