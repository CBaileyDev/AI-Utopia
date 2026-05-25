# AI Utopia — Milestone 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project foundation (repo layout, dependencies, identity DB, schemas, Chroma memory layer, Fabric mod scaffold with Py4J bridge, env wrapper, CLI, determinism harness scaffold, backup scripts) so that one command — `aiutopia agent spawn --role gatherer` — produces a Carpet fake player visible in a connected Minecraft client, with `observationsAll()` returning a valid JSON blob and PettingZoo `env.reset()` returning a valid Dict observation. M0 builds zero policy training; that is M1+.

**Architecture:** Python package `src/aiutopia/` (PEP-621 / hatchling); Fabric Java mod `fabric_mod/` forked from UnionClef Py4J pattern; two SQLite DBs (`identity.db`, `planner_state.db`); local Chroma. Critical-path constraint: all five §6-review bug fixes (ZGC, `additional_module_specs`, batched `observationsAll`, env `close()`, CUDA determinism fixture) are baked in from day 1 — they are foundation, not patches.

**Tech Stack:** Python 3.12, Ray RLlib 2.x (declared dep but not exercised in M0), PettingZoo Parallel API, Pydantic v2, Chroma, Typer, python-ulid, py4j, sentence-transformers (BGE-small-en-v1.5), pytest, ruff, mypy. Fabric Loader 0.16+, Fabric API for MC 1.21.1 (UnionClef baseline), Carpet, Lithium, FerriteCore.

**Spec reference:** `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`

---

## Spec §-touched-in-M0 vs deferred to M1+

| Spec § | Coverage | M0 task(s) |
|---|---|---|
| §1 Scope / hardware / budget | Documented in README + `pyproject.toml` metadata | T2, T36 |
| §2.1 Two-world topology | Production-side stub; training-side deferred | T19, T26 |
| §2.2 Per-agent three-tier brain | Tier 2 GoalSpecAdapter signature stubbed; Tier 1 LLM + Tier 3 RL stubs | T12, T18 |
| §2.3 Seed strategy | `fixed_easy` seed plumbed through env config; full mix-in is M4 | T26, T27 |
| §3.1 GoalSpecAdapter (BGE + hard dispatch) | Implemented (frozen BGE loads, adapter returns 512-d) | T18 |
| §3.2 Carpet fake-player embodiment | Implemented (`/player spawn` via Py4J) | T20, T31 |
| §3.3 ChatBridge mixin | Mixin compiles and registers; planner round-trip is M5 | T23 |
| §3.4 Planner state persistence | Schema migrated; startup-resync code is M5 | T9 |
| §3.5 Identity schema (Q5b death/succession) | Full 5-table schema + IdentityService; succession code is unit-tested with dry-run, live succession is M4 | T8, T10, T11 |
| §3.6 Death + succession sequence | Dry-run path only (no live Carpet kill yet) | T11 |
| §4.1–4.2 Per-role obs/action Dicts | Gatherer only; builder/farmer/defender deferred to their milestones | T25 |
| §4.3 Shared backbone | Stubbed module file showing intended structure; real impl M1 | T28 |
| §4.4 Pixel patches | Deferred entirely | M2/M4 |
| §4.5 Action masking | Gatherer mask only | T25, T26 |
| §4.6 Per-tick RL loop | Stub motor responses; real loop M1 | T27 |
| §4.7 Event-driven LLM planner | Stub event queue; real planner M5 | T16 |
| §4.8 Inter-agent comm channel | `CommBus.flushBatch()` stub on Java side | T22 |
| §4.9 Episodic memory write path | EpisodicMemoryWriter implemented (importance + Chroma write); LLM-summary generation is M5 | T17 |
| §5 Reward architecture | Deferred entirely (no training in M0) | M1+ |
| §6.1–6.5 Pydantic v2 schemas | Fully implemented | T13, T14, T15 |
| §6.6 DAG state machine | Stubbed enum + transition table; runtime is M5 | T15 |
| §6.7 Plan cache | Schema migrated; live cache logic M5 | T9 |
| §7.1 PPOConfig | Stub config file documenting M1 target shape | T28 |
| §7.2 RLModule (with `additional_module_specs`) | Stub file with class skeletons; real impl M1 | T28 |
| §7.3 EnvWrapper (with `observationsAll`, `close()`) | Implemented for gatherer | T26, T27 |
| §7.4 Training driver | Deferred entirely | M1 |
| §7.5 CLI tooling | `agent spawn`, `memory inspect`, `determinism check` implemented; others stubbed | T29, T30, T34 |
| §7.6 Production Compose (ZGC) | `docker-compose.production.yml` skeleton committed; real deploy M6 | T24, T36 |
| §7.7 Tauri dashboard SSE | Deferred entirely | M6 |
| §7.8 Determinism harness (CUDA fixture) | Scaffold + CUDA fixture in place; won't pass without weights | T32, T33 |

**Surfacing a spec inconsistency:** The user's M0 scope splits §3.5 tables: `roles`, `agents`, `agent_lives`, `funerals`, `policy_deployments` → `identity.db` (5 tables), and `chat_failures`, `planner_failures` → `planner_state.db` (alongside `planner_state` + `plan_cache`). The spec lumps all 7 tables under §3.5 without naming this physical split. **Decide before T8:** keep the split as planned here, or amend §3.5 to be explicit. The plan proceeds with the split because it matches data lineage (identity facts vs operational logs); flag if you want me to amend the spec.

**Early decision required at T19:** UnionClef targets MC 1.21.1. If you pin a later 1.21.x for Lithium/Carpet/Fabric API reasons, the UnionClef fork is non-trivial. T19 verifies compatibility first; on incompatibility, fall back to a clean `Py4JEntryPoint` reading UnionClef as reference (adds ~3–5 days).

---

## File structure (final M0 state)

```
.
├── IMPLEMENTATION_PLAN.md                                        (this file)
├── README.md                                                     (T36)
├── pyproject.toml                                                (T2)
├── ruff.toml                                                     (T3)
├── .python-version                                               (T1)
├── .gitignore                                                    (already committed)
├── docs/superpowers/specs/2026-05-25-…-design.md                 (already committed)
├── src/aiutopia/
│   ├── __init__.py                                               (T1)
│   ├── __version__.py                                            (T1)
│   ├── common/
│   │   ├── __init__.py
│   │   ├── ids.py                                                (T4)
│   │   ├── config.py                                             (T5)
│   │   └── logging.py                                            (T5)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── enums.py                                              (T13)
│   │   ├── plan.py                                               (T13, T14)
│   │   ├── failure.py                                            (T13)
│   │   ├── chat.py                                               (T13)
│   │   ├── state_machine.py                                      (T15)
│   │   └── versioning.py                                         (T15)
│   ├── identity/
│   │   ├── __init__.py
│   │   ├── service.py                                            (T10, T11)
│   │   ├── models.py                                             (T10)
│   │   ├── skin_pool.py                                          (T11)
│   │   └── migrations/
│   │       ├── identity_001_initial.sql                          (T8)
│   │       └── planner_state_001_initial.sql                     (T9)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── client.py                                             (T16)
│   │   ├── writer.py                                             (T17)
│   │   └── retriever.py                                          (T17)
│   ├── env/
│   │   ├── __init__.py
│   │   ├── spaces.py                                             (T25)
│   │   ├── action_mask.py                                        (T25)
│   │   ├── bridge.py                                             (T26)
│   │   └── wrapper.py                                            (T27)
│   ├── rl_module/                                                (stubs only — M1)
│   │   ├── __init__.py
│   │   └── stubs.py                                              (T28)
│   ├── planner/                                                  (stubs only — M5)
│   │   ├── __init__.py
│   │   └── stubs.py                                              (T28)
│   ├── determinism/
│   │   ├── __init__.py
│   │   └── harness.py                                            (T32)
│   └── cli/
│       ├── __init__.py
│       ├── app.py                                                (T29)
│       ├── agent.py                                              (T29, T30)
│       ├── memory.py                                             (T34)
│       └── determinism.py                                        (T33)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                                               (T7)
│   ├── unit/
│   │   ├── test_ids.py                                           (T4)
│   │   ├── test_schemas_plan.py                                  (T14)
│   │   ├── test_schemas_failure.py                               (T13)
│   │   ├── test_schemas_chat.py                                  (T13)
│   │   ├── test_state_machine.py                                 (T15)
│   │   ├── test_identity.py                                      (T11)
│   │   ├── test_skin_pool.py                                     (T11)
│   │   ├── test_memory_writer.py                                 (T17)
│   │   ├── test_memory_retriever.py                              (T17)
│   │   └── test_env_spaces.py                                    (T25)
│   ├── integration/
│   │   ├── test_chroma_smoke.py                                  (T16)
│   │   ├── test_env_smoke.py                                     (T27)
│   │   └── test_cli_spawn.py                                     (T30)
│   └── determinism/
│       ├── conftest.py                                           (T32)
│       └── test_seeded_replay_scaffold.py                        (T32)
├── fabric_mod/
│   ├── build.gradle
│   ├── gradle.properties                                         (T19)
│   ├── settings.gradle
│   ├── gradle/wrapper/                                           (T19 — gradle wrapper)
│   ├── src/main/java/dev/aiutopia/mod/
│   │   ├── AiUtopiaMod.java                                      (T20)
│   │   ├── Py4JEntryPoint.java                                   (T20)
│   │   ├── bridge/
│   │   │   ├── MotorBridge.java                                  (T20)
│   │   │   ├── CommBus.java                                      (T22)
│   │   │   └── WorldOps.java                                     (T20)
│   │   └── mixin/
│   │       ├── KickPlayerMixin.java                              (T21; PlayerListMixin deferred to M1)
│   │       └── ChatMessageMixin.java                             (T23)
│   └── src/main/resources/
│       ├── fabric.mod.json                                       (T19)
│       └── aiutopia.mixins.json                                  (T19)
├── scripts/
│   ├── backup-daily.sh                                           (T35)
│   ├── backup-weekly.sh                                          (T35)
│   ├── crontab.example                                           (T35)
│   └── smoke-test.sh                                              (T30)
├── docker-compose.production.yml                                 (T24 — skeleton only)
└── secrets/
    └── .gitkeep                                                  (T24)
```

---

## Tasks

### Task 1: Initialize Python package skeleton

**Files:**
- Create: `.python-version`
- Create: `src/aiutopia/__init__.py`
- Create: `src/aiutopia/__version__.py`
- Create: `src/aiutopia/common/__init__.py`
- Create: `src/aiutopia/schemas/__init__.py`
- Create: `src/aiutopia/identity/__init__.py`
- Create: `src/aiutopia/identity/migrations/__init__.py`
- Create: `src/aiutopia/memory/__init__.py`
- Create: `src/aiutopia/env/__init__.py`
- Create: `src/aiutopia/rl_module/__init__.py`
- Create: `src/aiutopia/planner/__init__.py`
- Create: `src/aiutopia/determinism/__init__.py`
- Create: `src/aiutopia/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/determinism/__init__.py`

- [ ] **Step 1: Pin Python version**

Create `.python-version`:
```
3.12
```

- [ ] **Step 2: Create top-level package init**

Create `src/aiutopia/__init__.py`:
```python
"""AI Utopia — multi-agent Minecraft AI village."""
from aiutopia.__version__ import __version__

__all__ = ["__version__"]
```

Create `src/aiutopia/__version__.py`:
```python
__version__ = "0.0.0+m0"
```

- [ ] **Step 3: Create empty submodule inits**

Each of these files contains only the module docstring. Create them with these contents:

`src/aiutopia/common/__init__.py`:
```python
"""Common utilities — IDs, config, logging."""
```

`src/aiutopia/schemas/__init__.py`:
```python
"""Pydantic v2 schemas for LLM planner artifacts (§6 of spec)."""
```

`src/aiutopia/identity/__init__.py`:
```python
"""Identity service + SQLite migrations (§3.5)."""
```

`src/aiutopia/identity/migrations/__init__.py`:
```python
"""SQLite migration scripts (raw SQL files; loader in service.py)."""
```

`src/aiutopia/memory/__init__.py`:
```python
"""Episodic memory + skill library backed by Chroma (§4.9, §5.6)."""
```

`src/aiutopia/env/__init__.py`:
```python
"""PettingZoo Parallel env wrapper + Py4J bridge (§7.3)."""
```

`src/aiutopia/rl_module/__init__.py`:
```python
"""RLlib module specs — stubs in M0; real impl in M1."""
```

`src/aiutopia/planner/__init__.py`:
```python
"""LLM planner — stubs in M0; real impl in M5."""
```

`src/aiutopia/determinism/__init__.py`:
```python
"""Determinism harness utilities (§7.8)."""
```

`src/aiutopia/cli/__init__.py`:
```python
"""Typer CLI surface (§7.5)."""
```

`tests/__init__.py`:
```python
```

`tests/unit/__init__.py`:
```python
```

`tests/integration/__init__.py`:
```python
```

`tests/determinism/__init__.py`:
```python
```

- [ ] **Step 4: Verify layout**

Run: `find src tests -type f -name "*.py" | sort`

Expected output (exactly these files):
```
src/aiutopia/__init__.py
src/aiutopia/__version__.py
src/aiutopia/cli/__init__.py
src/aiutopia/common/__init__.py
src/aiutopia/determinism/__init__.py
src/aiutopia/env/__init__.py
src/aiutopia/identity/__init__.py
src/aiutopia/identity/migrations/__init__.py
src/aiutopia/memory/__init__.py
src/aiutopia/planner/__init__.py
src/aiutopia/rl_module/__init__.py
src/aiutopia/schemas/__init__.py
tests/__init__.py
tests/determinism/__init__.py
tests/integration/__init__.py
tests/unit/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add .python-version src tests
git commit -m "chore: scaffold Python package layout"
```

---

### Task 2: Create `pyproject.toml` with all M0 dependencies

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling>=1.21"]
build-backend = "hatchling.build"

[project]
name = "aiutopia"
version = "0.0.0+m0"
description = "Multi-agent Minecraft AI village — persistent RL with LLM coordination"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = { text = "MIT" }
authors = [{ name = "Carte", email = "barker_carter@icloud.com" }]
keywords = ["minecraft", "multi-agent", "reinforcement-learning", "llm"]

dependencies = [
  # MARL stack (declared in M0; exercised in M1)
  "ray[rllib,tune]>=2.40.0",
  "pettingzoo>=1.24.3",
  "gymnasium>=1.0.0",
  "torch>=2.4.0",
  # Schema layer (§6)
  "pydantic>=2.9.0",
  "python-ulid>=3.0.0",
  # Memory layer (§4.9, §5.6)
  "chromadb>=0.5.20",
  "sentence-transformers>=3.2.0",   # BGE-small-en-v1.5
  # Bridge to Java (§7.3)
  "py4j>=0.10.9.9",
  # CLI (§7.5)
  "typer>=0.13.0",
  "rich>=13.9.0",
  # Numerics
  "numpy>=2.0.0",
  "scipy>=1.14.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3.0",
  "pytest-asyncio>=0.24.0",
  "pytest-mock>=3.14.0",
  "pytest-xdist>=3.6.0",
  "ruff>=0.7.0",
  "mypy>=1.13.0",
  "pre-commit>=4.0.0",
]
planner-runtime = [
  "anthropic>=0.39.0",              # Claude Haiku (M5)
  "fastapi>=0.115.0",                # dashboard SSE sidecar (M6)
  "sse-starlette>=2.1.0",
  "uvicorn[standard]>=0.32.0",
]

[project.scripts]
aiutopia = "aiutopia.cli.app:app"

[tool.hatch.build.targets.wheel]
packages = ["src/aiutopia"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = [
  "integration: marks tests requiring external services (Chroma, Fabric mod)",
  "determinism: marks determinism regression tests (require GPU + saved weights)",
  "slow: marks slow tests",
]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
warn_no_return = true
disallow_any_generics = true
disallow_untyped_defs = true
files = ["src/aiutopia"]

[[tool.mypy.overrides]]
module = ["py4j.*", "chromadb.*", "sentence_transformers.*", "ray.*", "pettingzoo.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Verify installable**

Run: `python -m pip install -e ".[dev]"`

Expected: installs without error; `aiutopia` script available.

Run: `aiutopia --help`

Expected: Typer help output (or `ModuleNotFoundError: aiutopia.cli.app` until T29 — that's fine for now; the install itself must succeed).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml with M0 dependencies"
```

---

### Task 3: Configure ruff + mypy + pre-commit

**Files:**
- Create: `ruff.toml`
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write ruff config**

Create `ruff.toml`:
```toml
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[lint]
select = [
  "E", "F", "W",         # pycodestyle / pyflakes
  "I",                    # isort
  "B",                    # bugbear
  "UP",                   # pyupgrade
  "SIM",                  # simplify
  "RUF",                  # ruff-specific
  "TID",                  # tidy imports
  "PL",                   # pylint subset
]
ignore = [
  "PLR0913",              # too many args (Pydantic models trip this)
  "PLR2004",              # magic value comparisons
]

[lint.per-file-ignores]
"tests/**/*.py" = ["PLR2004", "B018"]

[format]
quote-style = "double"
indent-style = "space"
```

- [ ] **Step 2: Write pre-commit config**

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.9.0
        args: [--config-file=pyproject.toml]
        files: ^src/aiutopia/
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=512]
```

- [ ] **Step 3: Install pre-commit hooks**

Run: `pre-commit install`

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 4: Run on all files (should pass on empty package)**

Run: `pre-commit run --all-files`

Expected: all hooks PASS (some may report "Files were modified by this hook" the first time — that's the formatter normalizing line endings; safe to commit).

- [ ] **Step 5: Commit**

```bash
git add ruff.toml .pre-commit-config.yaml
git commit -m "chore: add ruff, mypy, pre-commit configuration"
```

---

### Task 4: ULID utilities (`aiutopia.common.ids`)

**Files:**
- Create: `src/aiutopia/common/ids.py`
- Create: `tests/unit/test_ids.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ids.py`:
```python
import re

import pytest

from aiutopia.common.ids import (
    ULID_REGEX,
    is_ulid,
    new_agent_uuid,
    new_plan_id,
    new_subgoal_id,
    new_report_id,
    new_event_id,
    new_ulid,
    skill_library_id_for,
    memory_id_for,
)


def test_new_ulid_returns_valid_crockford_base32() -> None:
    value = new_ulid()
    assert re.fullmatch(ULID_REGEX, value), value


def test_factories_each_return_ulid_strings() -> None:
    for fn in (new_agent_uuid, new_plan_id, new_subgoal_id,
               new_report_id, new_event_id):
        assert is_ulid(fn())


def test_chroma_id_helpers_use_agent_uuid_verbatim() -> None:
    uuid = "01J0CABCDEFGHJKMNPQRSTVWXY"
    assert skill_library_id_for(uuid) == f"skill_lib_{uuid}"
    assert memory_id_for(uuid) == f"mem_{uuid}"


def test_is_ulid_rejects_uuidv4_and_garbage() -> None:
    assert not is_ulid("550e8400-e29b-41d4-a716-446655440000")
    assert not is_ulid("not-a-ulid")
    assert not is_ulid("")
    # I, L, O, U are excluded from Crockford base32
    assert not is_ulid("01J0CABCDEFGHIJKLMNPQRSTVW")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ids.py -v`

Expected: FAIL with `ImportError: cannot import name 'ULID_REGEX' from 'aiutopia.common.ids'` (module doesn't exist).

- [ ] **Step 3: Implement the module**

Create `src/aiutopia/common/ids.py`:
```python
"""ULID-only identifier helpers (§3.5 conventions)."""
from __future__ import annotations

import re

import ulid

# Crockford base32: 0-9 + A-HJKMNP-TV-Z (no I, L, O, U)
ULID_REGEX = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def new_ulid() -> str:
    """Return a new ULID as a 26-character Crockford-base32 string.

    NOTE: not monotonic across calls — each ULID's random component is
    independent. Fine at our event rate (orders of magnitude below the
    1M/sec where collisions become a concern). If strict ordering within
    the same millisecond is ever needed, switch to a MonotonicULIDFactory.
    """
    return str(ulid.ULID())


def is_ulid(value: str) -> bool:
    """True iff `value` matches the ULID regex."""
    return bool(ULID_REGEX.fullmatch(value))


# Domain factories (each returns a fresh ULID; named for grep-ability)
def new_agent_uuid() -> str:
    return new_ulid()


def new_plan_id() -> str:
    return new_ulid()


def new_subgoal_id() -> str:
    return new_ulid()


def new_report_id() -> str:
    return new_ulid()


def new_event_id() -> str:
    return new_ulid()


def skill_library_id_for(agent_uuid: str) -> str:
    """Chroma collection name for an agent's skill library (§3.5 convention)."""
    if not is_ulid(agent_uuid):
        raise ValueError(f"not a ULID: {agent_uuid!r}")
    return f"skill_lib_{agent_uuid}"


def memory_id_for(agent_uuid: str) -> str:
    """Chroma collection name for an agent's episodic memory (§3.5 convention)."""
    if not is_ulid(agent_uuid):
        raise ValueError(f"not a ULID: {agent_uuid!r}")
    return f"mem_{agent_uuid}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_ids.py -v`

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/common/ids.py tests/unit/test_ids.py
git commit -m "feat(common): ULID helpers + chroma collection name conventions"
```

---

### Task 5: Config + logging (`aiutopia.common.config`, `aiutopia.common.logging`)

**Files:**
- Create: `src/aiutopia/common/config.py`
- Create: `src/aiutopia/common/logging.py`

- [ ] **Step 1: Write config module**

Create `src/aiutopia/common/config.py`:
```python
"""Filesystem paths + environment-driven config (§7.6 layout)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout. All paths are absolute."""

    root:             Path
    identity_db:      Path
    planner_state_db: Path
    chroma_dir:       Path
    weights_dir:      Path
    goal_templates:   Path
    runs_dir:         Path
    schema_migrations: Path
    logs_dir:         Path
    secrets_dir:      Path

    @classmethod
    def from_env(cls) -> "Paths":
        root = _env_path("AIUTOPIA_ROOT", "/var/lib/aiutopia")
        return cls(
            root              = root,
            identity_db       = root / "identity.db",
            planner_state_db  = root / "planner_state.db",
            chroma_dir        = root / "chroma",
            weights_dir       = root / "weights",
            goal_templates    = root / "goal_templates",
            runs_dir          = root / "runs",
            schema_migrations = root / "schema_migrations" / "llm_plan",
            logs_dir          = root / "logs",
            secrets_dir       = root / "secrets",
        )

    def ensure(self) -> None:
        """Create directories that should exist at runtime. Idempotent."""
        for p in (self.root, self.chroma_dir, self.weights_dir,
                  self.goal_templates, self.runs_dir, self.schema_migrations,
                  self.logs_dir, self.secrets_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LLMConfig:
    model:                 str
    budget_hard_cap_usd:   float
    qwen_local_url:        str | None
    anthropic_api_key_file: Path | None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        key_file_str = os.environ.get("ANTHROPIC_API_KEY_FILE")
        return cls(
            model               = os.environ.get("LLM_MODEL", "claude-haiku"),
            budget_hard_cap_usd = float(os.environ.get(
                                    "LLM_BUDGET_HARD_CAP_USD_MONTH", "80")),
            qwen_local_url      = os.environ.get("QWEN_LOCAL_URL"),
            anthropic_api_key_file = Path(key_file_str) if key_file_str else None,
        )


@dataclass(frozen=True)
class Py4JConfig:
    training_ports: tuple[int, ...]
    production_port: int

    @classmethod
    def from_env(cls) -> "Py4JConfig":
        train = tuple(int(p) for p in
                      os.environ.get("PY4J_TRAINING_PORTS",
                                     "25001,25002,25003,25004").split(","))
        return cls(
            training_ports  = train,
            production_port = int(os.environ.get("PY4J_PRODUCTION_PORT", "25100")),
        )
```

- [ ] **Step 2: Write logging module**

Create `src/aiutopia/common/logging.py`:
```python
"""Structured logging setup. Default: human-readable to stderr; JSON when AIUTOPIA_LOG_JSON=1."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      time.time(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in {"args", "msg", "exc_info", "exc_text", "stack_info",
                     "lineno", "pathname", "filename", "funcName",
                     "module", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process", "name",
                     "levelname", "levelno", "created"}:
                continue
            payload[k] = v
        return json.dumps(payload, default=str)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger. Call once at process start."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("AIUTOPIA_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 3: Sanity-check both modules import**

Run: `python -c "from aiutopia.common.config import Paths, LLMConfig, Py4JConfig; from aiutopia.common.logging import setup_logging, get_logger; setup_logging(); get_logger('smoke').info('ok'); print(Paths.from_env())"`

Expected: prints `ok` line then a `Paths(...)` repr. No traceback.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/common/config.py src/aiutopia/common/logging.py
git commit -m "feat(common): paths, LLM config, py4j config, structured logging"
```

---

### Task 6: SQLite migration runner

**Files:**
- Create: `src/aiutopia/identity/migrations_runner.py`
- Create: `tests/unit/test_migrations_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_migrations_runner.py`:
```python
import sqlite3
from pathlib import Path

import pytest

from aiutopia.identity.migrations_runner import (
    apply_migrations,
    applied_migrations,
)


def test_apply_migrations_creates_versions_table_and_runs_files(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "001_create_widgets.sql").write_text(
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT);"
    )
    (mig_dir / "002_seed_widget.sql").write_text(
        "INSERT INTO widgets (name) VALUES ('hello');"
    )

    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)

    with sqlite3.connect(db) as conn:
        widgets = conn.execute("SELECT name FROM widgets").fetchall()
        applied = conn.execute(
            "SELECT name FROM _schema_migrations ORDER BY applied_at"
        ).fetchall()

    assert widgets == [("hello",)]
    assert [r[0] for r in applied] == [
        "001_create_widgets.sql",
        "002_seed_widget.sql",
    ]


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "001_create.sql").write_text(
        "CREATE TABLE t (id INTEGER PRIMARY KEY);"
    )
    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)
    apply_migrations(db, mig_dir)  # second run is a no-op
    assert applied_migrations(db) == ["001_create.sql"]


def test_apply_migrations_runs_in_filename_order(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "002_b.sql").write_text("CREATE TABLE b (id INTEGER PRIMARY KEY);")
    (mig_dir / "001_a.sql").write_text("CREATE TABLE a (id INTEGER PRIMARY KEY);")
    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)
    assert applied_migrations(db) == ["001_a.sql", "002_b.sql"]
```

- [ ] **Step 2: Run the test, expect failure**

Run: `pytest tests/unit/test_migrations_runner.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the runner**

Create `src/aiutopia/identity/migrations_runner.py`:
```python
"""Minimal forward-only SQLite migration runner.

Migrations are filename-ordered (e.g. `001_initial.sql`, `002_add_index.sql`).
A `_schema_migrations` table tracks which have been applied; running twice is
a no-op.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  INTEGER NOT NULL
)
"""


def applied_migrations(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_BOOTSTRAP_SQL)
        rows = conn.execute(
            "SELECT name FROM _schema_migrations ORDER BY applied_at, name"
        ).fetchall()
    return [r[0] for r in rows]


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into complete statements using sqlite3's parser.

    Walks the buffer with `sqlite3.complete_statement()` so triggers,
    multi-line expressions, and `;` inside string literals are handled
    correctly (string-split on `;` would mis-handle these)."""
    statements: list[str] = []
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            stripped = buf.strip()
            if stripped and not stripped.startswith("--"):
                statements.append(stripped)
            buf = ""
    tail = buf.strip()
    if tail and not tail.startswith("--"):
        statements.append(tail)
    return statements


def apply_migrations(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply every `*.sql` file in `migrations_dir` (sorted) not yet applied.

    Each migration runs as a SINGLE real transaction. We CANNOT use
    `executescript()` here because it issues an implicit COMMIT before
    running the script, defeating any wrapping BEGIN/ROLLBACK. Instead
    we parse the file via `sqlite3.complete_statement()` and execute
    statements one-by-one inside a real txn.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in migrations_dir.iterdir()
                   if p.is_file() and p.suffix == ".sql")
    newly_applied: list[str] = []

    with sqlite3.connect(db_path) as conn:
        # Bootstrap is a single idempotent CREATE TABLE IF NOT EXISTS;
        # `executescript`'s implicit COMMIT is harmless here.
        conn.executescript(_BOOTSTRAP_SQL)
        already = {r[0] for r in conn.execute(
            "SELECT name FROM _schema_migrations").fetchall()}

        for f in files:
            if f.name in already:
                continue
            sql = f.read_text(encoding="utf-8")
            statements = _split_statements(sql)
            try:
                conn.execute("BEGIN")
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO _schema_migrations (name, applied_at) "
                    "VALUES (?, ?)",
                    (f.name, int(time.time())),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass  # already auto-rolled-back; preserve original exc
                raise
            newly_applied.append(f.name)

    return newly_applied
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_migrations_runner.py -v`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/identity/migrations_runner.py tests/unit/test_migrations_runner.py
git commit -m "feat(identity): forward-only sqlite migration runner"
```

---

### Task 7: pytest fixtures (`tests/conftest.py`)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the fixture module**

Create `tests/conftest.py`:
```python
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
```

- [ ] **Step 2: Sanity-check by re-running an existing test**

Run: `pytest tests/unit/test_ids.py -v`

Expected: still PASS. (Fixtures are unused by `test_ids.py` but autouse logging must not break anything.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared fixtures (logging, tmp DBs, isolated AIUTOPIA_ROOT)"
```

---

### Task 8: `identity.db` schema migration (5 tables from §3.5)

**Files:**
- Create: `src/aiutopia/identity/migrations/identity_001_initial.sql`

- [ ] **Step 1: Write the migration**

Create `src/aiutopia/identity/migrations/identity_001_initial.sql`:
```sql
-- §3.5 — Identity DB initial schema (5 tables: roles, agents, agent_lives,
-- funerals, policy_deployments).
-- Note: chat_failures and planner_failures from §3.5 live in planner_state.db
-- per M0 scope (see IMPLEMENTATION_PLAN.md "Spec inconsistency surfaced").

CREATE TABLE roles (
    role_id                    TEXT PRIMARY KEY,
    display_name               TEXT NOT NULL,
    policy_weights_path        TEXT NOT NULL,
    policy_version             INTEGER NOT NULL,
    observation_schema_version INTEGER NOT NULL,
    action_schema_version      INTEGER NOT NULL,
    max_lives                  INTEGER NOT NULL DEFAULT 1,
    default_skin_pool          TEXT
);

CREATE TABLE agents (
    agent_uuid          TEXT PRIMARY KEY,
    role_id             TEXT NOT NULL REFERENCES roles(role_id),
    agent_name          TEXT NOT NULL,
    skill_library_id    TEXT NOT NULL,
    memory_id           TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('alive', 'dead')),
    born_at             INTEGER NOT NULL,
    died_at             INTEGER,
    spawn_position_json TEXT,
    current_skin        TEXT
);

CREATE UNIQUE INDEX idx_agent_name_alive
    ON agents(agent_name) WHERE status = 'alive';

CREATE TABLE agent_lives (
    life_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
    role_id        TEXT NOT NULL REFERENCES roles(role_id),
    born_at        INTEGER NOT NULL,
    died_at        INTEGER,
    cause_of_death TEXT
);

CREATE TABLE funerals (
    funeral_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    deceased_agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
    witness_agent_uuids_json TEXT NOT NULL,
    event_summary           TEXT NOT NULL,
    written_to_memory_at    INTEGER NOT NULL
);

CREATE TABLE policy_deployments (
    deployment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    role_id          TEXT NOT NULL REFERENCES roles(role_id),
    from_version     INTEGER,
    to_version       INTEGER NOT NULL,
    deployed_at      INTEGER NOT NULL,
    deployed_by      TEXT NOT NULL,
    notes            TEXT
);

-- Seed the 4 roles with stub policy paths (filled by promote-weights CLI later).
INSERT INTO roles (role_id, display_name, policy_weights_path, policy_version,
                   observation_schema_version, action_schema_version,
                   max_lives, default_skin_pool) VALUES
  ('gatherer', 'Gatherer', '', 0, 1, 1, 1,
   json('["Bjorn","Gunnar","Sigrid","Eirik","Astrid","Halvor","Frida","Magnus","Ingrid","Knut","Solveig","Olav"]')),
  ('builder',  'Builder',  '', 0, 1, 1, 1,
   json('["Bram","Lisa","Hugo","Maeve","Cedric","Nora","Oscar","Petra","Rolf","Saga","Tomas","Vera"]')),
  ('farmer',   'Farmer',   '', 0, 1, 1, 1,
   json('["Hannah","Idris","Jorah","Kara","Linus","Mira","Niko","Otto","Pia","Quinn","Reza","Sten"]')),
  ('defender', 'Defender', '', 0, 1, 1, 1,
   json('["Thora","Ulf","Vidar","Wilma","Xander","Yara","Zane","Anja","Bodvar","Cara","Dag","Eira"]'));
```

- [ ] **Step 2: Verify schema applies cleanly**

Run:
```bash
python -c "
from pathlib import Path
from aiutopia.identity.migrations_runner import apply_migrations
import importlib.resources as r

mig_dir = Path('src/aiutopia/identity/migrations')
applied = apply_migrations(Path('/tmp/aiutopia-smoke-identity.db'), mig_dir)
print('applied:', applied)
"
```

Expected: prints `applied: ['identity_001_initial.sql']` on first run. Second run prints `applied: []`.

- [ ] **Step 3: Verify seeded roles exist**

Run:
```bash
sqlite3 /tmp/aiutopia-smoke-identity.db "SELECT role_id, display_name, max_lives FROM roles ORDER BY role_id;"
```

Expected:
```
builder|Builder|1
defender|Defender|1
farmer|Farmer|1
gatherer|Gatherer|1
```

- [ ] **Step 4: Clean up smoke artifact**

Run: `rm -f /tmp/aiutopia-smoke-identity.db`

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/identity/migrations/identity_001_initial.sql
git commit -m "feat(identity): identity.db initial schema (5 tables) + seed roles"
```

---

### Task 9: `planner_state.db` schema migration

**Files:**
- Create: `src/aiutopia/identity/migrations/planner_state_001_initial.sql`

- [ ] **Step 1: Write the migration**

Create `src/aiutopia/identity/migrations/planner_state_001_initial.sql`:
```sql
-- §3.4 + §6.7 — planner_state.db initial schema:
--   planner_state, plan_cache, chat_failures, planner_failures.
-- This DB intentionally co-locates operational tables that change frequently
-- (vs identity.db which holds the slower-moving identity facts).

CREATE TABLE planner_state (
    plan_id              TEXT PRIMARY KEY,
    agent_uuid           TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN
                           ('active', 'completed', 'failed', 'paused',
                            'failed_migration')),
    dag_json             TEXT NOT NULL,
    current_subgoal_id   TEXT,
    pending_events_jsonl TEXT,
    llm_call_log_jsonl   TEXT,
    schema_version       TEXT NOT NULL,
    created_at           INTEGER NOT NULL,
    last_updated         INTEGER NOT NULL
);

CREATE INDEX idx_planner_active
    ON planner_state(status, agent_uuid)
    WHERE status IN ('active', 'paused');

CREATE TABLE plan_cache (
    cache_key            TEXT PRIMARY KEY,
    context_json         TEXT NOT NULL,
    prompt_text          TEXT NOT NULL,
    plan_json            TEXT NOT NULL,
    llm_model            TEXT NOT NULL,
    llm_call_latency_ms  INTEGER NOT NULL,
    llm_call_cost_usd    REAL,
    hit_count            INTEGER NOT NULL DEFAULT 1,
    created_at           INTEGER NOT NULL,
    last_hit_at          INTEGER NOT NULL,
    ttl_seconds          INTEGER NOT NULL DEFAULT 3600,
    invalidated          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_plan_cache_last_hit
    ON plan_cache(last_hit_at) WHERE invalidated = 0;

CREATE TABLE chat_failures (
    failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_uuid    TEXT NOT NULL,
    player_uuid   TEXT NOT NULL,
    text          TEXT NOT NULL,
    error_type    TEXT NOT NULL CHECK (error_type IN
                    ('timeout', 'api_error', 'qwen_unavail')),
    occurred_at   INTEGER NOT NULL
);

CREATE TABLE planner_failures (
    failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id       TEXT NOT NULL,
    failure_type  TEXT NOT NULL,
    detail_json   TEXT,
    occurred_at   INTEGER NOT NULL
);
```

- [ ] **Step 2: Verify schema applies cleanly**

Run:
```bash
python -c "
from pathlib import Path
from aiutopia.identity.migrations_runner import apply_migrations

mig_dir = Path('src/aiutopia/identity/migrations')
print('applied:', apply_migrations(Path('/tmp/aiutopia-smoke-planner.db'), mig_dir))
"
```

Expected: prints both migrations applied (the smoke DB is shared, so it will run BOTH `identity_001_initial.sql` and `planner_state_001_initial.sql`).

Note: in production these go to *different* DBs; the smoke runs them into one only because we're sharing the same `migrations/` directory for both DBs. Task 10 splits the migration dispatch by DB.

- [ ] **Step 3: Clean up**

Run: `rm -f /tmp/aiutopia-smoke-planner.db`

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/identity/migrations/planner_state_001_initial.sql
git commit -m "feat(identity): planner_state.db initial schema (4 tables)"
```

---

### Task 10: IdentityService — CRUD + dual-DB migration dispatch

**Files:**
- Create: `src/aiutopia/identity/models.py`
- Create: `src/aiutopia/identity/service.py`

- [ ] **Step 1: Write the models**

Create `src/aiutopia/identity/models.py`:
```python
"""Typed records for identity DB rows. Pydantic v2 used for validation
where data crosses a boundary; bare dataclasses for internal returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoleId = Literal["gatherer", "builder", "farmer", "defender"]
AgentStatus = Literal["alive", "dead"]


@dataclass(frozen=True, slots=True)
class Role:
    role_id:                    RoleId
    display_name:               str
    policy_weights_path:        str
    policy_version:             int
    observation_schema_version: int
    action_schema_version:      int
    max_lives:                  int
    default_skin_pool:          list[str]


@dataclass(frozen=True, slots=True)
class Agent:
    agent_uuid:          str
    role_id:             RoleId
    agent_name:          str
    skill_library_id:    str
    memory_id:           str
    status:              AgentStatus
    born_at:             int
    died_at:             int | None
    spawn_position_json: str | None
    current_skin:        str | None


@dataclass(frozen=True, slots=True)
class AgentLife:
    life_id:        int
    agent_uuid:     str
    role_id:        RoleId
    born_at:        int
    died_at:        int | None
    cause_of_death: str | None
```

- [ ] **Step 2: Write the service**

Create `src/aiutopia/identity/service.py`:
```python
"""IdentityService — CRUD over identity.db with §3.6 succession semantics.

In M0 the service is in-process synchronous SQLite. In M5+ it gains a
Py4J callback hook so on-death events from Fabric drive the same code
path. Death/spawn methods here are unit-tested with dry-runs only —
the Carpet `/player kill` and `/player spawn` calls are wired in CLI
T30 once the bridge is up.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aiutopia.common.ids import (
    is_ulid, memory_id_for, new_agent_uuid, skill_library_id_for,
)
from aiutopia.identity.migrations_runner import apply_migrations
from aiutopia.identity.models import Agent, AgentLife, Role, RoleId


# Subset of migrations belonging to identity.db (vs planner_state.db).
_IDENTITY_MIGRATIONS = ("identity_",)
_PLANNER_MIGRATIONS  = ("planner_state_",)


def _migrations_for(prefixes: tuple[str, ...], dir_path: Path) -> Path:
    """Materialize a temp dir containing only the matching migrations,
    so the shared runner only applies the right subset."""
    import shutil
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="aiutopia_mig_"))
    for f in dir_path.iterdir():
        if any(f.name.startswith(p) for p in prefixes):
            shutil.copy(f, out / f.name)
    return out


def init_identity_db(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply only the identity_*.sql migrations to db_path."""
    subset = _migrations_for(_IDENTITY_MIGRATIONS, migrations_dir)
    try:
        return apply_migrations(db_path, subset)
    finally:
        import shutil
        shutil.rmtree(subset)


def init_planner_state_db(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply only the planner_state_*.sql migrations to db_path."""
    subset = _migrations_for(_PLANNER_MIGRATIONS, migrations_dir)
    try:
        return apply_migrations(db_path, subset)
    finally:
        import shutil
        shutil.rmtree(subset)


class IdentityService:
    """All reads + writes for identity.db. Thread-safe per connection."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ───── reads ─────
    def get_role(self, role_id: RoleId) -> Role:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM roles WHERE role_id = ?", (role_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no such role: {role_id!r}")
        return Role(
            role_id=row["role_id"],
            display_name=row["display_name"],
            policy_weights_path=row["policy_weights_path"],
            policy_version=row["policy_version"],
            observation_schema_version=row["observation_schema_version"],
            action_schema_version=row["action_schema_version"],
            max_lives=row["max_lives"],
            default_skin_pool=json.loads(row["default_skin_pool"] or "[]"),
        )

    def get_agent(self, agent_uuid: str) -> Agent:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE agent_uuid = ?", (agent_uuid,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no such agent: {agent_uuid!r}")
        return Agent(
            agent_uuid=row["agent_uuid"],
            role_id=row["role_id"],
            agent_name=row["agent_name"],
            skill_library_id=row["skill_library_id"],
            memory_id=row["memory_id"],
            status=row["status"],
            born_at=row["born_at"],
            died_at=row["died_at"],
            spawn_position_json=row["spawn_position_json"],
            current_skin=row["current_skin"],
        )

    def list_living_agents(self) -> list[Agent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE status = 'alive' ORDER BY born_at"
            ).fetchall()
        return [self.get_agent(r["agent_uuid"]) for r in rows]

    # ───── writes ─────
    def spawn_agent(self, role_id: RoleId, agent_name: str, born_at: int,
                    spawn_position_json: str | None = None,
                    skin: str | None = None) -> Agent:
        agent_uuid = new_agent_uuid()
        skill_lib  = skill_library_id_for(agent_uuid)
        mem_id     = memory_id_for(agent_uuid)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agents
                       (agent_uuid, role_id, agent_name, skill_library_id,
                        memory_id, status, born_at, died_at,
                        spawn_position_json, current_skin)
                   VALUES (?, ?, ?, ?, ?, 'alive', ?, NULL, ?, ?)""",
                (agent_uuid, role_id, agent_name, skill_lib, mem_id,
                 born_at, spawn_position_json, skin),
            )
            conn.execute(
                """INSERT INTO agent_lives
                       (agent_uuid, role_id, born_at)
                   VALUES (?, ?, ?)""",
                (agent_uuid, role_id, born_at),
            )
        return self.get_agent(agent_uuid)

    def record_death(self, agent_uuid: str, died_at: int,
                     cause_of_death: str) -> None:
        if not is_ulid(agent_uuid):
            raise ValueError(f"not a ULID: {agent_uuid!r}")
        with self._conn() as conn:
            conn.execute(
                "UPDATE agents SET status='dead', died_at=? "
                "WHERE agent_uuid=? AND status='alive'",
                (died_at, agent_uuid),
            )
            conn.execute(
                "UPDATE agent_lives SET died_at=?, cause_of_death=? "
                "WHERE agent_uuid=? AND died_at IS NULL",
                (died_at, cause_of_death, agent_uuid),
            )

    def record_funeral(self, deceased_agent_uuid: str,
                       witness_uuids: list[str],
                       event_summary: str, written_at: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO funerals
                       (deceased_agent_uuid, witness_agent_uuids_json,
                        event_summary, written_to_memory_at)
                   VALUES (?, ?, ?, ?)""",
                (deceased_agent_uuid, json.dumps(witness_uuids),
                 event_summary, written_at),
            )
            return cur.lastrowid or -1
```

- [ ] **Step 3: Sanity-check it imports**

Run: `python -c "from aiutopia.identity.service import IdentityService, init_identity_db, init_planner_state_db; print('ok')"`

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/identity/models.py src/aiutopia/identity/service.py
git commit -m "feat(identity): IdentityService CRUD + per-DB migration dispatch"
```

---

### Task 11: Skin pool + identity service tests (incl. dry-run succession)

**Files:**
- Create: `src/aiutopia/identity/skin_pool.py`
- Create: `tests/unit/test_skin_pool.py`
- Create: `tests/unit/test_identity.py`

- [ ] **Step 1: Write the skin-pool failing test**

Create `tests/unit/test_skin_pool.py`:
```python
from aiutopia.identity.skin_pool import (
    pick_name, deterministic_skin_for_uuid, procedural_surname,
)


def test_pick_name_uses_pool_when_available() -> None:
    pool = ["Bjorn", "Gunnar", "Sigrid"]
    used = {"Bjorn"}
    name = pick_name(pool, used)
    assert name in {"Gunnar", "Sigrid"}


def test_pick_name_falls_back_to_first_plus_surname_when_exhausted() -> None:
    pool = ["Bjorn", "Gunnar"]
    used = {"Bjorn", "Gunnar"}
    name = pick_name(pool, used, seed=42)
    parts = name.split(" ")
    assert len(parts) == 2
    assert parts[0] in pool        # first name from pool
    assert parts[1].isalpha()      # procedural surname


def test_pick_name_never_returns_numbered_default() -> None:
    pool = ["Bjorn"]
    used = {"Bjorn"}
    for seed in range(50):
        name = pick_name(pool, used, seed=seed)
        assert "defender_" not in name
        assert "_" not in name      # no underscore-numbered names


def test_deterministic_skin_seeded_by_uuid_not_name() -> None:
    uuid_a = "01J0CABCDEFGHJKMNPQRSTVWXY"
    uuid_b = "01J0CABCDEFGHJKMNPQRSTVWX0"
    skins = ["Steve", "Alex", "Notch", "Herobrine"]
    sa = deterministic_skin_for_uuid(uuid_a, skins)
    sb = deterministic_skin_for_uuid(uuid_b, skins)
    assert sa in skins and sb in skins
    # Different UUIDs almost certainly map to different skins (4-element pool).
    # If this asserts equal, replace one UUID and rerun — birthday-paradox edge.


def test_procedural_surname_is_deterministic_with_seed() -> None:
    assert procedural_surname(seed=7) == procedural_surname(seed=7)
    assert procedural_surname(seed=7) != procedural_surname(seed=8)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/unit/test_skin_pool.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement skin_pool**

Create `src/aiutopia/identity/skin_pool.py`:
```python
"""Agent name + skin selection per §3.5 conventions.

- 12 names per role; on exhaustion fall back to "<first> <procedural-surname>".
- Skin selection is deterministic per agent_uuid (NOT per agent_name) so two
  agents that share a name (across lives) look different.
- Procedural surnames combine fixed roots and suffixes; never numeric.
"""
from __future__ import annotations

import hashlib
import random


_SURNAME_ROOTS = (
    "Iron", "Stone", "Oak", "Ash", "Pine", "Frost", "Storm", "River",
    "Hawk", "Wolf", "Bear", "Raven", "Fox", "Shadow", "Moon", "Sun",
)
_SURNAME_SUFFIXES = (
    "wood", "stone", "field", "hold", "bane", "ward", "blade",
    "thorn", "song", "claw", "fall", "gate", "vale", "mark",
)


def procedural_surname(seed: int) -> str:
    rng = random.Random(seed)
    return rng.choice(_SURNAME_ROOTS) + rng.choice(_SURNAME_SUFFIXES)


def pick_name(pool: list[str], used: set[str], seed: int | None = None) -> str:
    """Return an unused name from `pool`, or a procedural fallback when
    every pool name is in `used`.

    Determinism: when `seed` is given, fallback is reproducible. When None,
    a fresh random is used (production succession; succession_seed should
    be derived from current tick for replay).
    """
    available = [n for n in pool if n not in used]
    rng = random.Random(seed)
    if available:
        return rng.choice(available)
    first = rng.choice(pool)
    return f"{first} {procedural_surname(seed=rng.randrange(1 << 32))}"


def deterministic_skin_for_uuid(agent_uuid: str, skins: list[str]) -> str:
    """Deterministic pick from `skins` indexed by hash(agent_uuid)."""
    if not skins:
        raise ValueError("skin pool empty")
    h = hashlib.sha256(agent_uuid.encode()).digest()
    idx = int.from_bytes(h[:8], "big") % len(skins)
    return skins[idx]
```

- [ ] **Step 4: Verify skin-pool tests pass**

Run: `pytest tests/unit/test_skin_pool.py -v`

Expected: 5 PASSED.

- [ ] **Step 5: Write the identity-service failing test**

Create `tests/unit/test_identity.py`:
```python
from pathlib import Path

import pytest

from aiutopia.common.ids import is_ulid
from aiutopia.identity.service import (
    IdentityService, init_identity_db,
)


@pytest.fixture
def svc(identity_db_path: Path) -> IdentityService:
    init_identity_db(identity_db_path,
                     Path("src/aiutopia/identity/migrations"))
    return IdentityService(identity_db_path)


def test_seeded_roles_present(svc: IdentityService) -> None:
    for role_id in ("gatherer", "builder", "farmer", "defender"):
        role = svc.get_role(role_id)
        assert role.role_id == role_id
        assert role.max_lives == 1
        assert len(role.default_skin_pool) == 12


def test_spawn_agent_creates_alive_row_with_ulid_uuid(svc: IdentityService) -> None:
    agent = svc.spawn_agent("gatherer", "Bjorn", born_at=1_700_000_000)
    assert is_ulid(agent.agent_uuid)
    assert agent.status == "alive"
    assert agent.role_id == "gatherer"
    assert agent.agent_name == "Bjorn"
    assert agent.skill_library_id == f"skill_lib_{agent.agent_uuid}"
    assert agent.memory_id == f"mem_{agent.agent_uuid}"


def test_spawn_agent_rejects_duplicate_living_name(svc: IdentityService) -> None:
    svc.spawn_agent("gatherer", "Bjorn", born_at=1)
    with pytest.raises(Exception):
        svc.spawn_agent("gatherer", "Bjorn", born_at=2)


def test_dry_run_succession_role_persists_uuid_rotates(svc: IdentityService) -> None:
    # 1. Spawn Bjorn (defender)
    bjorn = svc.spawn_agent("defender", "Bjorn", born_at=100)

    # 2. Bjorn dies
    svc.record_death(bjorn.agent_uuid, died_at=200, cause_of_death="creeper")
    dead = svc.get_agent(bjorn.agent_uuid)
    assert dead.status == "dead"
    assert dead.died_at == 200

    # 3. Funeral written
    funeral_id = svc.record_funeral(
        deceased_agent_uuid=bjorn.agent_uuid,
        witness_uuids=[],
        event_summary="Bjorn the defender died defending the south wall.",
        written_at=201,
    )
    assert funeral_id > 0

    # 4. Successor spawns next morning — same role_id, NEW agent_uuid
    gunnar = svc.spawn_agent("defender", "Gunnar", born_at=24_000)
    assert gunnar.role_id == "defender"
    assert gunnar.agent_uuid != bjorn.agent_uuid
    assert gunnar.skill_library_id != bjorn.skill_library_id   # fresh memory
    assert gunnar.memory_id != bjorn.memory_id

    # 5. Both lives recorded
    living = svc.list_living_agents()
    assert [a.agent_name for a in living] == ["Gunnar"]


def test_record_death_rejects_non_ulid(svc: IdentityService) -> None:
    with pytest.raises(ValueError):
        svc.record_death("not-a-ulid", died_at=1, cause_of_death="x")
```

- [ ] **Step 6: Run identity tests**

Run: `pytest tests/unit/test_identity.py -v`

Expected: 5 PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/aiutopia/identity/skin_pool.py tests/unit/test_skin_pool.py tests/unit/test_identity.py
git commit -m "feat(identity): skin pool + dry-run succession unit tests"
```

---

### Task 12: Schema enums + base imports (`aiutopia.schemas.enums`)

**Files:**
- Create: `src/aiutopia/schemas/enums.py`

- [ ] **Step 1: Write the module**

Create `src/aiutopia/schemas/enums.py`:
```python
"""Closed-vocab Literal types shared across planner schemas (§6)."""
from __future__ import annotations

from typing import Literal

SCHEMA_VERSION_LLM_PLAN = "1.0.0"

RoleId = Literal["gatherer", "builder", "farmer", "defender"]

FailureType = Literal[
    "timeout", "health_critical", "tool_broken", "inventory_full",
    "path_blocked", "resource_unavailable", "attacked", "unknown",
]

PlannerSource = Literal[
    "claude-haiku", "local-qwen-14b", "stub-planner", "manual-cli",
]

ExpectedReplyType = Literal["text", "action_ack", "none"]

PlanStatus = Literal[
    "active", "completed", "failed", "paused", "failed_migration",
]

SubgoalState = Literal["pending", "active", "completed", "failed", "paused"]
```

- [ ] **Step 2: Sanity-check**

Run: `python -c "from aiutopia.schemas.enums import SCHEMA_VERSION_LLM_PLAN, RoleId; print(SCHEMA_VERSION_LLM_PLAN)"`

Expected: prints `1.0.0`.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/schemas/enums.py
git commit -m "feat(schemas): shared enums + SCHEMA_VERSION_LLM_PLAN"
```

---

### Task 13: `ChatEvent` and `FailureReport` Pydantic models

**Files:**
- Create: `src/aiutopia/schemas/chat.py`
- Create: `src/aiutopia/schemas/failure.py`
- Create: `tests/unit/test_schemas_chat.py`
- Create: `tests/unit/test_schemas_failure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_schemas_chat.py`:
```python
import pytest

from aiutopia.schemas.chat import ChatEvent


def test_minimal_chat_event_validates() -> None:
    e = ChatEvent(
        sender_player_uuid="player-mojang-uuid",
        sender_player_name="Carte",
        addressed_agent_uuid="01J0CABCDEFGHJKMNPQRSTVWXY",
        addressed_agent_name="Bjorn",
        text="where is the iron?",
        timestamp=1_700_000_000,
    )
    assert e.expected_reply_type == "text"
    assert e.suppressed_in_chat is True
    assert e.schema_version == "1.0.0"


def test_chat_event_rejects_too_long_text() -> None:
    with pytest.raises(Exception):
        ChatEvent(
            sender_player_uuid="x", sender_player_name="x",
            addressed_agent_uuid="x", addressed_agent_name="x",
            text="x" * 1_001, timestamp=1,
        )


def test_chat_event_rejects_empty_text() -> None:
    with pytest.raises(Exception):
        ChatEvent(
            sender_player_uuid="x", sender_player_name="x",
            addressed_agent_uuid="x", addressed_agent_name="x",
            text="", timestamp=1,
        )
```

Create `tests/unit/test_schemas_failure.py`:
```python
import pytest

from aiutopia.schemas.failure import (
    ExecutionTraceEntry, FailureDetails, FailureReport, PartialProgress,
)


def _make_report(**overrides) -> FailureReport:
    base = dict(
        plan_id="plan",
        subgoal_id="sg",
        role="gatherer",
        agent_uuid="01J0CABCDEFGHJKMNPQRSTVWXY",
        failure_details=FailureDetails(
            failure_type="timeout",
            failure_tick=1000,
            final_state_summary={"hunger": 4},
            descriptor_summary="ran out of time fetching wood",
        ),
        partial_progress=PartialProgress(progress_fraction=0.6),
        reported_at=1,
    )
    base.update(overrides)
    return FailureReport(**base)


def test_minimal_failure_report_validates() -> None:
    r = _make_report()
    assert r.status == "failed"
    assert r.failure_details.failure_type == "timeout"


def test_failure_type_must_be_in_closed_vocab() -> None:
    with pytest.raises(Exception):
        _make_report(failure_details=FailureDetails(
            failure_type="not_a_real_failure",   # type: ignore[arg-type]
            failure_tick=1, final_state_summary={},
            descriptor_summary="x",
        ))


def test_partial_progress_fraction_bounded() -> None:
    with pytest.raises(Exception):
        PartialProgress(progress_fraction=1.5)
    with pytest.raises(Exception):
        PartialProgress(progress_fraction=-0.1)


def test_execution_trace_capped_at_200() -> None:
    long_trace = [
        ExecutionTraceEntry(tick=i, action_summary="a",
                             observation_summary="o", reward=0.0)
        for i in range(201)
    ]
    with pytest.raises(Exception):
        FailureDetails(
            failure_type="timeout", failure_tick=1,
            final_state_summary={}, descriptor_summary="x",
            execution_trace=long_trace,
        )
```

- [ ] **Step 2: Verify failures**

Run: `pytest tests/unit/test_schemas_chat.py tests/unit/test_schemas_failure.py -v`

Expected: ImportError failures.

- [ ] **Step 3: Implement ChatEvent**

Create `src/aiutopia/schemas/chat.py`:
```python
"""§6.5 — ChatEvent."""
from __future__ import annotations

from pydantic import BaseModel, Field

from aiutopia.common.ids import new_event_id
from aiutopia.schemas.enums import ExpectedReplyType, SCHEMA_VERSION_LLM_PLAN


_ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"


class ChatEvent(BaseModel):
    event_id:             str  = Field(default_factory=new_event_id,
                                        pattern=_ULID_PATTERN)
    schema_version:       str  = SCHEMA_VERSION_LLM_PLAN
    sender_player_uuid:   str  = Field(..., description="Mojang UUID of player")
    sender_player_name:   str  = Field(..., min_length=1, max_length=16)
    addressed_agent_uuid: str  = Field(..., pattern=_ULID_PATTERN)
    addressed_agent_name: str  = Field(..., min_length=1, max_length=16)
    text:                 str  = Field(..., min_length=1, max_length=1000,
        description="raw chat text WITHOUT the leading @<agent_name> prefix")
    timestamp:            int
    expected_reply_type:  ExpectedReplyType = "text"
    suppressed_in_chat:   bool = True
```

> NOTE: `_ULID_PATTERN` is duplicated across `chat.py`, `failure.py`, and `plan.py` so each schema module is self-contained for type-checkers. The canonical regex lives in `aiutopia.common.ids.ULID_REGEX` — if you change it, change all four references.

- [ ] **Step 4: Implement FailureReport**

Create `src/aiutopia/schemas/failure.py`:
```python
"""§6.4 — FailureReport with closed-vocab failure_type."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from aiutopia.common.ids import new_report_id
from aiutopia.schemas.enums import FailureType, RoleId, SCHEMA_VERSION_LLM_PLAN


class ExecutionTraceEntry(BaseModel):
    tick:                int
    action_summary:      str = Field(..., max_length=200)
    observation_summary: str = Field(..., max_length=400)
    reward:              float


class PartialProgress(BaseModel):
    inventory_delta_achieved: dict[str, int] = Field(default_factory=dict)
    success_criteria_met:     list[str]      = Field(default_factory=list)
    progress_fraction:        float          = Field(..., ge=0.0, le=1.0)
    blueprint_status_summary: dict | None    = None
    crops_progressed:         int  | None    = None
    threats_neutralized:      int  | None    = None


class FailureDetails(BaseModel):
    failure_type:        FailureType
    failure_tick:        int
    final_state_summary: dict
    descriptor_summary:  str = Field(..., max_length=400)
    execution_trace:     list[ExecutionTraceEntry] = Field(
        default_factory=list, max_length=200,
    )


_ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"


class FailureReport(BaseModel):
    report_id:        str               = Field(default_factory=new_report_id,
                                                  pattern=_ULID_PATTERN)
    schema_version:   str               = SCHEMA_VERSION_LLM_PLAN
    plan_id:          str               = Field(..., pattern=_ULID_PATTERN)
    subgoal_id:       str               = Field(..., pattern=_ULID_PATTERN)
    role:             RoleId
    agent_uuid:       str               = Field(..., pattern=_ULID_PATTERN)
    status:           Literal["failed"] = "failed"
    failure_details:  FailureDetails
    partial_progress: PartialProgress
    reported_at:      int
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_schemas_chat.py tests/unit/test_schemas_failure.py -v`

Expected: 7 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/aiutopia/schemas/chat.py src/aiutopia/schemas/failure.py \
        tests/unit/test_schemas_chat.py tests/unit/test_schemas_failure.py
git commit -m "feat(schemas): ChatEvent + FailureReport with Pydantic v2 validators"
```

---

### Task 14: `Subgoal` / `LlmPlanOutput` with Kahn cycle detection

**Files:**
- Create: `src/aiutopia/schemas/plan.py`
- Create: `tests/unit/test_schemas_plan.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_schemas_plan.py`:
```python
import pytest

from aiutopia.schemas.plan import (
    Constraints, Dependency, GoalSpecification, LlmPlanOutput,
    Subgoal, TargetState, TerminationConditions,
)


def _sg(role: str = "gatherer", sgid: str | None = None,
        fallbacks: list[str] | None = None) -> Subgoal:
    return Subgoal(
        subgoal_id=sgid or "01J0SG0000123456789ABCDE0Y",
        role=role,
        goal_specification=GoalSpecification(
            target_state=TargetState(inventory_delta={"oak_log": 32}),
            termination_conditions=TerminationConditions(
                success_criteria=["inventory_meets_delta"],
                timeout_ticks=6000,
            ),
        ),
        constraints=Constraints(),
        fallback_subgoals=fallbacks or [],
        nl_summary="collect 32 oak_log",
    )


def test_minimal_plan_validates() -> None:
    plan = LlmPlanOutput(
        high_level_goal="build a small village",
        subgoals=[_sg()],
        created_at=1,
        created_by="stub-planner",
    )
    assert plan.schema_version == "1.0.0"
    assert plan.max_fallback_chain_depth == 3


def test_target_state_requires_at_least_one_field() -> None:
    with pytest.raises(Exception):
        TargetState()   # all fields empty/None


def test_timeout_ticks_capped_at_12000() -> None:
    with pytest.raises(Exception):
        TerminationConditions(
            success_criteria=["x"],
            timeout_ticks=12_001,
        )


def test_dependency_missing_subgoal_id_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            dependencies=[Dependency(before=a.subgoal_id,
                                      after="01J0SG0002000000000000000B")],
            created_at=1, created_by="stub-planner",
        )


def test_dependency_self_loop_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            dependencies=[Dependency(before=a.subgoal_id, after=a.subgoal_id)],
            created_at=1, created_by="stub-planner",
        )


def test_dag_cycle_detection_via_kahn() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    b = _sg(sgid="01J0SG0002000000000000000B")
    c = _sg(sgid="01J0SG0003000000000000000C")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a, b, c],
            dependencies=[
                Dependency(before=a.subgoal_id, after=b.subgoal_id),
                Dependency(before=b.subgoal_id, after=c.subgoal_id),
                Dependency(before=c.subgoal_id, after=a.subgoal_id),  # cycle
            ],
            created_at=1, created_by="stub-planner",
        )


def test_fallback_pointing_to_unknown_subgoal_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A",
            fallbacks=["01J0SG0099000000000000000Z"])
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            created_at=1, created_by="stub-planner",
        )


def test_nl_summary_max_length_1500() -> None:
    with pytest.raises(Exception):
        Subgoal(
            subgoal_id="01J0SG0000123456789ABCDE0Y",
            role="gatherer",
            goal_specification=GoalSpecification(
                target_state=TargetState(inventory_delta={"x": 1}),
                termination_conditions=TerminationConditions(
                    success_criteria=["x"], timeout_ticks=1,
                ),
            ),
            nl_summary="x" * 1501,
        )
```

- [ ] **Step 2: Verify failures**

Run: `pytest tests/unit/test_schemas_plan.py -v`

Expected: ImportError on all.

- [ ] **Step 3: Implement plan.py**

Create `src/aiutopia/schemas/plan.py`:
```python
"""§6.2–6.3 — LlmPlanOutput, Subgoal, GoalSpecification, Constraints,
TargetState, TerminationConditions, Dependency with full DAG validation
(Kahn cycle detection)."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from aiutopia.common.ids import new_plan_id, new_subgoal_id
from aiutopia.schemas.enums import (
    FailureType, PlannerSource, RoleId, SCHEMA_VERSION_LLM_PLAN,
)

_ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"


class TargetState(BaseModel):
    inventory_delta:    dict[str, int] = Field(default_factory=dict)
    spatial_target:     tuple[float, float, float] | None = None
    blueprint_target:   str | None = None
    threat_neutralized: bool | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "TargetState":
        # Distinguish "not set" (None / empty) from "set to False" — a
        # defender goal `threat_neutralized=False` ("ensure threat is NOT
        # in the neutralized state yet") is valid. Truthiness alone would
        # treat False as falsy and silently reject it.
        if (not self.inventory_delta
            and self.spatial_target is None
            and self.blueprint_target is None
            and self.threat_neutralized is None):
            raise ValueError("TargetState requires at least one target field")
        return self


class TerminationConditions(BaseModel):
    success_criteria: list[str]      = Field(..., min_length=1)
    timeout_ticks:    int            = Field(..., gt=0, le=12_000)
    failure_events:   list[FailureType] = Field(default_factory=list)


class Constraints(BaseModel):
    preserve_items:    list[str] = Field(default_factory=list)
    avoid_biomes:      list[str] = Field(default_factory=list)
    max_health_cost:   int | None = Field(None, ge=0, le=20)
    tool_requirements: list[str] = Field(default_factory=list)
    no_combat:         bool = False


class GoalSpecification(BaseModel):
    target_state:           TargetState
    termination_conditions: TerminationConditions


class Subgoal(BaseModel):
    subgoal_id:         str = Field(default_factory=new_subgoal_id,
                                     pattern=_ULID_PATTERN)
    role:               RoleId
    priority:           int = Field(default=5, ge=0, le=10)
    goal_specification: GoalSpecification
    constraints:        Constraints = Field(default_factory=Constraints)
    fallback_subgoals:  list[str]   = Field(default_factory=list)
    nl_summary:         str = Field(..., min_length=1, max_length=1500)


class Dependency(BaseModel):
    before: str = Field(..., pattern=_ULID_PATTERN)
    after:  str = Field(..., pattern=_ULID_PATTERN)


class LlmPlanOutput(BaseModel):
    plan_id:                     str = Field(default_factory=new_plan_id,
                                              pattern=_ULID_PATTERN)
    schema_version:              str = SCHEMA_VERSION_LLM_PLAN
    high_level_goal:             str = Field(..., min_length=1, max_length=400)
    high_level_goal_template_id: str | None = Field(None,
        description="references /var/lib/aiutopia/goal_templates/{id}.yaml; "
                    "None or 'freeform' = free-form goal")
    village_targets:             dict[str, int] | None = Field(default=None,
        description="Stage-3 inventory targets; null in stages 1-2 and "
                    "late-M4 stub pre-exposure")
    subgoals:                    list[Subgoal] = Field(..., min_length=1, max_length=32)
    dependencies:                list[Dependency] = Field(default_factory=list)
    max_fallback_chain_depth:    int = Field(default=3, ge=1, le=5)
    created_at:                  int
    created_by:                  PlannerSource
    notes:                       str | None = None

    @model_validator(mode="after")
    def _validate_dag(self) -> "LlmPlanOutput":
        ids = {s.subgoal_id for s in self.subgoals}
        for dep in self.dependencies:
            if dep.before not in ids:
                raise ValueError(f"dep.before {dep.before!r} not in subgoals")
            if dep.after not in ids:
                raise ValueError(f"dep.after {dep.after!r} not in subgoals")
            if dep.before == dep.after:
                raise ValueError("self-dependency forbidden")
        for sg in self.subgoals:
            for fb in sg.fallback_subgoals:
                if fb not in ids:
                    raise ValueError(f"fallback {fb!r} not in subgoals")
        # Kahn's topo sort for cycle detection
        in_deg = {sid: 0 for sid in ids}
        adj: dict[str, list[str]] = {sid: [] for sid in ids}
        for dep in self.dependencies:
            in_deg[dep.after] += 1
            adj[dep.before].append(dep.after)
        roots = [n for n, d in in_deg.items() if d == 0]
        seen = 0
        while roots:
            n = roots.pop()
            seen += 1
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    roots.append(m)
        if seen != len(ids):
            raise ValueError("DAG cycle detected in dependencies")
        return self
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_schemas_plan.py -v`

Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/schemas/plan.py tests/unit/test_schemas_plan.py
git commit -m "feat(schemas): Subgoal + LlmPlanOutput with Kahn cycle detection"
```

---

### Task 15: Subgoal state machine + schema-version loader scaffold

**Files:**
- Create: `src/aiutopia/schemas/state_machine.py`
- Create: `src/aiutopia/schemas/versioning.py`
- Create: `tests/unit/test_state_machine.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_state_machine.py`:
```python
import pytest

from aiutopia.schemas.state_machine import (
    can_transition, SubgoalTransitionError,
)


def test_pending_to_active_allowed() -> None:
    assert can_transition("pending", "active")


def test_active_to_completed_allowed() -> None:
    assert can_transition("active", "completed")


def test_active_to_failed_allowed() -> None:
    assert can_transition("active", "failed")


def test_active_to_paused_allowed() -> None:
    assert can_transition("active", "paused")


def test_paused_to_active_allowed() -> None:
    assert can_transition("paused", "active")


def test_failed_to_pending_allowed_fallback() -> None:
    assert can_transition("failed", "pending")


def test_completed_terminal() -> None:
    assert not can_transition("completed", "active")
    assert not can_transition("completed", "pending")


def test_pending_to_completed_forbidden() -> None:
    assert not can_transition("pending", "completed")


def test_assert_helper_raises_on_invalid() -> None:
    from aiutopia.schemas.state_machine import assert_transition
    with pytest.raises(SubgoalTransitionError):
        assert_transition("completed", "active")
```

- [ ] **Step 2: Implement state machine**

Create `src/aiutopia/schemas/state_machine.py`:
```python
"""§6.6 — Subgoal DAG state machine: allowed transitions only.
Runtime hooks (Py4J calls, episodic memory writes, EventQueue puts)
land in M5 alongside the real planner."""
from __future__ import annotations

from aiutopia.schemas.enums import SubgoalState


class SubgoalTransitionError(ValueError):
    pass


_ALLOWED: dict[SubgoalState, frozenset[SubgoalState]] = {
    "pending":   frozenset({"active"}),
    "active":    frozenset({"completed", "failed", "paused"}),
    "paused":    frozenset({"active"}),
    "failed":    frozenset({"pending"}),   # fallback substitution
    "completed": frozenset(),              # terminal
}


def can_transition(src: SubgoalState, dst: SubgoalState) -> bool:
    return dst in _ALLOWED.get(src, frozenset())


def assert_transition(src: SubgoalState, dst: SubgoalState) -> None:
    if not can_transition(src, dst):
        raise SubgoalTransitionError(f"invalid transition: {src!r} → {dst!r}")
```

- [ ] **Step 3: Implement versioning loader stub**

Create `src/aiutopia/schemas/versioning.py`:
```python
"""§6.1 — SemVer migration loader (forward-only).

In M0 there is exactly one MAJOR.MINOR.PATCH version (1.0.0); this module
exists so M5+ migrations have a stable import target. The mechanism:

  /var/lib/aiutopia/schema_migrations/llm_plan/X.Y.Z_to_X'.Y'.Z'.py
      exports `def migrate(data: dict) -> dict`

Loader algorithm:
  1. Read persisted plan_json
  2. Parse top-level schema_version
  3. If < CURRENT, walk migration scripts forward
  4. Re-validate with current pydantic model
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from aiutopia.schemas.enums import SCHEMA_VERSION_LLM_PLAN


def _parse_version(v: str) -> tuple[int, int, int]:
    major, minor, patch = (int(x) for x in v.split("."))
    return major, minor, patch


def _load_migration(script: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load migration {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "migrate"):
        raise AttributeError(f"migration {script} missing migrate()")
    return module.migrate  # type: ignore[no-any-return]


def migrate_plan_data(data: dict[str, Any], migrations_dir: Path,
                      target_version: str = SCHEMA_VERSION_LLM_PLAN
                      ) -> dict[str, Any]:
    """Walk migrations forward from data['schema_version'] to target_version.
    Returns the migrated dict (NOT the validated Pydantic model — caller
    validates with the current model)."""
    current = data.get("schema_version", "1.0.0")
    if current == target_version:
        return data

    cur = _parse_version(current)
    target = _parse_version(target_version)
    if cur > target:
        raise ValueError(f"persisted plan version {current} is newer than "
                         f"runtime {target_version}; downgrades not supported")

    while _parse_version(data.get("schema_version", "1.0.0")) < target:
        current_str = data["schema_version"]
        # Find next migration whose name starts with "{current_str}_to_"
        candidates = sorted(migrations_dir.glob(f"{current_str}_to_*.py"))
        if not candidates:
            raise FileNotFoundError(
                f"no migration script found from {current_str} in {migrations_dir}"
            )
        migrate_fn = _load_migration(candidates[0])
        data = migrate_fn(data)
    return data
```

- [ ] **Step 4: Run state-machine tests**

Run: `pytest tests/unit/test_state_machine.py -v`

Expected: 9 PASSED.

- [ ] **Step 5: Sanity-check versioning import**

Run: `python -c "from aiutopia.schemas.versioning import migrate_plan_data; print('ok')"`

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/aiutopia/schemas/state_machine.py src/aiutopia/schemas/versioning.py \
        tests/unit/test_state_machine.py
git commit -m "feat(schemas): subgoal state machine + schema versioning loader"
```

---

### Task 16: Chroma client wrapper + smoke test

**Files:**
- Create: `src/aiutopia/memory/client.py`
- Create: `tests/integration/test_chroma_smoke.py`

- [ ] **Step 1: Write the smoke test (will be marked integration)**

Create `tests/integration/test_chroma_smoke.py`:
```python
"""Marker-gated Chroma smoke test. Runs by default; skips if chromadb
is not installed (CI may want to install only [dev], not the full deps)."""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from aiutopia.common.ids import memory_id_for, new_agent_uuid
from aiutopia.memory.client import open_chroma


pytestmark = pytest.mark.integration


def test_chroma_roundtrip(chroma_dir):
    agent_uuid = new_agent_uuid()
    client = open_chroma(chroma_dir)
    coll = client.get_or_create_collection(memory_id_for(agent_uuid))
    coll.add(
        ids=["rec1"],
        documents=["Bjorn found 3 oak logs near the river."],
        metadatas=[{"importance_score": 0.6, "timestamp": 100,
                     "event_type": "harvest",
                     "participants_csv": f",agent-{agent_uuid},"}],
        embeddings=[[0.1] * 384],     # fake BGE-small vector
    )
    got = coll.query(query_embeddings=[[0.1] * 384], n_results=1,
                     include=["documents", "metadatas", "distances"])
    assert got["documents"][0][0] == "Bjorn found 3 oak logs near the river."
    assert got["metadatas"][0][0]["importance_score"] == 0.6
```

- [ ] **Step 2: Run the test (expect ImportError on `open_chroma`)**

Run: `pytest tests/integration/test_chroma_smoke.py -v`

Expected: FAIL on `cannot import name 'open_chroma'`.

- [ ] **Step 3: Implement the client wrapper**

Create `src/aiutopia/memory/client.py`:
```python
"""§5.6 — Chroma client wrapper. Local persistent store under chroma_dir."""
from __future__ import annotations

from pathlib import Path

import chromadb


def open_chroma(chroma_dir: Path) -> chromadb.ClientAPI:
    """Open a persistent local Chroma client rooted at chroma_dir."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/integration/test_chroma_smoke.py -v`

Expected: PASS (1 passed; first run may take ~10s while Chroma initializes the SQLite backend).

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/memory/client.py tests/integration/test_chroma_smoke.py
git commit -m "feat(memory): chroma client wrapper + roundtrip smoke test"
```

---

### Task 17: EpisodicMemoryWriter + MemoryRetriever

**Files:**
- Create: `src/aiutopia/memory/writer.py`
- Create: `src/aiutopia/memory/retriever.py`
- Create: `tests/unit/test_memory_writer.py`
- Create: `tests/unit/test_memory_retriever.py`

- [ ] **Step 1: Write writer tests**

Create `tests/unit/test_memory_writer.py`:
```python
from aiutopia.memory.writer import importance_score, IMPORTANCE_WEIGHTS


def test_importance_weights_sum_to_one() -> None:
    assert abs(sum(IMPORTANCE_WEIGHTS.values()) - 1.0) < 1e-9


def test_importance_score_all_zero_inputs_is_zero() -> None:
    s = importance_score(
        abs_reward_norm=0.0, novel_state=0.0, comm_norm=0.0,
        player_proximity=0.0, threat_level=0.0, planner_event=0.0,
    )
    assert s == 0.0


def test_importance_score_all_one_inputs_is_one() -> None:
    s = importance_score(
        abs_reward_norm=1.0, novel_state=1.0, comm_norm=1.0,
        player_proximity=1.0, threat_level=1.0, planner_event=1.0,
    )
    assert abs(s - 1.0) < 1e-9


def test_importance_score_weighted_combination_matches_spec() -> None:
    # Only abs_reward_norm = 1, others 0 → should equal 0.30 weight
    s = importance_score(
        abs_reward_norm=1.0, novel_state=0.0, comm_norm=0.0,
        player_proximity=0.0, threat_level=0.0, planner_event=0.0,
    )
    assert abs(s - IMPORTANCE_WEIGHTS["abs_reward"]) < 1e-9
```

- [ ] **Step 2: Write retriever tests**

Create `tests/unit/test_memory_retriever.py`:
```python
import numpy as np

from aiutopia.memory.retriever import recency_score, RECENCY_LAMBDA_GENERAL


def test_recency_score_at_zero_age_is_one() -> None:
    assert recency_score(now_tick=100, mem_tick=100,
                          recency_lambda=RECENCY_LAMBDA_GENERAL) == 1.0


def test_recency_score_decays_with_age() -> None:
    s_recent = recency_score(now_tick=200, mem_tick=100,
                               recency_lambda=RECENCY_LAMBDA_GENERAL)
    s_old    = recency_score(now_tick=200_000, mem_tick=100,
                               recency_lambda=RECENCY_LAMBDA_GENERAL)
    assert s_recent > s_old > 0.0


def test_recency_score_never_negative() -> None:
    # Even with negative age (memory from the future, shouldn't happen but…)
    s = recency_score(now_tick=0, mem_tick=100, recency_lambda=0.05)
    assert s >= 0.0


def test_recency_score_lambda_per_intent_constants_exist() -> None:
    from aiutopia.memory.retriever import (
        RECENCY_LAMBDA_LONG_TERM,
        RECENCY_LAMBDA_GENERAL,
        RECENCY_LAMBDA_TIME_SENSITIVE,
    )
    assert RECENCY_LAMBDA_LONG_TERM    < RECENCY_LAMBDA_GENERAL
    assert RECENCY_LAMBDA_GENERAL      < RECENCY_LAMBDA_TIME_SENSITIVE
```

- [ ] **Step 3: Run tests, expect failure**

Run: `pytest tests/unit/test_memory_writer.py tests/unit/test_memory_retriever.py -v`

Expected: ImportError on all.

- [ ] **Step 4: Implement writer**

Create `src/aiutopia/memory/writer.py`:
```python
"""§4.9 — Episodic memory write path with importance scoring.

In M0 only the scoring + batch-buffer logic is implemented (no LLM summary
generation; that's M5). The writer can persist to Chroma if a client is
passed; otherwise it batches in memory for tests."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# v1 weights — starting guesses. Exploitation hunt at end of M3 may revise.
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "abs_reward":      0.30,
    "novel_state":     0.15,
    "comm_norm":       0.10,
    "player_proximity":0.15,
    "threat_level":    0.15,
    "planner_event":   0.15,
}

# Tiered thresholds (§4.9)
HIGH_IMPORTANCE_THRESHOLD   = 0.70
MEDIUM_IMPORTANCE_THRESHOLD = 0.30

# Batch flush limits (§4.9)
BATCH_FLUSH_EVERY_TICKS   = 200
BATCH_FLUSH_MAX_RECORDS   = 50


def importance_score(*,
                     abs_reward_norm: float,
                     novel_state: float,
                     comm_norm: float,
                     player_proximity: float,
                     threat_level: float,
                     planner_event: float) -> float:
    """Weighted sum of clamped [0, 1] inputs → score in [0, 1]."""
    parts = {
        "abs_reward":       abs_reward_norm,
        "novel_state":      novel_state,
        "comm_norm":        comm_norm,
        "player_proximity": player_proximity,
        "threat_level":     threat_level,
        "planner_event":    planner_event,
    }
    return float(sum(IMPORTANCE_WEIGHTS[k] * max(0.0, min(1.0, v))
                     for k, v in parts.items()))


@dataclass
class EpisodicRecord:
    agent_uuid:       str
    timestamp:        int
    event_type:       str
    participants:     list[str]
    importance_score: float
    summary:          str
    embedding:        list[float] | None = None


@dataclass
class EpisodicMemoryWriter:
    """Buffers MEDIUM-importance records, immediate-writes HIGH ones.
    Real Chroma writes are wired in M5 alongside summary-generation LLM
    calls; in M0 this only buffers + counts so smoke tests are testable."""
    high_count:   int = 0
    medium_count: int = 0
    skipped_count: int = 0
    _buffer: dict[str, list[EpisodicRecord]] = field(
        default_factory=lambda: defaultdict(list))

    def maybe_write(self, record: EpisodicRecord) -> str:
        if record.importance_score >= HIGH_IMPORTANCE_THRESHOLD:
            self.high_count += 1
            return "high"
        if record.importance_score >= MEDIUM_IMPORTANCE_THRESHOLD:
            self.medium_count += 1
            self._buffer[record.agent_uuid].append(record)
            return "medium"
        self.skipped_count += 1
        return "skipped"

    def flush(self) -> int:
        flushed = sum(len(v) for v in self._buffer.values())
        self._buffer.clear()
        return flushed
```

- [ ] **Step 5: Implement retriever**

Create `src/aiutopia/memory/retriever.py`:
```python
"""§5.6 — Memory retrieval with tiered recency-decay per query intent."""
from __future__ import annotations

import math

# §5.6 tiered recency lambdas (decay per in-game day; 24_000 ticks/day)
RECENCY_LAMBDA_LONG_TERM      = 0.02     # decay at ~50 days
RECENCY_LAMBDA_GENERAL        = 0.04     # default
RECENCY_LAMBDA_TIME_SENSITIVE = 0.05     # decay at ~20 days

TICKS_PER_GAME_DAY = 24_000


def recency_score(now_tick: int, mem_tick: int, recency_lambda: float) -> float:
    """exp(-λ × age_days). Clamped to [0, 1]."""
    age_days = max(0.0, (now_tick - mem_tick) / TICKS_PER_GAME_DAY)
    return math.exp(-recency_lambda * age_days)


# Static query templates per intent (§5.6 — LLM-composed deferred to Phase 5+).
QUERY_TEMPLATES = {
    "general":          "general context for planning: goal={goal}, role={role}",
    "player_history":   "interactions with player {player_name}",
    "combat":           "combat danger threat involving {threat_types}",
    "funeral":          "death funeral memorial of {predecessor_agent_name}",
}


def render_query(intent: str, **kwargs: str) -> str:
    tpl = QUERY_TEMPLATES[intent]
    return tpl.format(**kwargs)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_memory_writer.py tests/unit/test_memory_retriever.py -v`

Expected: 8 PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/aiutopia/memory/writer.py src/aiutopia/memory/retriever.py \
        tests/unit/test_memory_writer.py tests/unit/test_memory_retriever.py
git commit -m "feat(memory): importance scorer + retriever utilities"
```

---

### Task 18: GoalSpecAdapter (frozen BGE + hard role dispatch)

**Files:**
- Create: `src/aiutopia/planner/goal_spec.py`
- Create: `tests/unit/test_goal_spec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_goal_spec.py`:
```python
"""Verifies §3.1 invariants without loading the actual BGE model
(the model is loaded lazily by the adapter; tests inject a fake encoder)."""
from __future__ import annotations

import numpy as np
import pytest

from aiutopia.planner.goal_spec import (
    GoalSpecAdapter, build_structured_features, build_nl_summary,
)
from aiutopia.schemas.plan import (
    Constraints, GoalSpecification, Subgoal, TargetState, TerminationConditions,
)


class _FakeBGE:
    """Deterministic stand-in: returns a constant 384-d vector regardless of text."""
    def encode(self, text: str) -> np.ndarray:
        # Different texts → slightly different vectors so we can distinguish.
        h = hash(text) % 1000
        v = np.full(384, h / 1000.0, dtype=np.float32)
        return v


def _sg() -> Subgoal:
    return Subgoal(
        role="gatherer",
        priority=7,
        goal_specification=GoalSpecification(
            target_state=TargetState(inventory_delta={"oak_log": 32}),
            termination_conditions=TerminationConditions(
                success_criteria=["inventory_meets_delta"],
                timeout_ticks=6000,
            ),
        ),
        constraints=Constraints(),
        nl_summary="collect 32 oak_log",
    )


def test_structured_features_shape_is_128() -> None:
    sg = _sg()
    feat = build_structured_features(sg)
    assert feat.shape == (128,)
    assert feat.dtype == np.float32


def test_role_onehot_set_correctly() -> None:
    sg = _sg()
    feat = build_structured_features(sg)
    # role_one_hot occupies indices 0-3 in our layout: gatherer=0
    assert feat[0] == 1.0
    assert feat[1] == 0.0


def test_goal_embedding_is_512_d() -> None:
    sg = _sg()
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    emb = adapter.embed(sg)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32


def test_role_dispatch_returns_correct_policy_name() -> None:
    sg = _sg()
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    assert adapter.policy_name_for(sg) == "gatherer_policy"


def test_invalid_role_raises() -> None:
    """If subgoal.role were ever forged outside the Literal, dispatch raises."""
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    class _Fake:
        role = "not_a_real_role"
    with pytest.raises(KeyError):
        adapter.policy_name_for(_Fake())


def test_build_nl_summary_uses_subgoal_nl_summary_verbatim() -> None:
    sg = _sg()
    s = build_nl_summary(sg)
    assert "collect 32 oak_log" in s
    assert "gatherer" in s
```

- [ ] **Step 2: Verify failures**

Run: `pytest tests/unit/test_goal_spec.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement the adapter**

Create `src/aiutopia/planner/goal_spec.py`:
```python
"""§3.1 — Tier 2 Goal Spec Adapter.

Frozen pre-trained BGE-small NL embedding (384-d) + structured features
(128-d) → 512-d goal_embedding. Hard role dispatch (string lookup), no
learned routing. BGE model is lazy-loaded on first call to keep cold
imports fast (tests inject a fake)."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from aiutopia.schemas.plan import Subgoal


class BGEEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


_ROLE_INDEX: dict[str, int] = {
    "gatherer": 0, "builder": 1, "farmer": 2, "defender": 3,
}
_ROLE_TO_POLICY: dict[str, str] = {
    "gatherer": "gatherer_policy",
    "builder":  "builder_policy",
    "farmer":   "farmer_policy",
    "defender": "defender_policy",
}


# Layout of the 128-d structured feature vector (deterministic):
#   [ 0:  4] role one-hot (4 dims; 1.0 at role index)
#   [ 4: 68] inventory_delta normalized (64 dims)
#   [68: 69] timeout_normalized (1 dim) — timeout_ticks / 12000
#   [69: 70] priority normalized   (1 dim) — priority / 10
#   [70:128] reserved flags        (58 dims; zeros for now, used in later milestones)
_STRUCTURED_DIM = 128


def build_structured_features(subgoal: Subgoal) -> np.ndarray:
    out = np.zeros(_STRUCTURED_DIM, dtype=np.float32)
    out[_ROLE_INDEX[subgoal.role]] = 1.0
    # inventory_delta: hash item name → bucket in [4, 68) so it's stable
    for item, qty in subgoal.goal_specification.target_state.inventory_delta.items():
        bucket = 4 + (hash(item) % 64)
        out[bucket] = max(-1.0, min(1.0, float(qty) / 64.0))
    out[68] = subgoal.goal_specification.termination_conditions.timeout_ticks / 12000.0
    out[69] = subgoal.priority / 10.0
    return out


def build_nl_summary(subgoal: Subgoal) -> str:
    """Compose the NL string that gets BGE-encoded.
    Includes role tag so the embedder sees role context."""
    return f"{subgoal.role}: {subgoal.nl_summary}"


class GoalSpecAdapter:
    def __init__(self, bge: BGEEncoder):
        self._bge = bge

    def embed(self, subgoal: Subgoal) -> np.ndarray:
        nl_vec = self._bge.encode(build_nl_summary(subgoal)).astype(np.float32)
        if nl_vec.shape != (384,):
            raise ValueError(f"BGE encoder returned shape {nl_vec.shape}, expected (384,)")
        struct = build_structured_features(subgoal)
        return np.concatenate([nl_vec, struct]).astype(np.float32)

    def policy_name_for(self, subgoal: Any) -> str:
        # `subgoal.role` is sufficient — tolerate ducktype for testability.
        return _ROLE_TO_POLICY[subgoal.role]


def load_bge_small() -> BGEEncoder:
    """Lazy loader for `BAAI/bge-small-en-v1.5`. Call once at process start;
    NEVER from inside a tight loop. Heavy import."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    class _Wrapper:
        def encode(self, text: str) -> np.ndarray:
            return model.encode(text, normalize_embeddings=True)

    return _Wrapper()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_goal_spec.py -v`

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/planner/goal_spec.py tests/unit/test_goal_spec.py
git commit -m "feat(planner): GoalSpecAdapter — frozen BGE + hard role dispatch (§3.1)"
```

---

### Task 19: Fabric mod scaffold + UnionClef compatibility verification

**Files:**
- Create: `fabric_mod/build.gradle`
- Create: `fabric_mod/gradle.properties`
- Create: `fabric_mod/settings.gradle`
- Create: `fabric_mod/src/main/resources/fabric.mod.json`
- Create: `fabric_mod/src/main/resources/aiutopia.mixins.json`
- Create: `fabric_mod/README.md`

- [ ] **Step 1: Verify UnionClef compatibility (decision gate)**

Run these checks before scaffolding:
```bash
# (a) Read UnionClef's gradle.properties for its target MC + loader versions:
curl -s https://raw.githubusercontent.com/3ndetz/unionclef/master/gradle.properties

# (b) Check current Fabric Loader stable:
curl -s https://meta.fabricmc.net/v2/versions/loader | python -c "import sys, json; d=json.load(sys.stdin); print(d[0])"

# (c) Check Carpet for MC 1.21.1 release:
curl -s "https://api.modrinth.com/v2/project/carpet/version" | python -c "import sys, json; v=[x for x in json.load(sys.stdin) if '1.21.1' in x['game_versions']]; print(v[0]['version_number'] if v else 'NONE')"
```

**Decision:**
- If UnionClef's `minecraft_version` is `1.21.1` AND Carpet has a `1.21.1` release: proceed with this scaffold pinned to MC 1.21.1.
- Otherwise: pause this task, document the version mismatch in `fabric_mod/README.md`, and fall back to scaffolding a clean `Py4JEntryPoint` reading UnionClef as reference (adds ~3–5 days; bridge surface is the same, only the build setup differs).

For this plan we assume MC 1.21.1 is the pin. Adjust the property values below if your check produced different versions.

- [ ] **Step 2: Write gradle.properties**

Create `fabric_mod/gradle.properties`:
```properties
# Pinned to UnionClef baseline (MC 1.21.1) — see IMPLEMENTATION_PLAN.md T19.
# If you bump these, re-verify UnionClef fork compatibility before building.

# Build
org.gradle.jvmargs=-Xmx2G
org.gradle.parallel=true

# Minecraft / Fabric Loader / Fabric API
minecraft_version=1.21.1
yarn_mappings=1.21.1+build.3
loader_version=0.16.5
fabric_version=0.103.0+1.21.1

# Mod metadata
mod_version=0.0.0-m0
maven_group=dev.aiutopia
archives_base_name=aiutopia-mod
```

- [ ] **Step 3: Write settings.gradle**

Create `fabric_mod/settings.gradle`:
```groovy
pluginManagement {
    repositories {
        maven { url = uri("https://maven.fabricmc.net/") }
        gradlePluginPortal()
    }
}

rootProject.name = "aiutopia-mod"
```

- [ ] **Step 4: Write build.gradle**

Create `fabric_mod/build.gradle`:
```groovy
plugins {
    id "fabric-loom" version "1.7-SNAPSHOT"
    id "maven-publish"
    id "java"
}

version = project.mod_version
group   = project.maven_group

base {
    archivesName = project.archives_base_name
}

repositories {
    maven { url = "https://maven.fabricmc.net/" }
    maven { url = "https://masa.dy.fi/maven" }                  // Carpet
    mavenCentral()
}

dependencies {
    minecraft "com.mojang:minecraft:${project.minecraft_version}"
    mappings  "net.fabricmc:yarn:${project.yarn_mappings}:v2"
    modImplementation "net.fabricmc:fabric-loader:${project.loader_version}"
    modImplementation "net.fabricmc.fabric-api:fabric-api:${project.fabric_version}"

    // Py4J — Python ↔ Java bridge (§7.3)
    implementation "net.sf.py4j:py4j:0.10.9.9"
    include        "net.sf.py4j:py4j:0.10.9.9"

    // Carpet API (compile-time; runtime users install the mod separately)
    modCompileOnly "carpet:fabric-carpet:1.4.147+v240613:deobf"
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
    withSourcesJar()
}

processResources {
    inputs.property "version", project.mod_version

    filesMatching("fabric.mod.json") {
        expand "version": project.mod_version
    }
}

tasks.withType(JavaCompile).configureEach {
    it.options.release = 21
}

jar {
    from("LICENSE") {
        rename { "${it}_${project.archives_base_name}" }
    }
}
```

- [ ] **Step 5: Write fabric.mod.json**

Create `fabric_mod/src/main/resources/fabric.mod.json`:
```json
{
  "schemaVersion": 1,
  "id": "aiutopia",
  "version": "${version}",
  "name": "AI Utopia",
  "description": "Multi-agent AI village bridge mod (Py4J + Carpet).",
  "authors": ["Carte"],
  "contact": { "homepage": "https://github.com/CBaileyDev/aiutopia" },
  "license": "MIT",
  "environment": "*",
  "entrypoints": {
    "main": [ "dev.aiutopia.mod.AiUtopiaMod" ]
  },
  "mixins": [
    "aiutopia.mixins.json"
  ],
  "depends": {
    "fabricloader": ">=0.16.0",
    "fabric-api":   "*",
    "minecraft":    "1.21.x",
    "java":         ">=21"
  },
  "suggests": {
    "carpet": "*"
  }
}
```

- [ ] **Step 6: Write the empty mixin manifest**

Create `fabric_mod/src/main/resources/aiutopia.mixins.json`:
```json
{
  "required": true,
  "minVersion": "0.8",
  "package": "dev.aiutopia.mod.mixin",
  "compatibilityLevel": "JAVA_21",
  "mixins": [],
  "client": [],
  "server": [],
  "injectors": {
    "defaultRequire": 1
  }
}
```

- [ ] **Step 7: Write the README**

Create `fabric_mod/README.md`:
```markdown
# AI Utopia Fabric Mod

Java side of the Python ↔ Minecraft bridge for the multi-agent village system.
This is a *server-side* mod by design — install on a Fabric 1.21.1 dedicated server.

## Components

- **`Py4JEntryPoint`** — Java methods exposed to Python via Py4J. Implements
  `observationsAll()`, `motorBridge()`, `commBus()`, `resetWorld()`,
  `advanceTickAwaitEvents()`. Forked from UnionClef.
- **Mixins (added incrementally per the task list):**
  - `KickPlayerMixin` (T21) — block `/kick` of Carpet fake players.
  - `ChatMessageMixin` (T23) — intercept `@<agent_name>` chat → emit ChatEvent.
  - `PlayerListMixin` — deferred to M1 (per-recipient `/list` filter requires non-trivial mapping work).

## Build

```bash
./gradlew build
# → build/libs/aiutopia-mod-<version>.jar
```

Install in `<server>/mods/` alongside:
- Fabric API
- Carpet
- Lithium
- FerriteCore
- Krypton (optional)

## Decision gate (T19)

Pinned to MC 1.21.1 because UnionClef baseline targets that version.
If you need a later 1.21.x, regenerate `gradle.properties` AND verify the
UnionClef fork still compiles — bridge surface may need patching.
```

- [ ] **Step 8: Verify Gradle wrapper bootstraps**

Run (from `fabric_mod/`):
```bash
cd fabric_mod
gradle wrapper --gradle-version 8.10
./gradlew --version
```

Expected: prints Gradle version info; no `BUILD FAILED`. (If Gradle is not installed system-wide, install it first or skip this step until you have it; the wrapper bootstraps once and then can be used standalone.)

- [ ] **Step 9: Commit**

```bash
git add fabric_mod/
git commit -m "feat(mod): fabric mod scaffold pinned to MC 1.21.1 (UnionClef baseline)"
```

---

### Task 20: `Py4JEntryPoint` + bridge stubs (Java)

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/AiUtopiaMod.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`

- [ ] **Step 1: Write the mod entry point**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/AiUtopiaMod.java`:
```java
package dev.aiutopia.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

public class AiUtopiaMod implements ModInitializer {
    public static final String MOD_ID = "aiutopia";
    public static final Logger LOG = LoggerFactory.getLogger(MOD_ID);

    private GatewayServer gateway;
    private Py4JEntryPoint entryPoint;

    @Override
    public void onInitialize() {
        LOG.info("AI Utopia mod initializing");

        int py4jPort = Integer.parseInt(
            System.getProperty("aiutopia.py4j.port", "25001")
        );

        this.entryPoint = new Py4JEntryPoint();
        this.gateway    = new GatewayServer.GatewayServerBuilder(entryPoint)
                            .javaPort(py4jPort)
                            .build();

        ServerLifecycleEvents.SERVER_STARTING.register(server -> {
            entryPoint.attachServer(server);
            gateway.start();
            LOG.info("AI Utopia Py4J gateway listening on port {}", py4jPort);
        });

        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            try {
                gateway.shutdown();
            } finally {
                entryPoint.detachServer();
            }
            LOG.info("AI Utopia Py4J gateway stopped");
        });
    }
}
```

- [ ] **Step 2: Write the Py4J entry point**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`:
```java
package dev.aiutopia.mod;

import dev.aiutopia.mod.bridge.CommBus;
import dev.aiutopia.mod.bridge.MotorBridge;
import dev.aiutopia.mod.bridge.WorldOps;
import net.minecraft.server.MinecraftServer;

/**
 * Surface exposed to Python via Py4J. Each method here is callable from
 * Python with normal type marshalling. Keep signatures stable — Python
 * code depends on them.
 *
 * §7.3 invariants:
 *   - observationsAll() returns ONE JSON blob per env per tick (batched).
 *   - motorBridge() handles SkillCompletionEvent via ack-based callbacks.
 *   - commBus().flushBatch(...) is called mid-tick by the env wrapper.
 */
public class Py4JEntryPoint {
    private MinecraftServer server;
    private final MotorBridge motor   = new MotorBridge();
    private final CommBus     commBus = new CommBus();
    private final WorldOps    world   = new WorldOps();

    public void attachServer(MinecraftServer server) {
        this.server = server;
        this.motor.attachServer(server);
        this.world.attachServer(server);
    }

    public void detachServer() {
        this.server = null;
        this.motor.detachServer();
        this.world.detachServer();
    }

    // ───── methods called from Python ─────

    /** Stub heartbeat — returns "ok" if the server is attached. */
    public String health() {
        return server == null ? "stopped" : "ok";
    }

    /** Batched observation read — one JSON blob containing all agents
     *  on this env's server. */
    public String observationsAll() {
        return world.observationsAll();
    }

    public MotorBridge motorBridge() { return motor; }
    public CommBus     commBus()     { return commBus; }

    /** Reset the world to the given seed. M0 stub. */
    public void resetWorld(long seed) {
        world.resetWorld(seed);
    }

    /** Advance one tick and return the list of agent_ids whose current
     *  skill completed this tick. Times out after timeoutMs and returns
     *  an empty list. M0 stub: returns [] immediately. */
    public java.util.List<String> advanceTickAwaitEvents(long timeoutMs) {
        return motor.advanceTickAwaitEvents(timeoutMs);
    }
}
```

- [ ] **Step 3: Write the bridge stubs**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java`:
```java
package dev.aiutopia.mod.bridge;

import java.util.Collections;
import java.util.List;
import net.minecraft.server.MinecraftServer;

/**
 * Motor module — dispatches parameterized skill primitives to the world.
 * M0 stubs all methods so the bridge surface is callable end-to-end;
 * real Baritone / Carpet integrations land in M1+.
 *
 * §6.3 invariants:
 *   - dispatchSkill(...) returns immediately; completion is signalled via
 *     advanceTickAwaitEvents().
 *   - Idempotency: skillInvocationId is unique per (agent, dispatch);
 *     duplicate IDs are silently de-duped.
 */
public class MotorBridge {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

    /** Dispatch a parameterized skill. M0 stub: enqueues a no-op on the
     *  server thread so we exercise the public scheduling API. M1 replaces
     *  the lambda body with the actual skill dispatch.
     *
     *  NOTE: `MinecraftServer.execute(Runnable)` is the public API for
     *  scheduling work on the main thread. `ServerTask` is package-private
     *  and cannot be instantiated from outside `net.minecraft.server`. */
    public void dispatchSkill(String agentId, String encodedAction,
                              String skillInvocationId) {
        if (server != null) {
            server.execute(() -> { /* no-op stub — M1 wires real motor */ });
        }
    }

    /** Advance one tick; return agent_ids that completed a skill this tick.
     *  M0 stub: returns empty list immediately. */
    public List<String> advanceTickAwaitEvents(long timeoutMs) {
        return Collections.emptyList();
    }
}
```

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`:
```java
package dev.aiutopia.mod.bridge;

import net.minecraft.server.MinecraftServer;

/** §7.3 world inspection + reset operations. M0 stubs return an empty JSON object. */
public class WorldOps {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

    /** Batched observation read — single JSON blob with every agent on
     *  this env. M0 stub: returns "{}" so Python can parse it. */
    public String observationsAll() {
        // TODO M1: assemble per-agent obs from world state + Carpet APIs.
        return "{}";
    }

    /** Reset world to the given seed. M0 stub. */
    public void resetWorld(long seed) {
        // TODO M1: kill all entities, regenerate world from seed.
    }
}
```

- [ ] **Step 4: Verify Java compiles (no mixins yet)**

Run (from `fabric_mod/`):
```bash
./gradlew compileJava
```

Expected: `BUILD SUCCESSFUL`. CommBus.java is referenced but not yet created — Gradle will error. Fix at T22.

If you want to compile right now without errors, add a stub `CommBus.java` with just `public class CommBus { public void flushBatch(java.util.List<Object> msgs) {} }` — T22 will replace it.

- [ ] **Step 5: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/
git commit -m "feat(mod): Py4JEntryPoint + MotorBridge + WorldOps stubs (§7.3 surface)"
```

---

### Task 21: AgentRegistry + KickPlayer guard mixin

> **Scope change vs initial sketch:** `PlayerListMixin` is **NOT** built in M0. The per-recipient `/list` filter requires either rewriting `PlayerListS2CPacket` per send or hooking `ServerPlayNetworkHandler.sendPacket` — both depend on mapping signatures that vary across Yarn revisions and are easy to silently no-op. Deferring to M1 (alongside the first non-trivial mixin work) avoids dead code in the M0 jar. `/list` will show fake players in M0; not a blocker for the smoke test.

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/mixin/KickPlayerMixin.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/agent/AgentRegistry.java`

- [ ] **Step 1: Write the in-process agent name registry**

`AgentRegistry` is the single source of truth for "is this player name an AI agent?" — the mixin consults it. It is populated when an agent is spawned via Py4J (`agent spawn` CLI). M0 stores in-memory; M5 backs to identity.db.

Create `fabric_mod/src/main/java/dev/aiutopia/mod/agent/AgentRegistry.java`:
```java
package dev.aiutopia.mod.agent;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** In-process registry of AI agent player names.
 *  Populated by Py4J (CLI `aiutopia agent spawn`). Mixins read this to
 *  decide whether to filter / block. Thread-safe; reads are lock-free. */
public final class AgentRegistry {
    private static final Set<String> AGENT_NAMES = ConcurrentHashMap.newKeySet();

    private AgentRegistry() {}

    public static void registerAgent(String playerName) {
        AGENT_NAMES.add(playerName);
    }
    public static void unregisterAgent(String playerName) {
        AGENT_NAMES.remove(playerName);
    }
    public static boolean isAgent(String playerName) {
        return AGENT_NAMES.contains(playerName);
    }
    public static Set<String> snapshot() {
        return Set.copyOf(AGENT_NAMES);
    }
}
```

- [ ] **Step 2: Verify the `KickCommand` mapping (HARD GATE — do this before Step 3)**

Vanilla `/kick`'s static handler is named `kick` (not `execute`) and takes `Collection<GameProfile>` (not `Collection<ServerPlayerEntity>`) in MC 1.21.1 Yarn. Verify before writing the mixin:

```bash
cd fabric_mod
./gradlew genSources
grep -A 12 "class KickCommand" \
    build/loom-cache/remappedSrc/net/minecraft/server/command/KickCommand.java
```

Expected: a static method like
```
private static int kick(ServerCommandSource source,
                         Collection<GameProfile> targets,
                         Text reason) throws CommandSyntaxException
```

If your Yarn version reports a different signature (different method name, different param order, or `Collection<ServerPlayerEntity>` instead of `Collection<GameProfile>`), update the mixin in Step 3 to match. **Do not skip this step** — a wrong `@Inject` target silently fails to apply and the kick guard becomes a no-op.

- [ ] **Step 3: Write the KickPlayer mixin (signature matched to MC 1.21.1)**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/mixin/KickPlayerMixin.java`:
```java
package dev.aiutopia.mod.mixin;

import com.mojang.authlib.GameProfile;
import dev.aiutopia.mod.agent.AgentRegistry;
import net.minecraft.server.command.KickCommand;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.Collection;

/** Block `/kick` of registered AI agents — for ALL permission levels.
 *
 *  Permission semantics: vanilla `/kick` already requires permission
 *  level 3, so any source that reaches this mixin is already op-3+.
 *  Filtering by permission level here would be a no-op. Instead we
 *  block ALL `/kick` of agents (including from full operators) and
 *  force the explicit `aiutopia agent kill <uuid>` CLI path, so
 *  accidental misclick kicks can never trigger permadeath.
 *
 *  Target verified in Step 2 against MC 1.21.1 Yarn mappings:
 *  KickCommand.kick(ServerCommandSource, Collection<GameProfile>, Text).
 */
@Mixin(KickCommand.class)
public abstract class KickPlayerMixin {

    @Inject(method = "kick", at = @At("HEAD"), cancellable = true)
    private static void aiutopia$blockKickOfAgents(
            ServerCommandSource source,
            Collection<GameProfile> targets,
            Text reason,
            CallbackInfoReturnable<Integer> cir) {
        for (GameProfile profile : targets) {
            if (AgentRegistry.isAgent(profile.getName())) {
                source.sendError(Text.literal(
                    "AI Utopia: cannot kick agent '" + profile.getName()
                    + "' via /kick. Use `aiutopia agent kill <uuid>` instead."));
                cir.setReturnValue(0);
                return;
            }
        }
    }
}
```

- [ ] **Step 4: Verify mixin manifest lists only `KickPlayerMixin`**

Edit `fabric_mod/src/main/resources/aiutopia.mixins.json` and ensure the `server` array contains exactly:
```json
  "server": [
    "KickPlayerMixin"
  ],
```

(If a stale entry for `PlayerListMixin` or `ChatMessageMixin` is present from earlier scaffolding, only `KickPlayerMixin` should be listed at the end of this task — `ChatMessageMixin` is added in T23.)

- [ ] **Step 5: Build verify**

Run (from `fabric_mod/`):
```bash
./gradlew compileJava
```

Expected: `BUILD SUCCESSFUL`. Confirm at the end of the log that Mixin applied successfully — look for `[Mixin] Mixing KickPlayerMixin from aiutopia.mixins.json into net.minecraft.server.command.KickCommand` (or equivalent). If you see `INVALID_INJECTION_DESC`, Step 2 verification was wrong; re-do it.

- [ ] **Step 6: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/agent/ \
        fabric_mod/src/main/java/dev/aiutopia/mod/mixin/KickPlayerMixin.java \
        fabric_mod/src/main/resources/aiutopia.mixins.json
git commit -m "feat(mod): AgentRegistry + KickPlayerMixin (block /kick of agents)"
```

---

### Task 22: `CommBus` stub

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/CommBus.java`

- [ ] **Step 1: Write the CommBus stub**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/CommBus.java`:
```java
package dev.aiutopia.mod.bridge;

import java.util.List;

/** §4.8 — Inter-agent communication bus. Mid-tick batched flush.
 *  M0 stub: accepts messages, logs count. Real routing M1+. */
public class CommBus {
    /** Flush a batch of CommMessage JSON strings to receivers. */
    public void flushBatch(List<Object> messages) {
        // TODO M1: parse JSON, route to receivers by role mask + spatial range,
        //          insert into per-agent 32-slot ring buffer.
    }
}
```

- [ ] **Step 2: Verify Java compiles cleanly**

Run (from `fabric_mod/`): `./gradlew compileJava`

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/CommBus.java
git commit -m "feat(mod): CommBus stub for mid-tick comm batch flush"
```

---

### Task 23: ChatMessage mixin (extract `@<agent_name>`, emit `ChatEvent`)

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/mixin/ChatMessageMixin.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/chat/ChatEventBuffer.java`

- [ ] **Step 1: Write the ChatEventBuffer (Java-side queue Py4J reads)**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/chat/ChatEventBuffer.java`:
```java
package dev.aiutopia.mod.chat;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/** Bounded FIFO of ChatEvent JSON blobs awaiting Python pickup.
 *  Python pulls via Py4J at the start of each planner tick. */
public final class ChatEventBuffer {
    private static final int MAX_SIZE = 256;
    private static final Deque<String> QUEUE = new ArrayDeque<>();

    private ChatEventBuffer() {}

    public static synchronized void push(String chatEventJson) {
        if (QUEUE.size() >= MAX_SIZE) QUEUE.pollFirst();   // drop oldest
        QUEUE.addLast(chatEventJson);
    }

    public static synchronized List<String> drainAll() {
        List<String> out = new ArrayList<>(QUEUE);
        QUEUE.clear();
        return out;
    }

    public static synchronized int size() { return QUEUE.size(); }
}
```

- [ ] **Step 2: Add buffer accessor to Py4JEntryPoint**

Append to `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java` inside the class:
```java
    /** Drain queued ChatEvents. Called from Python planner each tick. */
    public java.util.List<String> drainChatEvents() {
        return dev.aiutopia.mod.chat.ChatEventBuffer.drainAll();
    }
```

- [ ] **Step 3: Write the ChatMessage mixin**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/mixin/ChatMessageMixin.java`:
```java
package dev.aiutopia.mod.mixin;

import dev.aiutopia.mod.agent.AgentRegistry;
import dev.aiutopia.mod.chat.ChatEventBuffer;
import net.minecraft.network.message.SignedMessage;
import net.minecraft.server.network.ServerPlayNetworkHandler;
import net.minecraft.server.network.ServerPlayerEntity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** §3.3 — Intercept chat starting with `@<agent_name>`, emit ChatEvent to
 *  Python via ChatEventBuffer. Vanilla broadcast is suppressed when the
 *  pattern matches (suppressed_in_chat=true default in ChatEvent schema). */
@Mixin(ServerPlayNetworkHandler.class)
public abstract class ChatMessageMixin {
    @Shadow public ServerPlayerEntity player;

    private static final Pattern AGENT_MENTION =
        Pattern.compile("^@([A-Za-z0-9_]{1,16})\\s+(.+)$", Pattern.DOTALL);

    @Inject(method = "handleDecoratedMessage", at = @At("HEAD"), cancellable = true)
    private void aiutopia$interceptAgentMention(SignedMessage message, CallbackInfo ci) {
        String text = message.getSignedContent();
        if (text == null) return;

        Matcher m = AGENT_MENTION.matcher(text);
        if (!m.matches()) return;

        String agentName = m.group(1);
        String body      = m.group(2);
        if (!AgentRegistry.isAgent(agentName)) return;

        String json = String.format(
            "{\"sender_player_uuid\":\"%s\",\"sender_player_name\":\"%s\","
          + "\"addressed_agent_name\":\"%s\",\"text\":\"%s\","
          + "\"timestamp\":%d}",
          player.getUuidAsString(),
          escape(player.getGameProfile().getName()),
          escape(agentName),
          escape(body),
          System.currentTimeMillis() / 1000L
        );
        ChatEventBuffer.push(json);
        ci.cancel();    // suppress vanilla broadcast
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r");
    }
}
```

- [ ] **Step 4: HARD verification — confirm `handleDecoratedMessage` signature against your Yarn mappings**

`handleDecoratedMessage(SignedMessage)` is the 1.21.1 signed-chat hook in current Yarn, but the LVT slot order and the exact method name vary across Yarn revisions. Run before building:

```bash
cd fabric_mod
./gradlew genSources
grep -A 5 -E "handleDecoratedMessage|handleMessage|onChatMessage" \
    build/loom-cache/remappedSrc/net/minecraft/server/network/ServerPlayNetworkHandler.java \
    | head -30
```

Expected: one method whose body broadcasts the chat message (calls something like `MessageType.CHAT` or `playerManager.broadcast(...)`) — that's your target. If the method name is not `handleDecoratedMessage`, update `method = "..."` in the `@Inject` accordingly.

- [ ] **Step 5: Verify build + mixin registers**

Run (from `fabric_mod/`):
```bash
./gradlew build
```

Expected: `BUILD SUCCESSFUL`. Generated jar at `build/libs/aiutopia-mod-0.0.0-m0.jar`. Inspect the build log near the end and confirm the mixin actually applied — look for `[Mixin] Mixing ChatMessageMixin from aiutopia.mixins.json into net.minecraft.server.network.ServerPlayNetworkHandler` (or equivalent). If you see `INVALID_INJECTION_DESC` or `Could not find target method`, Step 4's verification was wrong; redo it.

Also update `fabric_mod/src/main/resources/aiutopia.mixins.json` `server` array to add `ChatMessageMixin`:
```json
  "server": [
    "KickPlayerMixin",
    "ChatMessageMixin"
  ],
```

- [ ] **Step 6: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/chat/ \
        fabric_mod/src/main/java/dev/aiutopia/mod/mixin/ChatMessageMixin.java \
        fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java \
        fabric_mod/src/main/resources/aiutopia.mixins.json
git commit -m "feat(mod): ChatMessageMixin intercepts @<agent> chat → ChatEventBuffer"
```

---

### Task 24: `docker-compose.production.yml` skeleton (with ZGC, not G1GC)

**Files:**
- Create: `docker-compose.production.yml`
- Create: `secrets/.gitkeep`
- Create: `secrets/README.md`

- [ ] **Step 1: Write the Compose skeleton**

Create `docker-compose.production.yml`:
```yaml
# Production deployment skeleton — full M6 deliverable.
# Committed in M0 so the JVM args (Generational ZGC, NOT G1GC) and the
# localhost-only port bindings live in source from day one.
#
# To use:
#   docker compose -f docker-compose.production.yml up
# To bring up the local-LLM fallback:
#   docker compose -f docker-compose.production.yml --profile llm-fallback up

services:
  fabric-prod:
    image: aiutopia/fabric-server:1.21.1
    container_name: aiutopia-fabric-prod
    cpuset: "12-13"
    mem_limit: 4g
    volumes:
      - ./world:/server/world
      - ./fabric_mod/build/libs:/server/mods/aiutopia:ro
      - ./logs/fabric:/server/logs
    ports:
      - "25565:25565"                          # MC client
      - "127.0.0.1:25100:25100"                # Py4J — localhost only
    environment:
      # CRITICAL: Generational ZGC, NOT G1GC. G1's 200ms pauses stall the
      # tick loop and silently destroy training stability.
      # QUOTED + single line because YAML's folding rules for unquoted
      # multi-line list items are parser-dependent — Docker Compose has
      # historically split the lines and silently dropped the ZGC flags,
      # leaving the container on default G1GC.
      - "JAVA_OPTS=-Xms3g -Xmx3g -XX:+UseZGC -XX:+ZGenerational -XX:+UnlockExperimentalVMOptions -XX:+UseTransparentHugePages -Daiutopia.py4j.port=25100"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "25100"]
      interval: 30s
      timeout: 5s
      retries: 3

  planner:
    image: aiutopia/planner:latest
    cpuset: "14"
    mem_limit: 4g
    depends_on: [fabric-prod, chroma]
    volumes:
      - ./var/lib/aiutopia:/var/lib/aiutopia
      - ./goal_templates:/var/lib/aiutopia/goal_templates:ro
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
      - PY4J_HOST=fabric-prod
      - PY4J_PORT=25100
      - CHROMA_URL=http://chroma:8000
      - LLM_MODEL=claude-haiku
      - LLM_BUDGET_HARD_CAP_USD_MONTH=80
      - QWEN_LOCAL_URL=http://planner-llm-local:8001
    secrets: [anthropic_key]
    restart: unless-stopped

  planner-llm-local:
    image: aiutopia/llm-local:qwen14b-int4
    cpuset: "14"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "127.0.0.1:8001:8001"
    profiles: [llm-fallback]

  chroma:
    image: chromadb/chroma:0.5.20
    cpuset: "15"
    mem_limit: 2g
    volumes:
      - ./var/lib/aiutopia/chroma:/chroma/chroma
    ports:
      - "127.0.0.1:8000:8000"

  dashboard-feed:
    image: aiutopia/dashboard-feed:latest
    cpuset: "15"
    mem_limit: 512m
    volumes:
      - ./var/lib/aiutopia:/var/lib/aiutopia:ro
    ports:
      - "127.0.0.1:8080:8080"                  # localhost only — NEVER expose

secrets:
  anthropic_key:
    file: ./secrets/anthropic_key
```

- [ ] **Step 2: Write secrets/README**

Create `secrets/.gitkeep` (empty file) and `secrets/README.md`:
```markdown
# Secrets

Do NOT commit secret contents to git. The `.gitignore` includes `secrets/`
except for this README and `.gitkeep`.

Required files (create locally before `docker compose up`):

- `anthropic_key` — single-line Anthropic API key (`sk-ant-...`)

Permissions:
```bash
chmod 600 secrets/anthropic_key
```
```

- [ ] **Step 3: Append to .gitignore (do NOT overwrite)**

Append these lines to the existing `.gitignore` (the file already exists from the early commit; do not replace it):

```bash
cat >> .gitignore <<'EOF'

# Secrets (allowlist README + .gitkeep)
secrets/*
!secrets/.gitkeep
!secrets/README.md
EOF
```

Verify with: `git diff .gitignore` — should show only the new lines added at the end.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.production.yml secrets/ .gitignore
git commit -m "feat(deploy): production compose skeleton (ZGC, localhost-only ports)"
```

---

### Task 25: Gatherer obs + action spaces + action mask

**Files:**
- Create: `src/aiutopia/env/spaces.py`
- Create: `src/aiutopia/env/action_mask.py`
- Create: `tests/unit/test_env_spaces.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_env_spaces.py`:
```python
import numpy as np
import pytest
from gymnasium.spaces import Dict as DictSpace

from aiutopia.env.spaces import (
    build_role_observation_space, build_role_action_space, CORE_KEYS, GATHERER_KEYS,
)


def test_gatherer_obs_space_has_core_plus_gatherer_keys() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert isinstance(space, DictSpace)
    for k in CORE_KEYS:
        assert k in space.spaces, f"missing core key {k}"
    for k in GATHERER_KEYS:
        assert k in space.spaces, f"missing gatherer key {k}"


def test_other_role_keys_NOT_in_gatherer_obs() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    for k in space.spaces:
        assert not k.startswith("b_"), f"builder key {k} leaked into gatherer obs"
        assert not k.startswith("f_"), f"farmer key {k} leaked into gatherer obs"
        assert not k.startswith("d_"), f"defender key {k} leaked into gatherer obs"


def test_comm_buffer_is_32_slots_not_8() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert space.spaces["comm_payloads"].shape == (32, 128)
    assert space.spaces["comm_metadata"].shape == (32, 8)


def test_goal_ticks_left_capped_at_12000() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert space.spaces["goal_ticks_left"].high.max() == 12_000


def test_action_space_has_universal_header() -> None:
    space = build_role_action_space("gatherer")
    for k in ("skill_type", "target_class", "spatial_param", "scalar_param",
              "comm_payload", "should_broadcast", "comm_target_mask"):
        assert k in space.spaces


def test_action_space_sample_roundtrip() -> None:
    space = build_role_action_space("gatherer")
    sample = space.sample()
    assert space.contains(sample)


def test_observation_space_contains_action_mask() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert "action_mask" in space.spaces
    mask = space.spaces["action_mask"]
    assert "skill_type" in mask.spaces
    assert "target_per_skill" in mask.spaces
```

- [ ] **Step 2: Verify failures**

Run: `pytest tests/unit/test_env_spaces.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement spaces**

Create `src/aiutopia/env/spaces.py`:
```python
"""§4.1, §4.2 — Per-role observation + action Dict spaces.

M0 implements gatherer only. Other roles raise NotImplementedError until
their milestones (builder M2, farmer M3, defender M4)."""
from __future__ import annotations

from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiBinary, MultiDiscrete
import numpy as np

# Fixed constants — must agree with Java side (motor_module.encode_action).
N_ITEMS         = 1024     # MC 1.21 item-id space (sparse OK; this is the cap)
N_BIOMES        = 64
INV_SLOTS       = 36
GOAL_EMBED_DIM  = 512
COMM_PAYLOAD_DIM= 128
COMM_BUFFER_SLOTS= 32      # §3 carry-forward (1.6 s history at 20 TPS)

N_GATHERER_SKILLS         = 6   # navigate, harvest, deposit_chest, search, wait, noop_broadcast
N_TARGET_CLASSES_PER_ROLE = 64  # block_pos / resource_id / chest_id / direction_bias index

CORE_KEYS = (
    "agent_uuid_embed", "role_one_hot", "tick_in_episode",
    "position", "velocity", "yaw_pitch", "health", "hunger",
    "saturation", "armor_value",
    "inv_slot_item_ids", "inv_slot_counts",
    "main_hand_item_id", "off_hand_item_id",
    "goal_embedding", "goal_ticks_left",
    "time_of_day", "weather", "biome_id", "light_level",
    "comm_payloads", "comm_metadata",
    "action_mask",
)
GATHERER_KEYS = (
    "g_resource_grid", "g_nearest_resources",
    "g_richness_score", "g_hostiles_nearby",
)


def _action_mask_space(n_skills: int, n_targets: int) -> DictSpace:
    return DictSpace({
        "skill_type":       MultiBinary(n_skills),
        "target_per_skill": MultiBinary((n_skills, n_targets)),
        "comm_payload":     MultiBinary(1),
        "should_broadcast": MultiBinary(2),
    })


def _core_space() -> dict:
    return {
        "agent_uuid_embed":  Box(-1, 1,  (384,), np.float32),
        "role_one_hot":      MultiBinary(4),
        "tick_in_episode":   Box(0, 24_000, (1,), np.int32),
        "position":          Box(-3e7, 3e7, (3,), np.float32),
        "velocity":          Box(-10, 10,  (3,), np.float32),
        "yaw_pitch":         Box(-180, 180, (2,), np.float32),
        "health":            Box(0, 20, (1,), np.float32),
        "hunger":            Box(0, 20, (1,), np.float32),
        "saturation":        Box(0, 20, (1,), np.float32),
        "armor_value":       Box(0, 20, (1,), np.float32),
        "inv_slot_item_ids": MultiDiscrete([N_ITEMS] * INV_SLOTS),
        "inv_slot_counts":   Box(0, 64, (INV_SLOTS,), np.int32),
        "main_hand_item_id": Discrete(N_ITEMS),
        "off_hand_item_id":  Discrete(N_ITEMS),
        "goal_embedding":    Box(-3, 3, (GOAL_EMBED_DIM,), np.float32),
        "goal_ticks_left":   Box(0, 12_000, (1,), np.int32),
        "time_of_day":       Box(0, 24_000, (1,), np.int32),
        "weather":           Discrete(3),
        "biome_id":          Discrete(N_BIOMES),
        "light_level":       Box(0, 15, (1,), np.int32),
        "comm_payloads":     Box(-1, 1, (COMM_BUFFER_SLOTS, COMM_PAYLOAD_DIM),
                                 np.float32),
        "comm_metadata":     Box(0, 1, (COMM_BUFFER_SLOTS, 8), np.float32),
    }


def _gatherer_overlay() -> dict:
    return {
        "g_resource_grid":     Box(0, 1, (32, 32, 6), np.float32),
        "g_nearest_resources": Box(-1, 1, (8, 6), np.float32),
        "g_richness_score":    Box(0, 1, (1,), np.float32),
        "g_hostiles_nearby":   Box(0, 1, (4, 4), np.float32),
    }


def build_role_observation_space(role: str, stage: int) -> DictSpace:
    if role != "gatherer":
        raise NotImplementedError(
            f"role {role!r} obs space not implemented in M0 (see milestone map)"
        )
    spaces = _core_space()
    spaces.update(_gatherer_overlay())
    spaces["action_mask"] = _action_mask_space(
        N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE
    )
    return DictSpace(spaces)


def build_role_action_space(role: str) -> DictSpace:
    if role != "gatherer":
        raise NotImplementedError(
            f"role {role!r} action space not implemented in M0"
        )
    return DictSpace({
        "skill_type":       Discrete(N_GATHERER_SKILLS),
        "target_class":     Discrete(N_TARGET_CLASSES_PER_ROLE),
        "spatial_param":    Box(-1, 1, (3,), np.float32),
        "scalar_param":     Box(0, 1, (1,), np.float32),
        "comm_payload":     Box(-1, 1, (COMM_PAYLOAD_DIM,), np.float32),
        "should_broadcast": Discrete(2),
        "comm_target_mask": MultiBinary(4),
    })
```

- [ ] **Step 4: Implement action mask**

Create `src/aiutopia/env/action_mask.py`:
```python
"""§4.5 — Hard action mask for gatherer (M0).

Mask is computed in Python from the symbolic obs (Python is the only place
that sees both the obs and the role rules). Java sends raw world facts;
Python decides which actions are legal.

The mask shape MUST match the action_mask sub-dict declared in spaces.py."""
from __future__ import annotations

import numpy as np

from aiutopia.env.spaces import N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE

# Skill indices for gatherer (must match Java motor side):
NAVIGATE        = 0
HARVEST         = 1
DEPOSIT_CHEST   = 2
SEARCH          = 3
WAIT            = 4
NOOP_BROADCAST  = 5


def compute_gatherer_action_mask(obs_raw: dict) -> dict[str, np.ndarray]:
    """Build the action_mask sub-dict from a raw observation dict.

    `obs_raw` is the Python-side parse of the JSON blob from
    `Py4JEntryPoint.observationsAll()` — has at minimum:
      - inv_slot_counts: list[int] of length 36
      - target_chest_in_range: bool
      - target_resource_in_range: bool
      - health: float
    """
    skill_mask  = np.ones(N_GATHERER_SKILLS, dtype=np.int8)
    target_mask = np.ones((N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE),
                          dtype=np.int8)

    inv_full = sum(obs_raw.get("inv_slot_counts", [0] * 36)) >= 36 * 64
    if inv_full:
        skill_mask[HARVEST] = 0

    if not obs_raw.get("target_chest_in_range", False):
        target_mask[DEPOSIT_CHEST, :] = 0
        skill_mask[DEPOSIT_CHEST]    = 0
    if not obs_raw.get("target_resource_in_range", False):
        target_mask[HARVEST, :] = 0
        if skill_mask[HARVEST] == 1:
            skill_mask[HARVEST] = 0

    # Fail-safe: if every skill is masked, allow WAIT (§4.5 guard).
    if not skill_mask.any():
        skill_mask[WAIT] = 1

    return {
        "skill_type":       skill_mask,
        "target_per_skill": target_mask,
        "comm_payload":     np.ones(1, dtype=np.int8),
        "should_broadcast": np.ones(2, dtype=np.int8),
    }
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_env_spaces.py -v`

Expected: 7 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/aiutopia/env/spaces.py src/aiutopia/env/action_mask.py \
        tests/unit/test_env_spaces.py
git commit -m "feat(env): gatherer obs/action spaces + hard action mask (§4.1/4.2/4.5)"
```

---

### Task 26: Py4J bridge wrapper (`aiutopia.env.bridge`)

**Files:**
- Create: `src/aiutopia/env/bridge.py`

- [ ] **Step 1: Implement bridge wrapper**

Create `src/aiutopia/env/bridge.py`:
```python
"""§7.3 — Py4J bridge wrapper.

Owns the JavaGateway lifecycle and the BATCHED observationsAll() call.
NOT per-agent observation(agent) — that pattern is forbidden per spec §4.6
(4× Py4J roundtrips/tick would cap throughput at ~300 agent-steps/sec).

`close()` is mandatory (PettingZoo lifecycle); without it, Ray worker
shutdown leaks Java processes that hold Py4J ports."""
from __future__ import annotations

import json
from typing import Any

from py4j.java_gateway import GatewayParameters, JavaGateway


class FabricBridge:
    """Single connection to one Fabric-side Py4J gateway."""

    def __init__(self, port: int):
        self.port = port
        self.gw: JavaGateway | None = None
        self.entry_point: Any = None

    def open(self) -> None:
        self.gw = JavaGateway(GatewayParameters(port=self.port, auto_field=True))
        self.entry_point = self.gw.entry_point

    def close(self) -> None:
        """Mandatory — see module docstring."""
        if self.gw is not None:
            try:
                self.gw.shutdown()
            finally:
                self.gw = None
                self.entry_point = None

    def __enter__(self) -> "FabricBridge":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ───── operations ─────
    def health(self) -> str:
        return str(self.entry_point.health())

    def observations_all(self) -> dict[str, dict]:
        """Single BATCHED call — returns dict mapping agent_id → obs_raw dict."""
        raw = str(self.entry_point.observationsAll())
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError(f"observationsAll must return a JSON object, got {type(parsed)}")
        return parsed

    def reset_world(self, seed: int) -> None:
        self.entry_point.resetWorld(int(seed))

    def advance_tick_await_events(self, timeout_ms: int = 30_000) -> list[str]:
        result = self.entry_point.advanceTickAwaitEvents(int(timeout_ms))
        return [str(x) for x in result]

    def dispatch_skill(self, agent_id: str, action_dict: dict,
                       skill_invocation_id: str) -> None:
        encoded = json.dumps(action_dict)
        self.entry_point.motorBridge().dispatchSkill(agent_id, encoded,
                                                      skill_invocation_id)

    def flush_comm_batch(self, messages: list[dict]) -> None:
        encoded = [json.dumps(m) for m in messages]
        # Py4J auto-converts Python list to java.util.List
        self.entry_point.commBus().flushBatch(encoded)

    def drain_chat_events(self) -> list[dict]:
        raw = self.entry_point.drainChatEvents()
        return [json.loads(str(x)) for x in raw]
```

- [ ] **Step 2: Sanity-check import**

Run: `python -c "from aiutopia.env.bridge import FabricBridge; print('ok')"`

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/env/bridge.py
git commit -m "feat(env): FabricBridge wrapper (batched observationsAll, close())"
```

---

### Task 27: `AiUtopiaPettingZooEnv` (gatherer-only M0)

**Files:**
- Create: `src/aiutopia/env/wrapper.py`
- Create: `tests/integration/test_env_smoke.py`

- [ ] **Step 1: Implement the env wrapper**

Create `src/aiutopia/env/wrapper.py`:
```python
"""§7.3 — PettingZoo Parallel env wrapper.

M0 limitations:
  - gatherer role only
  - reward computation deferred to M1 (returns 0.0 stub)
  - episodic memory writes deferred to M1 (stub no-op)
  - exploit detection deferred to M1 (stub no-op)
  - per_worker_seed_offset is honored
  - mid-tick comm flush is wired
  - close() is implemented and idempotent
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv

from aiutopia.env.action_mask import compute_gatherer_action_mask
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
    CORE_KEYS,
    GATHERER_KEYS,
    GOAL_EMBED_DIM,
)


log = logging.getLogger(__name__)


def _role_of(agent_id: str) -> str:
    return agent_id.split("_", 1)[0]


def _decode_obs(raw: dict, role: str, stage: int,
                 action_mask: dict[str, np.ndarray]) -> dict[str, Any]:
    """Coerce a raw JSON dict from Java into a Gymnasium-Dict-conforming
    obs dict. M0 fills missing fields with zeros (Java side may not
    populate all keys yet)."""
    space = build_role_observation_space(role, stage=stage)
    out: dict[str, Any] = {}
    for key, sub in space.spaces.items():
        if key == "action_mask":
            out[key] = action_mask
            continue
        if key in raw:
            out[key] = np.asarray(raw[key], dtype=sub.dtype if hasattr(sub, "dtype") else None)
        else:
            out[key] = np.zeros(sub.shape, dtype=sub.dtype) if hasattr(sub, "shape") else 0
    # Always emit goal_embedding even if Java omits it (zeros = "no goal")
    if "goal_embedding" not in raw:
        out["goal_embedding"] = np.zeros(GOAL_EMBED_DIM, dtype=np.float32)
    return out


class AiUtopiaPettingZooEnv(ParallelEnv):
    metadata = {"name": "aiutopia_minecraft_v0", "render_modes": []}

    def __init__(self, config: dict[str, Any]):
        self.cfg            = config
        self.active_roles   = list(config["active_roles"])
        self.agents_init    = [f"{r}_0" for r in self.active_roles]
        self.possible_agents = list(self.agents_init)
        self.agents: list[str] = []
        self.stage          = int(config["stage"])
        self.tick_warp      = bool(config.get("tick_warp", False))
        self.max_ticks      = int(config.get("max_episode_ticks", 6_000))
        self._tick          = 0

        # Pick port from worker index (defaults to first port for tests).
        ports = config["py4j_ports"]
        widx  = int(getattr(config, "worker_index",
                            config.get("worker_index", 0))) % len(ports)
        self.bridge = FabricBridge(port=ports[widx])
        self.bridge.open()

        self.skill_counters: dict[str, int] = {}
        self._prev_obs: dict[str, Any] = {}

    # ───── PettingZoo API ─────
    def observation_space(self, agent: str) -> DictSpace:
        return build_role_observation_space(_role_of(agent), stage=self.stage)

    def action_space(self, agent: str) -> DictSpace:
        return build_role_action_space(_role_of(agent))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self._next_seed_for_strategy()
        self.bridge.reset_world(seed)
        self.agents = list(self.agents_init)
        self.skill_counters = {a: 0 for a in self.agents}
        self._tick = 0
        obs = self._read_all_obs()
        self._prev_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, dict]):
        comm_msgs: list[dict] = []
        for agent, act in actions.items():
            self.skill_counters[agent] += 1
            invocation_id = f"{agent}-{self.skill_counters[agent]}"
            self.bridge.dispatch_skill(agent, act, invocation_id)
            if int(act.get("should_broadcast", 0)) == 1 and np.asarray(
                    act.get("comm_target_mask", [0, 0, 0, 0])).any():
                comm_msgs.append({"sender": agent, "action": act})
        self.bridge.flush_comm_batch(comm_msgs)            # mid-tick flush
        self.bridge.advance_tick_await_events(timeout_ms=30_000)

        new_obs = self._read_all_obs()
        rew  = {a: 0.0 for a in self.agents}               # M1: real reward stack
        term = {a: False for a in self.agents}
        trunc = {a: self._tick >= self.max_ticks for a in self.agents}
        info: dict[str, dict] = {a: {} for a in self.agents}

        self._prev_obs = new_obs
        self._tick += 1
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info

    def close(self) -> None:
        """Idempotent close. Without this Ray workers leak Java processes."""
        try:
            self.bridge.close()
        except Exception:        # nosec - close path
            log.exception("error closing FabricBridge")

    # ───── helpers ─────
    def _read_all_obs(self) -> dict[str, dict]:
        raw_all = self.bridge.observations_all()
        out: dict[str, dict] = {}
        for agent in self.agents:
            raw = raw_all.get(agent, {})
            mask = (compute_gatherer_action_mask(raw)
                    if _role_of(agent) == "gatherer"
                    else {})
            out[agent] = _decode_obs(raw, _role_of(agent), self.stage, mask)
        return out

    def _next_seed_for_strategy(self) -> int:
        strategy = self.cfg.get("seed_strategy", "fixed_easy")
        offset = (1
                  if self.cfg.get("per_worker_seed_offset")
                  else 0) * int(getattr(self.cfg, "worker_index",
                                         self.cfg.get("worker_index", 0)))
        seed_table = {
            "fixed_easy":   1 + offset,
            "fixed_medium": 2 + offset,
            "fixed_hard":   3 + offset,
        }
        return seed_table.get(strategy, 1 + offset)
```

- [ ] **Step 2: Write the env smoke test (uses live Fabric server, marked integration)**

Create `tests/integration/test_env_smoke.py`:
```python
"""End-to-end smoke: spin up the wrapper against a running Fabric server
on port PY4J_SMOKE_PORT (default 25099) and verify reset() + 1 step().

Skip if PY4J_SMOKE_PORT is not reachable — most contributors won't have a
server running, and that's fine."""
from __future__ import annotations

import os
import socket

import numpy as np
import pytest

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


pytestmark = pytest.mark.integration


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


@pytest.fixture
def smoke_port() -> int:
    return int(os.environ.get("PY4J_SMOKE_PORT", "25099"))


def test_env_reset_returns_valid_obs(smoke_port: int) -> None:
    if not _port_open("127.0.0.1", smoke_port):
        pytest.skip(f"no Py4J server on port {smoke_port} (set PY4J_SMOKE_PORT)")
    env = AiUtopiaPettingZooEnv({
        "stage": 1, "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy", "tick_warp": True,
        "py4j_ports": [smoke_port], "max_episode_ticks": 100,
        "per_worker_seed_offset": False, "worker_index": 0,
    })
    try:
        obs, info = env.reset(seed=1)
        assert "gatherer_0" in obs
        sample = obs["gatherer_0"]
        assert sample["goal_embedding"].shape == (512,)
        assert sample["comm_payloads"].shape == (32, 128)
        assert "action_mask" in sample

        # 1 step
        act = env.action_space("gatherer_0").sample()
        new_obs, rew, term, trunc, _info = env.step({"gatherer_0": act})
        assert "gatherer_0" in new_obs
        assert isinstance(rew["gatherer_0"], float)
    finally:
        env.close()
```

- [ ] **Step 3: Run smoke (will skip without a live server)**

Run: `pytest tests/integration/test_env_smoke.py -v`

Expected: 1 SKIPPED (assuming no live Fabric server). That's fine — T30 sets up the live smoke.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/env/wrapper.py tests/integration/test_env_smoke.py
git commit -m "feat(env): AiUtopiaPettingZooEnv (gatherer M0) with close() + mid-tick comm flush"
```

---

### Task 28: RLModule + Planner stub files (placeholders that document M1 / M5 shape)

**Files:**
- Create: `src/aiutopia/rl_module/stubs.py`
- Create: `src/aiutopia/planner/stubs.py`

- [ ] **Step 1: Write the RLModule stub**

Create `src/aiutopia/rl_module/stubs.py`:
```python
"""§7.2 — RLModule scaffold. NOT runnable in M0.

This file documents the intended shape so M1 implementation has a target
to fill in. Real implementation requires:
  - inheriting from ray.rllib.core.rl_module.torch.TorchRLModule
  - declaring shared submodules via MultiRLModuleSpec.additional_module_specs
    (NEVER module-level Python globals — Ray workers fork processes,
    Python globals don't share, and gradients silently never propagate
    to rollouts)
"""
from __future__ import annotations

# Intentionally NOT importing ray/torch here so cold imports stay light.
# M1 will replace this module with the real implementation.


class CoreEncoderModule:
    """§4.3 — universal core obs → 256-d feature. SHARED across roles via
    `additional_module_specs` on MultiRLModuleSpec."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1: implement against ray.rllib TorchRLModule")


class SharedBackboneModule:
    """§4.3 — Linear(448→384) + LSTM(384, hidden=256). SHARED across roles."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")


class CTDECriticModule:
    """§4.3 — two-stage encoder (per-agent → 128, then MLP). SHARED."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")


class AiUtopiaRoleRLModule:
    """§7.2 — per-role module: role encoder + (optional) pixel encoder + actor."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")
```

- [ ] **Step 2: Write the planner stub**

Create `src/aiutopia/planner/stubs.py`:
```python
"""§4.7 — LLM planner scaffold. NOT runnable in M0.

The real planner (M5):
  - reads EventQueue (priority 0 ChatEvent > 1 FailureReport > 2 World > 3 Phase)
  - composes prompt with memory retrieval (§5.6)
  - calls Claude Haiku (5 s timeout, exp backoff)
  - 3-tier degradation: Haiku → Qwen 14B → halt + alert
  - writes to planner_state with planner_001_initial schema
"""
from __future__ import annotations


class StubPlanner:
    """M0 placeholder. Replaced by HaikuPlanner in M5."""

    def emit_paired_subgoals(self, role_pair: tuple[str, str]) -> dict:
        """Return a 2-subgoal LlmPlanOutput dict for M2 cooperative training."""
        raise NotImplementedError("M2 wires this with hand-coded blueprints")
```

- [ ] **Step 3: Sanity-check imports**

Run: `python -c "from aiutopia.rl_module.stubs import CoreEncoderModule; from aiutopia.planner.stubs import StubPlanner; print('ok')"`

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/rl_module/stubs.py src/aiutopia/planner/stubs.py
git commit -m "feat(stubs): RLModule + Planner placeholders documenting M1/M5 shape"
```

---

### Task 29: CLI app + `agent spawn` command (registry-only path)

**Files:**
- Create: `src/aiutopia/cli/app.py`
- Create: `src/aiutopia/cli/agent.py`

- [ ] **Step 1: Implement the Typer app**

Create `src/aiutopia/cli/app.py`:
```python
"""§7.5 — Top-level Typer CLI surface."""
from __future__ import annotations

import typer

from aiutopia.cli import agent, memory, determinism
from aiutopia.common.logging import setup_logging


app = typer.Typer(
    name="aiutopia",
    help="AI Utopia — multi-agent Minecraft AI village.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(agent.app, name="agent",
              help="Spawn / kill / list AI agents.")
app.add_typer(memory.app, name="memory",
              help="Inspect agent episodic memory.")
app.add_typer(determinism.app, name="determinism",
              help="Determinism harness commands.")


@app.callback()
def _root_setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Implement `agent` subcommand**

Create `src/aiutopia/cli/agent.py`:
```python
"""`aiutopia agent <spawn|kill|list>` — minimal M0 surface.

`spawn` does THREE things:
  1. Insert a row in identity.db (so the agent has a persistent identity)
  2. Create the Chroma collections `mem_{uuid}` and `skill_lib_{uuid}`
     (so `aiutopia memory inspect <uuid>` returns "(empty)" instead of
     "no memory collection")
  3. Call Py4J to Carpet `/player <name> spawn` so the visible Carpet
     fake player appears in the connected MC client.

If --no-fabric is passed, step 3 is skipped (useful for unit tests + when
no Fabric server is running). Steps 1 and 2 always run."""
from __future__ import annotations

import time
from pathlib import Path

import typer

from aiutopia.common.config import Paths, Py4JConfig
from aiutopia.common.ids import memory_id_for, skill_library_id_for
from aiutopia.common.logging import get_logger
from aiutopia.env.bridge import FabricBridge
from aiutopia.identity.service import IdentityService, init_identity_db
from aiutopia.identity.skin_pool import deterministic_skin_for_uuid, pick_name
from aiutopia.memory.client import open_chroma


app = typer.Typer(no_args_is_help=True)
log = get_logger(__name__)


@app.command("spawn")
def spawn(
    role: str           = typer.Option(..., help="gatherer|builder|farmer|defender"),
    name: str | None    = typer.Option(None, help="explicit agent name (else pool pick)"),
    no_fabric: bool     = typer.Option(False, "--no-fabric",
                                        help="skip Py4J Carpet /player spawn"),
    py4j_port: int      = typer.Option(0,
                                        help="override Py4J port (else production port)"),
) -> None:
    paths = Paths.from_env(); paths.ensure()
    init_identity_db(paths.identity_db,
                     Path("src/aiutopia/identity/migrations"))
    svc = IdentityService(paths.identity_db)

    role_obj = svc.get_role(role)   # type: ignore[arg-type]
    living_names = {a.agent_name for a in svc.list_living_agents()}
    chosen_name  = name or pick_name(role_obj.default_skin_pool,
                                      used=living_names)

    agent = svc.spawn_agent(role, chosen_name, born_at=int(time.time()))
    skin = deterministic_skin_for_uuid(agent.agent_uuid,
                                        role_obj.default_skin_pool)

    typer.echo(f"identity: spawned {chosen_name} ({role}, uuid={agent.agent_uuid})")

    # Always create Chroma collections so memory inspect doesn't 404
    chroma = open_chroma(paths.chroma_dir)
    chroma.get_or_create_collection(memory_id_for(agent.agent_uuid))
    chroma.get_or_create_collection(skill_library_id_for(agent.agent_uuid))
    typer.echo(f"memory:   collections mem_{agent.agent_uuid} + "
               f"skill_lib_{agent.agent_uuid} ready")

    if no_fabric:
        typer.echo("--no-fabric set; skipping Carpet /player spawn")
        return

    # The Fabric Py4J call is wired in T30 (this is the M0 stub).
    typer.echo("(carpet spawn wiring lands in T30)")


@app.command("kill")
def kill(
    agent_uuid: str = typer.Argument(..., help="agent_uuid (ULID)"),
    cause: str      = typer.Option("manual_cli", help="cause_of_death string"),
) -> None:
    paths = Paths.from_env()
    svc = IdentityService(paths.identity_db)
    svc.record_death(agent_uuid, died_at=int(time.time()), cause_of_death=cause)
    typer.echo(f"agent {agent_uuid} marked dead (cause: {cause})")


@app.command("list")
def list_agents() -> None:
    paths = Paths.from_env()
    svc = IdentityService(paths.identity_db)
    for a in svc.list_living_agents():
        typer.echo(f"  {a.agent_name:16} {a.role_id:9} uuid={a.agent_uuid}")
```

- [ ] **Step 3: Verify CLI loads**

Run: `aiutopia --help`

Expected: prints help with `agent`, `memory`, `determinism` subcommands listed.

Run: `aiutopia agent --help`

Expected: prints `spawn`, `kill`, `list` subcommands.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/cli/app.py src/aiutopia/cli/agent.py
git commit -m "feat(cli): aiutopia entrypoint + agent spawn/kill/list (registry path)"
```

---

### Task 30: Carpet `/player spawn` from Py4J + end-to-end smoke test

**Files:**
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`
- Modify: `src/aiutopia/env/bridge.py`
- Modify: `src/aiutopia/cli/agent.py`
- Create: `scripts/smoke-test.sh`
- Create: `tests/integration/test_cli_spawn.py`

- [ ] **Step 1: Add `carpetSpawn(name, skin)` to Java side (skin applied via second command)**

Edit `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java` — add at end of class:
```java
    /** Spawn a Carpet fake player. Returns true on success.
     *  Requires Carpet to be installed on the running server.
     *
     *  If `skin` is non-null and non-empty, follows the spawn with
     *  `/player <name> loadProfile <skin>` so the agent appears with
     *  the chosen skin instead of the default. Failure to apply the
     *  skin is logged but does NOT fail the overall spawn — the
     *  agent still exists, just with the default skin. */
    public boolean carpetSpawn(String playerName, String skin) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm =
                server.getCommandManager();
            int result = cm.executeWithPrefix(
                server.getCommandSource(),
                "/player " + playerName + " spawn"
            );
            if (result <= 0) return false;
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName);

            if (skin != null && !skin.isEmpty()) {
                int skinResult = cm.executeWithPrefix(
                    server.getCommandSource(),
                    "/player " + playerName + " loadProfile " + skin
                );
                if (skinResult <= 0) {
                    dev.aiutopia.mod.AiUtopiaMod.LOG.warn(
                        "loadProfile {} failed for {}; using default skin",
                        skin, playerName);
                }
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "carpetSpawn failed for {}: {}", playerName, e.getMessage());
            return false;
        }
    }
```

- [ ] **Step 2: Expose it from `Py4JEntryPoint`**

Edit `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java` — add:
```java
    /** Spawn a Carpet fake player with optional skin. Returns true on success. */
    public boolean carpetSpawn(String playerName, String skin) {
        return world.carpetSpawn(playerName, skin);
    }
```

- [ ] **Step 3: Add Python-side method**

Edit `src/aiutopia/env/bridge.py` — add to `FabricBridge`:
```python
    def carpet_spawn(self, player_name: str, skin: str | None = None) -> bool:
        return bool(self.entry_point.carpetSpawn(player_name, skin or ""))
```

- [ ] **Step 4: Wire CLI to call it (pass skin)**

Edit `src/aiutopia/cli/agent.py` — replace the inside of the `else` branch (when `--no-fabric` is NOT set) with:
```python
    port = py4j_port or Py4JConfig.from_env().production_port
    log.info("connecting to Fabric Py4J on port %d", port)
    with FabricBridge(port=port) as bridge:
        if bridge.health() != "ok":
            typer.echo("ERROR: Fabric server reports unhealthy; aborting Carpet spawn",
                       err=True)
            raise typer.Exit(code=1)
        ok = bridge.carpet_spawn(chosen_name, skin=skin)
        if not ok:
            typer.echo(f"ERROR: Carpet /player {chosen_name} spawn failed", err=True)
            raise typer.Exit(code=2)
        typer.echo(f"carpet: /player {chosen_name} spawn (skin={skin}) → ok")
```

The `skin` variable was computed earlier in T29 via `deterministic_skin_for_uuid(agent.agent_uuid, role_obj.default_skin_pool)`. It is now wired through to Carpet — no longer dead code.

- [ ] **Step 5: Rebuild Fabric mod**

```bash
cd fabric_mod && ./gradlew build && cd ..
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 6: Write the smoke-test driver script**

Create `scripts/smoke-test.sh`:
```bash
#!/usr/bin/env bash
# M0 end-to-end smoke test.
#
# Prereqs (you set up these once):
#   1. Local Fabric 1.21.1 server with Carpet + Lithium + FerriteCore + our mod
#      jar, started with -Daiutopia.py4j.port=25099 on port 25565 (MC) /
#      25099 (Py4J).
#   2. A Minecraft Java client (matching version) connected to localhost:25565.
#
# This script:
#   - calls `aiutopia agent spawn --role gatherer --py4j-port 25099`
#   - asserts an identity row was inserted
#   - asserts Carpet /player spawn returned ok
#   - (manual) you see the new player appear in your MC client

set -euo pipefail

export AIUTOPIA_ROOT="${AIUTOPIA_ROOT:-/tmp/aiutopia-smoke}"
export PY4J_PRODUCTION_PORT="${PY4J_PRODUCTION_PORT:-25099}"

mkdir -p "$AIUTOPIA_ROOT"
rm -f "$AIUTOPIA_ROOT/identity.db"

echo "[smoke] spawning gatherer via aiutopia CLI…"
aiutopia agent spawn --role gatherer --py4j-port "$PY4J_PRODUCTION_PORT"

echo "[smoke] listing identity rows…"
aiutopia agent list

echo "[smoke] check your connected MC client — the gatherer should be visible."
echo "[smoke] PASS (manual verification required for visual confirmation)"
```

Make executable:
```bash
chmod +x scripts/smoke-test.sh
```

- [ ] **Step 7: Write integration test (skips without live server)**

Create `tests/integration/test_cli_spawn.py`:
```python
"""End-to-end test of `aiutopia agent spawn`. Skips when no live Fabric
server is reachable. The identity-only path (--no-fabric) is always tested."""
from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def test_spawn_no_fabric_creates_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "spawn",
         "--role", "gatherer", "--no-fabric"],
        capture_output=True, text=True, check=True,
    )
    assert "spawned" in out.stdout
    list_out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "list"],
        capture_output=True, text=True, check=True,
    )
    assert "gatherer" in list_out.stdout


@pytest.mark.skipif(not _port_open("127.0.0.1",
                                     int(os.environ.get("PY4J_SMOKE_PORT", "25099"))),
                    reason="no live Fabric server on PY4J_SMOKE_PORT")
def test_spawn_with_fabric_calls_carpet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    port = int(os.environ.get("PY4J_SMOKE_PORT", "25099"))
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "spawn",
         "--role", "gatherer", "--py4j-port", str(port)],
        capture_output=True, text=True, check=True,
    )
    assert "carpet: /player" in out.stdout
    assert "→ ok" in out.stdout
```

- [ ] **Step 8: Run the always-on portion**

Run: `pytest tests/integration/test_cli_spawn.py -v -k "not with_fabric"`

Expected: 1 passed, 1 skipped (the live one).

- [ ] **Step 9: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java \
        fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java \
        src/aiutopia/env/bridge.py src/aiutopia/cli/agent.py \
        scripts/smoke-test.sh tests/integration/test_cli_spawn.py
git commit -m "feat(cli): carpet_spawn wired end-to-end + smoke script + integration test"
```

---

### Task 31: ChatBridge wire-up — drain queue → log (M0 placeholder for M5 LLM round-trip)

**Files:**
- Create: `src/aiutopia/planner/chat_drain.py`
- Create: `tests/unit/test_chat_drain.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_chat_drain.py`:
```python
from aiutopia.planner.chat_drain import classify_reply_type


def test_question_mark_classifies_as_text() -> None:
    assert classify_reply_type("where is the iron?") == "text"


def test_imperative_verb_classifies_as_action_ack() -> None:
    for cmd in ("come help me", "bring wood here", "attack the zombie",
                "defend the wall", "stop digging", "move to spawn",
                "build me a tower", "gather more stone"):
        assert classify_reply_type(cmd) == "action_ack", cmd


def test_statement_classifies_as_none() -> None:
    assert classify_reply_type("thanks") == "none"
    assert classify_reply_type("nice work") == "none"
```

- [ ] **Step 2: Implement**

Create `src/aiutopia/planner/chat_drain.py`:
```python
"""§3.3 — ChatBridge reply-type heuristic + drain loop scaffold.

In M0 this drains queued ChatEvents from the Java side and prints them.
M5 replaces the print with the real LLM round-trip + /tellraw response.
The classifier is committed in M0 so the heuristic is testable now."""
from __future__ import annotations

from aiutopia.schemas.enums import ExpectedReplyType


_IMPERATIVE_VERBS = frozenset({
    "come", "bring", "stop", "attack", "defend", "gather", "build", "move",
    "follow", "wait", "go", "drop", "pick", "kill", "heal", "trade",
})


def classify_reply_type(text: str) -> ExpectedReplyType:
    """§3.3 — heuristic for `expected_reply_type`.

    Upgrade to LLM classifier in Phase 5+ if heuristic miscategorizes >10%
    of messages."""
    if "?" in text:
        return "text"
    first_words = text.strip().lower().split()[:3]
    if any(w in _IMPERATIVE_VERBS for w in first_words):
        return "action_ack"
    return "none"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_chat_drain.py -v`

Expected: 3 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/planner/chat_drain.py tests/unit/test_chat_drain.py
git commit -m "feat(planner): chat reply-type heuristic (§3.3) — M0 scaffold"
```

---

### Task 32: Determinism harness — CUDA fixture + scaffolded test

**Files:**
- Create: `src/aiutopia/determinism/harness.py`
- Create: `tests/determinism/conftest.py`
- Create: `tests/determinism/test_seeded_replay_scaffold.py`

- [ ] **Step 1: Implement the harness module**

Create `src/aiutopia/determinism/harness.py`:
```python
"""§7.8 — Determinism utilities. Two metrics per §5.10:
  - action_argmax_divergence < 0.05
  - continuous_param_L2     < 0.10
over a 1000-tick replay window."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


EPS_ARGMAX = 0.05
EPS_L2     = 0.10


def configure_cuda_determinism() -> None:
    """Pin every cuDNN / cuBLAS knob that controls non-determinism.
    Call once at process start before importing torch CUDA ops.

    Without this, cuDNN autotuner randomizes per-process and the
    determinism gate becomes a flaky test."""
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class ReplayDivergence:
    action_argmax_divergence: float
    continuous_param_l2:      float

    @property
    def passes(self) -> bool:
        return (self.action_argmax_divergence < EPS_ARGMAX
                and self.continuous_param_l2 < EPS_L2)


def compute_divergence(trace_a: list[dict],
                       trace_b: list[dict]) -> ReplayDivergence:
    """Each trace entry must have keys `action_argmax: int` and
    `continuous_params: np.ndarray`."""
    if len(trace_a) != len(trace_b):
        raise ValueError(f"trace lengths differ: {len(trace_a)} vs {len(trace_b)}")
    if not trace_a:
        return ReplayDivergence(0.0, 0.0)

    argmax_div = float(np.mean([
        ta["action_argmax"] != tb["action_argmax"]
        for ta, tb in zip(trace_a, trace_b)
    ]))
    l2_div = float(np.mean([
        np.linalg.norm(np.asarray(ta["continuous_params"]) -
                       np.asarray(tb["continuous_params"]))
        for ta, tb in zip(trace_a, trace_b)
    ]))
    return ReplayDivergence(argmax_div, l2_div)
```

- [ ] **Step 2: Write determinism conftest with CUDA fixture**

Create `tests/determinism/conftest.py`:
```python
"""CUDA determinism fixture — applied to every test in this directory."""
import pytest

from aiutopia.determinism.harness import configure_cuda_determinism


@pytest.fixture(autouse=True)
def _cuda_determinism() -> None:
    configure_cuda_determinism()
```

- [ ] **Step 3: Write the scaffolded replay test**

Create `tests/determinism/test_seeded_replay_scaffold.py`:
```python
"""§7.8 scaffold. Will NOT pass in M0 (no real weights yet). Tests that
the harness logic itself is correct using a synthetic deterministic agent;
M1+ replaces the dummy with a real RLlib policy."""
from __future__ import annotations

import numpy as np
import pytest

from aiutopia.determinism.harness import (
    compute_divergence, EPS_ARGMAX, EPS_L2,
)


pytestmark = pytest.mark.determinism


def _synthetic_trace(seed: int, length: int = 1000) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{
        "action_argmax":     int(rng.integers(0, 6)),
        "continuous_params": rng.uniform(-1, 1, size=4).astype(np.float32),
    } for _ in range(length)]


def test_identical_traces_pass() -> None:
    a = _synthetic_trace(seed=42)
    b = _synthetic_trace(seed=42)
    div = compute_divergence(a, b)
    assert div.passes
    assert div.action_argmax_divergence == 0.0
    assert div.continuous_param_l2 == 0.0


def test_different_seeds_fail() -> None:
    a = _synthetic_trace(seed=1)
    b = _synthetic_trace(seed=2)
    div = compute_divergence(a, b)
    assert not div.passes


def test_thresholds_match_spec() -> None:
    assert EPS_ARGMAX == 0.05
    assert EPS_L2     == 0.10
```

- [ ] **Step 4: Run**

Run: `pytest tests/determinism/ -v -m determinism`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/determinism/harness.py tests/determinism/
git commit -m "feat(determinism): harness + CUDA fixture + scaffold tests (§7.8)"
```

---

### Task 33: `aiutopia determinism check` CLI

**Files:**
- Create: `src/aiutopia/cli/determinism.py`

- [ ] **Step 1: Implement**

Create `src/aiutopia/cli/determinism.py`:
```python
"""`aiutopia determinism check` — runs the seeded-replay harness.

M0: prints a friendly "not runnable yet" message because there are no
trained weights to load. Wired fully in M1 once the first gatherer
checkpoint exists."""
from __future__ import annotations

from pathlib import Path

import typer

from aiutopia.determinism.harness import configure_cuda_determinism


app = typer.Typer(no_args_is_help=True)


@app.command("check")
def check(
    weights: Path  = typer.Option(..., help="path to policy checkpoint"),
    episodes: int  = typer.Option(10, help="number of seed pairs to compare"),
) -> None:
    configure_cuda_determinism()
    if not weights.exists():
        typer.echo(f"weights not found: {weights}", err=True)
        raise typer.Exit(code=2)
    typer.echo("M0: determinism harness scaffold is in place but cannot run "
               "without real RLlib weights. See IMPLEMENTATION_PLAN.md task "
               "M1.X (gatherer first checkpoint).")
    raise typer.Exit(code=3)
```

- [ ] **Step 2: Verify CLI surface**

Run: `aiutopia determinism --help`

Expected: prints `check` subcommand.

Run: `aiutopia determinism check --weights /tmp/nonexistent.ckpt`

Expected: exits with code 3 and prints the M0 message.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/cli/determinism.py
git commit -m "feat(cli): determinism check (M0 stub, fully wired in M1)"
```

---

### Task 34: `aiutopia memory inspect` CLI

**Files:**
- Create: `src/aiutopia/cli/memory.py`

- [ ] **Step 1: Implement**

Create `src/aiutopia/cli/memory.py`:
```python
"""`aiutopia memory inspect` — pretty-prints retrieval results.

Useful for "why did the planner think X" questions once M5 is live.
M0 returns raw collection contents (no LLM-summary generation yet)."""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from aiutopia.common.config import Paths
from aiutopia.common.ids import is_ulid, memory_id_for
from aiutopia.memory.client import open_chroma


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("inspect")
def inspect(
    agent_uuid: str = typer.Option(..., help="agent_uuid (ULID)"),
    query: str      = typer.Option("", help="optional NL query for ANN retrieval"),
    top_k: int      = typer.Option(10, help="number of results"),
) -> None:
    if not is_ulid(agent_uuid):
        typer.echo(f"not a ULID: {agent_uuid}", err=True)
        raise typer.Exit(code=2)

    paths = Paths.from_env()
    client = open_chroma(paths.chroma_dir)
    try:
        coll = client.get_collection(memory_id_for(agent_uuid))
    except Exception as exc:
        typer.echo(f"no memory collection for {agent_uuid}: {exc}", err=True)
        raise typer.Exit(code=3)

    if query:
        # M5 will wire BGE here; M0 returns by recency.
        typer.echo("(M0: ANN query not yet wired; falling back to most-recent)")

    got = coll.get(limit=top_k, include=["documents", "metadatas"])
    if not got["ids"]:
        typer.echo("(empty)")
        return

    table = Table(title=f"memory for {agent_uuid} (top {top_k})")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("type")
    table.add_column("imp")
    table.add_column("summary", overflow="fold")
    for i, (rid, meta, doc) in enumerate(zip(got["ids"],
                                              got["metadatas"],
                                              got["documents"])):
        table.add_row(rid,
                       str(meta.get("timestamp", "")),
                       str(meta.get("event_type", "")),
                       f"{meta.get('importance_score', 0):.2f}",
                       doc)
    console.print(table)
```

- [ ] **Step 2: Verify CLI**

Run: `aiutopia memory --help`

Expected: prints `inspect` subcommand.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/cli/memory.py
git commit -m "feat(cli): memory inspect — recency-ordered Chroma reader (M0)"
```

---

### Task 35: Backup scripts (daily rsync + weekly tar)

**Files:**
- Create: `scripts/backup-daily.sh`
- Create: `scripts/backup-weekly.sh`
- Create: `scripts/crontab.example`

- [ ] **Step 1: Write daily rsync script**

Create `scripts/backup-daily.sh`:
```bash
#!/usr/bin/env bash
# Daily incremental backup of /var/lib/aiutopia to a date-stamped directory
# on the configured backup target (AIUTOPIA_BACKUP_DIR).
#
# Retention: 7 dailies (script trims old ones).
#
# Usage:
#   AIUTOPIA_ROOT=/var/lib/aiutopia \
#   AIUTOPIA_BACKUP_DIR=/mnt/nas/aiutopia/daily \
#   scripts/backup-daily.sh

set -euo pipefail

: "${AIUTOPIA_ROOT:?must be set}"
: "${AIUTOPIA_BACKUP_DIR:?must be set}"

STAMP="$(date +%Y-%m-%d)"
TARGET="$AIUTOPIA_BACKUP_DIR/$STAMP"
LATEST_LINK="$AIUTOPIA_BACKUP_DIR/latest"

mkdir -p "$AIUTOPIA_BACKUP_DIR"

if [[ -e "$LATEST_LINK" ]]; then
  LINK_ARG=(--link-dest="$LATEST_LINK")
else
  LINK_ARG=()
fi

rsync -a --delete "${LINK_ARG[@]}" "$AIUTOPIA_ROOT/" "$TARGET/"

ln -sfn "$TARGET" "$LATEST_LINK"

# Trim to 7 dailies
ls -1d "$AIUTOPIA_BACKUP_DIR"/20* 2>/dev/null \
  | sort | head -n -7 | xargs -r rm -rf

echo "daily backup → $TARGET"
```

- [ ] **Step 2: Write weekly tar script**

Create `scripts/backup-weekly.sh`:
```bash
#!/usr/bin/env bash
# Weekly full tarball of /var/lib/aiutopia.
# Retention: 4 weeklies.
#
# Usage:
#   AIUTOPIA_ROOT=/var/lib/aiutopia \
#   AIUTOPIA_BACKUP_DIR=/mnt/nas/aiutopia/weekly \
#   scripts/backup-weekly.sh

set -euo pipefail

: "${AIUTOPIA_ROOT:?must be set}"
: "${AIUTOPIA_BACKUP_DIR:?must be set}"

STAMP="$(date +%Y-W%V)"
TARGET="$AIUTOPIA_BACKUP_DIR/aiutopia-$STAMP.tar.zst"

mkdir -p "$AIUTOPIA_BACKUP_DIR"

tar --use-compress-program=zstd \
    -cf "$TARGET" \
    -C "$(dirname "$AIUTOPIA_ROOT")" "$(basename "$AIUTOPIA_ROOT")"

# Trim to 4 weeklies
ls -1t "$AIUTOPIA_BACKUP_DIR"/aiutopia-*.tar.zst 2>/dev/null \
  | tail -n +5 | xargs -r rm -f

echo "weekly backup → $TARGET"
```

- [ ] **Step 3: Write crontab example**

Create `scripts/crontab.example`:
```cron
# AI Utopia backup schedule.
# Install with `crontab scripts/crontab.example` (after editing paths).

AIUTOPIA_ROOT=/var/lib/aiutopia
AIUTOPIA_BACKUP_DIR=/mnt/nas/aiutopia

# Daily at 04:00
0 4 * * *  AIUTOPIA_BACKUP_DIR=$AIUTOPIA_BACKUP_DIR/daily /opt/aiutopia/scripts/backup-daily.sh

# Weekly on Sunday at 04:30
30 4 * * 0 AIUTOPIA_BACKUP_DIR=$AIUTOPIA_BACKUP_DIR/weekly /opt/aiutopia/scripts/backup-weekly.sh
```

- [ ] **Step 4: Make scripts executable**

Run: `chmod +x scripts/backup-daily.sh scripts/backup-weekly.sh`

- [ ] **Step 5: Dry-run daily**

Run:
```bash
AIUTOPIA_ROOT=/tmp/aiutopia-fake-root \
AIUTOPIA_BACKUP_DIR=/tmp/aiutopia-backups \
mkdir -p /tmp/aiutopia-fake-root && \
echo "hello" > /tmp/aiutopia-fake-root/marker.txt && \
scripts/backup-daily.sh
```

Expected: prints `daily backup → /tmp/aiutopia-backups/<today>` and the directory contains `marker.txt`.

Clean up: `rm -rf /tmp/aiutopia-fake-root /tmp/aiutopia-backups`

- [ ] **Step 6: Commit**

```bash
git add scripts/backup-daily.sh scripts/backup-weekly.sh scripts/crontab.example
git commit -m "feat(ops): daily rsync + weekly zstd-tar backup scripts"
```

---

### Task 36: README + M0 completion checklist + handoff to M1

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:
```markdown
# AI Utopia — Multi-Agent Minecraft AI Village

Persistent multi-agent reinforcement-learning system. Four specialized agents
(gatherer, builder, farmer, defender) cooperatively grow and operate a village
on a private Minecraft Java server. Friends can join, are recognized as
friendly entities, and can NL-chat with agents via `@<agent_name>` messages.

## Status

**Milestone 0 — Infrastructure foundation** (this milestone).

After M0 you should be able to:

```bash
aiutopia agent spawn --role gatherer
```

…and see a Carpet fake player appear in your connected MC client. No
training has happened; the agent stands still. That's M1's job.

## Quickstart

```bash
# 1. Install Python deps
python -m pip install -e ".[dev]"

# 2. Run unit tests
pytest -v -m "not integration and not determinism"

# 3. (Optional, requires GPU) Run determinism harness scaffold
pytest -v -m determinism

# 4. Build the Fabric mod
cd fabric_mod && ./gradlew build && cd ..

# 5. Install a local Fabric 1.21.1 server with:
#      - Fabric API
#      - Carpet
#      - Lithium, FerriteCore
#      - fabric_mod/build/libs/aiutopia-mod-0.0.0-m0.jar
#    Start with: -Daiutopia.py4j.port=25099

# 6. Connect a Minecraft Java 1.21.1 client to localhost:25565

# 7. Run the end-to-end smoke test
PY4J_PRODUCTION_PORT=25099 scripts/smoke-test.sh
```

## Architecture

See `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`
for the full spec. Two-world topology (training instances + persistent
production server), three-tier brain (LLM planner → goal spec → per-role RL),
Level-D persistent identity, 22–28 weeks to M6.

## Repository layout

| Path | Role |
|---|---|
| `src/aiutopia/` | Python package — env wrapper, identity, schemas, memory, CLI, planner stubs |
| `fabric_mod/` | Java Fabric mod — Py4J bridge + mixins + Carpet integration |
| `tests/{unit,integration,determinism}/` | Test suites (markered for selective runs) |
| `scripts/` | Smoke test + backup automation |
| `docker-compose.production.yml` | M6 deployment skeleton (ZGC JVM args) |
| `docs/superpowers/specs/` | Design specs (committed history) |

## Constraints

- **Python 3.12** pinned (`.python-version`).
- **Minecraft Java 1.21.1** pinned (UnionClef baseline; see T19 in `IMPLEMENTATION_PLAN.md`).
- **Server stack:** Fabric + Carpet + Lithium + FerriteCore. **Generational ZGC**, not G1GC.
- **All identifiers:** ULID Crockford base32.

## Contributing

1. Follow `IMPLEMENTATION_PLAN.md` task-by-task — every change should be a focused commit on a single task's checkbox.
2. `pre-commit` hooks must pass (`ruff`, `mypy`, `pytest`).
3. Integration tests requiring a live Fabric server are marked `@pytest.mark.integration` and skip gracefully when the server isn't reachable.

## License

MIT.
```

- [ ] **Step 2: Append M0 completion checklist + M1 preview**

Append to `IMPLEMENTATION_PLAN.md` (this file) at the bottom — see "M0 completion checklist" section below this task.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: M0 README with quickstart + status"
```

- [ ] **Step 4: Run the full unit suite**

Run: `pytest -v -m "not integration and not determinism"`

Expected: all PASS. If any fail, fix and re-commit before declaring M0 done.

- [ ] **Step 5: Tag the M0 release**

```bash
git tag -a m0 -m "M0: infrastructure foundation complete"
git log --oneline | head -40    # sanity check
```

---

## M0 completion checklist

Before tagging `m0`, verify every box:

- [ ] `python -m pip install -e ".[dev]"` succeeds on a clean venv
- [ ] `pre-commit run --all-files` passes
- [ ] `pytest -m "not integration and not determinism" -v` is all green
- [ ] `pytest -m determinism -v` is all green (CUDA fixture not flaky)
- [ ] `aiutopia --help` lists `agent`, `memory`, `determinism`
- [ ] `aiutopia agent spawn --role gatherer --no-fabric` inserts an identity row
- [ ] `aiutopia agent list` shows the spawned agent
- [ ] `aiutopia agent kill <ulid>` marks dead
- [ ] `cd fabric_mod && ./gradlew build` produces `aiutopia-mod-0.0.0-m0.jar`
- [ ] With a live Fabric 1.21.1 server + Carpet + the mod jar: `scripts/smoke-test.sh` succeeds, agent visible in connected MC client
- [ ] Identity DB has the 5 §3.5 tables (`roles, agents, agent_lives, funerals, policy_deployments`)
- [ ] Planner DB has the 4 tables (`planner_state, plan_cache, chat_failures, planner_failures`)
- [ ] All 5 critical bugs verified in source:
  - [ ] ZGC in `docker-compose.production.yml` (no `G1GC`)
  - [ ] `additional_module_specs` mentioned in `rl_module/stubs.py` docstring (real impl deferred to M1)
  - [ ] `observations_all()` batched (no per-agent `observation(agent)`)
  - [ ] `close()` implemented in `AiUtopiaPettingZooEnv` and `FabricBridge`
  - [ ] `configure_cuda_determinism()` called in determinism conftest

## Handoff to M1

M1 deliverable: train a solo gatherer to 80% success on "collect 64 oak_log"
within 1000 env steps over 3 consecutive evals.

Pre-requisites M1 inherits from M0 (do NOT re-implement):
- All identity / schema / memory / bridge / env / CLI / determinism scaffolding above.
- Carpet `/player spawn` working end-to-end.
- ZGC + idempotent `close()` + batched obs + ULID convention + CUDA determinism.

What M1 adds:
- Real `AiUtopiaRoleRLModule` (per §7.2 — `additional_module_specs` for shared submodules)
- `CoreEncoder`, `SharedBackbone(LSTM)`, `CTDECritic` real implementations
- `compute_reward()` stage-1 (per §5.1 — gatherer signal + PBRS + universal penalties)
- `ExploitDetector` runtime hookup
- Training driver `scripts/train.py` with Ray Tune + `EvalGateStopCallback`
- Per-tick RL loop wired through `step()` (replace stub motor responses with real Baritone calls in `MotorBridge.dispatchSkill`)
- M1 eval scenarios (fixed seeds 1/2/3)
- First weight promotion via `aiutopia promote-weights`
- First passing determinism check on real weights

When M1 starts, invoke `superpowers:writing-plans` again with the spec
section pointer (`§5.1`, `§4.6`, `§7.2`, `§7.4`) and the M1 gate criteria.

---

## Self-review notes

Run after writing this plan:
- **Spec coverage:** every spec § in the "§-touched-in-M0 vs deferred" table maps to at least one task (or is explicitly deferred with a milestone). ✓
- **Placeholders:** no `TODO`/`TBD`/`fill in later` in tasks except inside Java stubs whose M1 work is explicitly scheduled. ✓
- **Type consistency:** `agent_uuid` is `str` everywhere; `role` is `Literal["gatherer"|"builder"|"farmer"|"defender"]` everywhere; `goal_embedding` is 512-d everywhere; `comm_payloads` is `(32, 128)` everywhere. ✓
- **Critical bugs baked in:** ZGC (T24), `additional_module_specs` (T28), batched `observationsAll` (T20, T26, T27), `close()` (T26, T27), CUDA fixture (T32). ✓
- **Spec inconsistency surfaced:** §3.5 7-table lump vs M0's 5+4 split called out in opening matter; user can decide whether to amend spec. ✓
- **UnionClef compatibility:** verified as decision gate at T19 with fallback path documented. ✓



