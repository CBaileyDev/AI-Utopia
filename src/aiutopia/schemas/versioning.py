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
