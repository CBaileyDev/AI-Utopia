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
