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
