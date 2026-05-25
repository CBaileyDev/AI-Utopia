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
