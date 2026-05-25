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
