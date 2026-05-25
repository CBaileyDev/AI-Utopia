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
