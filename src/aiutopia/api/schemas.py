"""Pydantic v2 response/request models matching gui/API_CONTRACT.md."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["gatherer", "builder", "farmer", "defender"]


# ── health ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    bridge: Literal["online", "offline"]
    py4j_port: int
    instances: int
    mc_version: str = "1.21.1"
    server_time: str


# ── agents ──────────────────────────────────────────────────────────────
class AgentModel(BaseModel):
    id: str
    name: str
    role: str
    status: str
    uuid: str
    skin: str | None = None
    born: int | None = None
    x: float | None = None
    z: float | None = None
    rewards: float = 0.0
    health: float | None = None
    hunger: float | None = None


class SpawnRequest(BaseModel):
    role: Role
    name: str | None = None


class SpawnResponse(BaseModel):
    ok: bool
    agent: AgentModel | None = None
    error: str | None = None


class KillRequest(BaseModel):
    cause: str | None = None


class OkResponse(BaseModel):
    ok: bool
    error: str | None = None


# ── training ────────────────────────────────────────────────────────────
class TrainingRun(BaseModel):
    run_id: str
    seed: int | None = None
    backend: str | None = None
    iters: int = 0
    last_return: float | None = None
    status: Literal["running", "done", "errored"]
    path: str


class TrainingMetrics(BaseModel):
    return_mean: float | None = None
    entropy: float | None = None
    kl: float | None = None
    clipfrac: float | None = None
    term_rate: float | None = None


class HistoryPoint(BaseModel):
    iter: int
    return_mean: float | None = None
    entropy: float | None = None
    kl: float | None = None


class TrainingStatus(BaseModel):
    running: bool
    run_id: str | None = None
    backend: str | None = None
    iter: int = 0
    max_iters: int = 0
    sps: float | None = None
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    history: list[HistoryPoint] = Field(default_factory=list)


class TrainingStartRequest(BaseModel):
    backend: Literal["sim", "real"] = "sim"
    iters: int = 50
    num_envs: int | None = None
    entropy_coeff: float | None = None
    spawn_jitter: float | None = None
    approach_shaping: bool | None = None
    force_masked_spawn: bool | None = None
    seed: int | None = None


class TrainingStartResponse(BaseModel):
    ok: bool
    pid: int | None = None
    run_id: str | None = None
    error: str | None = None


# ── rewards ─────────────────────────────────────────────────────────────
class RewardsConfig(BaseModel):
    log_value: dict[str, float]
    pbrs: dict[str, float]
    role_task_items: dict[str, list[str]]
    role_caps: dict[str, dict[str, int]]


class RewardsPutResponse(BaseModel):
    ok: bool
    saved_path: str | None = None
    error: str | None = None


# ── logs ────────────────────────────────────────────────────────────────
class LogEntry(BaseModel):
    ts: str
    type: Literal["AGENT", "TRAIN", "SYSTEM", "CHAT"]
    message: str


# generic envelope used by the global exception handler
class ErrorEnvelope(BaseModel):
    ok: bool = False
    error: str


def to_any(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()
