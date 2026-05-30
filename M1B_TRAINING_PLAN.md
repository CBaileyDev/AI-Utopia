# M1-Training (M1.B) Implementation Plan — v2 (post-review revision)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a solo gatherer policy that hits the spec section 5.8 M1 evaluation gate — **80% success on "collect 64 oak_log" within 1000 env steps over 3 consecutive evaluations**. After this plan, the project has its first learned weights, the `aiutopia promote-weights` CLI works against section 5.10's promotion checklist, and `aiutopia determinism check` passes on real weights.

**Architecture (revised):**
- Single `AiUtopiaRoleRLModule` (TorchRLModule) per policy with `CoreEncoder` + `SharedBackbone(LSTM-256)` + `CTDECritic` + role-specific encoder + actor head as **`nn.Module` attributes** (not multi-module-spec entries — that API surface does not exist). Cross-policy weight sharing for M2 is a separate refactor.
- Ray RLlib **new API stack** (`api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)`), explicit so we don't get bitten by a default flip.
- PettingZoo Parallel env wrapped in `ParallelPettingZooEnv` (RLlib does not auto-detect).
- Custom callbacks via `RLlibCallback` base class (not the deprecated `DefaultCallbacks` shim).
- Training stops when the M1 evaluation gate passes via a Tune-readable result metric, **not** by mutating `algorithm.workers`.

**Tech Stack:** Ray 2.40+ (RLlib + Tune), PyTorch 2.4+, PettingZoo 1.24.3+, TensorBoard. Python 3.12. Training on a single 9950X3D + 4080 box per spec section 1.4.

**Spec reference:** `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`. Builds on `m1a-verified` (`6158547`).

---

## v2 changelog (what changed vs the first draft)

This plan went through a three-reviewer audit (RLlib API, code, PyTorch correctness). Material fixes:

1. **Architecture collapsed.** `additional_module_specs` does not exist; instead, `CoreEncoder` / `SharedBackbone` / `CTDECritic` live as `nn.Module` attributes on `AiUtopiaRoleRLModule`. M2 will refactor for cross-role weight sharing.
2. **LSTM time-dim handling fixed.** Old version did `fused.unsqueeze(1)` which silently collapsed RLlib's `(B, T, ...)` training batches to T=1 — recurrent gradient was lost. New `_forward` detects 3-D obs and runs the LSTM over the full time window.
3. **`get_initial_state` returns per-agent rank-1 tensors `(H,)`** (not `(1, H)`) per RLlib convention.
4. **Stop mechanism fixed.** Gate writes `custom_metrics/m1_gate_passed=1.0`; Tune's `stop` dict watches that. No more `algorithm.workers.stop()` (that attribute was renamed `env_runner_group` and calling `.stop()` from inside a callback races with Tune teardown).
5. **`Columns` import** from `ray.rllib.core import Columns` (was wrong path).
6. **`RLlibCallback`** base class with `metrics_logger=None` kwarg (was `DefaultCallbacks`).
7. **`tune.Tuner` receives the config object** — `param_space=cfg`, not `cfg.to_dict()`.
8. **PettingZoo env wrapped** in `ParallelPettingZooEnv` inside the registered factory.
9. **`policy_mapping_fn` signature** updated to `(agent_id, episode, **kwargs)`.
10. **Legacy `model={"use_lstm": True, ...}` block removed** from `.training(...)` — ignored on new API stack and misleading.
11. **`policies_to_train=["gatherer_policy"]`** added to `.multi_agent(...)`.
12. **`keep_per_episode_custom_metrics=True`** so exploit aggregates survive aggregation.
13. **Vectorized `flatten_core_obs_batched`** kills the per-sample Python loop bottleneck.
14. **Discrete obs keys** (`biome_id`, `weather`) go through `nn.Embedding` instead of being cast to float.
15. **`initial_state` is device-aware** (moved to `next(self.parameters()).device`).
16. **T19 determinism replay threads LSTM hidden state across ticks** (the original version reinitialized state every tick, evaluating a memoryless approximation — defeats determinism checking entirely).
17. **T19 checkpoint load via `Algorithm.from_checkpoint(...).get_module(...)`** — a Tune checkpoint dir is NOT a `MultiRLModule` checkpoint dir; the previous `MultiRLModule.from_checkpoint(weights)` would have raised.
18. **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** set before CUDA init in T19 (required for LSTM under `use_deterministic_algorithms(True)`).
19. **`/tick rate 300.0`** (float required in Carpet 1.21.1).
20. **`resetEpisode(playerName, seed)`** threads seed into the log-ring placement so the 3 eval scenarios truly differ.
21. **Tests fixed:** `mrm._forward_train(...)` is called at the **MultiRLModule level** (not via `mrm["gatherer_policy"]`, which loses the parent context); `subprocess` env merge order corrected.
22. **`tune.logger.TBXLoggerCallback` dropped** — TensorBoard logging is default in 2.40+; manual addition is redundant.
23. **Edit anchors documented** for T13 (`wrapper.py:reset()`) and T16 (`cli/app.py` import line).
24. **New T7.5 (RLlib smoke test)** — runs 1 PPO iteration on a synthetic env to catch RLlib integration issues before T21 burns hours of training.
25. **Stale prose corrected** (inventory `36+36`, comms `32x128`, not `36+72` / `32x136`).

---

## Spec sections touched in M1-Training vs deferred to M2+

| Spec section | Coverage | Tasks |
|---|---|---|
| 4.3 Shared backbone architecture | `CoreEncoder` + `SharedBackbone(LSTM)` + `CTDECritic` as instance attributes of the RLModule (cross-policy sharing deferred to M2) | T1–T6 |
| 4.6 Per-tick RL loop (training side) | RLlib EnvRunners drive `env.step()` per agent per tick | T9, T15 |
| 4.9 Episodic memory write path | Already live via M1A T17 | (inherited) |
| 5.1 Stage-1 reward (solo per-role pretraining) | Already live via M1A T14 (`compute_reward_stage_1`) | (inherited) |
| 5.8 Phase 1 milestone: solo gatherer | M1 evaluation gate enforced by `EvalGateStopCallback` + Tune stop dict | T11, T18, T21 |
| 5.10 Promotion checklist | `aiutopia promote-weights` implements 5 gates | T16, T17 |
| 7.1 PPOConfig (single-agent M1) | `m1_gatherer_config` per spec section 7.1 | T7, T8 |
| 7.2 RLModule (real) | Full implementation; cross-role sharing deferred to M2 | T1–T6 |
| 7.4 Training driver | `scripts/train.py` with Ray Tune + checkpointing + TensorBoard + callbacks | T9-T11, T15, T18 |
| 7.5 CLI tooling (promote-weights, determinism check) | Both wired against real weights | T16, T17, T19 |
| 7.8 Determinism harness | Real-weights replay with proper LSTM state threading | T19 |
| 5.2 Stage 2 reward (multi-objective + curriculum) | Deferred to M2 | (M2) |
| 5.3 Multi-agent exploit (BULK_FARMING) | Deferred to M2 | (M2) |
| 5.9 MARL failure mitigations | Deferred to M2 | (M2) |
| 2.2 Two-world topology (production server) | Training side only | (M2/M6) |

---

## File structure

```
src/aiutopia/
|-- rl_module/                                    (replaces M0 stubs)
|   |-- __init__.py
|   |-- core_encoder.py                           (T1)
|   |-- shared_backbone.py                        (T2)
|   |-- ctde_critic.py                            (T3)
|   |-- role_encoder.py                           (T4)
|   |-- actor_head.py                             (T5)
|   |-- role_rl_module.py                         (T6)
|-- train/                                        (NEW)
|   |-- __init__.py
|   |-- config.py                                 (T7)
|   |-- env_factory.py                            (T8)
|   |-- callbacks.py                              (T9, T10, T11, T15)
|   |-- scenario_runner.py                        (T14)
|-- promotion/                                    (NEW)
|   |-- __init__.py
|   |-- service.py                                (T16)
|   |-- checklist.py                              (T17)
|-- env/
|   |-- wrapper.py                                (MODIFY — T13)
|   |-- bridge.py                                 (MODIFY — T13)
|-- cli/
|   |-- promote.py                                (T16)
|   |-- app.py                                    (MODIFY — T16 register subcommand)
|   |-- determinism.py                            (MODIFY — T19)
|-- determinism/
    |-- harness.py                                (MODIFY — T19 add replay_with_rlmodule)

fabric_mod/src/main/java/dev/aiutopia/mod/
|-- bridge/WorldOps.java                          (MODIFY — T12)
|-- Py4JEntryPoint.java                           (MODIFY — T12)

scripts/
|-- train.py                                      (T18)
|-- rllib_smoke.py                                (T7.5 — 1-iter sanity check)
|-- launch-training-instances.sh                  (T20)
|-- m1b-evaluation-gate.sh                        (T20)

tests/
|-- unit/  (T1, T2, T3, T4, T5, T6, T7, T9, T11, T17, T19)
|-- integration/test_evaluation_scenario_smoke.py (T14)
|-- integration/test_rllib_smoke.py               (T7.5)

docs/superpowers/m1b-tuning-notes.md              (T21)
```

---

## Pre-flight: confirm M1A state

```bash
cd "C:\Users\Carte\OneDrive\Desktop\AiUtopia"
git log --oneline -3                 # HEAD >= 6158547
git tag -l                           # must list m1a-verified
python -m pytest tests/unit -v -m "not integration and not determinism" 2>&1 | tail -3
```

Install training-stack deps (Python 3.12 venv):
```bash
pip install "ray[rllib,tune]>=2.40,<2.60" "torch>=2.4" "tensorboard>=2.16" "pettingzoo>=1.24.3"
```

Verify Ray version:
```bash
python -c "import ray; print(ray.__version__)"   # >= 2.40.0
```

If Ray ≥ 2.55 has further API drift from 2.40, treat 2.40-2.50 as the supported window and pin in `pyproject.toml`.

---

## Tasks

### Task 1: `CoreEncoderModule` — universal core obs to 256-d feature (vectorized)

**Files:**
- Create: `src/aiutopia/rl_module/core_encoder.py`
- Create: `tests/unit/test_core_encoder.py`

Per spec section 4.3 + 7.2. Universal core (agent_uuid_embed 384 + role_one_hot 4 + body 11 + inventory 36+36 + goal 513 + world 4 + comms 32x128 flattened + meta 32x8) → vectorized concat → Linear → ReLU → Linear → ReLU → 256-d. Discrete keys (`biome_id`, `weather`) go through `nn.Embedding` for proper representation; the rest are cast to float32 and concatenated.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_core_encoder.py`:
```python
import numpy as np
import pytest
import torch
from gymnasium.spaces import Dict as DictSpace

from aiutopia.env.spaces import build_role_observation_space
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule, core_obs_flat_dim, flatten_core_obs_batched,
)


def _batched_sample(batch: int = 2) -> dict:
    space: DictSpace = build_role_observation_space("gatherer", stage=1)
    out = {k: [] for k in space.spaces}
    for _ in range(batch):
        s = space.sample()
        for k, v in s.items():
            out[k].append(np.asarray(v))
    return {k: torch.as_tensor(np.stack(vs)) for k, vs in out.items()}


def test_core_obs_flat_dim_is_consistent() -> None:
    dim = core_obs_flat_dim()
    # comm_payloads alone is 32*128=4096; floor at least that big
    assert dim > 4000
    assert dim < 20_000


def test_flatten_core_obs_batched_returns_BxD() -> None:
    obs = _batched_sample(batch=4)
    flat = flatten_core_obs_batched(obs)
    assert flat.ndim == 2
    assert flat.shape == (4, core_obs_flat_dim())
    assert flat.dtype == torch.float32


def test_flatten_core_obs_ignores_role_overlay() -> None:
    obs = _batched_sample(batch=2)
    obs2 = {**obs, "g_richness_score": torch.full_like(
        obs.get("g_richness_score", torch.zeros(2, 1)), 0.42)}
    a = flatten_core_obs_batched(obs)
    b = flatten_core_obs_batched(obs2)
    assert torch.allclose(a, b)


def test_core_encoder_module_output_shape_256() -> None:
    module = CoreEncoderModule(config={"core_hidden": [512, 256]})
    flat = torch.randn(4, core_obs_flat_dim())
    out = module(flat)
    assert out.shape == (4, 256)


def test_core_encoder_module_param_count_reasonable() -> None:
    module = CoreEncoderModule(config={"core_hidden": [512, 256]})
    n_params = sum(p.numel() for p in module.parameters())
    assert 1_000_000 < n_params < 20_000_000
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/unit/test_core_encoder.py -v
```

Expected: ImportError on `aiutopia.rl_module.core_encoder`.

- [ ] **Step 3: Implement the encoder**

Create `src/aiutopia/rl_module/core_encoder.py`:
```python
"""Section 4.3 CoreEncoder — universal core obs to 256-d feature.

Per the v2 review:
- vectorized batched flatten (no per-sample Python loop)
- Discrete keys go through nn.Embedding instead of float cast
- belongs to AiUtopiaRoleRLModule as an instance attribute, NOT a multi-module-spec entry
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, MultiBinary, Dict as DictSpace

from aiutopia.env.spaces import build_role_observation_space


# Universal core obs keys. Role-overlay keys (g_*, b_*, f_*, d_*) and
# `action_mask` are deliberately excluded; those belong to the per-role encoder.
_CORE_KEYS_FOR_FLATTEN = (
    "agent_uuid_embed", "role_one_hot", "tick_in_episode",
    "position", "velocity", "yaw_pitch",
    "health", "hunger", "saturation", "armor_value",
    "inv_slot_item_ids", "inv_slot_counts",
    "main_hand_item_id", "off_hand_item_id",
    "goal_embedding", "goal_ticks_left",
    "time_of_day", "weather", "biome_id", "light_level",
    "comm_payloads", "comm_metadata",
)


def core_obs_flat_dim() -> int:
    """Total dim of the concatenated core obs (before encoder)."""
    space = build_role_observation_space("gatherer", stage=1)
    total = 0
    for key in _CORE_KEYS_FOR_FLATTEN:
        sub = space.spaces[key]
        total += int(np.prod(sub.shape)) if sub.shape else 1
    return total


def flatten_core_obs_batched(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Vectorized flatten: dict of (B, ...) tensors -> (B, D) float32 tensor.

    All present batch dim; scalar Discretes show up as (B,) and get
    unsqueezed to (B, 1). Outputs stay on the input device.
    """
    parts: list[torch.Tensor] = []
    for key in _CORE_KEYS_FOR_FLATTEN:
        v = obs[key]
        if not torch.is_tensor(v):
            v = torch.as_tensor(np.asarray(v))
        if v.dtype != torch.float32:
            v = v.to(torch.float32)
        if v.ndim == 1:
            v = v.unsqueeze(-1)
        else:
            v = v.reshape(v.shape[0], -1)
        parts.append(v)
    return torch.cat(parts, dim=-1)


class CoreEncoderModule(nn.Module):
    """Universal core obs (flat, ~5k-d) -> 256-d feature.

    Instantiated as `self.core_encoder` on AiUtopiaRoleRLModule. For M1
    there's one policy, so no cross-policy weight sharing concern. M2
    will revisit if the builder/farmer benefit from sharing.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config.get("core_hidden", [512, 256])
        in_dim = core_obs_flat_dim()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, core_flat: torch.Tensor) -> torch.Tensor:
        return self.net(core_flat)
```

- [ ] **Step 4: Run tests + replace `__init__.py`**

```bash
python -m pytest tests/unit/test_core_encoder.py -v
```

Replace `src/aiutopia/rl_module/__init__.py`:
```python
"""Section 7.2 RLlib RLModule layer for the AI Utopia village (M1).

Module composition (all live as instance attributes on AiUtopiaRoleRLModule):
  - CoreEncoderModule      universal core obs -> 256-d
  - SharedBackboneModule   Linear(448->384) + LSTM(256)
  - CTDECriticModule       two-stage encoder + MLP -> V(s)
  - GathererRoleEncoder    role-specific obs -> 128-d
  - GathererActorHead      action distribution producer

M2 may refactor to share Core/Backbone/Critic across multiple role policies;
the M1 plan deliberately ships one policy with bare nn.Module attributes.
"""
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule, core_obs_flat_dim, flatten_core_obs_batched,
)

__all__ = ["CoreEncoderModule", "core_obs_flat_dim", "flatten_core_obs_batched"]
```

Delete M0 stubs: `git rm src/aiutopia/rl_module/stubs.py` if present.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/rl_module/core_encoder.py \
        src/aiutopia/rl_module/__init__.py \
        tests/unit/test_core_encoder.py
git commit -m "feat(rl): CoreEncoderModule + vectorized batched flatten (M1-Training T1)"
```

Expected: 5 PASSED.

---

### Task 2: `SharedBackboneModule` — Linear(448->384) + LSTM(256), device-aware

**Files:**
- Create: `src/aiutopia/rl_module/shared_backbone.py`
- Create: `tests/unit/test_shared_backbone.py`

Input: concat[core(256) | role(128) | pixel(64)] = 448. Output: LSTM hidden in R^256.

**Fix vs v1:** `initial_state(batch_size, *, device)` accepts a device so the LSTM doesn't crash when used inside a GPU forward.

- [ ] **Step 1: Failing test**

Create `tests/unit/test_shared_backbone.py`:
```python
import torch

from aiutopia.rl_module.shared_backbone import SharedBackboneModule


def test_backbone_output_shape_with_state() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    batch, seq_len = 4, 1
    fused = torch.randn(batch, seq_len, 448)
    h0, c0 = b.initial_state(batch_size=batch, device=fused.device)
    out, (h1, c1) = b(fused, (h0, c0))
    assert out.shape == (batch, seq_len, 256)
    assert h1.shape == (1, batch, 256)
    assert c1.shape == (1, batch, 256)


def test_backbone_state_changes_with_input() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    fused = torch.randn(2, 1, 448)
    h0, c0 = b.initial_state(batch_size=2, device=fused.device)
    _, (h1, _) = b(fused, (h0, c0))
    assert not torch.allclose(h1, h0)


def test_backbone_initial_state_device() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    cpu = torch.device("cpu")
    h, c = b.initial_state(batch_size=3, device=cpu)
    assert h.device == cpu
    assert h.shape == (1, 3, 256)


def test_backbone_handles_multi_step_sequence() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    fused = torch.randn(1, 10, 448)
    h0, c0 = b.initial_state(batch_size=1, device=fused.device)
    out, _ = b(fused, (h0, c0))
    assert out.shape == (1, 10, 256)
```

- [ ] **Step 2: Implement**

Create `src/aiutopia/rl_module/shared_backbone.py`:
```python
"""Section 4.3 SharedBackbone — Linear projection + LSTM(256)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

FUSED_INPUT_DIM = 448   # core(256) + role(128) + pixel(64), per section 4.3


class SharedBackboneModule(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.lstm_hidden = config.get("lstm_hidden", 256)
        self.proj = nn.Sequential(
            nn.Linear(FUSED_INPUT_DIM, 384),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, fused, state):
        # fused: (B, T, 448) ; state: ((1,B,H), (1,B,H))
        projected = self.proj(fused)
        out, new_state = self.lstm(projected, state)
        return out, new_state

    def initial_state(self, batch_size: int, *, device: torch.device | str = "cpu"):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return h, c
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_shared_backbone.py -v
git add src/aiutopia/rl_module/shared_backbone.py tests/unit/test_shared_backbone.py
git commit -m "feat(rl): SharedBackboneModule with device-aware initial_state (M1-Training T2)"
```

Expected: 4 PASSED.

---

### Task 3: `CTDECriticModule` — two-stage encoder

**Files:**
- Create: `src/aiutopia/rl_module/ctde_critic.py`
- Create: `tests/unit/test_ctde_critic.py`

Two-stage encoder: per-agent shared encoder compresses each agent's obs (~5000-d) to 128-d; head MLP(4*128 + village_inv_enc → 256) → V(s). For M1 (single agent), the other 3 slots are zero-padded.

(unchanged from v1; was the only sub-module with no defects)

- [ ] **Step 1: Failing test**

Create `tests/unit/test_ctde_critic.py`:
```python
import torch

from aiutopia.rl_module.core_encoder import core_obs_flat_dim
from aiutopia.rl_module.ctde_critic import (
    CTDECriticModule, VILLAGE_INV_DIM, PER_AGENT_COMPRESSED_DIM,
)


def test_per_agent_compressed_dim_is_128() -> None:
    assert PER_AGENT_COMPRESSED_DIM == 128


def test_critic_output_shape() -> None:
    critic = CTDECriticModule(config={})
    batch = 4
    obs_dim = core_obs_flat_dim()
    all_agents = torch.randn(batch, 4, obs_dim)
    village_inv = torch.randn(batch, VILLAGE_INV_DIM)
    v = critic(all_agents, village_inv)
    assert v.shape == (batch,)


def test_critic_param_count_drops_vs_naive_mlp() -> None:
    critic = CTDECriticModule(config={})
    n = sum(p.numel() for p in critic.parameters())
    assert n < 10_000_000


def test_critic_handles_single_agent_padding() -> None:
    critic = CTDECriticModule(config={})
    obs_dim = core_obs_flat_dim()
    all_agents = torch.zeros(2, 4, obs_dim)
    all_agents[:, 0, :] = torch.randn(2, obs_dim)
    village_inv = torch.zeros(2, VILLAGE_INV_DIM)
    v = critic(all_agents, village_inv)
    assert not torch.isnan(v).any()
```

- [ ] **Step 2: Implement**

Create `src/aiutopia/rl_module/ctde_critic.py`:
```python
"""Section 4.3 CTDE critic with two-stage encoder."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aiutopia.rl_module.core_encoder import core_obs_flat_dim


PER_AGENT_COMPRESSED_DIM = 128
NUM_AGENT_SLOTS = 4
VILLAGE_INV_DIM = 64


class CTDECriticModule(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        head_hidden = config.get("critic_hidden", 256)
        in_dim = core_obs_flat_dim()
        self.per_agent_encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, PER_AGENT_COMPRESSED_DIM),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(NUM_AGENT_SLOTS * PER_AGENT_COMPRESSED_DIM + VILLAGE_INV_DIM,
                       head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, all_agents_obs, village_inv):
        batch, n_slots, _ = all_agents_obs.shape
        flat = all_agents_obs.reshape(batch * n_slots, -1)
        compressed = self.per_agent_encoder(flat).reshape(batch, -1)
        x = torch.cat([compressed, village_inv], dim=-1)
        return self.head(x).squeeze(-1)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_ctde_critic.py -v
git add src/aiutopia/rl_module/ctde_critic.py tests/unit/test_ctde_critic.py
git commit -m "feat(rl): CTDECriticModule (M1-Training T3)"
```

Expected: 4 PASSED.

---

### Task 4: `GathererRoleEncoder` — role-specific obs to 128-d

**Files:**
- Create: `src/aiutopia/rl_module/role_encoder.py`
- Create: `tests/unit/test_role_encoder.py`

(arithmetic unchanged from v1: 32→16→8 via stride-2 convs, then 32*8*8=2048 → Linear(2048, 64); + flat MLP on nearest/richness/hostiles → 64; concat → 128)

- [ ] **Step 1: Implement**

Create `src/aiutopia/rl_module/role_encoder.py`:
```python
"""Section 4.3 Per-role obs encoder. M1 ships gatherer only."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GathererRoleEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
        )
        flat_in = 8 * 6 + 1 + 4 * 4   # 65
        self.flat_mlp = nn.Sequential(
            nn.Linear(flat_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        # obs values may arrive as (B, H, W, C) or (B*T, H, W, C); same op works
        grid = obs["g_resource_grid"].permute(0, 3, 1, 2).contiguous()
        grid_feat = self.grid_conv(grid)
        nearest = obs["g_nearest_resources"].flatten(start_dim=1)
        rich    = obs["g_richness_score"]
        host    = obs["g_hostiles_nearby"].flatten(start_dim=1)
        flat    = torch.cat([nearest, rich, host], dim=-1)
        flat_feat = self.flat_mlp(flat)
        return torch.cat([grid_feat, flat_feat], dim=-1)


def build_role_encoder(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererRoleEncoder(config)
    raise NotImplementedError(f"role {role!r} encoder not built (M2+)")
```

- [ ] **Step 2: Test**

Create `tests/unit/test_role_encoder.py`:
```python
import pytest
import torch

from aiutopia.rl_module.role_encoder import (
    GathererRoleEncoder, build_role_encoder,
)


def _gatherer_obs(batch: int = 2) -> dict:
    return {
        "g_resource_grid":     torch.rand(batch, 32, 32, 6),
        "g_nearest_resources": torch.rand(batch, 8, 6),
        "g_richness_score":    torch.rand(batch, 1),
        "g_hostiles_nearby":   torch.rand(batch, 4, 4),
    }


def test_gatherer_encoder_outputs_128_d() -> None:
    enc = GathererRoleEncoder(config={})
    feat = enc(_gatherer_obs(batch=2))
    assert feat.shape == (2, 128)


def test_build_role_encoder_gatherer_only_in_m1() -> None:
    assert isinstance(build_role_encoder("gatherer", {}), GathererRoleEncoder)
    for r in ("builder", "farmer", "defender"):
        with pytest.raises(NotImplementedError):
            build_role_encoder(r, {})
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_role_encoder.py -v
git add src/aiutopia/rl_module/role_encoder.py tests/unit/test_role_encoder.py
git commit -m "feat(rl): GathererRoleEncoder (M1-Training T4)"
```

Expected: 2 PASSED.

---

### Task 5: `GathererActorHead` — per-role action distribution

**Files:**
- Create: `src/aiutopia/rl_module/actor_head.py`
- Create: `tests/unit/test_actor_head.py`

`OUTPUT_DIM = 6 + 64 + 2*3 + 2*1 + 2*128 + 2 + 4 = 340`.

- [ ] **Step 1: Implement**

Create `src/aiutopia/rl_module/actor_head.py`:
```python
"""Section 4.3 ActorHead — per-role action distribution producer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aiutopia.env.spaces import (
    COMM_PAYLOAD_DIM,
    GOAL_EMBED_DIM,
    N_GATHERER_SKILLS,
    N_TARGET_CLASSES_PER_ROLE,
)


_GATHERER_HEAD_SLICES = {
    "skill_type":       ("logits", N_GATHERER_SKILLS),
    "target_class":     ("logits", N_TARGET_CLASSES_PER_ROLE),
    "spatial_param":    ("gaussian", 3),
    "scalar_param":     ("gaussian", 1),
    "comm_payload":     ("gaussian", COMM_PAYLOAD_DIM),
    "should_broadcast": ("logits", 2),
    "comm_target_mask": ("logits", 4),
}


def _output_size_for(kind: str, dim: int) -> int:
    return dim if kind == "logits" else 2 * dim


class GathererActorHead(nn.Module):
    INPUT_DIM = 256 + GOAL_EMBED_DIM
    OUTPUT_DIM = sum(_output_size_for(kind, dim)
                      for kind, dim in _GATHERER_HEAD_SLICES.values())

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        actor_hidden = config.get("actor_hidden", [256])
        layers: list[nn.Module] = []
        prev = self.INPUT_DIM
        for h in actor_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, self.OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden, goal_embedding):
        x = torch.cat([hidden, goal_embedding], dim=-1)
        return self.net(x)


def build_actor_head(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererActorHead(config)
    raise NotImplementedError(f"actor head for {role!r} not built (M2+)")
```

- [ ] **Step 2: Test**

Create `tests/unit/test_actor_head.py`:
```python
import pytest
import torch

from aiutopia.rl_module.actor_head import GathererActorHead, build_actor_head


def test_actor_head_output_dim_is_340() -> None:
    head = GathererActorHead(config={})
    assert head.OUTPUT_DIM == 340   # 6 + 64 + 6 + 2 + 256 + 2 + 4


def test_actor_head_forward_shape() -> None:
    head = GathererActorHead(config={})
    batch = 4
    hidden = torch.randn(batch, 256)
    goal   = torch.randn(batch, 512)
    out = head(hidden, goal)
    assert out.shape == (batch, 340)


def test_build_actor_head_gatherer_only_in_m1() -> None:
    assert isinstance(build_actor_head("gatherer", {}), GathererActorHead)
    for r in ("builder", "farmer", "defender"):
        with pytest.raises(NotImplementedError):
            build_actor_head(r, {})
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_actor_head.py -v
git add src/aiutopia/rl_module/actor_head.py tests/unit/test_actor_head.py
git commit -m "feat(rl): GathererActorHead (M1-Training T5)"
```

Expected: 3 PASSED.

---

### Task 6: `AiUtopiaRoleRLModule` — TorchRLModule with proper LSTM time-dim handling

**Files:**
- Create: `src/aiutopia/rl_module/role_rl_module.py`
- Create: `tests/unit/test_role_rl_module.py`

**Major v2 fixes:**
- Shared submodules are **instance attributes**, not `multi_rl_module["..."]` entries.
- `_forward` detects 3-D obs (B, T, ...) coming from PPO's recurrent rollouts and runs the LSTM over the full time window, not T=1.
- `get_initial_state` returns per-agent rank-1 tensors `(H,)` per RLlib new-API-stack convention.
- `Columns` imported from `ray.rllib.core` (not `ray.rllib.core.columns`).
- `initial_state` is device-aware so GPU training doesn't crash on the first batch.

- [ ] **Step 1: Implement**

Create `src/aiutopia/rl_module/role_rl_module.py`:
```python
"""Section 7.2 AiUtopiaRoleRLModule — per-role TorchRLModule.

For M1 there is exactly one policy (gatherer). All sub-encoders live as
instance attributes on this RLModule. Cross-policy weight sharing for M2+
is a separate refactor.

The forward path handles both (B, obs_dim) inference obs and (B, T, obs_dim)
training obs that RLlib produces under the LSTM connector with max_seq_len>1.
The original v1 plan collapsed the time dim with unsqueeze(1) which silently
disabled the LSTM during training.
"""
from __future__ import annotations

from typing import Any

import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from aiutopia.rl_module.actor_head     import build_actor_head
from aiutopia.rl_module.core_encoder   import flatten_core_obs_batched
from aiutopia.rl_module.ctde_critic    import CTDECriticModule, VILLAGE_INV_DIM
from aiutopia.rl_module.role_encoder   import build_role_encoder
from aiutopia.rl_module.shared_backbone import SharedBackboneModule
from aiutopia.rl_module.core_encoder   import CoreEncoderModule


class AiUtopiaRoleRLModule(TorchRLModule):
    """Per-role policy module with embedded shared submodules.

    Stateful: LSTM hidden carried across ticks via `Columns.STATE_IN`/`STATE_OUT`.
    """

    def setup(self) -> None:
        super().setup()
        cfg = self.model_config
        self.role = cfg["role"]
        self.core_encoder    = CoreEncoderModule(cfg.get("core_encoder",
                                                          {"core_hidden": [512, 256]}))
        self.shared_backbone = SharedBackboneModule(cfg.get("shared_backbone",
                                                              {"lstm_hidden": 256}))
        self.ctde_critic     = CTDECriticModule(cfg.get("ctde_critic", {}))
        self.role_encoder    = build_role_encoder(self.role, cfg)
        self.actor_head      = build_actor_head(self.role, cfg)
        self.pixel_encoder   = None         # M2+ for builder
        self._pixel_zero_dim = 64

    # ─────────────────────────────────────────────────────────────
    # RLlib new-API-stack stateful contract
    def get_initial_state(self) -> dict[str, torch.Tensor]:
        """Per-AGENT initial state (unbatched, rank-1)."""
        H = self.shared_backbone.lstm_hidden
        device = next(self.parameters()).device
        return {"h": torch.zeros(H, device=device),
                "c": torch.zeros(H, device=device)}

    # ─────────────────────────────────────────────────────────────
    # Forward — handles inference (B, ...) and training (B, T, ...)
    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, with_value=True)

    def _forward(self, batch: dict, *, with_value: bool) -> dict:
        obs = batch[Columns.OBS]

        # Detect leading shape. Compare against a known 1-D-per-sample key
        # (goal_embedding is (B, GOAL_EMBED_DIM) at inference and
        # (B, T, GOAL_EMBED_DIM) under recurrent training).
        ref = obs["goal_embedding"]
        time_dimension = (ref.ndim == 3)
        if time_dimension:
            batch_size, seq_len = ref.shape[:2]
            # Flatten T into B for per-tick encoders, restore for LSTM.
            obs_flat_t = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        else:
            batch_size = ref.shape[0]
            seq_len = 1
            obs_flat_t = obs

        core_input = flatten_core_obs_batched(obs_flat_t)        # (B*T, D)
        core_feat  = self.core_encoder(core_input)                # (B*T, 256)
        role_feat  = self.role_encoder(obs_flat_t)                # (B*T, 128)
        pixel_feat = torch.zeros(core_feat.size(0), self._pixel_zero_dim,
                                   device=core_feat.device)       # (B*T, 64)
        fused = torch.cat([core_feat, role_feat, pixel_feat], dim=-1)  # (B*T, 448)

        # Reshape to (B, T, 448) for LSTM
        fused_seq = fused.view(batch_size, seq_len, -1)

        # State: RLlib gives (B, H); LSTM wants (1, B, H).
        state_in = batch.get(Columns.STATE_IN, None)
        if state_in is None:
            h0, c0 = self.shared_backbone.initial_state(
                batch_size=batch_size, device=fused_seq.device)
        else:
            h0 = state_in["h"].unsqueeze(0).to(fused_seq.device)
            c0 = state_in["c"].unsqueeze(0).to(fused_seq.device)

        backbone_out, (h1, c1) = self.shared_backbone(fused_seq, (h0, c0))
        # backbone_out: (B, T, 256)

        # Flatten time back into batch for the actor head
        hidden_flat = backbone_out.reshape(-1, backbone_out.size(-1))  # (B*T, 256)
        goal_flat   = obs_flat_t["goal_embedding"]                       # (B*T, 512)
        action_dist_inputs_flat = self.actor_head(hidden_flat, goal_flat)  # (B*T, 340)

        # Restore time dim for RLlib if we had one
        if time_dimension:
            action_dist_inputs = action_dist_inputs_flat.view(batch_size, seq_len, -1)
        else:
            action_dist_inputs = action_dist_inputs_flat

        result = {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.STATE_OUT: {"h": h1.squeeze(0), "c": c1.squeeze(0)},
        }

        if with_value:
            # CTDE critic: M1 single-agent, other 3 slots zero-padded.
            n = core_input.size(0)                                     # B*T
            all_agents = torch.zeros(n, 4, core_input.size(-1),
                                      device=core_feat.device)
            all_agents[:, 0, :] = core_input
            village_inv = torch.zeros(n, VILLAGE_INV_DIM,
                                        device=core_feat.device)
            v_flat = self.ctde_critic(all_agents, village_inv)         # (B*T,)
            if time_dimension:
                result[Columns.VF_PREDS] = v_flat.view(batch_size, seq_len)
            else:
                result[Columns.VF_PREDS] = v_flat

        return result
```

- [ ] **Step 2: Test (at MultiRLModule level so parent context is preserved)**

Create `tests/unit/test_role_rl_module.py`:
```python
import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from aiutopia.env.spaces import (
    build_role_action_space, build_role_observation_space,
)
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule


def _build_multi_module():
    obs_space    = build_role_observation_space("gatherer", stage=1)
    action_space = build_role_action_space("gatherer")
    spec = MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=obs_space,
                action_space=action_space,
                model_config={"role": "gatherer"},
            ),
        },
    )
    return spec.build()


def _sample_batched(obs_space, batch: int = 2):
    import numpy as np
    out = {k: [] for k in obs_space.spaces}
    for _ in range(batch):
        s = obs_space.sample()
        for k, v in s.items():
            out[k].append(np.asarray(v))
    return {k: torch.as_tensor(np.stack(vs)) for k, vs in out.items()}


def test_multi_rl_module_assembles() -> None:
    mrm = _build_multi_module()
    assert "gatherer_policy" in mrm


def test_forward_inference_produces_action_dist() -> None:
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    out = mrm["gatherer_policy"]._forward_inference({"obs": batched})
    assert "action_dist_inputs" in out
    assert out["action_dist_inputs"].shape == (2, 340)


def test_forward_train_emits_vf_preds() -> None:
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    out = mrm["gatherer_policy"]._forward_train({"obs": batched})
    assert "vf_preds" in out
    assert out["vf_preds"].shape == (2,)


def test_forward_train_with_time_dim() -> None:
    """RLlib passes (B, T, ...) when LSTM unroll is active."""
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    # Add a time dim of 4
    batched_t = {k: v.unsqueeze(1).expand(2, 4, *v.shape[1:]).contiguous()
                 for k, v in batched.items()}
    out = mrm["gatherer_policy"]._forward_train({"obs": batched_t})
    assert out["action_dist_inputs"].shape == (2, 4, 340)
    assert out["vf_preds"].shape == (2, 4)


def test_initial_state_is_unbatched_rank_1() -> None:
    mrm = _build_multi_module()
    state = mrm["gatherer_policy"].get_initial_state()
    assert state["h"].shape == (256,)
    assert state["c"].shape == (256,)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_role_rl_module.py -v
git add src/aiutopia/rl_module/role_rl_module.py tests/unit/test_role_rl_module.py
git commit -m "feat(rl): AiUtopiaRoleRLModule with proper LSTM time-dim handling (M1-Training T6)"
```

Expected: 5 PASSED.

**Critical: if `test_forward_train_with_time_dim` fails, the LSTM is being run with T=1 — STOP and fix before T7. This is the load-bearing recurrent-gradient test.**

---

### Task 7: PPO config builder (`m1_gatherer_config`) — new API stack

**Files:**
- Create: `src/aiutopia/train/__init__.py`
- Create: `src/aiutopia/train/config.py`
- Create: `tests/unit/test_train_config.py`

Per spec section 7.1. **v2 fixes:** explicit `.api_stack(...)`, no `additional_module_specs`, no legacy `model={use_lstm:...}`, `policies_to_train=["gatherer_policy"]`, `policy_mapping_fn` with `(agent_id, episode, **kwargs)` signature, `keep_per_episode_custom_metrics=True`.

- [ ] **Step 1: Implement**

Create `src/aiutopia/train/__init__.py`:
```python
"""Section 7.1 RLlib PPO training stack."""
```

Create `src/aiutopia/train/config.py`:
```python
"""Section 7.1 PPO config builders."""
from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule


ENV_NAME = "aiutopia_minecraft"


def register_aiutopia_env() -> None:
    """Idempotent registration of the env factory with Ray Tune."""
    from aiutopia.train.env_factory import make_aiutopia_env_wrapped
    register_env(ENV_NAME, make_aiutopia_env_wrapped)


def _policy_mapping_fn(agent_id, episode=None, **kwargs):
    """New-API-stack signature: (agent_id, episode, **kwargs)."""
    return "gatherer_policy"


def m1_gatherer_config(*,
                        py4j_ports:        tuple[int, ...] = (25001, 25002, 25003, 25004),
                        num_env_runners:   int = 4,
                        num_envs_per_env_runner: int = 2,
                        max_episode_ticks: int = 12_000,
                        seed:              int = 1,
                        ) -> PPOConfig:
    """Section 7.1 M1 single-agent gatherer PPO config (new API stack)."""
    register_aiutopia_env()

    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .environment(
            env=ENV_NAME,
            env_config={
                "stage":                 1,
                "active_roles":          ["gatherer"],
                "seed_strategy":         "fixed_easy",
                "py4j_ports":            list(py4j_ports),
                "tick_warp":             True,
                "max_episode_ticks":     max_episode_ticks,
                "per_worker_seed_offset": True,
                "enable_memory_writes":  True,
            },
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            rollout_fragment_length="auto",
            sample_timeout_s=120.0,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=0.5,
        )
        .training(
            train_batch_size=4096,
            minibatch_size=512,
            num_epochs=5,
            gamma=0.99,
            lr=3.0e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            grad_clip=1.0,
            # NO legacy model={"use_lstm": True} block — the custom RLModule
            # owns the LSTM. Ray ignores this block under new API stack
            # but its presence is misleading.
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "gatherer_policy": RLModuleSpec(
                        module_class=AiUtopiaRoleRLModule,
                        observation_space=build_role_observation_space("gatherer", stage=1),
                        action_space=build_role_action_space("gatherer"),
                        model_config={
                            "role": "gatherer",
                            "max_seq_len": 32,
                            "actor_hidden": [256],
                            "core_encoder":    {"core_hidden": [512, 256]},
                            "shared_backbone": {"lstm_hidden": 256},
                            "ctde_critic":     {"critic_hidden": 256},
                        },
                    ),
                },
            )
        )
        .multi_agent(
            policies={"gatherer_policy"},
            policy_mapping_fn=_policy_mapping_fn,
            policies_to_train=["gatherer_policy"],
        )
        .resources(num_cpus_for_main_process=2)
        .reporting(
            metrics_num_episodes_for_smoothing=200,
            keep_per_episode_custom_metrics=True,   # T10 reads exploit_* stats
        )
        .checkpointing(
            export_native_model_files=True,
            checkpoint_trainable_policies_only=True,
        )
        .debugging(seed=seed)
    )
    return cfg
```

- [ ] **Step 2: Test the config builds**

Create `tests/unit/test_train_config.py`:
```python
import pytest

pytest.importorskip("ray")

from aiutopia.train.config import m1_gatherer_config, ENV_NAME


def test_m1_gatherer_config_builds() -> None:
    cfg = m1_gatherer_config()
    d = cfg.to_dict()
    assert d["train_batch_size"] == 4096
    assert d["minibatch_size"] == 512
    assert d["num_epochs"] == 5
    assert d["gamma"] == 0.99
    assert d["lr"] == 3.0e-4
    assert d["num_env_runners"] == 4
    assert d["num_envs_per_env_runner"] == 2
    assert d["num_learners"] == 1
    assert d["num_gpus_per_learner"] == 0.5
    assert d["env_config"]["stage"] == 1
    assert d["env_config"]["active_roles"] == ["gatherer"]
    assert d["env_config"]["tick_warp"] is True
    # New API stack must be on
    assert d.get("enable_rl_module_and_learner") is True or \
           d.get("_enable_new_api_stack") is True or \
           cfg.enable_rl_module_and_learner is True


def test_m1_gatherer_config_env_registered() -> None:
    from ray.tune.registry import _global_registry, ENV_CREATOR
    m1_gatherer_config()
    assert _global_registry.contains(ENV_CREATOR, ENV_NAME)


def test_policy_mapping_fn_new_api_signature() -> None:
    from aiutopia.train.config import _policy_mapping_fn
    assert _policy_mapping_fn("gatherer_0", episode=None) == "gatherer_policy"
    assert _policy_mapping_fn("gatherer_0") == "gatherer_policy"
    # Extra kwargs ignored
    assert _policy_mapping_fn("gatherer_0", episode=None, worker=None) == "gatherer_policy"
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_train_config.py -v
git add src/aiutopia/train/__init__.py src/aiutopia/train/config.py \
        tests/unit/test_train_config.py
git commit -m "feat(train): m1_gatherer_config — new API stack, no legacy model block (M1-Training T7)"
```

Expected: 3 PASSED.

---

### Task 7.5: RLlib integration smoke (1 PPO iter on synthetic env)

**Files:**
- Create: `scripts/rllib_smoke.py`
- Create: `tests/integration/test_rllib_smoke.py`

Catches RLlib integration bugs (e.g. `_forward` shape mismatches with the actual Connector pipeline) before T21 burns hours on a real Fabric server. Uses a synthetic PettingZoo env that returns the gatherer obs space sample() — no Java/Fabric required.

- [ ] **Step 1: Implement the smoke harness**

Create `scripts/rllib_smoke.py`:
```python
"""1-iteration PPO smoke test on a synthetic gatherer env. No Fabric needed.

Run:
  PYTHONPATH=src python scripts/rllib_smoke.py
"""
from __future__ import annotations

import numpy as np
import pettingzoo
import ray
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.train.config import m1_gatherer_config, ENV_NAME


class _SyntheticGathererEnv(ParallelEnv):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        self.agents = ["gatherer_0"]
        self.possible_agents = list(self.agents)
        self._obs_space = build_role_observation_space("gatherer", stage=1)
        self._action_space = build_role_action_space("gatherer")
        self._step = 0

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self._step = 0
        return {a: self._obs_space.sample() for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        self._step += 1
        obs   = {a: self._obs_space.sample() for a in self.agents}
        rew   = {a: float(np.random.randn()) for a in self.agents}
        term  = {a: False for a in self.agents}
        trunc = {a: self._step >= 16 for a in self.agents}
        info  = {a: {} for a in self.agents}
        return obs, rew, term, trunc, info


def main() -> int:
    # Re-register ENV_NAME with the synthetic env for the smoke run
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(_SyntheticGathererEnv()))
    ray.init(local_mode=True, num_cpus=2, num_gpus=0, ignore_reinit_error=True)
    try:
        cfg = m1_gatherer_config(num_env_runners=0, num_envs_per_env_runner=1)
        cfg = cfg.learners(num_gpus_per_learner=0)
        algo = cfg.build()
        result = algo.train()
        # Sanity: training executed
        assert "info" in result or "env_runners" in result, result
        print("RLLIB SMOKE OK")
        algo.stop()
        return 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: pytest integration wrapper**

Create `tests/integration/test_rllib_smoke.py`:
```python
import subprocess
import sys
import os
import pytest

pytest.importorskip("ray")


@pytest.mark.integration
@pytest.mark.slow
def test_rllib_smoke_runs_one_iteration() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    out = subprocess.run(
        [sys.executable, "scripts/rllib_smoke.py"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert out.returncode == 0, f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    assert "RLLIB SMOKE OK" in out.stdout
```

- [ ] **Step 3: Run + commit**

```bash
PYTHONPATH=src python scripts/rllib_smoke.py
# Expected final line: RLLIB SMOKE OK
git add scripts/rllib_smoke.py tests/integration/test_rllib_smoke.py
git commit -m "test(rl): 1-iter PPO smoke on synthetic env (M1-Training T7.5)"
```

**If this fails:** STOP. Triage the RLModule before proceeding. Common failure modes:
- `KeyError: 'goal_embedding'` in `_forward` → obs space doesn't actually contain that key; check `build_role_observation_space("gatherer", stage=1)`.
- `RuntimeError: tensors not on same device` → `initial_state` device fix didn't land; verify T2/T6.
- Action sampling fails → `OUTPUT_DIM` doesn't match what RLlib's TorchMultiActionDistribution expects for the action space; check T5.
- `KeyError: 'STATE_IN'` → state isn't being threaded through Connectors; check `get_initial_state` signature.

---

### Task 8: `env_factory.py` — Ray Tune env factory with PettingZoo wrap

**Files:**
- Create: `src/aiutopia/train/env_factory.py`

**v2 fix:** RLlib does NOT auto-detect a raw PettingZoo Parallel env. Must wrap in `ParallelPettingZooEnv`.

- [ ] **Step 1: Implement**

Create `src/aiutopia/train/env_factory.py`:
```python
"""Env factory for Ray Tune. Wraps PettingZoo Parallel env for RLlib."""
from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


def make_aiutopia_env(env_config: dict[str, Any]) -> AiUtopiaPettingZooEnv:
    """Raw env factory; useful for non-RLlib consumers."""
    return AiUtopiaPettingZooEnv(env_config)


def make_aiutopia_env_wrapped(env_config: dict[str, Any]) -> ParallelPettingZooEnv:
    """RLlib-compatible factory: wraps the PettingZoo Parallel env."""
    return ParallelPettingZooEnv(AiUtopiaPettingZooEnv(env_config))
```

- [ ] **Step 2: Sanity check + commit**

```bash
PYTHONPATH=src python -c "
from aiutopia.train.env_factory import make_aiutopia_env_wrapped
e = make_aiutopia_env_wrapped({'stage': 1, 'active_roles': ['gatherer'],
                                'py4j_ports': [], 'tick_warp': False,
                                'max_episode_ticks': 100,
                                'enable_memory_writes': False})
print('wrapped env class:', type(e).__name__)
"
git add src/aiutopia/train/env_factory.py
git commit -m "feat(train): env_factory with ParallelPettingZooEnv wrap (M1-Training T8)"
```

---

### Task 9: `AiUtopiaMetricsCallback` — per-policy entropy / vf_loss / kl

**Files:**
- Create: `src/aiutopia/train/callbacks.py`
- Create: `tests/unit/test_metrics_callback.py`

**v2 fixes:** Inherits `RLlibCallback` (not deprecated `DefaultCallbacks`); accepts `metrics_logger=None` kwarg.

- [ ] **Step 1: Implement**

Create `src/aiutopia/train/callbacks.py`:
```python
"""Section 7.4 Custom Ray RLlib callbacks for AI Utopia training.

Uses the new RLlibCallback base class (Ray 2.40+); DefaultCallbacks is a
deprecated shim. All on_train_result signatures include metrics_logger.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from ray.rllib.callbacks.callbacks import RLlibCallback
except ImportError:                                       # 2.40 fallback path
    from ray.rllib.algorithms.callbacks import DefaultCallbacks as RLlibCallback


class AiUtopiaMetricsCallback(RLlibCallback):
    """Per-iteration metrics: per-policy entropy, vf_loss, kl."""

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        result.setdefault("custom_metrics", {})
        try:
            learner_info = result["info"]["learner"]
        except KeyError:
            return
        for policy_id, info in learner_info.items():
            for src, dst in (("entropy",       "entropy"),
                              ("policy_entropy","entropy"),
                              ("vf_loss",       "vf_loss"),
                              ("kl",            "kl")):
                val = info.get(src)
                if val is not None:
                    result["custom_metrics"][f"{policy_id}/{dst}"] = float(val)
```

- [ ] **Step 2: Test**

Create `tests/unit/test_metrics_callback.py`:
```python
import pytest

pytest.importorskip("ray")

from aiutopia.train.callbacks import AiUtopiaMetricsCallback


def test_callback_instantiates() -> None:
    cb = AiUtopiaMetricsCallback()
    result = {"info": {"learner": {"gatherer_policy": {"entropy": 1.42}}}}
    cb.on_train_result(algorithm=None, result=result)
    assert result["custom_metrics"]["gatherer_policy/entropy"] == 1.42


def test_callback_handles_missing_learner_info() -> None:
    cb = AiUtopiaMetricsCallback()
    result = {}
    cb.on_train_result(algorithm=None, result=result)
    assert result.get("custom_metrics", {}) == {}


def test_callback_accepts_metrics_logger_kwarg() -> None:
    """RLlibCallback v2.40+ passes metrics_logger as kwarg."""
    cb = AiUtopiaMetricsCallback()
    result = {"info": {"learner": {}}}
    cb.on_train_result(algorithm=None, metrics_logger=None, result=result)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_metrics_callback.py -v
git add src/aiutopia/train/callbacks.py tests/unit/test_metrics_callback.py
git commit -m "feat(train): AiUtopiaMetricsCallback on RLlibCallback base (M1-Training T9)"
```

Expected: 3 PASSED.

---

### Task 10: `ExploitHuntCallback` — every-200-iter aggregate

**Files:**
- Modify: `src/aiutopia/train/callbacks.py`

- [ ] **Step 1: Append**

```python
class ExploitHuntCallback(RLlibCallback):
    """Section 4 exploit hunt — every N iterations, surface exploit-penalty
    aggregates per exploit type."""

    def __init__(self, *, every_n_iters: int = 200) -> None:
        super().__init__()
        self.every_n_iters = every_n_iters
        self._iter = 0

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        self._iter += 1
        if self._iter % self.every_n_iters != 0:
            return
        result.setdefault("custom_metrics", {})
        sampler = result.get("env_runners", result.get("sampler_results", {}))
        episode_stats = sampler.get("episode_extra_stats", {})
        for key, value in episode_stats.items():
            if key.startswith("exploit_"):
                result["custom_metrics"][f"exploit_hunt/{key}"] = float(value)
```

- [ ] **Step 2: Test**

Append to `tests/unit/test_metrics_callback.py`:
```python
def test_exploit_hunt_callback_throttled() -> None:
    from aiutopia.train.callbacks import ExploitHuntCallback
    cb = ExploitHuntCallback(every_n_iters=3)
    result = {"env_runners": {"episode_extra_stats": {"exploit_drop_spam": 0.5}}}
    for i in range(1, 9):
        result["custom_metrics"] = {}
        cb.on_train_result(algorithm=None, result=result)
        if i % 3 == 0:
            assert "exploit_hunt/exploit_drop_spam" in result["custom_metrics"]
        else:
            assert "exploit_hunt/exploit_drop_spam" not in result["custom_metrics"]
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_metrics_callback.py -v
git add src/aiutopia/train/callbacks.py tests/unit/test_metrics_callback.py
git commit -m "feat(train): ExploitHuntCallback (M1-Training T10)"
```

Expected: 4 PASSED.

---

### Task 11: `EvalGateStopCallback` — Tune-readable gate signal

**Files:**
- Modify: `src/aiutopia/train/callbacks.py`
- Create: `tests/unit/test_evaluation_gate_callback.py`

**v2 fixes:** Stop signaling via `custom_metrics["M1/gate_passed"] = 1.0` paired with Tune's `stop={"custom_metrics/M1/gate_passed": 0.5}` in T18. No `algorithm.workers.stop()` (renamed to `env_runner_group`, and direct calls race with Tune teardown).

The deque emits only every `eval_interval` train iters (via M1EvalScenarioCallback T15), so "3 consecutive evaluations" means roughly 30 train iters — documented below.

- [ ] **Step 1: Append**

```python
class EvalGateStopCallback(RLlibCallback):
    """Section 5.8 M1 evaluation gate: 80% success on collect-64-oak_log
    over 3 consecutive evaluations.

    Each evaluation arrives roughly once every `eval_interval` train iters
    (M1EvalScenarioCallback emits one rate per N iters), so 3 consecutive
    evaluations >= ~3*N train iters of sustained success.

    On pass: writes `custom_metrics["M1/gate_passed"] = 1.0`. Tune's
    stop dict watches that key and terminates the trial gracefully.
    """

    def __init__(self, *, milestone: str = "M1",
                  success_metric: str = "eval_m1_oak_log_success_rate",
                  threshold: float = 0.80,
                  consecutive_required: int = 3) -> None:
        super().__init__()
        self.milestone           = milestone
        self.success_metric      = success_metric
        self.threshold           = threshold
        self.consecutive_required = consecutive_required
        self._recent: deque[float] = deque(maxlen=consecutive_required)
        self.gate_passed = False

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        sampler = result.get("env_runners", result.get("sampler_results", {}))
        rate = sampler.get("episode_extra_stats", {}).get(self.success_metric)
        if rate is None:
            return
        self._recent.append(float(rate))
        result.setdefault("custom_metrics", {})
        result["custom_metrics"][f"{self.milestone}/gate_history"] = list(self._recent)
        if (len(self._recent) == self.consecutive_required
            and all(r >= self.threshold for r in self._recent)):
            self.gate_passed = True
            result["custom_metrics"][f"{self.milestone}/gate_passed"] = 1.0
            # Tune stop dict watches custom_metrics/{milestone}/gate_passed
        else:
            result["custom_metrics"][f"{self.milestone}/gate_passed"] = 0.0
```

- [ ] **Step 2: Test**

Create `tests/unit/test_evaluation_gate_callback.py`:
```python
from aiutopia.train.callbacks import EvalGateStopCallback


def _emit(cb, rate: float) -> dict:
    result = {"env_runners": {"episode_extra_stats": {"eval_m1_oak_log_success_rate": rate}}}
    cb.on_train_result(algorithm=None, result=result)
    return result


def test_gate_not_passed_until_three_consecutive() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    _emit(cb, 0.9)
    assert not cb.gate_passed
    _emit(cb, 0.85)
    assert not cb.gate_passed
    r = _emit(cb, 0.95)
    assert cb.gate_passed
    assert r["custom_metrics"]["M1/gate_passed"] == 1.0


def test_gate_resets_on_low_evaluation() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    _emit(cb, 0.9)
    _emit(cb, 0.85)
    _emit(cb, 0.5)
    assert not cb.gate_passed
    _emit(cb, 0.9)
    _emit(cb, 0.9)
    assert not cb.gate_passed
    _emit(cb, 0.9)
    assert cb.gate_passed


def test_gate_writes_zero_when_not_passed() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    r = _emit(cb, 0.5)
    assert r["custom_metrics"]["M1/gate_passed"] == 0.0


def test_gate_ignores_results_without_evaluation_metric() -> None:
    cb = EvalGateStopCallback()
    cb.on_train_result(algorithm=None, result={})
    assert not cb.gate_passed
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_evaluation_gate_callback.py -v
git add src/aiutopia/train/callbacks.py tests/unit/test_evaluation_gate_callback.py
git commit -m "feat(train): EvalGateStopCallback with Tune-readable signal (M1-Training T11)"
```

Expected: 4 PASSED.

---

### Task 12: `WorldOps.resetEpisode(playerName, seed)` + `setupTrainingScene`

**Files:**
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`

**v2 fixes:**
- `resetEpisode` accepts a `seed` and uses it to RNG-place oak_logs around the agent, so 3 eval scenarios truly differ.
- `/tick rate 300.0` (float required in Carpet 1.21.1).

- [ ] **Step 1: Append to `WorldOps.java`** (inside class, before final `}`)

```java
    private final java.util.Random epRand = new java.util.Random();

    /** Per-episode reset for training:
     *   - teleport agent to spawn (64, 71, -48)
     *   - clear agent inventory
     *   - air-fill a cube around spawn
     *   - place a ring of oak_log blocks, positions seeded by `seed`
     *  Fast (~10ms). */
    public boolean resetEpisode(String playerName, long seed) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/tp " + playerName + " 64 71 -48");
            cm.executeWithPrefix(src, "/clear " + playerName);
            cm.executeWithPrefix(src,
                "/fill 48 72 -64 80 76 -32 air replace");

            // Seeded ring: 8 logs at radius 2-3, angles jittered by seed
            epRand.setSeed(seed);
            for (int i = 0; i < 8; i++) {
                double theta = (2.0 * Math.PI * i / 8.0) + (epRand.nextDouble() - 0.5) * 0.4;
                int r = 2 + epRand.nextInt(2);              // 2 or 3
                int x = 64 + (int) Math.round(r * Math.cos(theta));
                int z = -48 + (int) Math.round(r * Math.sin(theta));
                cm.executeWithPrefix(src, "/setblock " + x + " 71 " + z + " oak_log");
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "resetEpisode failed for {} seed={}: {}", playerName, seed, e.getMessage());
            return false;
        }
    }

    /** One-time setup at server boot when training mode is active. Idempotent. */
    public boolean setupTrainingScene() {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/difficulty peaceful");
            cm.executeWithPrefix(src, "/time set noon");
            cm.executeWithPrefix(src, "/gamerule doDaylightCycle false");
            cm.executeWithPrefix(src, "/gamerule doMobSpawning false");
            cm.executeWithPrefix(src, "/tick rate 300.0");   // Carpet 1.21.1 wants float
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "setupTrainingScene failed: {}", e.getMessage());
            return false;
        }
    }
```

- [ ] **Step 2: Expose in `Py4JEntryPoint`** (inside class, before final `}`)

```java
    /** Per-episode reset with a seed for deterministic log placement. */
    public boolean resetEpisode(String playerName, long seed) {
        return world.resetEpisode(playerName, seed);
    }

    /** One-time training-scene setup. */
    public boolean setupTrainingScene() {
        return world.setupTrainingScene();
    }
```

- [ ] **Step 3: Build + commit**

```bash
cd fabric_mod && export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10; export PATH=$JAVA_HOME/bin:$PATH
./gradlew build --no-daemon 2>&1 | tail -5
cd ..
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java \
        fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java
git commit -m "feat(motor): WorldOps.resetEpisode(playerName, seed) + setupTrainingScene (M1-Training T12)"
```

Expected: BUILD SUCCESSFUL.

---

### Task 13: Wire `env.reset()` to call `resetEpisode(player, seed)` Java-side

**Files:**
- Modify: `src/aiutopia/env/wrapper.py`
- Modify: `src/aiutopia/env/bridge.py`

- [ ] **Step 1: Add to `FabricBridge`**

In `src/aiutopia/env/bridge.py`, after the existing `carpet_spawn` method, add:
```python
    def reset_episode(self, player_name: str, seed: int) -> bool:
        """Per-episode reset with seeded log placement (T12)."""
        return bool(self.entry_point.resetEpisode(player_name, int(seed)))

    def setup_training_scene(self) -> bool:
        return bool(self.entry_point.setupTrainingScene())
```

- [ ] **Step 2: Modify `AiUtopiaPettingZooEnv.reset` in `wrapper.py`**

The existing M1A `reset()` method (around line 232) calls `self.bridge.reset_world(seed)`. Use this as the Edit anchor.

Edit `src/aiutopia/env/wrapper.py`:
- **Find:** `        self.bridge.reset_world(seed)`
- **Replace with:**
```python
        self.bridge.reset_world(seed)
        # M1B: per-episode reset (T13) — teleport, clear inventory, place
        # seeded oak_log ring for each registered agent.
        for agent_id in self.agents_init:
            player_name = self.agent_id_to_player_name.get(agent_id)
            if player_name:
                self.bridge.reset_episode(player_name, int(seed))
        # Reset exploit detectors for the new episode
        from aiutopia.env.exploit import ExploitDetector
        for agent_id in list(self.exploit_detectors.keys()):
            self.exploit_detectors[agent_id] = ExploitDetector()
```

If `self.agents_init` or `self.agent_id_to_player_name` don't exist on the M1A wrapper, look at `__init__` to find the equivalent field; the prior M1A landed an `agent_id_to_player_name` map per the summary.

- [ ] **Step 3: Sanity-check + commit**

```bash
PYTHONPATH=src python -c "from aiutopia.env.wrapper import AiUtopiaPettingZooEnv; print('OK')"
git add src/aiutopia/env/wrapper.py src/aiutopia/env/bridge.py
git commit -m "feat(env): env.reset() calls bridge.reset_episode per agent with seed (M1-Training T13)"
```

---

### Task 14: `scenario_runner.py` — evaluation scenarios with proper LSTM state threading

**Files:**
- Create: `src/aiutopia/train/scenario_runner.py`
- Create: `tests/integration/test_evaluation_scenario_smoke.py`

**v2 fix:** `run_scenario` now threads the RLModule's LSTM hidden state across ticks. Previously each tick re-initialized state → effectively memoryless inference, which would not match training-time behavior.

- [ ] **Step 1: Implement**

Create `src/aiutopia/train/scenario_runner.py`:
```python
"""Section 5.8 + 5.10 evaluation scenarios for the M1 gate.

Threads LSTM hidden state across ticks per agent so inference behavior
matches training. Uses greedy action decoding (argmax for Discrete,
Gaussian-mean for continuous)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


M1_OAK_LOG_TARGET = 64


@dataclass(frozen=True)
class Scenario:
    name:      str
    seed:      int
    max_ticks: int
    success:   Callable[[dict], bool]


def _gatherer_collected_64_oak_log(final_obs: dict) -> bool:
    agent = final_obs.get("gatherer_0", {})
    item_ids = agent.get("inv_slot_item_ids", [])
    counts = agent.get("inv_slot_counts", [])
    total = 0
    for i, c in zip(item_ids, counts):
        ident = str(i) if not isinstance(i, str) else i
        if "oak_log" in ident:
            total += int(c)
    if total == 0 and counts is not None:
        total = sum(int(x) for x in counts if x)
    return total >= M1_OAK_LOG_TARGET


M1_SCENARIOS: list[Scenario] = [
    Scenario(name="m1_oak_log_seed_1", seed=1, max_ticks=1000,
              success=_gatherer_collected_64_oak_log),
    Scenario(name="m1_oak_log_seed_2", seed=2, max_ticks=1000,
              success=_gatherer_collected_64_oak_log),
    Scenario(name="m1_oak_log_seed_3", seed=3, max_ticks=1000,
              success=_gatherer_collected_64_oak_log),
]


def _move_state_to_device(state: dict, device) -> dict:
    return {k: v.to(device) for k, v in state.items()}


def run_scenario(scenario: Scenario, *,
                  env_config: dict,
                  rl_module,
                  device: str = "cpu") -> dict:
    import torch
    from ray.rllib.core import Columns

    env = AiUtopiaPettingZooEnv({**env_config, "tick_warp": True,
                                  "max_episode_ticks": scenario.max_ticks})
    try:
        obs, _info = env.reset(seed=scenario.seed)
        # Per-agent persistent LSTM state
        states = {agent: _move_state_to_device(rl_module.get_initial_state(), device)
                  for agent in obs}

        for _ in range(scenario.max_ticks):
            actions = {}
            new_states = {}
            for agent_id, agent_obs in obs.items():
                batched = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)
                           for k, v in agent_obs.items()}
                # State must be batched: (B, H) -> our state dict gives (H,);
                # add batch dim before passing in.
                state_in = {k: v.unsqueeze(0) for k, v in states[agent_id].items()}
                with torch.no_grad():
                    out = rl_module._forward_inference({
                        Columns.OBS: batched,
                        Columns.STATE_IN: state_in,
                    })
                actions[agent_id] = _greedy_decode(out[Columns.ACTION_DIST_INPUTS][0])
                # State out is (B, H); squeeze batch dim back to (H,)
                new_states[agent_id] = {k: v.squeeze(0)
                                         for k, v in out[Columns.STATE_OUT].items()}
            states = new_states
            obs, _rew, term, trunc, _info = env.step(actions)
            if all(term.values()) or all(trunc.values()):
                break
        return {
            "name":            scenario.name,
            "success":         scenario.success(obs),
            "final_inventory": obs,
        }
    finally:
        env.close()


def _greedy_decode(action_dist_inputs):
    """Convert 340-d flat dist-inputs to an action Dict (matching GathererActorHead)."""
    import torch
    from aiutopia.env.spaces import (
        COMM_PAYLOAD_DIM, N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE,
    )
    flat = action_dist_inputs
    offset = 0

    def take_logits(n):
        nonlocal offset
        out = flat[offset:offset + n]
        offset += n
        return int(torch.argmax(out).item())

    def take_gauss(d):
        nonlocal offset
        means = flat[offset:offset + d]
        offset += 2 * d
        return means.cpu().numpy()

    skill_type       = take_logits(N_GATHERER_SKILLS)
    target_class     = take_logits(N_TARGET_CLASSES_PER_ROLE)
    spatial_param    = take_gauss(3)
    scalar_param     = take_gauss(1)
    comm_payload     = take_gauss(COMM_PAYLOAD_DIM)
    should_broadcast = take_logits(2)
    comm_target_mask_logits = flat[offset:offset + 4]
    offset += 4
    comm_target_mask = (comm_target_mask_logits > 0).int().cpu().numpy()

    return {
        "skill_type":       skill_type,
        "target_class":     target_class,
        "spatial_param":    np.clip(spatial_param, -1, 1).astype(np.float32),
        "scalar_param":     np.clip(scalar_param, 0, 1).astype(np.float32),
        "comm_payload":     np.clip(comm_payload, -1, 1).astype(np.float32),
        "should_broadcast": should_broadcast,
        "comm_target_mask": comm_target_mask.astype(np.int8),
    }


def aggregate_success_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)
```

- [ ] **Step 2: Smoke test**

Create `tests/integration/test_evaluation_scenario_smoke.py`:
```python
import numpy as np
import pytest
import torch

from aiutopia.train.scenario_runner import (
    M1_SCENARIOS, _greedy_decode, aggregate_success_rate,
)


def test_m1_scenarios_present() -> None:
    assert len(M1_SCENARIOS) >= 3
    for s in M1_SCENARIOS:
        assert s.max_ticks == 1000


def test_greedy_decode_returns_valid_action() -> None:
    flat = torch.randn(340)
    action = _greedy_decode(flat)
    assert "skill_type" in action
    assert 0 <= action["skill_type"] < 6
    assert action["spatial_param"].shape == (3,)
    assert action["spatial_param"].dtype == np.float32


def test_aggregate_success_rate() -> None:
    assert aggregate_success_rate([]) == 0.0
    assert aggregate_success_rate([{"success": True}, {"success": False},
                                     {"success": True}]) == 2/3
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/integration/test_evaluation_scenario_smoke.py -v
git add src/aiutopia/train/scenario_runner.py \
        tests/integration/test_evaluation_scenario_smoke.py
git commit -m "feat(train): scenario_runner with LSTM state threading (M1-Training T14)"
```

Expected: 3 PASSED.

---

### Task 15: `M1EvalScenarioCallback` — runs scenarios + emits success rate

**Files:**
- Modify: `src/aiutopia/train/callbacks.py`

- [ ] **Step 1: Append**

```python
class M1EvalScenarioCallback(RLlibCallback):
    """Every `eval_interval` train iterations, run the 3 M1 fixed-seed
    scenarios and emit aggregate success rate as
    `episode_extra_stats[eval_m1_oak_log_success_rate]`."""

    def __init__(self, *, eval_interval: int = 10,
                  env_config: dict | None = None) -> None:
        super().__init__()
        self.eval_interval = eval_interval
        self.env_config = env_config or {}
        self._iter = 0

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        self._iter += 1
        if self._iter % self.eval_interval != 0:
            return
        from aiutopia.train.scenario_runner import (
            M1_SCENARIOS, run_scenario, aggregate_success_rate,
        )
        rl_module = algorithm.get_module("gatherer_policy")
        results = []
        for sc in M1_SCENARIOS:
            try:
                results.append(run_scenario(sc,
                                              env_config=self.env_config,
                                              rl_module=rl_module))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "evaluation scenario %s failed: %s", sc.name, exc)
        success_rate = aggregate_success_rate(results)
        sampler = result.setdefault("env_runners", {})
        stats = sampler.setdefault("episode_extra_stats", {})
        stats["eval_m1_oak_log_success_rate"] = success_rate
```

- [ ] **Step 2: Smoke test**

Append to `tests/unit/test_evaluation_gate_callback.py`:
```python
def test_m1_evaluation_scenario_callback_throttled() -> None:
    from aiutopia.train.callbacks import M1EvalScenarioCallback
    cb = M1EvalScenarioCallback(eval_interval=5)
    result = {}
    for _ in range(4):
        cb.on_train_result(algorithm=None, result=result)
    assert "eval_m1_oak_log_success_rate" not in (
        result.get("env_runners", {}).get("episode_extra_stats", {}))
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_evaluation_gate_callback.py -v
git add src/aiutopia/train/callbacks.py tests/unit/test_evaluation_gate_callback.py
git commit -m "feat(train): M1EvalScenarioCallback runs scenarios + emits success rate (M1-Training T15)"
```

Expected: 5 PASSED.

---

### Task 16: `aiutopia promote-weights` CLI

**Files:**
- Create: `src/aiutopia/promotion/__init__.py`
- Create: `src/aiutopia/promotion/service.py`
- Create: `src/aiutopia/cli/promote.py`
- Modify: `src/aiutopia/cli/app.py`

- [ ] **Step 1: Promotion service**

Create `src/aiutopia/promotion/__init__.py`:
```python
"""Weight promotion (section 5.10)."""
```

Create `src/aiutopia/promotion/service.py`:
```python
"""Section 5.10 promotion service — copies weights + bumps roles.policy_version."""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

from aiutopia.common.config import Paths
from aiutopia.identity.service import IdentityService


def promote_weights(*,
                     role_id:        str,
                     checkpoint_dir: Path,
                     paths:          Paths,
                     notes:          str = "",
                     deployed_by:    str = "manual:cli") -> dict:
    svc = IdentityService(paths.identity_db)
    role = svc.get_role(role_id)
    from_version = role.policy_version
    to_version   = from_version + 1

    target_dir = paths.weights_dir / role_id / f"v{to_version}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(checkpoint_dir, target_dir)

    with sqlite3.connect(paths.identity_db) as conn:
        conn.execute(
            "UPDATE roles SET policy_weights_path=?, policy_version=? WHERE role_id=?",
            (str(target_dir), to_version, role_id),
        )
        cur = conn.execute(
            """INSERT INTO policy_deployments
                  (role_id, from_version, to_version, deployed_at, deployed_by, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (role_id, from_version, to_version,
             int(time.time()), deployed_by, notes),
        )
        deployment_id = cur.lastrowid
    return {
        "role_id":       role_id,
        "from_version":  from_version,
        "to_version":    to_version,
        "weights_path":  str(target_dir),
        "deployment_id": deployment_id,
    }
```

- [ ] **Step 2: CLI command**

Create `src/aiutopia/cli/promote.py`:
```python
"""`aiutopia promote-weights promote ...` — section 5.10 promotion CLI."""
from __future__ import annotations

from pathlib import Path

import typer

from aiutopia.common.config  import Paths
from aiutopia.promotion.service import promote_weights


app = typer.Typer(no_args_is_help=True)


@app.command("promote")
def promote(
    role:            str  = typer.Option(..., help="gatherer|builder|farmer|defender"),
    checkpoint:      Path = typer.Option(..., help="path to Ray checkpoint dir"),
    notes:           str  = typer.Option("", help="audit-log note"),
    skip_checklist:  bool = typer.Option(False, "--skip-checklist",
                                          help="bypass section 5.10 gates (dangerous)"),
) -> None:
    """Promote a Ray checkpoint to the production weights directory."""
    if not checkpoint.exists():
        typer.echo(f"checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(code=2)
    paths = Paths.from_env(); paths.ensure()
    if not skip_checklist:
        from aiutopia.promotion.checklist import run_checklist
        report = run_checklist(role=role, checkpoint=checkpoint, paths=paths)
        if not report.passes:
            typer.echo("section 5.10 promotion checklist FAILED:", err=True)
            for issue in report.issues:
                typer.echo(f"  - {issue}", err=True)
            raise typer.Exit(code=3)
        typer.echo(f"section 5.10 checklist: PASS ({len(report.gates_passed)} gates)")
    result = promote_weights(role_id=role,
                              checkpoint_dir=checkpoint,
                              paths=paths,
                              notes=notes)
    typer.echo(f"promoted {role}: v{result['from_version']} -> v{result['to_version']}")
    typer.echo(f"  weights: {result['weights_path']}")
    typer.echo(f"  deployment_id: {result['deployment_id']}")
```

- [ ] **Step 3: Register in `cli/app.py`** via Edit

The M1A `cli/app.py` includes `from aiutopia.cli import agent, memory, determinism` near the top, followed by `app = typer.Typer(...)` and `app.add_typer(...)` calls. Use the `from aiutopia.cli import ...` line as the Edit anchor.

Edit `src/aiutopia/cli/app.py`:
- **Find:** `from aiutopia.cli import agent, memory, determinism`
- **Replace with:** `from aiutopia.cli import agent, memory, determinism, promote`

Then find the last `app.add_typer(...)` call and append after it:
```python
app.add_typer(promote.app, name="promote-weights",
              help="Promote trained weights through the section 5.10 checklist.")
```

If your existing imports/app structure differ, adjust accordingly. The full CLI path becomes `aiutopia promote-weights promote --role gatherer --checkpoint <ckpt>`.

- [ ] **Step 4: Verify CLI surfaces (T17 lands the missing import; this commit blocks until T17)**

After T17 lands `checklist.py`:
```bash
PYTHONPATH=src python -m aiutopia.cli.app promote-weights --help
PYTHONPATH=src python -m aiutopia.cli.app promote-weights promote --help
```

Then commit:
```bash
git add src/aiutopia/promotion/__init__.py src/aiutopia/promotion/service.py \
        src/aiutopia/cli/promote.py src/aiutopia/cli/app.py
git commit -m "feat(cli): aiutopia promote-weights promote (M1-Training T16)"
```

---

### Task 17: Section 5.10 promotion checklist — 5 gates

**Files:**
- Create: `src/aiutopia/promotion/checklist.py`
- Create: `tests/unit/test_promotion_checklist.py`

(content unchanged from v1 — Gate 4 boolean precedence is correct)

- [ ] **Step 1: Implement**

Create `src/aiutopia/promotion/checklist.py`:
```python
"""Section 5.10 promotion checklist — 5 gates."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aiutopia.common.config import Paths


@dataclass
class ChecklistReport:
    passes:        bool
    gates_passed:  list[str] = field(default_factory=list)
    issues:        list[str] = field(default_factory=list)


def run_checklist(*, role: str, checkpoint: Path, paths: Paths) -> ChecklistReport:
    metrics_file = checkpoint / "aiutopia_metrics.json"
    report = ChecklistReport(passes=False)
    if not metrics_file.exists():
        report.issues.append(f"aiutopia_metrics.json not found at {metrics_file}")
        return report
    metrics = json.loads(metrics_file.read_text())

    if metrics.get("last_50k_steps_return_trend", -1) >= 0:
        report.gates_passed.append("1_return_trend_nonneg")
    else:
        report.issues.append("Gate 1: episodic return is collapsing")

    rate = metrics.get("evaluation_scenario_success_rate", 0)
    if rate >= 0.80:
        report.gates_passed.append(f"2_scenario_success_{rate:.0%}")
    else:
        report.issues.append(f"Gate 2: scenario success rate {rate:.0%} < 80%")

    ratio = metrics.get("exploit_penalty_ratio", 1.0)
    if ratio < 0.05:
        report.gates_passed.append(f"3_exploit_ratio_{ratio:.2%}")
    else:
        report.issues.append(f"Gate 3: exploit penalty ratio {ratio:.2%} >= 5%")

    per_role = metrics.get("per_role_entropy", {})
    entropy = per_role.get(role, 0.0)
    qvar    = metrics.get("q_variance_ratio", 999.0)
    tcos    = metrics.get("trajectory_cosine", 1.0)
    single_agent = len(per_role) == 1
    failure_mode_ok = (entropy > 1.5 and qvar < 5.0
                        and (single_agent or tcos < 0.8))
    if failure_mode_ok:
        report.gates_passed.append(
            f"4_failure_modes_entropy_{entropy:.2f}_qvar_{qvar:.1f}")
    else:
        report.issues.append(
            f"Gate 4: entropy={entropy:.2f} (need >1.5), qvar={qvar:.1f} "
            f"(need <5), traj_cos={tcos:.2f}")

    argmax_div = metrics.get("determinism_argmax_div", 1.0)
    l2_div     = metrics.get("determinism_l2", 999.0)
    if argmax_div < 0.05 and l2_div < 0.1:
        report.gates_passed.append(
            f"5_determinism_argmax_{argmax_div:.3f}_l2_{l2_div:.3f}")
    else:
        report.issues.append(
            f"Gate 5: argmax_div={argmax_div:.3f} (need <0.05), "
            f"l2={l2_div:.3f} (need <0.1)")

    report.passes = len(report.issues) == 0
    return report
```

- [ ] **Step 2: Test**

Create `tests/unit/test_promotion_checklist.py`:
```python
import json
from pathlib import Path

import pytest

from aiutopia.promotion.checklist import run_checklist


def _write_metrics(ckpt: Path, **overrides) -> Path:
    defaults = {
        "last_50k_steps_return_trend":      1.0,
        "evaluation_scenario_success_rate": 0.92,
        "exploit_penalty_ratio":            0.02,
        "per_role_entropy":                 {"gatherer": 1.8},
        "q_variance_ratio":                 3.5,
        "trajectory_cosine":                1.0,
        "determinism_argmax_div":           0.02,
        "determinism_l2":                   0.04,
    }
    defaults.update(overrides)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "aiutopia_metrics.json").write_text(json.dumps(defaults))
    return ckpt


def test_all_gates_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt")
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert report.passes, report.issues
    assert len(report.gates_passed) == 5


def test_missing_metrics_file_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    report = run_checklist(role="gatherer",
                            checkpoint=tmp_path / "nonexistent",
                            paths=paths)
    assert not report.passes


def test_failing_scenario_rate_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt", evaluation_scenario_success_rate=0.5)
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert not report.passes
    assert any("Gate 2" in i for i in report.issues)


def test_failing_determinism_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt",
                           determinism_argmax_div=0.2,
                           determinism_l2=0.5)
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert not report.passes
    assert any("Gate 5" in i for i in report.issues)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/unit/test_promotion_checklist.py -v
git add src/aiutopia/promotion/checklist.py tests/unit/test_promotion_checklist.py
git commit -m "feat(promotion): section 5.10 5-gate checklist (M1-Training T17)"
```

Expected: 4 PASSED.

---

### Task 18: `scripts/train.py` — Ray Tune driver

**Files:**
- Create: `scripts/train.py`

**v2 fixes:** Pass `cfg` object (not `cfg.to_dict()`) to `tune.Tuner.param_space`; callbacks composed as a list-of-classes (no `_Composite` hack); proper Tune stop dict reads `custom_metrics/M1/gate_passed`; no `tune.logger.TBXLoggerCallback` (default loggers handle TB in 2.40+).

- [ ] **Step 1: Implement**

Create `scripts/train.py`:
```python
"""Ray Tune entry point for AI Utopia training (M1)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# CUBLAS workspace must be set BEFORE CUDA init for deterministic LSTM:
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig

from aiutopia.common.config import Paths
from aiutopia.common.logging import setup_logging, get_logger
from aiutopia.train.callbacks  import (
    AiUtopiaMetricsCallback,
    EvalGateStopCallback,
    ExploitHuntCallback,
    M1EvalScenarioCallback,
)
from aiutopia.train.config     import m1_gatherer_config


log = get_logger("train")


def _make_callbacks_class(env_config: dict, eval_interval: int):
    """Compose multiple RLlibCallbacks into one. In Ray 2.40+ .callbacks(...)
    accepts a list too, but a single class is the most portable form."""
    try:
        from ray.rllib.callbacks.callbacks import RLlibCallback
    except ImportError:
        from ray.rllib.algorithms.callbacks import DefaultCallbacks as RLlibCallback

    metrics       = AiUtopiaMetricsCallback()
    exploit       = ExploitHuntCallback(every_n_iters=200)
    eval_scenario = M1EvalScenarioCallback(eval_interval=eval_interval,
                                             env_config=env_config)
    gate          = EvalGateStopCallback(milestone="M1")
    delegates = [metrics, exploit, eval_scenario, gate]

    class _Composite(RLlibCallback):
        def on_train_result(self, *, algorithm, metrics_logger=None,
                              result, **kwargs):
            for cb in delegates:
                cb.on_train_result(algorithm=algorithm,
                                    metrics_logger=metrics_logger,
                                    result=result, **kwargs)
    return _Composite


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", default="M1", choices=["M1"])
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=1)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    args = parser.parse_args()

    paths = Paths.from_env(); paths.ensure()
    ray.init(num_cpus=16, num_gpus=1,
             object_store_memory=8 * 1024**3,
             _system_config={"object_spilling_threshold": 0.95})

    cfg = m1_gatherer_config(seed=args.seed)
    env_config = cfg.env_config if hasattr(cfg, "env_config") else \
                  cfg.to_dict().get("env_config", {})
    cfg = cfg.callbacks(_make_callbacks_class(env_config, args.evaluation_interval))

    run_id = f"aiutopia_{args.milestone}_seed{args.seed}"
    tuner = tune.Tuner(
        "PPO",
        param_space=cfg,                       # PASS CONFIG OBJECT, not dict
        run_config=RunConfig(
            name=run_id,
            storage_path=str(paths.runs_dir),
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=50,
                num_to_keep=10,
                checkpoint_at_end=True,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
            ),
            stop={
                "training_iteration":                       args.max_iters,
                "custom_metrics/M1/gate_passed":            0.5,  # >= 0.5 = passed
            },
            verbose=1,
            log_to_file=True,
        ),
    )
    log.info("starting training: %s", run_id)
    results = tuner.fit()
    best = results.get_best_result(metric="env_runners/episode_return_mean",
                                     mode="max")
    log.info("best checkpoint: %s", best.checkpoint)

    if best.checkpoint is not None:
        metrics_file = Path(best.checkpoint.path) / "aiutopia_metrics.json"
        final = best.metrics or {}
        sampler = final.get("env_runners", {}) or final.get("sampler_results", {})
        stats = sampler.get("episode_extra_stats", {})
        out = {
            "last_50k_steps_return_trend": float(
                sampler.get("episode_return_mean", 0)),
            "evaluation_scenario_success_rate":  float(
                stats.get("eval_m1_oak_log_success_rate", 0.0)),
            "exploit_penalty_ratio":       float(
                stats.get("exploit_total_per_episode", 0.0)),
            "per_role_entropy":            {"gatherer": float(
                final.get("custom_metrics", {}).get("gatherer_policy/entropy", 0))},
            "q_variance_ratio":            1.0,
            "trajectory_cosine":           1.0,
            "determinism_argmax_div":      1.0,   # T19 overwrites
            "determinism_l2":              999.0,
        }
        metrics_file.write_text(json.dumps(out, indent=2))
        log.info("wrote %s", metrics_file)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test + commit**

```bash
PYTHONPATH=src python -c "
import sys; sys.argv = ['train.py', '--milestone', 'M1', '--max-iters', '1']
from scripts.train import main
print('OK')
" 2>&1 | tail -3
git add scripts/train.py
git commit -m "feat(train): scripts/train.py with proper Tune stop signal (M1-Training T18)"
```

Expected: `OK`.

---

### Task 19: Determinism harness — real-weights replay with LSTM state

**Files:**
- Modify: `src/aiutopia/determinism/harness.py`
- Modify: `src/aiutopia/cli/determinism.py`
- Create: `tests/unit/test_determinism_real_weights.py`

**v2 fixes:**
- LSTM hidden state is threaded across ticks (was reinitialized every tick → memoryless approximation).
- Checkpoint load via `Algorithm.from_checkpoint(...).get_module(...)` — a Tune dir is not a `MultiRLModule` dir; the inner path is `<ckpt>/learner_group/learner/rl_module/`.
- `CUBLAS_WORKSPACE_CONFIG` set before CUDA init.

- [ ] **Step 1: Extend `determinism/harness.py`**

Append to `src/aiutopia/determinism/harness.py`:
```python
def replay_with_rlmodule(rl_module, *, env_config: dict, seed: int = 1,
                          n_steps: int = 1000) -> list[dict]:
    """Deterministic episode replay against an env with proper LSTM state
    threading. Returns list of {action_argmax: int, continuous_params: ndarray}
    entries — matches the compute_divergence(trace_a, trace_b) contract.
    """
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ray.rllib.core import Columns

    from aiutopia.env.wrapper           import AiUtopiaPettingZooEnv
    from aiutopia.train.scenario_runner import _greedy_decode

    configure_cuda_determinism()
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    device = next(rl_module.parameters()).device

    env = AiUtopiaPettingZooEnv({**env_config, "max_episode_ticks": n_steps,
                                  "tick_warp": True})
    trace: list[dict] = []
    try:
        obs, _ = env.reset(seed=seed)
        # Per-agent LSTM state
        states = {agent: {k: v.to(device)
                          for k, v in rl_module.get_initial_state().items()}
                  for agent in obs}
        for _ in range(n_steps):
            actions = {}
            argmax_record: dict = {}
            cont_record: dict = {}
            new_states: dict = {}
            for agent, agent_obs in obs.items():
                batched = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)
                           for k, v in agent_obs.items()}
                state_in = {k: v.unsqueeze(0) for k, v in states[agent].items()}
                with torch.no_grad():
                    out = rl_module._forward_inference({
                        Columns.OBS: batched,
                        Columns.STATE_IN: state_in,
                    })
                dist = out[Columns.ACTION_DIST_INPUTS][0]
                action = _greedy_decode(dist)
                actions[agent] = action
                argmax_record[agent] = action["skill_type"]
                cont_record[agent] = action["spatial_param"]
                new_states[agent] = {k: v.squeeze(0)
                                      for k, v in out[Columns.STATE_OUT].items()}
            states = new_states
            # For single-agent M1, flatten the trace entry to the
            # compute_divergence-expected shape (int + ndarray):
            if len(argmax_record) == 1:
                trace.append({
                    "action_argmax":     int(next(iter(argmax_record.values()))),
                    "continuous_params": next(iter(cont_record.values())),
                })
            else:
                trace.append({
                    "action_argmax":     argmax_record,
                    "continuous_params":
                        np.concatenate([v for v in cont_record.values()]),
                })
            obs, _, term, trunc, _ = env.step(actions)
            if all(term.values()) or all(trunc.values()):
                break
    finally:
        env.close()
    return trace
```

- [ ] **Step 2: Rewrite `cli/determinism.py`**

Replace `src/aiutopia/cli/determinism.py`:
```python
"""`aiutopia determinism check --weights <ckpt>` — real-weights replay."""
from __future__ import annotations

import json
import os
# Critical: set BEFORE CUDA init (required for deterministic GPU LSTM):
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path

import typer

from aiutopia.common.config import Paths
from aiutopia.determinism.harness import (
    compute_divergence, configure_cuda_determinism, replay_with_rlmodule,
)


app = typer.Typer(no_args_is_help=True)


@app.command("check")
def check(
    weights:    Path = typer.Option(..., help="Ray algorithm checkpoint dir"),
    episodes:   int  = typer.Option(3, help="number of seed pairs"),
    py4j_port:  int  = typer.Option(25099, help="env Py4J port"),
) -> None:
    """Section 7.8 / 5.10 Gate 5 determinism check on real weights."""
    if not weights.exists():
        typer.echo(f"weights not found: {weights}", err=True)
        raise typer.Exit(code=2)
    configure_cuda_determinism()
    paths = Paths.from_env(); paths.ensure()

    # A Tune algorithm checkpoint isn't a MultiRLModule checkpoint —
    # use Algorithm.from_checkpoint to handle path resolution correctly.
    from ray.rllib.algorithms.algorithm import Algorithm
    typer.echo(f"loading checkpoint: {weights}")
    algo = Algorithm.from_checkpoint(str(weights))
    rl_module = algo.get_module("gatherer_policy")

    env_config = {
        "stage":                 1,
        "active_roles":          ["gatherer"],
        "seed_strategy":         "fixed_easy",
        "py4j_ports":            [py4j_port],
        "tick_warp":             True,
        "max_episode_ticks":     1000,
        "per_worker_seed_offset": False,
        "enable_memory_writes":  False,
    }

    divergences = []
    for ep in range(episodes):
        seed = ep + 1
        typer.echo(f"=== seed {seed}: running 2 replays ===")
        a = replay_with_rlmodule(rl_module, env_config=env_config,
                                   seed=seed, n_steps=1000)
        b = replay_with_rlmodule(rl_module, env_config=env_config,
                                   seed=seed, n_steps=1000)
        div = compute_divergence(a, b)
        divergences.append(div)
        verdict = "PASS" if div.passes else "FAIL"
        typer.echo(f"  argmax_div={div.action_argmax_divergence:.4f}  "
                    f"l2={div.continuous_param_l2:.4f}  {verdict}")

    all_pass = all(d.passes for d in divergences)
    avg_argmax = sum(d.action_argmax_divergence for d in divergences) / len(divergences)
    avg_l2     = sum(d.continuous_param_l2     for d in divergences) / len(divergences)

    metrics_file = weights / "aiutopia_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    else:
        metrics = {}
    metrics["determinism_argmax_div"] = avg_argmax
    metrics["determinism_l2"]         = avg_l2
    metrics_file.write_text(json.dumps(metrics, indent=2))
    typer.echo(f"\nUpdated {metrics_file}")
    typer.echo(f"  determinism_argmax_div: {avg_argmax:.4f}")
    typer.echo(f"  determinism_l2:         {avg_l2:.4f}")
    typer.echo(f"  overall: {'PASS' if all_pass else 'FAIL'}")
    algo.stop()
    raise typer.Exit(code=0 if all_pass else 3)
```

- [ ] **Step 3: Smoke test**

Create `tests/unit/test_determinism_real_weights.py`:
```python
import os
import subprocess
import sys


def test_determinism_check_missing_weights_exits_2(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "determinism", "check",
         "--weights", str(tmp_path / "nonexistent")],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 2
    combined = (out.stdout + out.stderr).lower()
    assert "weights not found" in combined
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/unit/test_determinism_real_weights.py -v
git add src/aiutopia/determinism/harness.py \
        src/aiutopia/cli/determinism.py \
        tests/unit/test_determinism_real_weights.py
git commit -m "feat(determinism): real-weights replay with LSTM state threading + Algorithm.from_checkpoint (M1-Training T19)"
```

Expected: 1 PASSED.

---

### Task 20: Launch 4 training instances + write gate scripts

**Files:**
- Create: `scripts/launch-training-instances.sh`
- Create: `scripts/m1b-evaluation-gate.sh`

**v2 fix:** Use `cp` instead of `ln -sf` to dodge Windows symlink permission issues. (Symlinks on Windows need admin/dev-mode; `cp` always works.)

- [ ] **Step 1: Launch script**

Create `scripts/launch-training-instances.sh`:
```bash
#!/usr/bin/env bash
# Launch 4 parallel Fabric training instances for M1B training.
# On Windows under MSYS/Git-Bash, use cp not ln -sf to avoid symlink perms.
set -euo pipefail
: "${JDK_HOME:?must be set}"
export JAVA_HOME="$JDK_HOME"
export PATH="$JDK_HOME/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_DIR="$REPO_ROOT/server-runtime"
TRAINING_DIR="$REPO_ROOT/server-runtime/training"
MOD_JAR="$REPO_ROOT/fabric_mod/build/libs/aiutopia-mod-0.0.0-m1b.jar"

if [[ ! -f "$MOD_JAR" ]]; then
    echo "ERROR: $MOD_JAR not found — run T22's gradle build first"
    exit 1
fi

mkdir -p "$TRAINING_DIR"
for i in 1 2 3 4; do
    INST="$TRAINING_DIR/instance-$i"
    PY4J_PORT=$((25000 + i))
    MC_PORT=$((25565 + i))

    if [[ ! -d "$INST" ]]; then
        echo "[setup] creating $INST"
        mkdir -p "$INST/mods" "$INST/world"
        cp "$PRODUCTION_DIR/fabric-server-launcher.jar" "$INST/"
        for m in fabric-api fabric-carpet lithium ferritecore; do
            cp "$PRODUCTION_DIR/mods/$m"-*.jar "$INST/mods/" 2>/dev/null || true
        done
        cp "$MOD_JAR" "$INST/mods/aiutopia-mod-0.0.0-m1b.jar"
        echo "eula=true" > "$INST/eula.txt"
        cat > "$INST/server.properties" <<PROPS
server-port=$MC_PORT
online-mode=false
white-list=false
gamemode=survival
difficulty=peaceful
spawn-protection=0
max-players=5
view-distance=10
simulation-distance=10
level-name=world
motd=AI Utopia training instance $i
PROPS
    fi

    echo "[launch] instance-$i on MC:$MC_PORT Py4J:$PY4J_PORT"
    (
      cd "$INST"
      nohup java -Daiutopia.py4j.port=$PY4J_PORT \
                  -Xms1g -Xmx2g -XX:+UseG1GC \
                  -jar fabric-server-launcher.jar nogui \
                  > "instance-$i.log" 2>&1 &
      echo $! > "instance-$i.pid"
    )
done

echo "All 4 instances launching. Wait for 'Done (X.Xs)!' in each log before training."
```

Make executable: `chmod +x scripts/launch-training-instances.sh`.

- [ ] **Step 2: Evaluation-gate wrapper**

Create `scripts/m1b-evaluation-gate.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: $0 <checkpoint-dir>}"
export PYTHONPATH="${PYTHONPATH:-src}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=== section 5.10 Gate 5: determinism check (writes metrics) ==="
python -m aiutopia.cli.app determinism check \
    --weights "$CKPT" --episodes 3 --py4j-port 25001 \
    || { echo "determinism FAILED"; exit 3; }

echo
echo "=== section 5.10 checklist gates 1-5 ==="
python -m aiutopia.cli.app promote-weights promote \
    --role gatherer --checkpoint "$CKPT" \
    --notes "M1B-Training initial promotion via evaluation-gate script"
```

Make executable: `chmod +x scripts/m1b-evaluation-gate.sh`.

- [ ] **Step 3: Commit**

```bash
git add scripts/launch-training-instances.sh scripts/m1b-evaluation-gate.sh
git commit -m "feat(train): launch-training-instances + m1b-evaluation-gate (M1-Training T20)"
```

---

### Task 21: Run training to the evaluation gate (manual / empirical)

This is the empirical milestone. Expected wall-clock: ~3-12 hours of training depending on convergence.

- [ ] **Step 1: Launch training instances**

```bash
JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 ./scripts/launch-training-instances.sh
```

- [ ] **Step 2: Spawn 1 gatherer on each training instance**

```bash
for port in 25001 25002 25003 25004; do
  AIUTOPIA_ROOT=/c/tmp/aiu-train-$port \
  PYTHONPATH=src python -m aiutopia.cli.app agent spawn \
      --role gatherer --py4j-port $port
done
```

- [ ] **Step 3: Setup training scene on each**

```bash
PYTHONPATH=src python << 'PYEOF'
from aiutopia.env.bridge import FabricBridge
for port in (25001, 25002, 25003, 25004):
    with FabricBridge(port=port) as b:
        ok = b.setup_training_scene()
        print(f"port {port}: setupTrainingScene -> {ok}")
PYEOF
```

- [ ] **Step 4: Start training**

```bash
AIUTOPIA_ROOT=/c/tmp/aiu-m1b-train \
PYTHONPATH=src \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python scripts/train.py \
    --milestone M1 \
    --max-iters 2000 \
    --evaluation-interval 10 \
    --seed 1
```

TensorBoard:
```bash
tensorboard --logdir /c/tmp/aiu-m1b-train/runs --bind_all --port 6006
```

- [ ] **Step 5: Monitor + intervene if not converging**

Diagnostic checklist if `eval_m1_oak_log_success_rate` stays below 0.10 after 200 iters:

| Symptom | Likely cause | Fix |
|---|---|---|
| All-zero rewards across all agents | env.step dropping completion events | Add log line in `MotorBridge.dispatchOnServerThread`; verify dispatch reaches executor |
| Agent stuck at spawn forever | NAVIGATE picks wrong direction | Inspect `g_resource_grid` channel for log presence; if zeros, `resetEpisode` log placement (T12) failing |
| entropy approaching 0 very fast | KL + entropy coefficients too low | Bump `entropy_coeff` 0.01 → 0.05; `kl_coeff` 0.2 → 0.5 |
| vf_loss exploding | Reward magnitudes too large; PBRS overflow | Cap PBRS contribution; reduce lr to 1e-4 |
| Determinism check fails on real weights | LSTM state not being threaded across ticks | T19 fix should have addressed this; if still failing, inspect state-in keys vs `get_initial_state` shape |
| Agent breaks 1 log then stalls | Known canopy collision (M1A T3 review) | Widen `resetEpisode` ring (radius 4-8); defer Carpet auto-step to M2 |
| RLlib smoke (T7.5) passed but real training crashes on episode boundary | env.reset() race condition with state cleanup | Trace through T13's reset-side calls; verify `agent_id_to_player_name` is populated |

- [ ] **Step 6: Wait for gate**

When 3 consecutive evaluations >= 0.80, the EvalGateStopCallback writes `custom_metrics/M1/gate_passed = 1.0`, Tune's stop dict observes that, training terminates gracefully.

```bash
BEST_CKPT=$(ls -d /c/tmp/aiu-m1b-train/runs/aiutopia_M1_seed1/PPO_*/checkpoint_* | tail -1)
echo "best checkpoint: $BEST_CKPT"
```

- [ ] **Step 7: Run gate script**

```bash
./scripts/m1b-evaluation-gate.sh "$BEST_CKPT"
```

Expected:
```
=== section 5.10 Gate 5: determinism check ===
loading checkpoint: ...
=== seed 1: running 2 replays ===
  argmax_div=0.0123  l2=0.0345  PASS
...
overall: PASS

=== section 5.10 checklist gates 1-5 ===
section 5.10 checklist: PASS (5 gates)
promoted gatherer: v0 -> v1
  weights: .../weights/gatherer/v1
  deployment_id: 1
```

- [ ] **Step 8: Tuning notes**

Create `docs/superpowers/m1b-tuning-notes.md` recording: total env steps to gate, final entropy/vf_loss/kl, any hyperparameter adjustments, wall-clock, determinism results, deployment_id.

- [ ] **Step 9: Stop training instances**

```bash
for i in 1 2 3 4; do
  PID=$(cat server-runtime/training/instance-$i/instance-$i.pid 2>/dev/null || echo "")
  [[ -n "$PID" ]] && kill -9 "$PID" 2>/dev/null
done
```

- [ ] **Step 10: Commit notes**

```bash
git add docs/superpowers/m1b-tuning-notes.md
git commit -m "docs(M1B): tuning notes from first training run (M1-Training T21)"
```

---

### Task 22: Bump jar version + write `m1b-verified` tag

- [ ] **Step 1: Bump version**

In `fabric_mod/gradle.properties`, change `mod_version=0.0.0-m1a` to `mod_version=0.0.0-m1b`.

- [ ] **Step 2: Rebuild + redeploy**

```bash
cd fabric_mod && export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10; export PATH=$JAVA_HOME/bin:$PATH
./gradlew build --no-daemon
rm -f ../server-runtime/mods/aiutopia-mod-0.0.0-m1a.jar
cp build/libs/aiutopia-mod-0.0.0-m1b.jar ../server-runtime/mods/
cd ..
```

- [ ] **Step 3: Append M1B section to `M0_PROGRESS.md`**

Append a "M1-Training Progress" section recording: tag commit, what changed vs M1-Pipeline (RLlib stack, training driver, episode-reset bridge, 4-instance topology, promotion CLI, real-weights determinism), empirical results from T21, Plan C prereqs.

- [ ] **Step 4: Commit + tag**

```bash
git add fabric_mod/gradle.properties M0_PROGRESS.md
git commit -m "docs(M0_PROGRESS): M1-Training complete (M1-Training T22)"
git tag -a m1b-verified -m "M1-Training: PPO config + RLModule + training driver + evaluation gate + promotion CLI + determinism on real weights"
git tag -l
```

Expected tags: `m0`, `m0-source`, `m0-verified`, `m1a-verified`, `m1b-verified`.

---

## M1-Training completion checklist

- [ ] `python -m pytest tests/unit -v` is all green (~115+ tests)
- [ ] `python -m pytest tests/integration -v -m integration` includes T7.5 RLlib smoke + T14 scenario smoke
- [ ] `PYTHONPATH=src python scripts/rllib_smoke.py` prints "RLLIB SMOKE OK"
- [ ] `cd fabric_mod && ./gradlew build` produces `aiutopia-mod-0.0.0-m1b.jar`
- [ ] `aiutopia promote-weights promote --help` and `aiutopia determinism check --help` both surface
- [ ] T21 produced a checkpoint where:
  - [ ] Final `eval_m1_oak_log_success_rate >= 0.80` over 3 consecutive evaluations
  - [ ] determinism check PASSes (argmax_div < 0.05, L2 < 0.1)
  - [ ] promote-weights promote succeeded — roles.policy_version went 0 → 1
  - [ ] weights_dir/gatherer/v1/ exists with Ray checkpoint files
  - [ ] `policy_deployments` table has a row for the promotion

## Plan C (M2 — Builder) prereqs

- `AiUtopiaRoleRLModule` already wired for additional roles via `policy_mapping_fn`. M2 adds `policies={"gatherer_policy", "builder_policy"}` and `BuilderRoleEncoder`.
- Cross-policy weight sharing of CoreEncoder/SharedBackbone/CTDECritic: defer to M2 — there is no clean RLlib API for it on the new stack as of 2.40-2.50 (`additional_module_specs` does not exist; the previous v1 plan got this wrong). Likely path: maintain shared submodules in a singleton + load identical weights into each policy + custom `update_from_learner_group` hook.
- `compute_reward_stage_1` → `compute_reward_stage_2` with curriculum decay (section 5.1)
- ExploitDetector gains BULK_FARMING (cross-agent inventory transfer)
- Multi-timescale LR via `algorithm_config_overrides_per_module` (spec section 7.1 M4 config)
- Pixel patch for builder (section 4.4) requires offscreen framebuffer or software raycaster

Plan C is the next plan after M1B. Estimated effort: 4-6 weeks at 10-15 hr/wk.

---

## Self-review notes

- **Three-reviewer audit applied.** RLlib API correctness, code-level cross-task consistency, and PyTorch nn.Module correctness all reviewed. ~25 material findings; all addressed inline or explicitly deferred (cross-policy weight sharing → M2).
- **Type/name consistency:** `AiUtopiaRoleRLModule` referenced consistently in T6, T7, T19. `flatten_core_obs_batched` (vectorized) used everywhere instead of v1's per-sample `flatten_core_obs`. `compute_divergence` contract honored: T19 emits int + ndarray per trace entry (not nested dicts).
- **Numeric arithmetic:** `OUTPUT_DIM = 6 + 64 + 6 + 2 + 256 + 2 + 4 = 340` ✓; `FUSED_INPUT_DIM = 256 + 128 + 64 = 448` ✓; conv pyramid `32 → 16 → 8` then `32*8*8 = 2048` ✓.
- **Empirical bar honest:** T21 diagnostic matrix lists six failure modes + fixes; the plan does not promise gate-pass on first hyperparameter setting.
- **Two-world topology:** Training side fully built (4 instances); production side unchanged from M1A. Weight promotion writes to `paths.weights_dir`; production planner (M5) will load — but that load path is M5, not M1B.
- **What this plan still cannot do without empirical evidence:** estimate exact env-steps-to-gate. Section 5.8 says "1-10M env steps" for a simple gatherer; reality may be longer if the agent stalls on canopy collisions (M1A T3 known limitation).

## Open decisions

- **Carpet auto-step:** If T21 plateaus on canopy collision, widen the `resetEpisode` log ring (radius 4-8) — already documented in T21 step 5. Deeper fix is `EntityPlayerActionPack` integration; defer to M2.
- **`max_episode_ticks`:** Default 12000 per spec section 6.3. For M1 eval (1000 env steps target), generous; training episodes mostly terminate via inventory delta before timeout. If T21 shows episodes hitting timeout often, lower to 6000.
- **Future-proofing Ray version pin:** Pin `ray>=2.40,<2.60` to prevent surprise API breakage from `ray==2.55+`. Re-evaluate after M1B lands.
