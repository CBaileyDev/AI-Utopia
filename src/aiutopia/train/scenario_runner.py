"""Section 5.8 + 5.10 evaluation scenarios for the M1 gate.

Threads LSTM hidden state across ticks per agent so inference behavior
matches training. Uses greedy action decoding (argmax for Discrete,
Gaussian-mean for continuous, per-bit argmax for MultiBinary).

Deviations from M1B_TRAINING_PLAN.md v1:
  - Slice order is ALPHABETICAL (matches `_GATHERER_HEAD_SLICES` in
    `actor_head.py` post-T7.5), not insertion order.
  - `comm_target_mask` is decoded as TorchMultiCategorical with 8 logits
    reshaped to (4, 2) and argmax along dim=-1 — not 4 raw logits with
    `> 0` thresholding. This brings the total flat width to 344 (was
    erroneously 340 in v1).
"""
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
    Scenario(name="m1_oak_log_seed_1", seed=1, max_ticks=300,
              success=_gatherer_collected_64_oak_log),
    Scenario(name="m1_oak_log_seed_2", seed=2, max_ticks=300,
              success=_gatherer_collected_64_oak_log),
    Scenario(name="m1_oak_log_seed_3", seed=3, max_ticks=300,
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
        # Per-agent persistent LSTM state — initialized ONCE at episode
        # start, then threaded through ticks via STATE_IN/STATE_OUT.
        states = {agent: _move_state_to_device(rl_module.get_initial_state(), device)
                  for agent in obs}

        def _batch_value(v):
            """N17: recursively add batch dim. The obs space contains nested
            dicts (`action_mask` has `skill_type` / `target_per_skill` /
            `comm_payload` sub-arrays); the original `torch.as_tensor(
            np.asarray(v))` crashes on those because `np.asarray(dict)`
            produces a 0-dim object array torch can't ingest."""
            if isinstance(v, dict):
                return {k: _batch_value(vv) for k, vv in v.items()}
            return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)

        for _ in range(scenario.max_ticks):
            actions = {}
            new_states = {}
            for agent_id, agent_obs in obs.items():
                batched = {k: _batch_value(v) for k, v in agent_obs.items()}
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
    """Convert 344-d flat dist-inputs to an action Dict (matching GathererActorHead).

    Slice order is alphabetical (post-T7.5 fix):
      comm_payload, comm_target_mask, scalar_param, should_broadcast,
      skill_type, spatial_param, target_class.
    """
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
        return means.detach().cpu().numpy()

    def take_multi_binary(d):
        """MultiBinary(d) -> 2*d logits; reshape (d, 2), argmax along last."""
        nonlocal offset
        logits = flat[offset:offset + 2 * d].view(d, 2)
        offset += 2 * d
        return torch.argmax(logits, dim=-1).detach().cpu().numpy()

    # Alphabetical consumption (matches _GATHERER_HEAD_SLICES dict order).
    comm_payload     = take_gauss(COMM_PAYLOAD_DIM)           # +256
    comm_target_mask = take_multi_binary(4)                    # +8
    scalar_param     = take_gauss(1)                           # +2
    should_broadcast = take_logits(2)                          # +2
    skill_type       = take_logits(N_GATHERER_SKILLS)          # +6
    spatial_param    = take_gauss(3)                           # +6
    target_class     = take_logits(N_TARGET_CLASSES_PER_ROLE)  # +64

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
