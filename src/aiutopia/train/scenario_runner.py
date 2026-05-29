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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# NOTE: `AiUtopiaPettingZooEnv` is imported LAZILY inside `run_scenario`
# (not at module level) so that importing this module — and in particular
# unit-testing the `_gatherer_collected_64_oak_log` predicate below — does
# NOT drag in the heavy wrapper dependency chain (FabricBridge, chroma,
# pettingzoo, planner). This keeps the predicate test deterministic and
# isolated from edits to wrapper.py.
#
# `reward._inventory_from_obs` IS imported at module scope: reward.py is a
# leaf module (only stdlib imports — `typing`; the wrapper imports reward,
# not the reverse), so it adds no heavy deps and keeps the predicate test
# isolated. The gate predicate REUSES this exact function so the gate
# measures precisely the inventory the reward function rewards — both
# resolve integer item IDs through the same `_ITEM_ID_TO_NAME` table.
from aiutopia.env.reward import _inventory_from_obs

M1_OAK_LOG_TARGET = 64


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    max_ticks: int
    success: Callable[[dict], bool]


def _gatherer_collected_64_oak_log(final_obs: dict) -> bool:
    """M1 gate predicate: did gatherer_0 collect >= 64 oak_log?

    Phase-0 fix #3(a) — FALSE-POSITIVE removed: the prior implementation had
    a FALSE-PASS fallback — if it found zero oak_log it fell back to summing
    ALL slot counts, so an inventory full of 64 cobblestone (or any other
    item) would FALSELY pass the gate. That fallback is DELETED.

    Phase-0 fix #3 — latent FALSE-NEGATIVE also removed: the obs space
    declares `inv_slot_item_ids` as `MultiDiscrete` (see spaces.py line 62),
    i.e. INTEGER item IDs, not strings. The old `"oak_log" in str(i)` check
    therefore matched nothing on real obs (e.g. `"oak_log" in "132"` is
    False) — the fallback was the ONLY reason this predicate ever returned
    True in production. Deleting the fallback without fixing the ID
    resolution would have converted the gate into a guaranteed
    false-negative (never passes no matter how much oak_log is collected).

    FIX: resolve the slot arrays into a canonical, namespace-stripped
    `{item_name: count}` dict via reward._inventory_from_obs — the SAME
    function (and SAME `_ITEM_ID_TO_NAME` table) the reward function uses.
    This guarantees the gate measures exactly the inventory the reward
    rewards (no gate-vs-reward skew), and it handles BOTH int IDs (mapped
    via the table) and string IDs ("minecraft:oak_log" → "oak_log"). Match
    is on the EXACT canonical name "oak_log", so cobblestone, oak_planks,
    and stripped_oak_log all correctly contribute nothing.
    """
    agent = final_obs.get("gatherer_0", {})
    inventory = _inventory_from_obs(agent)
    total = sum(count for name, count in inventory.items() if name == "oak_log")
    return total >= M1_OAK_LOG_TARGET


# Phase-0 fix #3(b): budget reconciliation.
#
# The M1 gate spec (M1B_TRAINING_PLAN.md line 5) is "collect 64 oak_log
# within 1000 ENV STEPS". Both relevant counters are measured in env steps,
# NOT Minecraft ticks (despite the field name "max_episode_ticks"):
#   * run_scenario's `for _ in range(scenario.max_ticks)` dispatches exactly
#     ONE action == ONE env.step() per iteration -> loop counter = env steps.
#   * wrapper.AiUtopiaPettingZooEnv.step() does `self._tick += 1` once at the
#     end of each step() (one increment per env step), and truncates via
#     `trunc[agent] = self._tick >= self.max_ticks` -> truncation counter =
#     env steps too. (Each env step internally advances the MC world by many
#     ticks via advance_tick_await_events, but `_tick` does NOT count those.)
# Both counters therefore share the same unit (env steps), and because the
# env is constructed with `max_episode_ticks = scenario.max_ticks` (see
# run_scenario), they stay in lockstep automatically — neither can cap the
# episode below the other.
#
# The previous value (300) prematurely capped every eval episode at <1/3 of
# the gate horizon, so a competent policy that needs >=64 successful HARVEST
# env-steps PLUS navigation/search env-steps (the seeded world now provides
# 64 reachable logs) could be truncated before reaching the target. Set the
# budget to the full gate horizon of 1000 env steps so the eval actually
# measures what the gate specifies.
M1_GATE_ENV_STEP_BUDGET = 1000

M1_SCENARIOS: list[Scenario] = [
    Scenario(
        name="m1_oak_log_seed_1",
        seed=1,
        max_ticks=M1_GATE_ENV_STEP_BUDGET,
        success=_gatherer_collected_64_oak_log,
    ),
    Scenario(
        name="m1_oak_log_seed_2",
        seed=2,
        max_ticks=M1_GATE_ENV_STEP_BUDGET,
        success=_gatherer_collected_64_oak_log,
    ),
    Scenario(
        name="m1_oak_log_seed_3",
        seed=3,
        max_ticks=M1_GATE_ENV_STEP_BUDGET,
        success=_gatherer_collected_64_oak_log,
    ),
]


def _move_state_to_device(state: dict, device) -> dict:
    return {k: v.to(device) for k, v in state.items()}


def run_scenario(scenario: Scenario, *, env_config: dict, rl_module, device: str = "cpu") -> dict:
    import torch
    from ray.rllib.core import Columns

    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    env = AiUtopiaPettingZooEnv(
        {**env_config, "tick_warp": True, "max_episode_ticks": scenario.max_ticks}
    )
    try:
        obs, _info = env.reset(seed=scenario.seed)
        # Seed `final_obs` from the reset obs so it is always bound even if
        # the loop body never runs (e.g. max_ticks == 0); each completed
        # step overwrites it with the post-step obs.
        final_obs = obs
        # Per-agent persistent LSTM state — initialized ONCE at episode
        # start, then threaded through ticks via STATE_IN/STATE_OUT.
        states = {
            agent: _move_state_to_device(rl_module.get_initial_state(), device) for agent in obs
        }

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
                    out = rl_module._forward_inference(
                        {
                            Columns.OBS: batched,
                            Columns.STATE_IN: state_in,
                        }
                    )
                actions[agent_id] = _greedy_decode(
                    out[Columns.ACTION_DIST_INPUTS][0], agent_obs.get("action_mask")
                )
                # State out is (B, H); squeeze batch dim back to (H,)
                new_states[agent_id] = {k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()}
            states = new_states
            obs, _rew, term, trunc, _info = env.step(actions)
            # `obs` is the observation read AFTER this env step (wrapper
            # `_read_all_obs()` runs before the agent is dropped from
            # `self.agents`), so it still carries gatherer_0's final
            # inventory even on the terminating step. Hold onto it as the
            # success-evaluation obs.
            final_obs = obs
            # Phase-0 fix #3(c): break on EITHER termination OR truncation
            # for the gatherer.
            #   * truncation == budget exhausted or strayed out of arena.
            #   * termination == goal-met success-termination (a parallel
            #     wrapper edit may now set terminations[agent]=True when the
            #     inventory delta is satisfied) OR death.
            # CRITICAL: the termination flag is NOT trusted as success —
            # death also terminates. Success is decided solely by the
            # inventory predicate on `final_obs` below, never by the flag.
            # Use `.get(...)` (not `all(...)`) so this is robust to the
            # wrapper pruning terminated agents from its dicts in a later
            # step: an empty `term` dict makes `all()` vacuously True, which
            # would mask the real per-agent signal.
            agent_terminated = bool(term.get("gatherer_0", False))
            agent_truncated = bool(trunc.get("gatherer_0", False))
            if agent_terminated or agent_truncated:
                break
        return {
            "name": scenario.name,
            # Success is the inventory predicate on the final obs — independent
            # of why the loop ended (goal-met, death, truncation, or budget).
            "success": scenario.success(final_obs),
            "final_inventory": final_obs,
        }
    finally:
        env.close()


def _greedy_decode(action_dist_inputs, action_mask=None):
    """Convert 344-d flat dist-inputs to an action Dict (matching GathererActorHead).

    Slice order is alphabetical (post-T7.5 fix):
      comm_payload, comm_target_mask, scalar_param, should_broadcast,
      skill_type, spatial_param, target_class.

    ``action_mask`` (optional, the obs ``action_mask`` dict): when given, the
    ``skill_type`` argmax respects the mask (masked skills get -inf), so greedy
    decode can't pick a known-invalid skill — e.g. HARVEST when no resource is in
    reach is masked, forcing NAVIGATE so the policy actually explores. Without this
    the rare-but-correct NAVIGATE the policy learned is suppressed by argmax (the
    N22 decision-core greedy-vs-sampled gap).
    """
    import torch

    from aiutopia.env.spaces import (
        COMM_PAYLOAD_DIM,
        N_GATHERER_SKILLS,
        N_TARGET_CLASSES_PER_ROLE,
    )

    flat = action_dist_inputs
    offset = 0

    def take_logits(n, mask=None):
        nonlocal offset
        out = flat[offset : offset + n]
        offset += n
        if mask is not None:
            mt = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=out.device)
            if bool(mt.any()):  # don't blank everything if the mask is all-zero
                out = out.masked_fill(~mt, float("-inf"))
        return int(torch.argmax(out).item())

    def take_gauss(d):
        nonlocal offset
        means = flat[offset : offset + d]
        offset += 2 * d
        return means.detach().cpu().numpy()

    def take_multi_binary(d):
        """MultiBinary(d) -> 2*d logits; reshape (d, 2), argmax along last."""
        nonlocal offset
        logits = flat[offset : offset + 2 * d].view(d, 2)
        offset += 2 * d
        return torch.argmax(logits, dim=-1).detach().cpu().numpy()

    # Alphabetical consumption (matches _GATHERER_HEAD_SLICES dict order).
    comm_payload = take_gauss(COMM_PAYLOAD_DIM)  # +256
    comm_target_mask = take_multi_binary(4)  # +8
    scalar_param = take_gauss(1)  # +2
    should_broadcast = take_logits(2)  # +2
    _skill_mask = action_mask.get("skill_type") if action_mask else None
    skill_type = take_logits(N_GATHERER_SKILLS, mask=_skill_mask)  # +6 (mask-aware)
    spatial_param = take_gauss(3)  # +6
    target_class = take_logits(N_TARGET_CLASSES_PER_ROLE)  # +64

    return {
        "skill_type": skill_type,
        "target_class": target_class,
        "spatial_param": np.clip(spatial_param, -1, 1).astype(np.float32),
        "scalar_param": np.clip(scalar_param, 0, 1).astype(np.float32),
        "comm_payload": np.clip(comm_payload, -1, 1).astype(np.float32),
        "should_broadcast": should_broadcast,
        "comm_target_mask": comm_target_mask.astype(np.int8),
    }


def aggregate_success_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)
