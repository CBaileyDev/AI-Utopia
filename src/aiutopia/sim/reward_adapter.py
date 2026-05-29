"""Reward adapter for the gatherer fast-sim.

``step_reward(obs_prev, obs_curr, action, env_meta) -> float`` is a thin
FORWARDING wrapper: it does NOT reimplement any reward math. It calls the
real, authoritative stage-1 reward function
``aiutopia.env.reward.compute_reward_stage_1`` with ``role="gatherer"`` and the
args passed through unchanged. This makes reward parity with real Minecraft
*free* -- the same pure function scores both the sim and the live env -- so all
reward correctness rides on obs/inventory faithfulness (covered by the
golden-trace obs parity test, Phase A Task 3b), not on this file.

``env_meta`` is assembled one layer up (the SimEnv, Phase A Task 5) from the
skill-completion dict and forwarded here untouched. Keys consumed by
``compute_reward_stage_1``:
  - ``died_this_tick``: bool (always False in the M1B gatherer arena)
  - ``n_clipped_param_axes``: int 0..4 (popcount of the skill's clipped bitset)
  - ``exploit_penalties``: list[(name, penalty)] (empty for Phase A)

IMPORT-LIGHT by contract: this module imports only ``aiutopia.env.reward``
(itself pure: no chromadb / py4j / torch / sentence_transformers -- verified at
0.004s cold import) -- never the wrapper, the bridge, or any heavy dependency.
Verify with ``py -3.11 -c "import aiutopia.sim.reward_adapter"`` staying fast.
"""

from __future__ import annotations

from aiutopia.env.reward import compute_reward_stage_1


def step_reward(obs_prev: dict, obs_curr: dict, action: dict, env_meta: dict) -> float:
    """Forward to ``compute_reward_stage_1`` for the gatherer role.

    Pure pass-through: ``obs_prev``, ``obs_curr``, ``action``, and ``env_meta``
    are handed to the real reward function unmodified, so this adapter and a
    direct call to ``compute_reward_stage_1(role="gatherer", ...)`` return the
    identical scalar. No reward math is reimplemented here.
    """
    return compute_reward_stage_1(
        role="gatherer",
        obs_prev=obs_prev,
        obs_curr=obs_curr,
        action=action,
        env_meta=env_meta,
    )
