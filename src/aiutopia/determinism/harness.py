"""§7.8 — Determinism utilities. Two metrics per §5.10:
  - action_argmax_divergence < 0.05
  - continuous_param_L2     < 0.10
over a 1000-tick replay window."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


EPS_ARGMAX = 0.05
EPS_L2     = 0.10


def configure_cuda_determinism() -> None:
    """Pin every cuDNN / cuBLAS knob that controls non-determinism.
    Call once at process start before importing torch CUDA ops.

    Without this, cuDNN autotuner randomizes per-process and the
    determinism gate becomes a flaky test."""
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class ReplayDivergence:
    action_argmax_divergence: float
    continuous_param_l2:      float

    @property
    def passes(self) -> bool:
        return (self.action_argmax_divergence < EPS_ARGMAX
                and self.continuous_param_l2 < EPS_L2)


def compute_divergence(trace_a: list[dict],
                       trace_b: list[dict]) -> ReplayDivergence:
    """Each trace entry must have keys `action_argmax: int` and
    `continuous_params: np.ndarray`."""
    if len(trace_a) != len(trace_b):
        raise ValueError(f"trace lengths differ: {len(trace_a)} vs {len(trace_b)}")
    if not trace_a:
        return ReplayDivergence(0.0, 0.0)

    argmax_div = float(np.mean([
        ta["action_argmax"] != tb["action_argmax"]
        for ta, tb in zip(trace_a, trace_b)
    ]))
    l2_div = float(np.mean([
        np.linalg.norm(np.asarray(ta["continuous_params"]) -
                       np.asarray(tb["continuous_params"]))
        for ta, tb in zip(trace_a, trace_b)
    ]))
    return ReplayDivergence(argmax_div, l2_div)


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
