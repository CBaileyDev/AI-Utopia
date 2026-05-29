"""Phase C diagnostic probe — WHY did the sim-trained policy collect 0/64 in
real MC? Settles three hypotheses with direct measurement (not wall-clock
inference):

  H1 (test-setup / world-fidelity): the warm instance's world wasn't seeded /
     reset didn't run resetEpisode -> no oak_log near spawn. Evidence: post-reset
     `position` not at (64.5,65,-47.5), or `g_nearest_resources`/`g_resource_grid`
     show no oak_log nearby.
  H2 (skill-reach fidelity + policy-never-navigates): logs ARE there, but real
     HARVEST returns IMMEDIATE_FAILURE / FAILED_TIMEOUT and the greedy policy
     never NAVIGATEs to reposition. Evidence: logs present in obs, but
     skill_completion.resultCode != COMPLETED and oak_log stays 0.
  H3 (tick_warp not engaged): ~30s/step == the advance_tick_await_events 30s
     ceiling, not a 400-tick budget. Evidence: per-step wall time + completion
     resultCode (FAILED_TIMEOUT with a high tick count) vs the 30s await ceiling.

Captures per step: chosen skill, skill_completion.resultCode + failureReason,
position, oak_log. Resets at seed 1, dumps post-reset obs, runs ~12 steps.

Run AFTER the eval has fully stopped (port 25001 must be free of the eval client):
  PYTHONPATH=src AIUTOPIA_ROOT=C:/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/transfer_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path

import numpy as np

# `scripts` is not a package on PYTHONPATH (=src only); add this dir so the
# sibling transfer_eval module imports as a top-level module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_eval import (  # noqa: E402 — reuse proven loader + decode helpers
    GATHERER_MODULE_DIR,
    SKILL_NAMES,
    load_gatherer_module,
)

# Seed + step count are CLI-overridable: `transfer_probe.py <seed> <n_steps>`.
# Default seed 1, 12 steps. Short seeds-2/3 confirmation runs use ~3 steps.
PROBE_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_PROBE_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _oak_log(agent_obs: dict) -> int:
    from aiutopia.env.reward import _inventory_from_obs
    inv = _inventory_from_obs(agent_obs)
    return int(sum(c for n, c in inv.items() if n == "oak_log"))


def _dump_reset_obs(obs: dict) -> None:
    a = obs.get("gatherer_0", {})
    pos = np.asarray(a.get("position", []))
    _p(f"  position           = {pos.tolist() if pos.size else 'MISSING'}  "
       f"(sim/Java spawn = [64.5, 65.0, -47.5])")
    _p(f"  inventory oak_log  = {_oak_log(a)}")

    # g_nearest_resources: nearest-resource feature vector (relative offsets +
    # type). Print raw so we can see if ANY resource is encoded near spawn.
    gnr = a.get("g_nearest_resources", None)
    if gnr is not None:
        gnr = np.asarray(gnr)
        _p(f"  g_nearest_resources shape={gnr.shape}  "
           f"nonzero={int(np.count_nonzero(gnr))}  "
           f"first8={np.round(gnr.reshape(-1)[:8], 3).tolist()}")
    else:
        _p("  g_nearest_resources = MISSING")

    # g_resource_grid: local occupancy/type grid. Count nonzero cells = how many
    # resource blocks the obs builder sees around the agent.
    grid = a.get("g_resource_grid", None)
    if grid is not None:
        grid = np.asarray(grid)
        _p(f"  g_resource_grid    shape={grid.shape}  "
           f"nonzero_cells={int(np.count_nonzero(grid))}  "
           f"sum={float(np.sum(grid)):.1f}  max={float(np.max(grid)):.1f}")
    else:
        _p("  g_resource_grid    = MISSING")

    gr = a.get("g_richness_score", None)
    if gr is not None:
        _p(f"  g_richness_score   = {np.asarray(gr).reshape(-1).tolist()}")
    gh = a.get("g_hostiles_nearby", None)
    if gh is not None:
        _p(f"  g_hostiles_nearby  = {np.asarray(gh).reshape(-1).tolist()}")


def main() -> int:
    _p("=" * 72)
    _p("Phase C PROBE — diagnosing the 0/64 real-MC HARVEST collapse")
    _p("=" * 72)

    module = load_gatherer_module()

    import torch
    from ray.rllib.core import Columns
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    from aiutopia.train.scenario_runner import _greedy_decode

    env_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "max_episode_ticks": 1000,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
    }

    _p(f"[probe] constructing real env (Py4J 25001) + reset(seed={PROBE_SEED}) …")
    env = AiUtopiaPettingZooEnv(env_config)
    device = "cpu"

    def batch(v):
        if isinstance(v, dict):
            return {k: batch(x) for k, x in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)

    try:
        t_reset = time.time()
        obs, _info = env.reset(seed=PROBE_SEED)
        _p(f"[probe] reset done in {time.time()-t_reset:.1f}s")
        _p(f"[probe] === POST-RESET OBS (seed={PROBE_SEED}) ===")
        _dump_reset_obs(obs)

        states = {
            agent: {k: v.to(device) for k, v in module.get_initial_state().items()}
            for agent in obs
        }

        _p(f"[probe] === {N_PROBE_STEPS} GREEDY STEPS "
           f"(skill / resultCode / reason / pos / oak_log / dt) ===")
        for i in range(N_PROBE_STEPS):
            actions = {}
            new_states = {}
            chosen = {}
            for agent_id, agent_obs in obs.items():
                b = {k: batch(v) for k, v in agent_obs.items()}
                state_in = {k: v.unsqueeze(0) for k, v in states[agent_id].items()}
                with torch.no_grad():
                    out = module._forward_inference(
                        {Columns.OBS: b, Columns.STATE_IN: state_in}
                    )
                act = _greedy_decode(out[Columns.ACTION_DIST_INPUTS][0])
                actions[agent_id] = act
                chosen[agent_id] = act
                new_states[agent_id] = {
                    k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()
                }
            states = new_states

            t0 = time.time()
            obs, _rew, term, trunc, info = env.step(actions)
            dt = time.time() - t0

            a = obs.get("gatherer_0", {})
            pos = np.asarray(a.get("position", [])).reshape(-1)
            pos_s = [round(float(x), 2) for x in pos.tolist()] if pos.size else "?"
            comp = info.get("gatherer_0", {}).get("skill_completion", {}) or {}
            rc = comp.get("resultCode", "?")
            reason = comp.get("failureReason", "") or comp.get("reason", "")
            ticks_used = comp.get("ticksUsed", comp.get("ticks", "?"))
            act = chosen.get("gatherer_0", {})
            scalar = np.asarray(act.get("scalar_param", [0.0])).reshape(-1)
            target_cls = int(np.asarray(act.get("target_class", -1)).item())
            spatial = np.round(np.asarray(act.get("spatial_param", [])).reshape(-1), 2).tolist()
            _p(f"  step {i+1:>2}  {SKILL_NAMES.get(int(np.asarray(act['skill_type']).item()),'?'):<14} "
               f"rc={rc:<18} oak={_oak_log(a)}  pos={pos_s}  "
               f"scalar={round(float(scalar[0]),3)} target_cls={target_cls} spatial={spatial} "
               f"ticks={ticks_used} dt={dt:.1f}s")
            if reason:
                _p(f"          reason: {reason}")
            if bool(term.get("gatherer_0", False)) or bool(trunc.get("gatherer_0", False)):
                _p(f"          (episode end: term={term.get('gatherer_0')} "
                   f"trunc={trunc.get('gatherer_0')})")
                break
    finally:
        env.close()
        _p("[probe] env closed (JVM left running)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
