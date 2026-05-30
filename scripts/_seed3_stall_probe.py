"""THROWAWAY (investigation-only, do not commit). Seed-3 stall diagnosis on the
REAL env (Py4J 25001). Drives the greedy sim-trained gatherer policy ~20 steps,
characterizes the post-55 HARVEST stall, and runs the NAVIGATE-unstick test.

Per step logs: position, oak_log, nearest-remaining-log distance + dx/dy/dz
(decoded from g_nearest_resources row 0: dx=row0*16 dy=row1*8 dz=row2*16),
skill_completion.resultCode + failureReason + ticksUsed VERBATIM.

After the stall: dispatches a manual NAVIGATE toward the remaining logs (aimed
from sim seed-3 world.logs, since stranded logs are beyond the 16-block obs
scan and thus zeroed in g_nearest_resources), then HARVEST — confirms whether
navigate-and-repeat unsticks it.

Run AFTER the eval has stopped (port 25001 free):
  PYTHONPATH=src AIUTOPIA_ROOT=C:/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/_seed3_stall_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_eval import SKILL_NAMES, load_gatherer_module  # noqa: E402

SEED = 3
N_STEPS = 20


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _oak_log(agent_obs: dict) -> int:
    from aiutopia.env.reward import _inventory_from_obs
    inv = _inventory_from_obs(agent_obs)
    return int(sum(c for n, c in inv.items() if n == "oak_log"))


def _decode_nearest(agent_obs: dict):
    """Decode g_nearest_resources row 0 -> (dist, dx, dy, dz) in blocks.
    Row layout (obs_adapter): [dx/16, dy/8, dz/16, 0, 1, 1]. All-zero row = no
    resource within the 16-block scan (stranded)."""
    gnr = agent_obs.get("g_nearest_resources", None)
    if gnr is None:
        return None
    gnr = np.asarray(gnr).reshape(8, 6)
    row0 = gnr[0]
    dx, dy, dz = float(row0[0]) * 16.0, float(row0[1]) * 8.0, float(row0[2]) * 16.0
    if np.count_nonzero(row0) == 0:
        return (None, 0.0, 0.0, 0.0)  # no resource in scan radius
    dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return (dist, dx, dy, dz)


def _log_step(tag, i, act, agent_obs, comp, dt):
    pos = np.asarray(agent_obs.get("position", [])).reshape(-1)
    pos_s = [round(float(x), 2) for x in pos.tolist()] if pos.size else "?"
    rc = comp.get("resultCode", "?")
    reason = comp.get("failureReason", "") or comp.get("reason", "")
    ticks = comp.get("ticksUsed", comp.get("ticks", comp.get("ticksRemaining", "?")))
    near = _decode_nearest(agent_obs)
    skill = SKILL_NAMES.get(int(np.asarray(act["skill_type"]).item()), "?")
    tcls = int(np.asarray(act.get("target_class", -1)).item())
    scal = round(float(np.asarray(act.get("scalar_param", [0.0])).reshape(-1)[0]), 3)
    if near is None:
        near_s = "g_nearest=MISSING"
    elif near[0] is None:
        near_s = "nearest=NONE(beyond16)"
    else:
        near_s = (f"nearest dist={near[0]:.2f} "
                  f"d=({near[1]:+.1f},{near[2]:+.1f},{near[3]:+.1f})")
    _p(f"  {tag} {i:>2}  {skill:<13} rc={rc:<18} oak={_oak_log(agent_obs):>2}  "
       f"pos={pos_s}  {near_s}  tcls={tcls} scal={scal} ticks={ticks} dt={dt:.1f}s")
    if reason:
        _p(f"          reason: {reason!r}")
    return _oak_log(agent_obs), pos


def main() -> int:
    _p("=" * 74)
    _p("SEED-3 STALL PROBE — real MC (Py4J 25001), greedy sim-trained gatherer")
    _p("=" * 74)

    module = load_gatherer_module()

    import torch
    from ray.rllib.core import Columns
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    from aiutopia.train.scenario_runner import _greedy_decode
    from aiutopia.sim.world import SimWorld

    # Sim seed-3 arena (for NAVIGATE-unstick aiming + arena-fidelity cross-check).
    simw = SimWorld()
    simw.reset(SEED)
    sim_logs = simw.logs.copy()

    env_config = {
        "stage": 1, "active_roles": ["gatherer"], "seed_strategy": "fixed_easy",
        "py4j_ports": [25001], "tick_warp": True, "max_episode_ticks": 1000,
        "per_worker_seed_offset": False, "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
    }

    device = "cpu"

    def batch(v):
        if isinstance(v, dict):
            return {k: batch(x) for k, x in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)

    env = AiUtopiaPettingZooEnv(env_config)
    try:
        t0 = time.time()
        obs, _info = env.reset(seed=SEED)
        _p(f"[reset] seed={SEED} done in {time.time()-t0:.1f}s")
        a0 = obs.get("gatherer_0", {})
        # confirm nearest_resource_distance is dropped (reconstruct from row0)
        _p(f"[reset] obs keys include nearest_resource_distance? "
           f"{'nearest_resource_distance' in a0}")
        _p(f"[reset] pos={np.asarray(a0.get('position')).tolist()}  "
           f"oak={_oak_log(a0)}  nearest(row0)={_decode_nearest(a0)}")

        states = {ag: {k: v.to(device) for k, v in module.get_initial_state().items()}
                  for ag in obs}

        _p(f"[drive] {N_STEPS} GREEDY steps:")
        last_pos = None
        last_action = None
        for i in range(N_STEPS):
            actions, new_states = {}, {}
            for ag, ao in obs.items():
                b = {k: batch(v) for k, v in ao.items()}
                st = {k: v.unsqueeze(0) for k, v in states[ag].items()}
                with torch.no_grad():
                    out = module._forward_inference(
                        {Columns.OBS: b, Columns.STATE_IN: st})
                act = _greedy_decode(out[Columns.ACTION_DIST_INPUTS][0])
                actions[ag] = act
                new_states[ag] = {k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()}
            states = new_states
            last_action = actions["gatherer_0"]

            ts = time.time()
            obs, _rew, term, trunc, info = env.step(actions)
            dt = time.time() - ts
            comp = info.get("gatherer_0", {}).get("skill_completion", {}) or {}
            ologs, last_pos = _log_step("step", i + 1, actions["gatherer_0"],
                                        obs.get("gatherer_0", {}), comp, dt)
            if term.get("gatherer_0") or trunc.get("gatherer_0"):
                _p(f"          (episode end term={term.get('gatherer_0')} "
                   f"trunc={trunc.get('gatherer_0')})")
                break
            # Detect the stall: 3 consecutive HARVESTs with no oak gain after >40.
            # We'll just run the full N_STEPS to characterize; unstick test after.

        # ── NAVIGATE-unstick test (task 3d) ──
        _p("")
        _p("[unstick] === NAVIGATE-unstick test ===")
        final_obs = obs.get("gatherer_0", {})
        cur_oak = _oak_log(final_obs)
        cur_pos = np.asarray(final_obs.get("position", [])).reshape(-1).astype(np.float64)
        _p(f"[unstick] pre: oak={cur_oak} pos={[round(float(x),2) for x in cur_pos]}")
        near = _decode_nearest(final_obs)
        _p(f"[unstick] obs nearest(row0)={near}  "
           f"(None/beyond16 => must aim from sim world.logs)")

        # Aim NAVIGATE at the nearest sim log that the real agent likely hasn't
        # collected. We don't know real-alive from obs (stranded => zeroed), so
        # target the sim log nearest to cur_pos that is >REACH from it.
        centers = sim_logs.astype(np.float64) + 0.5
        d = np.sqrt(((centers - cur_pos) ** 2).sum(axis=1))
        # candidate = nearest sim log that is beyond reach (an uncollected tail log)
        order = np.argsort(d)
        target = None
        for j in order:
            if d[j] > 4.5:
                target = centers[j]
                _p(f"[unstick] aiming NAVIGATE at sim log#{int(j)} center="
                   f"{[round(float(x),1) for x in target]} dist={d[j]:.1f}")
                break
        if target is None:
            _p("[unstick] no sim log beyond reach to aim at; skipping")
        else:
            for rnd in range(4):
                cur_pos = np.asarray(obs["gatherer_0"].get("position", cur_pos)).reshape(-1).astype(np.float64)
                delta = (target - cur_pos) / np.array([32.0, 8.0, 32.0])
                sp = np.clip(delta, -1.0, 1.0).astype(np.float32)
                nav = {**last_action, "skill_type": np.array(0, dtype=np.int64),
                       "spatial_param": sp}
                ts = time.time()
                obs, _r, term, trunc, info = env.step({"gatherer_0": nav})
                dt = time.time() - ts
                comp = info.get("gatherer_0", {}).get("skill_completion", {}) or {}
                _log_step("NAV ", rnd + 1, nav, obs.get("gatherer_0", {}), comp, dt)
                if term.get("gatherer_0") or trunc.get("gatherer_0"):
                    _p("          (episode end during nav)")
                    break
                # HARVEST after each nav
                hv = {**last_action, "skill_type": np.array(1, dtype=np.int64),
                      "target_class": np.array(0, dtype=np.int64),
                      "scalar_param": np.array([1.0], dtype=np.float32)}
                ts = time.time()
                obs, _r, term, trunc, info = env.step({"gatherer_0": hv})
                dt = time.time() - ts
                comp = info.get("gatherer_0", {}).get("skill_completion", {}) or {}
                new_oak, _ = _log_step("HARV", rnd + 1, hv,
                                       obs.get("gatherer_0", {}), comp, dt)
                if new_oak > cur_oak:
                    _p(f"[unstick] >>> NAVIGATE+HARVEST collected "
                       f"{new_oak - cur_oak} more (oak {cur_oak}->{new_oak}) "
                       f"<<< navigate-and-repeat UNSTICKS the stall")
                    cur_oak = new_oak
                if new_oak >= 64:
                    _p("[unstick] reached 64/64")
                    break
                if term.get("gatherer_0") or trunc.get("gatherer_0"):
                    break
    finally:
        env.close()
        _p("[probe] env closed (JVM left running)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
