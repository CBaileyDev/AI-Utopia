"""Microbenchmark for AiUtopiaSimEnv.step decomposition.

Splits the env step into apply_skill / build_gatherer_obs / step_reward, across
three action regimes (WAIT floor, cap=1 HARVEST, cap=64 field-clear), and times a
full representative episode for seeds 1/2/3. Robust to CPU contention: many reps,
report min + median, lean on relative breakdown.

Run: PYTHONPATH=src py -3.11 scripts/sim_step_microbench.py
"""

from __future__ import annotations

import time
import statistics
import numpy as np

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.world import SimWorld
from aiutopia.sim.skills import apply_skill
from aiutopia.sim.obs_adapter import build_gatherer_obs
from aiutopia.sim.reward_adapter import step_reward

AGENT = "gatherer_0"
CFG = {"stage": 1, "active_roles": ["gatherer"], "max_episode_ticks": 6000}


def _harvest_action(scalar: float) -> dict:
    return {
        "skill_type": np.int64(1),  # HARVEST
        "scalar_param": np.array([scalar], dtype=np.float32),
        "spatial_param": np.zeros(3, dtype=np.float32),
        "target_class": np.int64(0),  # oak_log
    }


def _wait_action() -> dict:
    return {
        "skill_type": np.int64(4),  # WAIT
        "scalar_param": np.array([0.0], dtype=np.float32),
        "spatial_param": np.zeros(3, dtype=np.float32),
        "target_class": np.int64(0),
    }


def _timeit(fn, reps: int) -> tuple[float, float]:
    """Return (min_ms, median_ms) per call over `reps`."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), statistics.median(samples)


def bench_full_episode():
    """Time AiUtopiaSimEnv.step over a full episode, seeds 1/2/3.

    Drives a HARVEST policy with scalar default (cap=1 per the default action) so
    each step harvests one log; episode runs until success (64 logs) or trunc.
    This is the closest synthetic proxy to a converged greedy gatherer.
    """
    print("\n=== FULL EPISODE: AiUtopiaSimEnv.step (cap=1 HARVEST greedy) ===")
    for seed in (1, 2, 3):
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=seed)
        act = {AGENT: _harvest_action(1.0 / 64.0)}  # cap=1
        step_ms = []
        steps = 0
        while env.agents:
            t0 = time.perf_counter()
            obs, rew, term, trunc, info = env.step(act)
            step_ms.append((time.perf_counter() - t0) * 1e3)
            steps += 1
            if steps > 8000:
                break
        inv = info[AGENT]["skill_completion"]
        done = term.get(AGENT) or trunc.get(AGENT)
        print(
            f"  seed={seed}: steps={steps} total={sum(step_ms):.1f}ms "
            f"mean={statistics.mean(step_ms):.4f}ms median={statistics.median(step_ms):.4f}ms "
            f"min={min(step_ms):.4f}ms max={max(step_ms):.4f}ms "
            f"term={term.get(AGENT)} trunc={trunc.get(AGENT)}"
        )
    env.close()


def bench_step_regimes():
    """Whole AiUtopiaSimEnv.step under 3 action regimes (fresh world each rep)."""
    print("\n=== WHOLE STEP by regime (ms/step, min / median) ===")
    reps = 2000

    # WAIT floor: rebuild obs + reward every step, no skill movement.
    def wait_step():
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=1)
        env.step({AGENT: _wait_action()})

    # cap=1 HARVEST: walk to nearest + break 1 log.
    def h1_step():
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=1)
        env.step({AGENT: _harvest_action(1.0 / 64.0)})

    # cap=64 field-clear: walk the whole field, break all 64 in ONE step.
    def h64_step():
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=1)
        env.step({AGENT: _harvest_action(1.0)})

    # Note: these include reset+__init__ overhead; isolate below. Here we time
    # only the .step() by pre-building the env (warm) for a tighter floor.
    def make_warm(action):
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=1)
        return lambda: (env.reset(seed=1), env.step({AGENT: action}))

    for label, action in [
        ("WAIT (obs-only floor)", _wait_action()),
        ("HARVEST cap=1", _harvest_action(1.0 / 64.0)),
        ("HARVEST cap=64 (field-clear)", _harvest_action(1.0)),
    ]:
        env = AiUtopiaSimEnv(dict(CFG))
        env.reset(seed=1)

        def one():
            env.reset(seed=1)
            env.step({AGENT: action})

        # subtract reset cost
        def reset_only():
            env.reset(seed=1)

        mn, md = _timeit(one, reps)
        rmn, rmd = _timeit(reset_only, reps)
        print(
            f"  {label:32s} step+reset min={mn:.4f} med={md:.4f} | "
            f"reset min={rmn:.4f} med={rmd:.4f} | step_only(med-med)~{md - rmd:.4f}ms"
        )


def bench_components():
    """Decompose into apply_skill / build_gatherer_obs / step_reward.

    For each regime we set up a fresh world, then time each component in
    isolation. apply_skill mutates the world, so for repeated timing we re-seed
    the world per rep where the skill has side effects (HARVEST).
    """
    print("\n=== COMPONENT BREAKDOWN (ms/call, min / median) ===")
    reps = 3000

    # --- build_gatherer_obs: pure function of world; run on a full (64-log) world.
    w_full = SimWorld()
    w_full.reset(1)
    mn, md = _timeit(lambda: build_gatherer_obs(w_full), reps)
    print(f"  build_gatherer_obs (64 logs alive)      min={mn:.4f} med={md:.4f}")

    # build_gatherer_obs on a near-empty world (most logs harvested) -> fewer
    # nearby entries; isolates how much the per-log loop / sort costs.
    w_empty = SimWorld()
    w_empty.reset(1)
    w_empty.log_alive[:] = False
    w_empty.log_alive[0] = True
    mn, md = _timeit(lambda: build_gatherer_obs(w_empty), reps)
    print(f"  build_gatherer_obs (1 log alive)        min={mn:.4f} med={md:.4f}")

    # --- step_reward: needs prev/curr obs.
    obs_prev = build_gatherer_obs(w_full)
    w2 = SimWorld()
    w2.reset(1)
    w2.inventory["oak_log"] = 1
    obs_curr = build_gatherer_obs(w2)
    act = _harvest_action(1.0 / 64.0)
    env_meta = {"died_this_tick": False, "n_clipped_param_axes": 0, "exploit_penalties": []}
    mn, md = _timeit(lambda: step_reward(obs_prev, obs_curr, act, env_meta), reps)
    print(f"  step_reward                             min={mn:.4f} med={md:.4f}")

    # --- apply_skill WAIT (no movement).
    def wait_skill():
        w = SimWorld()
        w.reset(1)
        apply_skill(w, _wait_action())

    # isolate reset cost to subtract
    def reset_only():
        w = SimWorld()
        w.reset(1)

    rmn, rmd = _timeit(reset_only, reps)
    mn, md = _timeit(wait_skill, reps)
    print(f"  apply_skill WAIT  (incl world reset)    min={mn:.4f} med={md:.4f}  [reset med={rmd:.4f}]")

    # --- apply_skill HARVEST cap=1: 1 nearest-scan + 1 walk loop (~up to ~75 ticks).
    def h1_skill():
        w = SimWorld()
        w.reset(1)
        apply_skill(w, _harvest_action(1.0 / 64.0))

    mn, md = _timeit(h1_skill, reps)
    print(f"  apply_skill HARVEST cap=1 (incl reset)  min={mn:.4f} med={md:.4f}  -> skill~{md - rmd:.4f}ms")

    # --- apply_skill HARVEST cap=64: 64 nearest-scans + 64 walk loops (the chain).
    def h64_skill():
        w = SimWorld()
        w.reset(1)
        apply_skill(w, _harvest_action(1.0))

    mn, md = _timeit(h64_skill, reps // 5)
    print(f"  apply_skill HARVEST cap=64 (incl reset) min={mn:.4f} med={md:.4f}  -> skill~{md - rmd:.4f}ms")

    # Count walk-loop iterations for cap=64 to ground the "Python loop" claim.
    from aiutopia.sim import skills as _sk
    orig = _sk._walk_into_reach
    counter = {"calls": 0, "iters": 0}

    def counting_walk(world, target):
        counter["calls"] += 1
        reach_sq = _sk.HARVEST_REACH * _sk.HARVEST_REACH + 1e-3
        for i in range(_sk._WALK_TICK_BUDGET):
            delta = target - world.agent_pos
            if float(delta @ delta) <= reach_sq:
                return
            dist = float(np.sqrt(delta @ delta))
            direction = delta / dist
            world.agent_pos = world.agent_pos + direction * _sk.WALK_PER_TICK
            counter["iters"] += 1

    _sk._walk_into_reach = counting_walk
    w = SimWorld()
    w.reset(1)
    apply_skill(w, _harvest_action(1.0))
    _sk._walk_into_reach = orig
    print(
        f"  [cap=64 walk-loop instrumentation] walk calls={counter['calls']} "
        f"total walk-tick iters={counter['iters']} "
        f"(avg {counter['iters'] / max(1, counter['calls']):.1f} iters/log)"
    )


if __name__ == "__main__":
    print("AiUtopiaSimEnv.step microbenchmark (NOTE: run under CPU contention; "
          "prefer min + relative breakdown)")
    bench_components()
    bench_step_regimes()
    bench_full_episode()
