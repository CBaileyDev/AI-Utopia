"""Phase C — SIM->REAL transfer eval for the M1B gatherer gate.

Headline question: does a policy trained ENTIRELY in the headless sim
(checkpoint_000001, episode_return_mean 127.0/127.36 = collects all 64
oak_log nearly every episode in sim) clear the REAL Minecraft M1B gate
(>=64 oak_log within 1000 env-steps, on 3 fixed seeds, 80% pass == 3/3)?

This script:
  1. Loads `gatherer_policy` from the checkpoint via the LIGHTWEIGHT path
     `RLModule.from_checkpoint(<rl_module/gatherer_policy subdir>)` — no
     Algorithm/env-runner/Ray-cluster spin-up, so no GPU/Ray contention.
  2. NOT-RANDOM guard: builds a fresh random-init module and asserts the
     loaded module's dist-inputs differ — proves weights actually applied,
     so a FAIL verdict can't be a silent random-init artifact.
  3. SIM CONTROL episode (cheap, no live MC): if the loaded policy collects
     64 in sim, loading is correct and any REAL failure is a genuine
     transfer / wrapper-artifact gap — not a loading bug. Partitions the
     verdict's failure space.
  4. Runs the 3 M1_SCENARIOS against the REAL env on Py4J port 25001 with an
     INSTRUMENTED per-step loop (own loop, not bare run_scenario): captures
     per-step chosen skill + oak_log inventory trajectory + timestamps, and a
     WALL-CLOCK cap that breaks and preserves the trace (so a looping policy
     never hangs ~2h at max_ticks=1000 * multi-second steps).

Run:
  PYTHONPATH=src AIUTOPIA_ROOT=C:/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/transfer_eval.py
"""
from __future__ import annotations

import os

# Pin CUDA determinism knobs BEFORE torch CUDA init (load-bearing per the
# determinism harness; harmless on CPU). Eval runs on CPU anyway.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path

import numpy as np

# Resolve the checkpoint's per-module subdir as an ABSOLUTE path. RLModule
# .from_checkpoint feeds the path to pyarrow.fs.FileSystem.from_uri, which
# rejects a relative path ("URI has empty scheme") — must be absolute.
_REPO = Path(__file__).resolve().parent.parent
# Auto-resolve the NEWEST sim-trained checkpoint's gatherer_policy module (so a
# re-train after a fidelity fix is picked up without editing this path).
_SIM_CKPTS = sorted(
    (_REPO / "runs" / "aiutopia_M1_seed1").glob("PPO_aiutopia_sim_*/checkpoint_*"),
    key=lambda p: p.stat().st_mtime,
)
if not _SIM_CKPTS:
    raise FileNotFoundError("no PPO_aiutopia_sim checkpoint under runs/aiutopia_M1_seed1")
GATHERER_MODULE_DIR = (
    _SIM_CKPTS[-1] / "learner_group" / "learner" / "rl_module" / "gatherer_policy"
).resolve()

# Skill index -> name (spaces.py: N_GATHERER_SKILLS=6, agent.py docstring).
SKILL_NAMES = {
    0: "NAVIGATE",
    1: "HARVEST",
    2: "DEPOSIT_CHEST",
    3: "SEARCH",
    4: "WAIT",
    5: "NOOP_BROADCAST",
}

# Per-scenario wall-clock cap. The task budgets ~20 min/scenario worst case;
# a good policy success-terminates in a few minutes (~64-200 steps). This cap
# only fires on a pathological loop so we report behavior instead of hanging.
WALL_BUDGET_S = 20 * 60


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ────────────────────────────────────────────────────────────────────
# Loading
def load_gatherer_module():
    """Load the trained gatherer_policy RLModule (lightweight, CPU, eval)."""
    import torch  # noqa: F401  (imported for device placement / no_grad later)
    from ray.rllib.core.rl_module.rl_module import RLModule

    if not GATHERER_MODULE_DIR.exists():
        raise FileNotFoundError(f"module subdir not found: {GATHERER_MODULE_DIR}")

    _p(f"[load] RLModule.from_checkpoint({GATHERER_MODULE_DIR})")
    try:
        module = RLModule.from_checkpoint(GATHERER_MODULE_DIR)
    except Exception as exc:  # noqa: BLE001 — fall back to MultiRLModule subdir
        _p(f"[load] single-module load failed ({type(exc).__name__}: {exc});"
           f" falling back to MultiRLModule.from_checkpoint")
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
        multi = MultiRLModule.from_checkpoint(GATHERER_MODULE_DIR.parent)
        module = multi["gatherer_policy"]

    module.eval()
    # Force CPU (eval is CPU; avoids any GPU contention with anything else).
    module.to("cpu")
    _p(f"[load] loaded {type(module).__name__} role={getattr(module, 'role', '?')} "
       f"params={sum(p.numel() for p in module.parameters())}")
    return module


def verify_not_random(trained) -> None:
    """Guard against silently evaluating a random-init module.

    Build a fresh random-init module with the same config and confirm the
    trained module's dist-inputs differ on a fixed obs. A small/zero diff
    would mean the weights did NOT apply -> any FAIL verdict downstream would
    be a loading artifact, not a transfer result.
    """
    import torch
    from ray.rllib.core import Columns

    from aiutopia.env.spaces import build_role_observation_space, build_role_action_space
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule

    obs_space = build_role_observation_space("gatherer", stage=1)
    act_space = build_role_action_space("gatherer")

    # Contract parity: the loaded module must expose the canonical gatherer
    # obs/action spaces (it loads + runs against the real env unchanged).
    assert sorted(obs_space.spaces.keys()) == sorted(trained.observation_space.spaces.keys()), \
        "loaded obs space keys != canonical gatherer obs space"
    assert sorted(act_space.spaces.keys()) == sorted(trained.action_space.spaces.keys()), \
        "loaded act space keys != canonical gatherer act space"

    fresh = AiUtopiaRoleRLModule(
        observation_space=obs_space,
        action_space=act_space,
        model_config={
            "role": "gatherer",
            "max_seq_len": 32,
            "actor_hidden": [256],
            "core_encoder": {"core_hidden": [512, 256]},
            "shared_backbone": {"lstm_hidden": 256},
            "ctde_critic": {"critic_hidden": 256},
        },
    )
    fresh.eval()

    obs_space.seed(0)
    sample = obs_space.sample()

    def batch(v):
        if isinstance(v, dict):
            return {k: batch(x) for k, x in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0)

    b = {k: batch(v) for k, v in sample.items()}
    st_t = {k: v.unsqueeze(0) for k, v in trained.get_initial_state().items()}
    st_f = {k: v.unsqueeze(0) for k, v in fresh.get_initial_state().items()}
    with torch.no_grad():
        ot = trained._forward_inference({Columns.OBS: b, Columns.STATE_IN: st_t})
        of = fresh._forward_inference({Columns.OBS: b, Columns.STATE_IN: st_f})
    diff = float(torch.abs(ot[Columns.ACTION_DIST_INPUTS][0]
                           - of[Columns.ACTION_DIST_INPUTS][0]).max())
    # skill_type slice: alphabetical offsets comm_payload(256)+comm_target_mask(8)
    # +scalar_param(2)+should_broadcast(2) = 268, width 6.
    skill_logits = [round(x, 3) for x in ot[Columns.ACTION_DIST_INPUTS][0][268:274].tolist()]
    top = int(np.argmax(skill_logits))
    _p(f"[guard] not-random max|trained-fresh|={diff:.4f}  "
       f"trained skill_logits={skill_logits}  top={SKILL_NAMES[top]}")
    if diff <= 1e-3:
        raise RuntimeError(
            "NOT-RANDOM GUARD FAILED: trained module's dist-inputs match a "
            "fresh random-init module — weights did NOT load. Aborting (a FAIL "
            "verdict here would be a loading bug, not a transfer result)."
        )


# ────────────────────────────────────────────────────────────────────
# Instrumented episode loop (own loop; reuses production decode/predicate)
def run_instrumented(
    scenario,
    *,
    env_factory,
    env_config: dict,
    rl_module,
    device: str = "cpu",
    wall_budget_s: float = WALL_BUDGET_S,
    trace_first_n: int = 30,
) -> dict:
    """Run ONE scenario with per-step instrumentation + a wall-clock cap.

    Mirrors scenario_runner.run_scenario's LSTM-threaded greedy-decode loop and
    its success semantics (inventory predicate on the final obs), but adds:
      * per-step (skill chosen, oak_log count, cumulative wall time),
      * a wall-clock guard that breaks and preserves the trace,
      * env-steps-used + wall-time in the result.

    `env_factory(cfg) -> env` lets us run the SAME loop against the sim env
    (control) and the real wrapper (gate) without duplicating logic.
    """
    import torch
    from ray.rllib.core import Columns

    from aiutopia.env.reward import _inventory_from_obs

    # Lazy import of the production decode (keeps scenario_runner pristine).
    from aiutopia.train.scenario_runner import _greedy_decode

    def _oak_log(agent_obs: dict) -> int:
        inv = _inventory_from_obs(agent_obs)
        return int(sum(c for n, c in inv.items() if n == "oak_log"))

    def _batch_value(v):
        if isinstance(v, dict):
            return {k: _batch_value(vv) for k, vv in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)

    env = env_factory({**env_config, "tick_warp": True,
                       "max_episode_ticks": scenario.max_ticks})

    trace: list[dict] = []
    steps_used = 0
    capped_reason = None
    t0 = time.time()
    try:
        obs, _info = env.reset(seed=scenario.seed)
        final_obs = obs
        states = {
            agent: {k: v.to(device) for k, v in rl_module.get_initial_state().items()}
            for agent in obs
        }
        for _ in range(scenario.max_ticks):
            now = time.time()
            if now - t0 > wall_budget_s:
                capped_reason = f"wall-clock cap ({wall_budget_s:.0f}s) hit"
                break

            actions = {}
            new_states = {}
            step_skill = {}
            for agent_id, agent_obs in obs.items():
                batched = {k: _batch_value(v) for k, v in agent_obs.items()}
                state_in = {k: v.unsqueeze(0) for k, v in states[agent_id].items()}
                with torch.no_grad():
                    out = rl_module._forward_inference(
                        {Columns.OBS: batched, Columns.STATE_IN: state_in}
                    )
                action = _greedy_decode(out[Columns.ACTION_DIST_INPUTS][0])
                actions[agent_id] = action
                step_skill[agent_id] = int(action["skill_type"])
                new_states[agent_id] = {
                    k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()
                }
            states = new_states

            pre_logs = _oak_log(obs.get("gatherer_0", {}))
            obs, _rew, term, trunc, _info = env.step(actions)
            final_obs = obs
            steps_used += 1
            post_logs = _oak_log(obs.get("gatherer_0", {}))

            trace.append({
                "step": steps_used,
                "skill": SKILL_NAMES.get(step_skill.get("gatherer_0"), "?"),
                "oak_log_pre": pre_logs,
                "oak_log_post": post_logs,
                "t": round(time.time() - t0, 2),
                "term": bool(term.get("gatherer_0", False)),
                "trunc": bool(trunc.get("gatherer_0", False)),
            })

            agent_terminated = bool(term.get("gatherer_0", False))
            agent_truncated = bool(trunc.get("gatherer_0", False))
            if agent_terminated or agent_truncated:
                break
    finally:
        env.close()

    wall = time.time() - t0
    final_logs = 0
    if final_obs:
        from aiutopia.env.reward import _inventory_from_obs as _inv
        inv = _inv(final_obs.get("gatherer_0", {}))
        final_logs = int(sum(c for n, c in inv.items() if n == "oak_log"))

    return {
        "name": scenario.name,
        "seed": scenario.seed,
        "success": scenario.success(final_obs),
        "oak_log": final_logs,
        "steps_used": steps_used,
        "wall_s": round(wall, 1),
        "capped": capped_reason,
        "trace": trace,
        "trace_first_n": trace_first_n,
    }


def _print_result(r: dict) -> None:
    head = (f"  [{r['name']}] seed={r['seed']}  oak_log={r['oak_log']}/64  "
            f"success={r['success']}  steps={r['steps_used']}  wall={r['wall_s']}s")
    if r.get("capped"):
        head += f"  CAPPED({r['capped']})"
    _p(head)
    # Print the first ~N steps of the trace whenever the scenario did NOT
    # success-terminate cleanly (capped, truncated, or short on logs) — the
    # pathology evidence the task asks for.
    show_trace = r.get("capped") or not r["success"]
    if show_trace and r["trace"]:
        _p(f"    first {min(r['trace_first_n'], len(r['trace']))} steps "
           f"(skill / oak_log pre->post / t):")
        from collections import Counter
        for e in r["trace"][: r["trace_first_n"]]:
            _p(f"      step {e['step']:>3}  {e['skill']:<14} "
               f"oak {e['oak_log_pre']}->{e['oak_log_post']}  t={e['t']}s"
               f"{'  TERM' if e['term'] else ''}{'  TRUNC' if e['trunc'] else ''}")
        skills = Counter(e["skill"] for e in r["trace"])
        _p(f"    skill histogram (all {len(r['trace'])} steps): {dict(skills)}")
        last_log = max((e["oak_log_post"] for e in r["trace"]), default=0)
        _p(f"    max oak_log seen across episode: {last_log}")


# ────────────────────────────────────────────────────────────────────
def main() -> int:
    _p("=" * 72)
    _p("Phase C — SIM->REAL transfer eval (M1B gatherer gate)")
    _p("=" * 72)

    # 1. Load
    module = load_gatherer_module()
    # 2. Not-random guard (aborts if weights didn't apply)
    verify_not_random(module)

    from aiutopia.train.scenario_runner import M1_SCENARIOS

    # ── 3. SIM CONTROL (cheap; partitions loading-bug vs transfer-gap) ──
    _p("")
    _p("[control] running SIM control episode (seed 1) with the LOADED policy …")
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env
    sim_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "per_worker_seed_offset": False,
    }
    sim_scn = M1_SCENARIOS[0]
    sim_result = run_instrumented(
        sim_scn,
        env_factory=lambda cfg: make_aiutopia_sim_env(cfg),
        env_config=sim_config,
        rl_module=module,
        device="cpu",
        wall_budget_s=300,  # sim is fast; this is generous
    )
    sim_result["name"] = "SIM_control_seed_1"
    _print_result(sim_result)
    if sim_result["success"]:
        _p("[control] SIM control PASSED -> loading is correct; any REAL "
           "failure is a genuine transfer / wrapper-artifact gap.")
    else:
        _p("[control] SIM control FAILED -> the LOADED policy does not even "
           "collect 64 in sim. This points at a LOADING bug (or a config "
           "mismatch vs the trained module), NOT a sim->real transfer gap. "
           "REAL results below are still run for completeness.")

    # ── 4. REAL gate scenarios ──
    real_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
    }

    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    _p("")
    _p("[real] running 3 M1_SCENARIOS against REAL Minecraft (Py4J 25001) …")
    real_results = []
    for scn in M1_SCENARIOS:
        _p(f"[real] -> {scn.name} (seed={scn.seed}, max_ticks={scn.max_ticks}, "
           f"wall_cap={WALL_BUDGET_S/60:.0f}min) …")
        r = run_instrumented(
            scn,
            env_factory=lambda cfg: AiUtopiaPettingZooEnv(cfg),
            env_config=real_config,
            rl_module=module,
            device="cpu",
            wall_budget_s=WALL_BUDGET_S,
        )
        _print_result(r)
        real_results.append(r)

    # ── Verdict ──
    n_pass = sum(1 for r in real_results if r["success"])
    rate = n_pass / len(real_results) if real_results else 0.0
    _p("")
    _p("=" * 72)
    _p("VERDICT")
    _p("=" * 72)
    _p(f"  SIM control (loaded policy): success={sim_result['success']} "
       f"oak_log={sim_result['oak_log']}/64")
    for r in real_results:
        _p(f"  REAL {r['name']}: success={r['success']} oak_log={r['oak_log']}/64 "
           f"steps={r['steps_used']} wall={r['wall_s']}s"
           + (f" CAPPED({r['capped']})" if r.get("capped") else ""))
    _p(f"  REAL pass rate: {n_pass}/{len(real_results)} = {rate:.0%} "
       f"(gate = 80% == 3/3)")
    gate_pass = (n_pass == len(real_results)) and len(real_results) == 3
    _p(f"  >>> SIM->REAL M1B GATE: {'PASS' if gate_pass else 'FAIL'} <<<")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
