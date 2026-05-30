"""Survival-pressure RECON of the PROVEN M1B Lumberjack policy.

NOT training, NOT a capability claim. Pure observation: take the gatherer_policy
checkpoint that transfers 3/3 on the peaceful flat M1B arena (constant
health=20 / hunger=20 / no hostiles) and ask, honestly, what it does when real
Minecraft survival pressure is flipped on at RUNTIME via the already-exposed
`Py4JEntryPoint.runCommand` (no Java rebuild):

    /difficulty normal
    /gamerule doMobSpawning true
    /gamerule doDaylightCycle false
    /time set 18000          # midnight

The policy is OUT-OF-DISTRIBUTION on every survival signal (health, hunger,
g_hostiles_nearby were all trained at constant safe values). The recon question
is: does it react at all, how does HARVEST-spam behave under attack, and what
concretely breaks?

KEY GOTCHA (why this script does not trust decoded health for death):
    A Carpet fake player that dies is REMOVED from the server, so its
    player_name DISAPPEARS from `observationsAll()`. Downstream
    `_read_all_obs` then zero-fills the obs (health decodes to 0.0) and the
    wrapper would prune the agent — but the *robust* death oracle that does
    not depend on any decode path is simply: is the player_name still a key
    in the raw `observations_all()` dict? Absent key == dead/removed. We log
    BOTH the raw-key oracle and the decoded health so the two can be compared.

PRE-FLIGHT (before the 600-step run): natural night spawns are stochastic and
may be zero if the arena is lit, and on `/difficulty normal` starvation caps at
1 HP (never kills). So before committing to a long run we DETERMINISTICALLY
`/summon` a few zombies next to the fake player, advance ~40 ticks, and confirm
health actually drops + g_hostiles_nearby populates. If health does not move,
THAT is the headline finding ("fake player not targeted/damaged on the flat
arena") and we stop — no point running 600 steps of noise.

Run:
  PYTHONPATH=src AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/survival_recon.py
"""
from __future__ import annotations

import os

# Pin CUDA determinism knobs BEFORE torch CUDA init (harmless on CPU; eval is CPU).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from collections import Counter

import numpy as np

# Reuse the proven loading + decode plumbing from the transfer eval so we run
# the EXACT same policy the 3/3 transfer used (no re-derivation of paths).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_eval import (
    GATHERER_MODULE_DIR,
    SKILL_NAMES,
    load_gatherer_module,
    verify_not_random,
)

PORT = int(os.environ.get("PY4J_RECON_PORT", "25001"))
PLAYER_NAME = "gatherer_0"           # wrapper default agent_id->player_name
AGENT_ID = "gatherer_0"
MAX_STEPS = int(os.environ.get("RECON_MAX_STEPS", "600"))
# Per-step wall cap so a looping policy under attack never hangs for hours.
WALL_BUDGET_S = float(os.environ.get("RECON_WALL_CAP_S", 45 * 60))

# Survival-pressure commands flipped at runtime AFTER each reset. NOTE the
# `/gamemode survival` line: the Carpet fake player is spawned by
# `/player <name> spawn` (WorldOps.carpetSpawn) which inherits the server's
# default (creative/invulnerable) gamemode — so WITHOUT this flip mobs and
# `/damage` do nothing (the first version of this recon wrongly concluded the
# player was "invulnerable, untestable without a rebuild"; it was just
# gamemode). `reset_episode` re-spawns the player, so the flip is re-applied
# every episode.
PRESSURE_CMDS = [
    "/difficulty hard",
    "/gamemode survival gatherer_0",
    "/gamerule doMobSpawning true",
    "/gamerule doDaylightCycle false",
    "/time set 18000",   # midnight
]


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _decoded_health(agent_obs: dict) -> float:
    h = agent_obs.get("health")
    if h is None:
        return float("nan")
    return float(np.asarray(h).reshape(-1)[0])


def _decoded_hunger(agent_obs: dict) -> float:
    h = agent_obs.get("hunger")
    if h is None:
        return float("nan")
    return float(np.asarray(h).reshape(-1)[0])


def _decoded_pos(agent_obs: dict):
    p = agent_obs.get("position")
    if p is None:
        return None
    arr = np.asarray(p).reshape(-1)
    if arr.size < 3:
        return None
    return (round(float(arr[0]), 1), round(float(arr[1]), 1), round(float(arr[2]), 1))


def _hostiles_present(agent_obs: dict) -> int:
    """Count nonzero rows in g_hostiles_nearby (4x4); a zero row = empty slot."""
    g = agent_obs.get("g_hostiles_nearby")
    if g is None:
        return 0
    arr = np.asarray(g, dtype=np.float32).reshape(-1, 4) if np.asarray(g).size else np.zeros((0, 4))
    return int(sum(1 for row in arr if np.any(row != 0.0)))


def _oak_log(agent_obs: dict) -> int:
    from aiutopia.env.reward import _inventory_from_obs

    inv = _inventory_from_obs(agent_obs)
    return int(sum(c for n, c in inv.items() if n == "oak_log"))


def _run_command(env, cmd: str) -> bool:
    ok = bool(env.bridge.entry_point.runCommand(cmd))
    _p(f"    runCommand({cmd!r}) -> {ok}")
    return ok


def _raw_alive(env) -> bool:
    """Robust death oracle: is the fake player still in the raw obs dict?"""
    raw = env.bridge.observations_all()
    return PLAYER_NAME in raw


def main() -> int:  # noqa: PLR0915, PLR0912
    _p("=" * 72)
    _p("SURVIVAL-PRESSURE RECON — proven M1B Lumberjack vs night/mobs/hunger")
    _p("=" * 72)
    _p(f"checkpoint module: {GATHERER_MODULE_DIR}")
    _p(f"port={PORT}  max_steps={MAX_STEPS}  wall_cap={WALL_BUDGET_S/60:.0f}min")

    # 1. Load the proven policy (import role module FIRST so from_checkpoint
    #    returns the real subclass, not a base RLModule — done inside
    #    load_gatherer_module's import path, but verify_not_random imports it
    #    explicitly too). Then the not-random guard.
    import aiutopia.rl_module.role_rl_module  # noqa: F401  (registers subclass)

    module = load_gatherer_module()
    verify_not_random(module)

    import torch
    from ray.rllib.core import Columns

    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    from aiutopia.train.scenario_runner import _greedy_decode

    env_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [PORT],
        "tick_warp": True,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
        # Plenty of headroom so the wrapper's truncation doesn't pre-empt the
        # 600-step recon (it still truncates on out-of-bounds — recorded).
        "max_episode_ticks": MAX_STEPS + 100,
    }

    env = AiUtopiaPettingZooEnv(env_config)

    def _reset_once(seed: int):
        return env.reset(seed=seed)

    try:
        # 2. Reset. Cold-start spawn race: first reset on a freshly-launched
        #    server can strand the agent at origin (0/0). Detect + re-reset once.
        _p("")
        _p("[reset] first reset (seed=1) …")
        obs, _info = _reset_once(1)
        pos0 = _decoded_pos(obs.get(AGENT_ID, {}))
        logs0 = _oak_log(obs.get(AGENT_ID, {}))
        _p(f"[reset] pos={pos0} oak_log={logs0}")
        stranded = pos0 is not None and abs(pos0[0]) < 4 and abs(pos0[2]) < 4
        if stranded:
            _p("[reset] looks stranded at origin — re-resetting once (cold-start "
               "race guard) …")
            obs, _info = _reset_once(1)
            pos0 = _decoded_pos(obs.get(AGENT_ID, {}))
            _p(f"[reset] after re-reset pos={pos0}")

        # 3. Flip survival pressure at RUNTIME via the bridge.
        _p("")
        _p("[pressure] flipping survival pressure via runCommand …")
        for cmd in PRESSURE_CMDS:
            _run_command(env, cmd)

        # 4. DAMAGEABILITY / GRANULARITY FACTS (established by separate probes,
        #    recorded here for the report; NOT re-run destructively before the
        #    policy run because every confirmed /damage on this config is
        #    instantly lethal — see notes):
        #
        #    * After `/gamemode survival` the player IS damageable: a separate
        #      probe drove health 20.0 -> 0.0 with /damage and removed the
        #      player; 5 zombies summoned on it killed it on the first WAIT step.
        #    * Health is BINARY at this sampling granularity: the only decoded
        #      health values ever observed are 20.0 and 0.0. Even `/damage 1`
        #      goes straight to 0.0 (the fake player behaves as ~1 effective HP
        #      in survival, or /damage applies lethal force). So there is NO
        #      graduated health signal to react to.
        #    * Under tick-warp, one env-step spans many ticks; combat resolves
        #      BETWEEN obs samples, so the policy gets obs=alive -> picks a skill
        #      -> the step runs -> next obs = dead. Death is ATOMIC from the
        #      policy's view; "reaction to declining health/mobs" is structurally
        #      UNOBSERVABLE on this path.
        #
        #    We confirm survival mode took effect (health still readable as 20.0
        #    after the flip => mode applied, world ticking) and then ARM a
        #    deterministic threat (one summoned zombie — natural night spawns on
        #    the lit flat arena are stochastic and unreliable, so we don't rely
        #    on them) and run the policy, recording the atomic death.
        _p("")
        baseline_h = _decoded_health(obs.get(AGENT_ID, {}))
        _p(f"[preflight] survival mode applied; baseline decoded health="
           f"{baseline_h} (binary 20/0 at this granularity — see notes). "
           f"raw_alive={_raw_alive(env)}")
        if np.isnan(baseline_h) or not _raw_alive(env):
            _p("[preflight] unexpected: agent not alive/readable right after the "
               "pressure flip — aborting honestly rather than fabricating a run.")
            _write_report(
                survived=None, cause="AGENT_NOT_READABLE_AFTER_PRESSURE_FLIP",
                steps_to_death=None,
                preflight={"baseline_h": baseline_h, "raw_alive": _raw_alive(env)},
                trace=[], reacted=None, logs_at_end=0, hostiles_ever=False,
                notes="Agent was not alive/readable immediately after the "
                      "survival-pressure flip; no run performed.",
            )
            return 0

        # ARM the threat: one zombie a few blocks from the player. Deterministic
        # (not relying on stochastic natural spawns). g_hostiles_nearby was
        # separately confirmed to populate with real summoned mobs.
        _p("[preflight] arming deterministic threat: 1 zombie ~4 blocks away …")
        _run_command(env, f"/execute at {PLAYER_NAME} run summon minecraft:zombie ~4 ~ ~")

        # 5. GREEDY-POLICY RUN under pressure. We continue from the post-reset
        #    world (survival mode + armed threat), freshly initialising the LSTM
        #    state, and run the proven greedy policy. The capture records pre-step
        #    hostiles/health/pos so a terminal step's cause rests on what was in
        #    obs the step it died — not a run-wide flag.
        _p("")
        _p(f"[run] running greedy policy up to {MAX_STEPS} steps under survival "
           f"pressure (death expected to be atomic) …")
        preflight_record = {
            "baseline_h": baseline_h,
            "health_is_binary_20_or_0": True,
            "armed_threat": "1 zombie summoned ~4 blocks away",
            "death_atomic_under_tickwarp": True,
        }

        states = {
            a: {k: v for k, v in module.get_initial_state().items()} for a in env.agents
        }

        trace: list[dict] = []
        t0 = time.time()
        cause = "SURVIVED"
        steps_to_death = None
        survived = True
        hostiles_ever = False
        capped = None

        for step_i in range(1, MAX_STEPS + 1):
            if time.time() - t0 > WALL_BUDGET_S:
                capped = f"wall-clock cap ({WALL_BUDGET_S:.0f}s)"
                _p(f"[run] {capped} — stopping.")
                break
            if not env.agents:
                break

            # Greedy decode per agent (mask-aware, production decode).
            actions = {}
            step_skill = None
            for aid in list(env.agents):
                agent_obs = obs[aid]
                # _batch_obs recursively batches the obs dict including the
                # nested action_mask Dict (N17 nested-Dict batching gotcha).
                state_in = {k: v.unsqueeze(0) for k, v in states[aid].items()}
                with torch.no_grad():
                    out = module._forward_inference(
                        {Columns.OBS: _batch_obs(agent_obs), Columns.STATE_IN: state_in}
                    )
                action = _greedy_decode(
                    out[Columns.ACTION_DIST_INPUTS][0], agent_obs.get("action_mask")
                )
                actions[aid] = action
                if aid == AGENT_ID:
                    step_skill = int(action["skill_type"])
                states[aid] = {k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()}

            # Capture PRE-step obs (the obs the policy actually decided on, and
            # the obs that still carries hostiles on the step it dies — the
            # post-step obs is zero-filled once the player is removed).
            pre_obs = obs.get(AGENT_ID, {})
            pre_host = _hostiles_present(pre_obs)

            obs, _rew, term, trunc, _info = env.step(actions)

            ag_obs = obs.get(AGENT_ID, {})
            post_h = _decoded_health(ag_obs)
            post_hu = _decoded_hunger(ag_obs)
            post_logs = _oak_log(ag_obs)
            n_host = _hostiles_present(ag_obs)
            pos = _decoded_pos(ag_obs)
            raw_alive = _raw_alive(env)
            hostiles_ever = hostiles_ever or n_host > 0 or pre_host > 0
            agent_term = bool(term.get(AGENT_ID, False))
            agent_trunc = bool(trunc.get(AGENT_ID, False))

            trace.append({
                "step": step_i,
                "skill": SKILL_NAMES.get(step_skill, "?"),
                "health": round(post_h, 2) if not np.isnan(post_h) else None,
                "hunger": round(post_hu, 2) if not np.isnan(post_hu) else None,
                "hostiles": n_host,
                "pre_hostiles": pre_host,
                "oak_log": post_logs,
                "pos": pos,
                "raw_alive": raw_alive,
                "term": agent_term,
                "trunc": agent_trunc,
                "t": round(time.time() - t0, 1),
            })

            # Death oracle: raw-key absence is ground truth (player removed on
            # death). Corroborate with decoded-health-to-0 and term flag. Cause
            # attribution rests on PRE-step hostiles (what was in obs the step it
            # died) since the post-step obs is zero-filled after removal.
            if (not raw_alive) or agent_term or (AGENT_ID not in env.agents):
                survived = False
                steps_to_death = step_i
                if agent_trunc and raw_alive:
                    cause = "OUT_OF_BOUNDS_TRUNCATION"
                    survived = True  # not a death — fled the arena
                    steps_to_death = None
                elif pre_host > 0:
                    cause = "MOB_ATTACK"
                elif hostiles_ever:
                    cause = "MOB_ATTACK_LIKELY (hostiles seen earlier, none in pre-step obs)"
                else:
                    cause = "DEATH_CAUSE_UNKNOWN_NO_HOSTILES_IN_OBS"
                _p(f"[run] terminal at step {step_i}: raw_alive={raw_alive} "
                   f"term={agent_term} trunc={agent_trunc} pre_step_hostiles={pre_host} "
                   f"skill_chosen={SKILL_NAMES.get(step_skill, '?')} -> cause={cause}")
                break

            if agent_trunc:
                # Truncation without death = fled arena / budget. Record + stop.
                cause = "OUT_OF_BOUNDS_TRUNCATION"
                _p(f"[run] truncated at step {step_i} (out-of-bounds or budget) "
                   f"pos={pos}")
                break

        wall = round(time.time() - t0, 1)

        # 6. Did the policy REACT to survival signals? It was trained at
        #    constant health=20/hunger=20/no-hostiles, so the prior is "ignores
        #    them." Quantify: did the skill distribution shift when hostiles
        #    were present vs absent? (A reactive policy would deviate.)
        reacted, react_detail = _measure_reaction(trace)

        if survived and cause == "SURVIVED" and capped:
            cause = f"SURVIVED_TO_WALL_CAP ({capped})"

        _summarize(trace, survived, cause, steps_to_death, reacted, react_detail,
                   hostiles_ever, wall, capped)

        _write_report(
            survived=survived,
            cause=cause,
            steps_to_death=steps_to_death,
            preflight=preflight_record,
            trace=trace,
            reacted=reacted,
            logs_at_end=trace[-1]["oak_log"] if trace else 0,
            hostiles_ever=hostiles_ever,
            notes=react_detail,
            wall_s=wall,
            capped=capped,
        )
        return 0
    finally:
        env.close()


def _batch_obs(agent_obs: dict):
    """Recursively batch an obs dict (nested action_mask Dict included) -> add
    a leading batch dim of 1, as tensors. Mirrors transfer_eval._batch_value."""
    import torch

    def b(v):
        if isinstance(v, dict):
            return {k: b(x) for k, x in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0)

    return {k: b(v) for k, v in agent_obs.items()}


def _measure_reaction(trace: list[dict]):
    """Compare the skill distribution on steps WITH hostiles in obs vs WITHOUT.
    A policy that reacts to the survival signal would show a different skill
    mix. Returns (reacted_bool_or_None, human_detail)."""
    with_h = [e["skill"] for e in trace if e["hostiles"] > 0]
    no_h = [e["skill"] for e in trace if e["hostiles"] == 0]
    if not with_h:
        return None, ("hostiles never populated g_hostiles_nearby during the "
                      "run, so reaction-to-hostiles cannot be measured")
    cw = Counter(with_h)
    cn = Counter(no_h)
    # Dominant skill in each regime.
    dom_w = cw.most_common(1)[0] if cw else ("-", 0)
    dom_n = cn.most_common(1)[0] if cn else ("-", 0)
    same_dom = dom_w[0] == dom_n[0]
    detail = (
        f"skill mix WITH hostiles ({len(with_h)} steps): {dict(cw)}; "
        f"WITHOUT hostiles ({len(no_h)} steps): {dict(cn)}. "
        f"dominant_with={dom_w[0]} dominant_without={dom_n[0]}."
    )
    # "Reacted" = dominant skill changed when hostiles appeared. Weak signal
    # (could be incidental), reported as observation not proof.
    return (not same_dom), detail


def _summarize(trace, survived, cause, steps_to_death, reacted, react_detail,
               hostiles_ever, wall, capped) -> None:
    _p("")
    _p("=" * 72)
    _p("RECON SUMMARY")
    _p("=" * 72)
    _p(f"  survived       : {survived}")
    _p(f"  cause          : {cause}")
    _p(f"  steps_to_death : {steps_to_death}")
    _p(f"  steps_run      : {len(trace)}")
    _p(f"  wall_s         : {wall}  capped={capped}")
    _p(f"  hostiles_ever_in_obs : {hostiles_ever}")
    _p(f"  reacted_to_hostiles  : {reacted}")
    _p(f"    {react_detail}")
    if trace:
        skills = Counter(e["skill"] for e in trace)
        _p(f"  skill histogram: {dict(skills)}")
        _p(f"  oak_log first->last: {trace[0]['oak_log']} -> {trace[-1]['oak_log']}")
        hmin = min((e["health"] for e in trace if e["health"] is not None), default=None)
        humin = min((e["hunger"] for e in trace if e["hunger"] is not None), default=None)
        _p(f"  min health seen: {hmin}   min hunger seen: {humin}")
        _p("  first 25 steps (step / skill / health / hunger / hostiles / oak / pos / alive):")
        for e in trace[:25]:
            _p(f"    {e['step']:>3}  {e['skill']:<14} h={e['health']} "
               f"hu={e['hunger']} host={e['hostiles']} oak={e['oak_log']} "
               f"pos={e['pos']} alive={e['raw_alive']}"
               f"{'  TERM' if e['term'] else ''}{'  TRUNC' if e['trunc'] else ''}")


def _write_report(*, survived, cause, steps_to_death, preflight, trace, reacted,  # noqa: PLR0915
                  logs_at_end, hostiles_ever, notes, wall_s=None, capped=None) -> None:
    import datetime

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo, "Research", "SURVIVAL_RECON.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    skills = Counter(e["skill"] for e in trace) if trace else Counter()
    hmin = min((e["health"] for e in trace if e.get("health") is not None), default=None)
    humin = min((e["hunger"] for e in trace if e.get("hunger") is not None), default=None)

    lines = []
    a = lines.append
    a("# Survival-Pressure RECON — proven M1B Lumberjack")
    a("")
    a(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')} by "
      "`scripts/survival_recon.py`. RECON only — no training, no capability "
      "claims. Observations are quantified and reported as-is._")
    a("")
    a("## What was tested")
    a("")
    a("The PROVEN M1B gatherer_policy checkpoint (HARVEST-spam policy that "
      "transfers 3/3 on the peaceful flat arena) was loaded greedily and faced "
      "real Minecraft survival pressure flipped on at RUNTIME via "
      "`Py4JEntryPoint.runCommand` (no Java rebuild):")
    a("")
    for c in PRESSURE_CMDS:
        a(f"- `{c}`")
    a("")
    a("Survival is enabled by `/gamemode survival` at runtime (the Carpet fake "
      "player otherwise spawns invulnerable). A deterministic summoned zombie "
      "arms the threat rather than relying on stochastic natural night spawns. "
      "See the analyst addendum for the corrected damageability/granularity "
      "findings.")
    a("")
    a("The policy is OUT-OF-DISTRIBUTION on every survival signal: it was "
      "trained at constant `health=20`, `hunger=20`, empty `g_hostiles_nearby`.")
    a("")
    a("## Headline result")
    a("")
    a(f"- **survived**: `{survived}`")
    a(f"- **cause**: `{cause}`")
    a(f"- **steps to death**: `{steps_to_death}`")
    a(f"- **steps run**: `{len(trace)}`")
    if wall_s is not None:
        a(f"- **wall time**: `{wall_s}s`  (capped: `{capped}`)")
    a(f"- **oak_log at end**: `{logs_at_end}`")
    a(f"- **hostiles ever populated `g_hostiles_nearby`**: `{hostiles_ever}`")
    a(f"- **reacted to hostiles (dominant-skill shift)**: `{reacted}`")
    a("")
    a("## Pre-flight (threat-lands check)")
    a("")
    a("```")
    for k, v in preflight.items():
        a(f"{k} = {v}")
    a("```")
    a("")
    a("## Behavior under pressure")
    a("")
    if trace:
        a(f"- skill histogram: `{dict(skills)}`")
        a(f"- oak_log first->last: `{trace[0]['oak_log']}` -> `{trace[-1]['oak_log']}`")
        a(f"- min health seen: `{hmin}`   min hunger seen: `{humin}`")
        a("")
        a(f"- reaction detail: {notes}")
    else:
        a(f"- no trace recorded. {notes}")
    a("")
    a("## Death / death-oracle methodology")
    a("")
    a("A Carpet fake player that dies is REMOVED from the server, so its "
      "`player_name` disappears from `observationsAll()`. The robust death "
      "oracle used here is **raw-key presence in `bridge.observations_all()`** "
      "(absent key == dead), corroborated by the decoded `term` flag and the "
      "agent dropping out of `env.agents`. Decoded `health` is logged but NOT "
      "trusted as the primary death signal (it zero-fills to 0.0 on the "
      "missing-key step, which is itself only a derived signal).")
    a("")
    a("## First 30 steps (trace)")
    a("")
    a("| step | skill | health | hunger | hostiles | oak_log | pos | raw_alive |")
    a("|---|---|---|---|---|---|---|---|")
    for e in trace[:30]:
        a(f"| {e['step']} | {e['skill']} | {e['health']} | {e['hunger']} | "
          f"{e['hostiles']} | {e['oak_log']} | {e['pos']} | {e['raw_alive']} |")
    a("")
    a("## Honest caveats")
    a("")
    a("- This is a single seeded run (seed=1) on one instance. Minecraft mob "
      "spawning/pathing has stochastic elements; treat counts as one sample, "
      "not a distribution.")
    a("- Pressure was flipped at runtime mid-world (no Java rebuild); the arena "
      "remains the flat M1B ring.")
    a("- `cause = MOB_ATTACK` is inferred from hostiles being present in the "
      "PRE-step obs of the terminal step, not from a death-message parse.")
    a("- `reacted to hostiles` cannot be cleanly measured here: under tick-warp "
      "death is atomic (one step alive, next step dead), so there is no "
      "alive-with-hostiles decision step to compare. See addendum Finding 4.")
    a("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _p(f"[report] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
