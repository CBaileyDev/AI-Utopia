# Next Session Handoff — Fast-Sim & Sim→Real Transfer (2026-05-29)

This supersedes the earlier post-v20 handoff. Read `PROJECT_CONTEXT.md` for the
big picture; this captures the current frontier.

## TL;DR

We pivoted from grinding M1B on real Minecraft (~0.9 env-steps/s, ~50 days to
the gate) to a **fast headless sim** (the RLGym/RocketSim + Craftax pattern). The
sim trains the gatherer to convergence in **~90 s (≈170× real-MC)**, and a
sim-trained policy **transfers to the real M1B gate on 2 of 3 seeds (64/64
oak_log)**. The remaining seed is blocked by a now-diagnosed fidelity gap (being
closed) plus a possible real-MC motor-reach limit.

## Where things stand

**Done + on `origin/main`:**
- Phase 0 (real-MC fixes that made M1B winnable at all): `a191a2e` and prior.
- N-runner scaling (`638a97d`): `--num-env-runners N` derives ports; batch 768.
- Sim Phase A (`a191a2e` ... actually `0535ec7`): `src/aiutopia/sim/` — `SimWorld`
  (byte-faithful `java.util.Random` arena), `apply_skill`, `obs_adapter`
  (byte-faithful obs, golden-trace validated), `reward_adapter`, `AiUtopiaSimEnv`;
  `env/_embeds.py` (light shared embeds); golden-trace fixture.
- Sim Phase B (`318bae5`): `AiUtopiaSimEnv` wired into RLlib;
  `m1_gatherer_config(backend="sim")` + `train.py --backend sim`;
  `sim_env_factory.py`. Sim converges `ret_mean → 127` (all 64 logs) in ~90 s.
- Transfer fidelity fix #1 (`624da53`): the sim HARVEST now honors `target_class`
  (was ignored → policy learned a degenerate `target_class=54` → real MC mapped
  it to "stone" → 0 collected). After this, **transfer = 2/3 (seeds 1,2 = 64/64;
  seed 3 = 55/64)**.

## UPDATE 2026-05-29 (N21) — gap #2 root cause CONFIRMED, fix re-training

The bounded-skills work below is committed. New, decisive findings this session:

- **v5 (PBRS distance shaping) was insufficient.** `scripts/sim_rollout_check.py`
  (greedy sim rollout, all 3 gate seeds) showed the policy is **100% HARVEST**:
  seed1=64/64 (2 dispatches), seed2=62/64, seed3=55/64 — never NAVIGATEs.
- **Exact root cause (geometrically confirmed):** HARVEST only searches within
  `MAX_SEARCH_RADIUS=16` (`skills.py:163`). Once HARVEST auto-walks through the
  near cluster, the **tail logs fall OUTSIDE 16 b** (seed2: 19.5 b, seed3: 16.1 b),
  so every later HARVEST is `IMMEDIATE_FAILURE` and the agent is stuck. Shaping
  toward the nearest log doesn't help because auto-walking HARVEST earns the same
  shaping NAVIGATE would.
- **The gate IS achievable in-sim — it is a POLICY-LEARNING gap, not a motor
  limit.** Proven: a single forced `NAVIGATE→HARVEST` clears the tail to **64/64
  on BOTH seed2 and seed3**. This **resolves the prior "real-MC dy=+3 motor-reach
  caveat"** as the sim-side blocker — navigation simply works in sim. (A real-MC
  reach limit may still exist and must be checked separately during real transfer,
  but it is no longer the suspected cause of the sim stall.)
- **Why training didn't learn it:** HARVEST auto-walks, so on most *randomized*
  layouts HARVEST-spam clears the field; the "tail >16 b away" stall state is
  rare/under-explored, and greedy eval always argmaxes HARVEST.
- **Fix (committed `5e7fdb0`, training-only, eval unchanged):** added
  `failure_penalty=0.5` (sim_env.step penalizes `IMMEDIATE_FAILURE`/
  `FAILED_TIMEOUT`) so no-op HARVEST-spam at a stall is costly and NAVIGATE (which
  the shaping rewards toward the far tail) becomes the better action. Composes
  with `distance_shaping`.
- **v6 re-training NOW** (`Research/train-sim-v6.log`, background task). When done:
  1. `py -3.11 scripts/sim_rollout_check.py` — expect `NAVIGATE>0` and 64/64 on
     all 3 seeds. **If still HARVEST-only**, escalate: a curriculum that forces
     far-apart clusters every episode (make `randomize_layout` bias toward
     nav-requiring layouts), raise the penalty / entropy_coeff, or sampled (not
     greedy) eval.
  2. If sim clears 3/3, re-run `scripts/transfer_eval.py` against real MC
     (instance-1, port 25001). Watch for a genuine real-MC reach limit on the
     tail logs — distinct from the (now-solved) sim nav gap.

**Two environment gotchas burned time this session — heed them:**
- **Windows PYTHONPATH uses `;`, not `:`.** `PYTHONPATH=src:scripts` is ONE bogus
  entry → `aiutopia` not importable → `RLModule.from_checkpoint` silently returns
  a useless base `RLModule`. Use `PYTHONPATH=src` alone (a script in `scripts/`
  is auto on `sys.path`), or `src;scripts`.
- **Cold `RLModule.from_checkpoint` returns a base class** unless the concrete
  module is imported first. `import aiutopia.rl_module.role_rl_module` before
  loading (see `sim_rollout_check._load_module`).

---

**In progress (uncommitted at time of writing):** fidelity fix #2.
- The sim's HARVEST/NAVIGATE used a free, unlimited per-target walk budget, so one
  HARVEST chained the whole field → the optimal sim policy was **open-loop
  (HARVEST-only, never NAVIGATE)**, which stalls at 55/64 on layouts whose tail
  ends up >16 blocks away (seed 3).
- Fix (in `src/aiutopia/sim/skills.py`): a **shared 400-tick per-dispatch budget**
  for HARVEST *and* NAVIGATE (matches the wrapper's injected `timeout_ticks=400`,
  `wrapper.py` L154/L346). One sim dispatch now collects ~55–62 (seed 3 = 55,
  matching real exactly), so the policy *must* learn navigate-and-repeat.
- Plus **layout diversity** in training (`AiUtopiaSimEnv` `randomize_layout`, set
  in the sim `env_config`) so the policy sees nav-requiring layouts, not a
  single-seed overfit.
- A re-train under these changes was running when this was written
  (`Research/train-sim-v4.log`).

## The path to 3/3 (M1B fully solved via sim→real)

1. Confirm the re-trained policy actually NAVIGATEs (sim rollout: it should
   HARVEST a cluster, NAVIGATE to the tail, HARVEST again — not HARVEST-spam).
2. Re-transfer (`scripts/transfer_eval.py`, auto-finds the newest sim checkpoint;
   `TRANSFER_SEEDS=3 TRANSFER_WALL_CAP_S=600` runs just the failing seed fast).
   Seeds 1,2 should still pass; seed 3 should improve.
3. **Real-MC motor caveat (likely the last blocker):** the seed-3 diagnosis found
   the residual tail logs read `dy=+3` and even manual NAVIGATE+HARVEST didn't
   recover them — `HarvestSkill.findNearest`'s ground-preference (`dy∈[-2,+1]`) +
   REACH=4.5 may make them unreachable by the current motor. If seed 3 still
   fails after the nav-trained policy, this is a **Java-side motor fix**
   (larger vertical reach / a jump), `minecraft-modding-and-server-specialist`
   territory — NOT a sim issue. Confirm whether it's a real arena property or an
   artifact of the agent's post-walk position first.
4. If budget+diversity alone don't make the policy learn NAVIGATE (delayed-credit
   local optimum), add a **distance-to-nearest-log PBRS shaping** in
   `sim_env.step` (potential Φ = −W·dist, W≈0.05, training-only, policy-invariant)
   to make moving toward a log immediately rewarding.

## How to operate

```bash
# Train in sim (fast):
PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
  py -3.11 scripts/train.py --milestone M1 --max-iters 200 \
  --evaluation-interval 999 --num-env-runners 0 --backend sim

# Transfer-eval a sim checkpoint against real MC (instance-1 on port 25001):
PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
  py -3.11 scripts/transfer_eval.py            # full 3-seed gate
TRANSFER_SEEDS=3 TRANSFER_WALL_CAP_S=600 ... py -3.11 scripts/transfer_eval.py  # fast single-seed
```
Real MC needs a warm Fabric instance: `NUM_INSTANCES=1 JDK_HOME=... bash
scripts/launch-training-instances.sh` (instance-1, port 25001). Cold-start spawn
race: the first reset can strand the agent at world origin; it self-corrects on
the next reset (the harness re-resets).

## Key files
- Sim: `src/aiutopia/sim/{world,skills,obs_adapter,reward_adapter,sim_env}.py`
- Shared light embeds: `src/aiutopia/env/_embeds.py`
- Training: `src/aiutopia/train/{config.py (backend=sim), sim_env_factory.py}`, `scripts/train.py`
- Transfer: `scripts/transfer_eval.py` (gate harness), `scripts/transfer_probe.py`
- Diagnostics (throwaway, uncommitted): `scripts/n20_*.py`, `scripts/_seed3_*.py`
- Spec/plan: `docs/superpowers/specs/2026-05-29-gatherer-fast-sim-spike-design.md`,
  `docs/superpowers/plans/2026-05-29-gatherer-fast-sim-phase-a.md`
- Memory: `m1b-blocked-by-broken-task`, `ai-utopia-fast-sim-plan`

## Why this matters
The sim + the iterative sim→real fidelity loop (find gap in a real eval → fix the
sim cheaply → re-train in minutes → re-validate) is the capability that makes the
whole multi-role roadmap (builder/farmer/defender/explorer) tractable — each new
role is a sim module behind the same obs/action/reward contract, validated the
same way. M1B is the proof-of-concept; it's ~1 fidelity gap from done.
