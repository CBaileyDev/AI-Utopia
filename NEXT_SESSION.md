# Next Session Handoff — Fast-Sim & Sim→Real Transfer (2026-05-29)

This supersedes the earlier post-v20 handoff. Read `PROJECT_CONTEXT.md` for the
big picture; this captures the current frontier.

## ⏩ START HERE — overnight 2026-05-30: oracle-ablation characterization (READ `Research/MORNING_BRIEF.md`)

A sim-only oracle ablation (4 cells × 3 seeds × 200 iters) measured how much of
the decision-core's blind-explore clearance is the POLICY vs the env oracle.
**Headline: a zero-learning scripted follower ≥ the trained PPO policy in EVERY
cell.** When PPO doesn't collapse it merely reproduces the follower; the
perception MASK is the single load-bearing crutch (remove it → greedy 0/5).

| cell | mask | cue | follower | policy (per seed) | gap |
|---|---|---|---|---|---|
| full | ON | ON | 5/5 | {5,0,5}=3.3 | −1.7 |
| mask_only | ON | off | 5/5 | {3,0,1}=1.3 | −3.7 |
| cue_only | off | ON | 5/5 | {0,0,0}=0.0 | −5.0 |
| neither | off | off | 5/5 | {0,0,0}=0.0 (oak=32, NAV=0) | −5.0 |

**Defensible:** follower ≥ policy every cell; full working seeds ARE the follower
(NAV=1, SUCCESS); PPO unstable (non-finite KL under kl_coeff=0.2; seeds collapse).
**Do NOT over-claim "PPO can't learn search" — UNTESTED + confounded:** (1) KL was
never set to 0 (training broken); (2) the follower clears even `neither` 5/5 with a
fixed −z heading → the eval arena doesn't REQUIRE directed search; (3) mask-off
cells were greedy-only (HARVEST-when-blind decode artifact). A KL=0 de-confound on
the full cell is running (`Research/dc_kl0_full.json`).

**DESIGN FORK (yours to call — see brief):** (a) thin reactive controller + put
intelligence in PRODUCERS (real Explorer emits the cue from partial info; keep the
mask); (b) genuine search learning — untested, needs a stabilized trainer AND a
search-requiring arena (fixed-heading must fail).

**External research (Kimi swarm, in `Research/okcomputer/`) CONVERGED on Fork A,
staged to B** — and corroborates the ablation (flat RL failing long-horizon
exploration = field consensus; the decision-core already = Plan4MC's Finding-skill;
the mask is the correct producer→controller interface, not a crutch). Highest-leverage
**UNGATED** next experiment (sim-only, no Java/deploy/promotion): re-run the
decision-core with the FULL stabilization recipe (kl=0 + **entropy-decay schedule** +
**pre-softmax FLOAT_MIN masking** + log-std clamp `[-5,2]`) + a real **Level-1 frontier
scout** (occupancy grid + WFD, score `size/(dist+1)`) replacing the oracle cue, gated
by the scripted-follower discriminator. NOTE our kl=0 de-confound already showed kl=0
ALONE is insufficient → the lever is entropy-annealing+masking, untested. Then RE3
(dim64,k=4) or `(x,z)` count bonus. Full synthesis + falsifiable criteria in
`Research/MORNING_BRIEF.md` §"External research synthesis". **Promotion deferred** (user-gated:
ambiguous checkpoint + unmeetable §5.10 + deploy-adjacent). **Phase D Java scoped,
NOT built** (`docs/.../2026-05-29-phase-d-decision-core-java-scoping.md`;
build/deploy/restart reserved for you).

Commits this run: `bf89c97` (AIUTOPIA_DATA_DIR), `7ad92c4` (decouple mask/cue),
`c48af03` (dc_ablation harness), `3c20f83` (Phase D scoping), + this handoff.

**⚡ 12h-autonomous-run update (2026-05-30) — scout built; "validated" WITHDRAWN.**
Built a partial-info `FrontierScout` + a decisive no-training follower test
(`scripts/dc_scout_follower.py`). Findings: (a) the old `clusters` arena was
DEGENERATE (cluster B always south — `_cluster_bases` dirs all dz=−1 — a fixed
heading cleared it 10/10, inflating ALL prior blind-explore results); (b) on a new
non-degenerate `clusters_omni` arena, the WFD scout got 14/24, a committed
open-loop `SweepScout` got 50/50, fixed-heading 21/50. The 50/50 was initially
committed (`c1083a9`) as "Fork-A VALIDATED" — **CORRECTED**: SweepScout ignores
perception (open-loop lawnmower) and bakes in B's radius, so the win is exhaustive
coverage tuned to the toy, NOT partial-info scouting. **Partial-info scouting (the
real Explorer question) remains UNTESTED**; a non-gameable test needs B at
variable/unknown radius + a steps-to-clear metric. See `Research/MORNING_BRIEF.md`
"UPDATE 2". Lesson (5th over-claim this session): the sim toy keeps rewarding
whatever exploits its structure — verify the metric isn't gameable before claiming.

---

## 🟡 M2 DECISION-CORE (N22, 2026-05-29) — MECHANISM works; policy learning UNVERIFIED

Executed the advisor/workflow-recommended **Option B pivot** (make the POLICY decide,
not the skill), sim-only. **The MECHANISM is sound** (pointer-MINE, skill demoted, no
chaining — keep it). **But the policy's learning contribution is ~zero and the gate
numbers measure the env scaffolding, not the network** (see discriminator below). The
"5/5 held-out" / "learned to explore" framing from earlier commits is OVER-CLAIMED.

⚠️ **Two caveats (advisor) — do NOT over-trust this:**
1. **The perception-based HARVEST mask is load-bearing**, not the policy alone: when
   nothing is visible the mask *forces* NAVIGATE, so the env (not a learned explore
   drive) decides "explore now"; the policy only has to not steer the one nav hop badly.
   The tell that the policy's own nav is WEAK: the OOD **trees=60/64 with NAVIGATE=985/
   MINE=15** — when the mask doesn't cleanly gate (trunks drift in/out of perception),
   it thrashes nav and strands a trunk. Honest claim: "decision-core + perception-mask
   clears the TRAINED layout greedily," not "the policy learned to explore."
2. **Single seed-1 run on ONE hand-built 2-cluster geometry** = demonstrated, not
   validated. The OOD trees result shows it's fit to that geometry. Harden before
   building on it: 2–3 training seeds + held-out cluster placements.

**Built + committed (`e3b8dbb`, `579af7d`):**
- **Decision-core**: `target_class` reinterpreted as an instance pointer (slot into
  `g_nearest_resources`); HARVEST demoted to mine ONLY the pointed trunk
  (`skills.mine_instance`) — no findNearest, no chaining. **Zero action-space /
  RLModule change** (the key simplification). `decision_core` config flag (default
  OFF; the proven survival-forest path is untouched).
- `obs_adapter.gatherer_nearest_columns`: single source of truth for both
  g_nearest_resources AND the pointer→world-column map (golden trace intact).
- **2-cluster blind-explore arena** (`world.py arena_mode="clusters"`, sim-only): 8
  trunks visible at spawn (cluster A), 8 beyond the 16-block perception (cluster B,
  ~22 b south) → forces ONE blind explore hop. `arena_half` widens the OOB box.
- PBRS distance-shaping re-added (training-only; legit now that the policy navigates).
- `m1_gatherer_config(decision_core=True)` + `train.py --decision-core`.
- Tools: `scripts/n21_decision_core_gonogo.py` (scripted demonstrator),
  `scripts/decision_core_rollout.py` (eval).

**🟡 RESULT — mechanism clears blind-explore arenas, but via env oracle (not learned).**
- v2 (fixed clusters, commit `6053556`): greedy clears 64/64 — MINE cluster A → ONE
  NAVIGATE hop (mask-forced when blind) → MINE cluster B.
- v4 (RANDOMIZED clusters + blind-only shaping, commit `f355c9c`): clears 64/64 across
  3 randomized geometries; **HELD-OUT (novel high-seed geometries) = 3/5** — the 2/5
  failures were the explore-DIRECTION (blind search, no directional signal → the policy
  learns a direction *prior* that thrashes for uncommon directions).
- v5/v7 (clusters + ground-truth bearing cue): greedy held-out 5/5.
- ⛔ **DISCRIMINATOR (the honest finding, `scripts/_dc_follower.py`): a SCRIPTED
  ZERO-LEARNING follower** — `HARVEST target_class=0 when the mask allows; else NAVIGATE
  toward the g_hostiles_nearby[0] cue` — **also clears 5/5, identical to the trained
  policy.** So PPO learned ~nothing here: the env does the deciding — the perception-mask
  picks WHEN (HARVEST if a trunk's visible, else NAVIGATE) and the GROUND-TRUTH bearing
  cue picks WHICH WAY. The "5/5" measures the oracle scaffolding, not the network.
- **Honest claim:** decision-core MECHANISM (pointer-MINE, no chaining) works; with a
  perfect oracle bearing + action mask, a reactive (scriptable) policy clears novel
  geometries. **NOT "Explorer→Lumberjack validated"** — the Explorer's HARD job is
  PRODUCING accurate bearings from partial info, which is exactly what's stubbed to
  ground-truth here. Validated the easy half, assumed the hard half.
- Caveats: cue is ground-truth oracle; mask is load-bearing; trees OOD (~60/64); single
  PPO seed; **sim-only — no real-MC transfer (the only real test) attempted.**
- ⛔ DO NOT train on "mixed" (trees+clusters) — trees dilutes the explore signal; mixed
  REGRESSED to held-out 0/5 even with the cue (v6). Train explore on clusters-only.

The earlier "needs BC, 4× confirmed" framing was WRONG — a greedy-eval artifact, not
a learning failure (the advisor caught it from training `ep_len≈90 < cap 300` =
episodes dying by OOB, which requires NAVIGATE). Three real bugs, all fixed:
1. **OOB-truncation punished exploration** (wander out → episode dies → policy learns
   NAVIGATE is fatal → greedy MINE-spam). Fix: CLAMP the agent at the arena wall.
2. **PBRS drip** (Φ=−W·dist ≤0 with γ<1 paid ~+0.02/step to stand still). Fix:
   distance-reduction form (zero inaction drip).
3. **Greedy decode ignored the action mask** so it picked masked HARVEST instead of
   NAVIGATE. Fix: `_greedy_decode` applies the skill mask; in decision_core the
   HARVEST mask is **perception-based** (valid if a trunk is VISIBLE, since MINE
   walks to it). → greedy MINEs visible trunks, NAVIGATEs only when blind.

Scripted go/no-go also PASSES both arenas; sampled eval 64/64. (Trees=60/64 is OOD —
the policy was trained on clusters, not the trees layout.)

**NEXT STEPS (the REAL open work — sim-variant tuning is exhausted; 7 runs, every "win"
came from the env doing more):**
1. **Decide what the policy must actually LEARN.** Right now the env (mask + oracle cue)
   does the deciding and a scriptable follower matches the net. Two honest paths:
   (a) accept the decision-core as a thin reactive controller and put the intelligence in
   the **producers** (the action-mask + a real Explorer/memory that emits bearings from
   PARTIAL info — the hard half); or (b) remove the oracle (no ground-truth cue, no
   forced mask) and make the policy genuinely learn search — which earlier needed BC.
   Pick deliberately; don't keep adding oracle assists that hide the question.
2. **Real-MC transfer (Phase D) is the only remaining real test of the MECHANISM** —
   Java HarvestSkill demoted to mine the pointed instance (`target_class` = k-th nearest
   column, matching GathererOverlayBuilder order) + perception mask; deploy + transfer.
   Watchable. (Intricate Java; a fresh-session chunk.)
3. **Real Explorer/Scout role** — produces the bearing cue from partial info (the hard
   half currently stubbed to ground-truth). Design-gated (brainstorm).
Spec/plan: `docs/superpowers/{specs,plans}/2026-05-29-gatherer-survival-forest-fidelity*`.

---

## ✅✅✅ SURVIVAL-FOREST FIDELITY MILESTONE COMPLETE (2026-05-29, N21)

The Lumberjack now harvests **real vertical oak trees at survival speed** and the
sim→real gate **PASSES 3/3 on the tree forest** — the fast-sim methodology proven
under genuine survival mechanics. Two increments, both transferred with **NO
retrain** (the HARVEST-spam policy is robust to the OOD tree obs):

- **Increment 1 — survival break-timing + stone axe** (`877789a`): `HarvestSkill`
  mines each oak log for 15 ticks (stone axe, ~0.75 s) instead of instant/creative;
  `WorldOps` equips a stone axe each episode. Real gate **3/3** (~23 s/scenario),
  deterministic. Spike-verified (`scripts/n21_breaktiming_determinism.py`).
- **Increment 2 — real capped-height bare oak trees** (`99e7b96`, `cec7d1b`,
  `12b248b`): flat 8×8 log grid → **16 vertical 4-tall bare oak trunks** (64 logs)
  on a 4×4 jittered grid, in both `WorldOps.java` and `sim/world.py` with
  byte-faithful `_JavaRandom` parity (**verified seed-for-seed on seeds 1/2/3**:
  16 identical trunk columns). Obs parity fixed (golden trace regenerated +
  un-skipped, 204 tests green): real `GathererOverlayBuilder` reports per-(x,z)-
  column **topmost log within the ±3 vertical scan** (4-tall trunk → dy+3=0.375,
  the Y69 crown is above the window, never seen) + the stone axe in inv slot 0
  (id 826). Real gate **3/3** on trees (~38 s/scenario), deterministic.

Reproduce: warm instance-1 (`NUM_INSTANCES=1 JDK_HOME=… bash
scripts/launch-training-instances.sh`), then `scripts/transfer_eval.py`. The
survival-forest jar is deployed to **all 12 instance dirs**.

### ⚠️ Known fidelity gaps (harmless now; bite at M2 — read before extending)
These are **masked only because the policy ignores the obs and the skill does all
the spatial/sequencing work** (so the gate validates skill+sim PARITY, not policy
learning). They become real the moment a future (M2) policy *consumes* the obs or
*decides* harvest sequencing:
1. **The sim still instant-breaks** (plan Task 3 was skipped — transfer passed
   without it). The sim does NOT model survival harvest *timing*; only the real
   `HarvestSkill` does. Sim episodes are shorter than real per dispatch.
2. **The sim agent moves in full 3D** (`_walk_into_reach` can float up toward a
   log) while the real fake player is **ground-bound**. They agree on *outcome*
   for ≤4-tall trunks ONLY because real reaches the top log by walking into the
   cleared column after breaking the base. A taller trunk (or a policy that
   reasons about reach) would expose the divergence — do NOT raise trunk height
   past 4 without a climb model + making the sim ground-bound.
Spec/plan: `docs/superpowers/specs/2026-05-29-gatherer-survival-forest-fidelity-design.md`,
`docs/superpowers/plans/2026-05-29-gatherer-survival-forest-fidelity.md`.

---

## ✅✅ M1B SOLVED — sim→real gate PASSES 3/3 (2026-05-29, N21)

A gatherer policy **trained entirely in the headless sim clears the REAL Minecraft
M1B gate on all three seeds** (64/64 oak_log each, ~8–9 s/scenario):

```
REAL m1_oak_log_seed_1: 64/64 ✓   seed_2: 64/64 ✓   seed_3: 64/64 ✓   →  3/3 = 100% PASS
```

Reproduce: instance-1 must be warm (`NUM_INSTANCES=1 JDK_HOME=… bash
scripts/launch-training-instances.sh`), then `PYTHONPATH=src AIUTOPIA_DATA_DIR=…
py -3.11 scripts/transfer_eval.py`. **NB the first reset on a freshly-launched
server hits the cold-start spawn race (seed_1 → 0/64 in 1 step); just re-run — it
clears on the warm pass.** The fast-sim + iterative fidelity-loop methodology is
now proven end-to-end; this is the template for every future role.

**What closed it (full arc, N21):** the bounded-skills detour (v4) and three
reward-tuning re-trains (v5 PBRS / v6 +penalty / v7 +completion-bonus) were all
dead ends — the policy is *correctly* HARVEST-only (real HARVEST chains
internally), and the real blocker was a **second 16-block search-radius limit**
in `HarvestSkill` (`MAX_SEARCH_RADIUS`), distinct from the obs scan. Fix = a
symmetric `16→48` one-liner in `HarvestSkill.java:48` + `sim/skills.py:58`
(commit `57894a9`); jar rebuilt + redeployed to all 12 instance dirs; **no
re-train needed**. The reverted real-faithful sim (commit `d4d3985`) + this radius
fix = the whole solution. Two earlier "blockers" were misreads: the "dy=+3 motor
limit" (logs are all Y=66=dy+1) and the "obs blindness blocks navigation" (true,
but nav was never needed — chaining does the work).

**Server state:** instance-1 is RUNNING (MC 25566 / Py4J 25001) with the new jar.
The other 11 instance dirs have the new jar deployed but are stopped.

## Next frontier (M1B is done — pick up here)
- **Promote the gatherer weights** (the §5.10 checklist / `aiutopia.cli.promote_weights`)
  and tag M1B verified (M1A_PIPELINE_PLAN T22 / M0_PROGRESS).
- **Scale-out for future roles:** see `PROJECT_CONTEXT.md §9.1` (added this session) —
  the bottleneck is now RLlib framework+policy overhead, not the sim or GPU; the
  next throughput unlock is the Phase-D JAX vectorized sim (~200–500×), worth
  building only when M2+ multi-role load can consume it. For M1B, no new hardware
  is warranted.
- **M2 / new roles** (Lumberjack→Miner/Farmer/Soldier/Scout): each is a new sim
  module behind the same obs/action/reward contract, validated by the same
  sim→real transfer loop just proven here.

---

## (Historical) TL;DR — the arc that led to the 3/3 pass

We pivoted from grinding M1B on real Minecraft (~0.9 env-steps/s) to a **fast
headless sim** (the RLGym/RocketSim + Craftax pattern). The sim trains the gatherer
to convergence in **~90 s (≈170–250× real-MC)**; the sim→real transfer went
2/3 → **3/3** once the `MAX_SEARCH_RADIUS` skill-search limit was fixed (above).

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

## ⛔ UPDATE 2026-05-29 (N21 FINAL) — TRUE root cause: OBSERVATION BLINDNESS. STOP reward tuning. DESIGN DECISION NEEDED.

Three reward-engineering re-trains (v5 PBRS shaping, v6 +failure_penalty, v7
+completion_bonus +entropy) **all produced the identical HARVEST-only greedy
policy** (seed1=64 ✓, seed2=62, seed3=55, `end=TICK_LIMIT`, NAVIGATE=0). That
invariance was the tell. The real cause, confirmed directly:

**At the stall the observation is ALL ZEROS.** Probed seed2/seed3 stall states:
`g_nearest_resources` all-zero, `g_resource_grid` all-zero, `g_richness_score=0`
— the tail logs (19.5 b / 16.1 b away) are **invisible to the policy**. Cause:
`obs_adapter.py:47-48` `GRID_RADIUS = SCAN_RADIUS = 16`, and the docstring says
this **mirrors the real Java `GathererOverlayBuilder` (golden-trace validated)**.
So **both sim and real are blind beyond 16 blocks.** You cannot reward a policy
to navigate to something it cannot perceive — v5/v6/v7 were doomed a priori.

**This also means the bounded-skills change (v4, `624da53`) was a wrong turn.**
The REAL HarvestSkill *internally chains* (it re-scans + walks as it runs, so the
SKILL sees logs the policy's obs can't), clearing the whole field in ONE dispatch
on easy layouts — that's how the original transfer hit 2/3 (seeds 1,2 = 64/64,
real). v4 bounded HARVEST to one 400-tick dispatch to "force navigation," which
(a) the obs can't support, and (b) doesn't match how real MC clears the field.
The real seed3 residual (55/64) is the **Java HarvestSkill motor/reach limit**
(the `dy=+3` tail + ground-preference `dy∈[-2,+1]`, REACH 4.5) — a skill/motor
gap, NOT a policy navigation gap. NB: the sim's flat-y world doesn't even model
that dy limit, so sim seed3 can reach 64 via forced nav while real cannot — a
separate fidelity gap.

### DESIGN FORK — needs a human call (do NOT auto-spend on this)
- **(A) Real-faithful path — CHOSEN, in progress.** Revert the bounded-skills
  change → sim HARVEST chains like real (clears the field in one dispatch). DONE
  (`d4d3985`). Accept the policy is HARVEST-only (correct — that's what real does).
  The 3/3 closer is now PINNED (modding investigation): seed3's residual is a
  **second 16-block search-radius limit** — `HarvestSkill.MAX_SEARCH_RADIUS = 16`
  (`HarvestSkill.java:48`), the SKILL's own chaining-scan radius (distinct from the
  obs `SCAN_RADIUS`). After chaining through the field the agent rests and the
  remaining tail logs are >16 b away → `findNearest` returns empty → chaining
  halts. NOT a motor/reach limit: all logs sit at Y=66 = dy=+1 (the earlier
  "dy=+3" was a stale-arena misread); `REACH=4.5` and the `dy∈[-2,+1]` band are
  not implicated. **Fix = a symmetric one-liner**: `MAX_SEARCH_RADIUS` 16→48 in
  BOTH `HarvestSkill.java:48` and `src/aiutopia/sim/skills.py:58` (48 ≈ arena
  diagonal). Verified in-sim: all 3 seeds → 64/64. No test/golden-trace pins the
  constant (obs `SCAN_RADIUS`/`GRID_RADIUS=16` stay unchanged). Requires a mod
  rebuild + redeploy to all 4 instances (Fabric won't hot-reload) + a live cap=64
  HARVEST probe on seed3 to confirm 64/64 (catches a possible real-only collision
  residual: straight-line NAVIGATE rams intact logs, but HARVEST's nearest-first
  chain + the `STALL_TICK_BUDGET=20` watchdog should self-heal). **This Java
  rebuild+restart is the step to clear with the user before executing.**
- **(B) Navigation path.** Widen the observation (Java `GathererOverlayBuilder`
  AND `obs_adapter` to ±24 arena) so the policy can see + navigate to distant
  resources; keep bounded skills; re-validate golden trace; re-train. Bigger
  change, alters the obs contract — but the wider obs is exactly what the future
  **Explorer/Scout** role needs, so it may be worth doing once, deliberately.
- **(C) SEARCH-based blind exploration** when obs is empty — most complex, defer.

### State on `origin/main` (all committed + pushed, `851aff7`)
- `scripts/sim_rollout_check.py` — greedy rollout w/ skill histogram + `end=`.
- Sim has `distance_shaping`/`failure_penalty`/`completion_bonus` flags (all
  training-only, eval-neutral). **They are harmless but inert given the obs
  blindness** — leave them off (or remove) under path (A). Config currently has
  them ON for the sim backend.
- v5/v6/v7 checkpoints under `runs/aiutopia_M1_seed1/PPO_aiutopia_sim_*`.

### Two env gotchas that burned time (heed)
- **Windows PYTHONPATH uses `;` not `:`** — `src:scripts` is one bogus entry →
  `aiutopia` unimportable → `RLModule.from_checkpoint` silently returns a base
  class. Use `PYTHONPATH=src` (script dir auto on `sys.path`).
- **Cold `from_checkpoint` returns a base RLModule** unless the concrete module
  is imported first (`import aiutopia.rl_module.role_rl_module`).

---

## UPDATE 2026-05-29 (N21) — gap #2 root cause CONFIRMED, fix re-training
(SUPERSEDED by the FINAL update above — the "policy-learning gap" framing was
correct that nav is needed, but missed that the obs makes nav unlearnable.)

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

**⚠️ KNOWN RISK in v6 — the failure_penalty may backfire (read v6 carefully).**
`oak_log` is worth **+1.0/log, no terminal completion bonus** (`reward.py:16`),
so the tail is worth only ~+2 (seed2). But `failure_penalty=0.5` per step accrues
to truncation (~0.5 × ~298 ≈ **−150/episode**). The cheapest escape from that is a
single large NAVIGATE **out of the ±24 arena** (`MAX_NAV_RANGE=32` exits in one
dispatch → OOB truncation → penalty stops) — strictly easier to discover than
"NAVIGATE to the exact tail + follow-up HARVEST". So v6 may converge to *grab the
near cluster, then walk off the edge*. `sim_rollout_check.py` now prints an
`end=` field — **read it**:
  - `end=OOB_ESCAPE@<n>` (short episodes) → escape optimum confirmed. Fix:
    **drop `failure_penalty` to ~0.05 AND add an OOB penalty** (or only penalize
    the first failed dispatch per stall), NOT more exploration.
  - `end=TICK_LIMIT`, still 100% HARVEST → no discovery → raise `entropy_coeff`
    and/or the shaping weight (`_SHAPING_W`).
  - `NAVIGATE>0` but wanders, never clears → direction-learning → shaping weight.
**Cleanest alternative to the penalty entirely:** a **terminal completion bonus**
for reaching 64 makes the tail worth a lot (so navigate-to-collect dominates and
escape is never tempting) — this is likely the more robust fix than tuning a
per-step penalty. Consider it first next cycle. A second full re-train is a
**user check-in point**, not autonomous spend.

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
