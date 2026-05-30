# Morning Brief — 2026-05-30 (overnight autonomous run)

**For: the design call that's yours to make.** This summarizes an overnight,
sim-only characterization of the M2 decision-core. No live MC was touched; no
weights promoted; no Java built/deployed (all of that is reserved for you).

---

## TL;DR

- The decision-core's blind-explore clearance is the **env oracle**, not the
  network. Across a 4-cell oracle ablation (×3 seeds, 200 iters each), a
  **zero-learning scripted follower beats or ties the trained PPO policy in every
  cell**. When PPO doesn't collapse, it merely *reproduces* the follower.
- **The perception mask is the single load-bearing crutch.** Remove it and the
  greedy policy goes to **0/5** regardless of the bearing cue.
- **PPO training is unstable here** — non-finite-KL warnings in every run; some
  seeds collapse to "never explore" (0/5).
- **I did NOT conclude "PPO can't learn search."** That would be confounded (see
  *What's untested*). The honest status of fork (b) is **untested**.
- Net: this is a measurement, banked honestly. The design fork below is yours.

---

## What ran overnight (all committed + pushed)

| commit | what |
|---|---|
| `bf89c97` | `AIUTOPIA_DATA_DIR` — relocate volatile WAL SQLite off OneDrive (the at-risk uncommitted config fix). |
| `7ad92c4` | Decouple the perception-mask from the bearing-cue in the sim (back-compat; 204 tests + golden trace green) so the two crutches can be ablated independently. |
| `c48af03` | `scripts/dc_ablation.py` — the oracle-ablation harness (trains each cell, greedy held-out eval, per-cell scripted-follower baseline). |
| `3c20f83` | `docs/.../2026-05-29-phase-d-decision-core-java-scoping.md` — READ-ONLY Java scoping for the real-MC mirror (not implemented; build = your call). |

Artifacts: `Research/dc_ablation_results.json` (+ `dc_ablation.log`).

---

## The ablation result (the headline data)

decision_core is ON in all cells (it's the mechanism under test). The two oracle
crutches are toggled: **mask** = perception HARVEST mask (forces NAVIGATE when
blind → tells the policy *WHEN*); **cue** = ground-truth bearing in
`g_hostiles_nearby[0]` (tells it *WHICH WAY*). 5 held-out novel cluster
geometries; "policy" = greedy clearance averaged over 3 training seeds.

| cell | mask | cue | scripted follower | trained policy (per seed) | mean | **gap (policy − follower)** |
|---|---|---|---|---|---|---|
| full | ON | ON | **5/5** | {5, 0, 5} | 3.3/5 | **−1.7** |
| mask_only | ON | off | **5/5** | {3, 0, 1} | 1.3/5 | **−3.7** |
| cue_only | off | ON | **5/5** | {0, 0, 0} | 0.0/5 | **−5.0** |
| neither | off | off | **5/5** | {0, 0, 0} | 0.0/5 | **−5.0** |

Reading it:
- **Policy ≤ follower in every cell.** The network adds no value over a reactive
  script; often less, because PPO collapses on some seeds.
- **The "working" seeds ARE the follower.** full s1/s3 cleared with exactly
  `NAVIGATE=1` per episode, all SUCCESS — i.e. mine cluster A, take the one
  mask-forced explore hop toward the cue, mine cluster B. Identical to the
  scripted follower's policy. full s2 collapsed to `NAVIGATE=0` / all TICK_LIMIT
  (never explores → 0/5).
- **Mask off → total collapse (cue_only, neither = 0/5).** Without the mask the
  greedy policy is never reliably steered off HARVEST-when-blind. `neither` is
  clean evidence: every held-out seed gets **oak=32 with NAVIGATE=0** — mines
  the visible cluster A, never explores to B. (But see confound #3 — greedy-only.)

---

## What's defensible vs what's untested (read before deciding)

**Defensible (the bankable findings):**
1. A reactive, zero-learning follower ≥ the trained policy in every cell.
2. The full-cell working seeds reproduce the follower exactly (NAVIGATE=1, SUCCESS).
3. PPO does not converge on this task. The failing seed (full s2) oscillates its
   return wildly (−23…+40) across all 200 iters with **persistently high entropy
   (~12, no decay)** and small finite per-iter KL — i.e. NOT an entropy collapse
   and NOT a KL blowup, but a trainer with no stable gradient toward a
   better-than-reactive policy. (The intermittent non-finite-KL *warning* is a
   masked-zero-prob-action symptom, not the failure mode.) Corollary: greedy
   (argmax) eval of such a high-entropy non-converged policy is unreliable — a
   sampled-eval control (confound #3) would be fairer.

**Untested — do NOT read the ablation as "PPO can't learn search":**
1. **Training is broken at a level deeper than kl_coeff.** Non-finite KL fires in
   every run; some seeds collapse to never-explore (0/5). I ran the de-confound
   (full cell at kl_coeff=0): it reproduced the baseline {5,0,5} *exactly* and
   STILL threw non-finite-KL — so the warning's stock remedy doesn't fix it. A
   genuinely stabilized trainer (entropy bump / masked-comm-head `log_std` fix /
   the deterministic-env zero-prob cause) is the real prerequisite — **untested**.
   Until that, collapses may be instability, not a learning ceiling.
2. **The eval arena doesn't require search.** The follower clears even the
   no-cue `neither` cell 5/5 with a *fixed −z heading*. If a fixed heading
   solves it, there's no directed search to learn — so the no-cue cells can't
   prove the policy "failed to learn search."
3. **Mask-off cells were greedy-only.** Greedy + no mask = HARVEST-when-blind,
   the exact greedy-decode artifact fixed earlier with mask-aware decode. No
   sampled-eval control was run, so 0/5 there is partly a decode artifact.

So: "whether a **stabilized-KL** PPO on a **search-requiring** arena could learn
search" is **open**. This run neither had a stable trainer nor a search-requiring
arena.

---

## The design fork (yours to call)

Both are legitimate; the data above supports presenting (a), not killing (b).

**(a) Thin reactive controller + smart producers.** Accept the decision-core as a
reactive controller (it works *for this task* — a script suffices) and put the
intelligence in the **producers**: keep the load-bearing perception mask, and
build a **real Explorer/Scout** that emits the bearing cue from *partial* info
(the hard half currently stubbed to ground-truth). The Lumberjack stays simple;
the village's "find the forest" intelligence lives in a dedicated role + memory.

**(b) Genuine search learning.** Make the policy itself learn to search with no
oracle. This run does *not* show this is impossible — it shows it's untested.
Testing it needs **both**: a stabilized trainer (kl_coeff=0 or entropy bump —
fix the non-finite KL first) **and** an arena where a fixed-heading script
*fails* (randomized B direction with a real coverage requirement). Likely also
benefits from BC warm-start, but that's a hypothesis, not a finding. Arena
geometry is a design choice — your call, not an unattended one.

**My recommendation (weak):** for the *immediate* Lumberjack, (a) — a reactive
controller fed by a real producer is the lower-risk path to a working village
role, and the mask+cue already define the producer interface. But (b) is the
more interesting research question and is genuinely open; if you want to know
whether these agents can *learn* to search (vs be scripted to), it's worth the
stabilized-trainer + harder-arena experiment. Pick based on whether the near-term
goal is "a working village" (a) or "agents that genuinely learn" (b).

---

## External research synthesis (Kimi swarm, 2026-05-30) — converges on Fork A, staged to B

You ran a research swarm; output is in `Research/okcomputer/` (final report
`ai_utopia_exploration_report.agent.final.md`, 11 dimension deep-dives under
`research/`, synthesis in `…_sec04.md`, insights in `research/ai_utopia_insight.md`).
It independently lands on **Fork A with a staged path to B** — and corroborates the
ablation rather than contradicting it. Key points (their confidence in brackets):

- **The ablation validates the architecture, doesn't condemn it [High].** "Flat RL
  underperforms scripted/hybrid controllers on long-horizon sparse-reward
  navigation" is the established field consensus (MineRL BASALT winners, DreamerV3,
  Plan4MC). We rediscovered a known limit, not a novel failure — the solution space
  is mapped.
- **The decision-core already IS the right mechanism [High].** It matches Plan4MC's
  "Finding-skill" (a high-level policy that emits a goal for a low-level
  controller). It needs a *producer* + a *stabilized trainer*, not a redesign. The
  perception mask is the **correct producer→controller interface**, not a crutch —
  drive it from a real scout's bearings instead of the hardcoded oracle.
- **Field consensus (5 dimensions converge) [High]:** high-level goal producer →
  mid-level bearing/waypoint → low-level reactive controller. Fork A = that
  consensus.
- **Fork B / BC-from-the-follower is weak [Medium]:** BC from the scripted follower
  just memorizes the follower (matches our own discriminator + the advisor's
  "cloning the oracle"). Genuine end-to-end search needs intrinsic motivation +
  hierarchy or real human demos — a real scope increase, and learned exploration is
  fragile sim→real.
- **Staged hybrid is highest-ROI [Medium]:** build Fork A now; its explorer
  trajectories become Fork B's BC warm-start data later. Forks aren't exclusive.

**Concrete staged plan they recommend** (sim-only, fast-sim-friendly):
1. *Week 1–2 — stabilize PPO + Level-1 scout.* `kl_coeff=0`, **mask logits
   pre-softmax (FLOAT_MIN)**, **entropy-decay schedule `[[0,0.01],[1e6,0.001]]`**,
   log-std clamp `[-5,2]`, `grad_clip=0.5`. Scout = 2D occupancy grid (sparse
   hash-map) + Wavefront Frontier Detection, scoring frontiers `size/(dist+1)`,
   emitting the top centroid as a bearing (NO learning — clean sim→real).
2. *Week 3–6 — light exploration bonus + collect demos.* RE3 (fixed random encoder
   dim 64, k=4, ~1.05× overhead) or `(x,z)` count `β=0.01/√N`. Both trivial, proven
   on MiniGrid.
3. *Week 6+ — BC→PPO* (two-phase critic warmup) toward Fork B, only once Fork A has
   produced rich-coverage trajectories.

**Falsifiable success criteria (theirs, good):** Fork A wins if stabilized PPO +
real frontier bearings trains stably (finite KL, no seed collapse) and **exceeds the
scripted-follower baseline** on held-out geometries. Fork B wins if RE3+stabilized
PPO on a *search-requiring* arena (fixed-heading must fail) beats the follower on
3+ seeds. If Fork A can't beat the follower even with a real scout, the arena may
lack learnable structure.

### ⚠️ Reconciliation with our own data (don't take the recipe at face value)
The research bullet "PPO instability is a 3-line fix: just `kl_coeff=0`" is **too
strong, and we already half-falsified it**: the kl=0 de-confound reproduced the
{5,0,5} collapse *exactly* (§ KL de-confound above). BUT we only tested the *kl*
piece — NOT the entropy-decay schedule or pre-softmax masking. And our measured
failure mode (return oscillates, **entropy pinned ~12, never anneals**) is precisely
what an entropy-decay schedule + proper masking address. So data and research are
consistent: the lever is **entropy annealing + masking, not the KL term**. The honest
first experiment is the **FULL recipe**, watching whether entropy actually decays and
return converges — not just whether kl=0 was set. Clean, cheap, sim-only; does NOT
need the user-gated Java/deploy/promotion.

### Updated recommendation (research + our data now agree)
Lean **Fork A, staged to B** — strengthened from "weak" to corroborated by external
consensus *and* consistent with our ablation. Highest-leverage, fully **ungated**
next experiment: re-run the decision-core with the **full** stabilization recipe
(kl=0 + entropy-decay + pre-softmax masking + log-std clamp) and a real Level-1
frontier scout replacing the oracle cue, then gate it through the scripted-follower
discriminator on held-out geometries. Still your call — but the evidence converged.

---

## Two queued items, both deferred to you (not done unattended)

**Promotion / M1B-verified tag (§5.10) — DEFERRED, user-gated.** Findings:
- The M1B-verified policy is the newest **non**-decision-core run
  (`runs/.../PPO_aiutopia_sim_98045_..._13-55-35`); the 17:31+ runs are the M2
  experiment and must NOT be promoted as M1B.
- The §5.10 checklist can't pass as-is: that checkpoint has no/empty
  `aiutopia_metrics.json`, and Gate 5 (determinism) needs the GPU weights-replay
  harness (train.py writes an `l2=999` placeholder that fails the gate).
- `promote_weights` mutates `identity.db` + is deploy-adjacent. Correct path:
  confirm the M1B checkpoint, re-verify it live (instance-1), run the determinism
  harness for Gate 5, then promote. Left for you.

**Phase D (real-MC decision-core mirror) — SCOPED, not built.** Full scoping in
`docs/superpowers/specs/2026-05-29-phase-d-decision-core-java-scoping.md`. Key
points: mine the k-th nearest column via a **shared-method extraction** from
`GathererOverlayBuilder` (parity by construction); the clusters arena needs a
**forceload+grass widen** (the single most likely "works-in-sim, blank-in-MC"
bug); the perception mask is a Python wrapper toggle (≈ zero Java). **Build +
deploy + restart is the step you reserved — not done.** Also: post the
follower finding, real-MC transfer tests *fidelity*, not the open learning
question — so weigh it against (b) before spending the build.

---

## Repro / pointers

```bash
# the ablation matrix (sim-only):
PYTHONPATH=src AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
  AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
  DC_ITERS=200 DC_SEEDS=1,2,3 py -3.11 scripts/dc_ablation.py
# optional KL de-confound (full cell only): add DC_KL_COEFF=0 DC_ABLATIONS=full
```

- Discriminator (zero-learning follower): `scripts/_dc_follower.py`.
- Sim env knobs added: `harvest_perception_mask`, `harvest_mask_off` (sim_env.py);
  `DC_KL_COEFF` knob in the harness.
- **KL de-confound (DONE):** full cell ×3 seeds at `kl_coeff=0`
  (`Research/dc_kl0_full.json`) → **{5, 0, 5} = identical to the kl=0.2 baseline**,
  same seed-2 collapse (NAV=0), and **3 non-finite-KL warnings still appeared**.
  ⇒ **kl_coeff is NOT the instability driver** — the warning's own suggested
  remedy (kl=0) does not fix the collapse. So claim (iii) sharpens: the instability
  is *deeper* than the KL term (candidates: `entropy_coeff`, the masked-comm-head
  `log_std`, or the deterministic-env near-zero-prob action the warning describes).
  This does NOT change confound #1's verdict — a *properly stabilized* trainer
  (the real fix, not just kl=0) on a search-requiring arena is still untested. It
  does mean "just set kl=0" is not that fix.
- **Entropy probe (full s2, the collapsing seed, to 200 iters):** entropy stays
  **~12 flat** (13.9→12.1, no decay), per-iter KL stays small/finite (~0.004–0.02),
  return **oscillates −23…+40 and never converges**. ⇒ the 0/5 is non-convergence +
  greedy-mode misalignment on a still-stochastic policy — NOT entropy collapse, NOT
  KL. So the honest lead for fork (b) is "the trainer never finds a stable
  better-than-reactive gradient here" — which points at the reward/exploration
  signal + a search-requiring arena, NOT at kl/entropy hyperparameter tweaks alone.
  (Probe was a single diagnostic run, not a sweep.)
