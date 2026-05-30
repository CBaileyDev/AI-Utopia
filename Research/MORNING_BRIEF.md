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
3. PPO is unstable under the current config (`kl_coeff=0.2`): non-finite KL in
   every run, reproducible, with seed-level collapses.

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
