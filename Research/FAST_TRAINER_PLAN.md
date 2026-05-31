# Super-Fast Trainer — measured plan (path A: vectorized numpy + lean PPO)

## Why (measured 2026-05-31)
- Pure scalar sim: **6,270 steps/s** single env.
- Same sim under RLlib (Windows, num_learners=0): **~190 steps/s** — a **33x framework
  tax** (connectors, per-env serialization, single-core learner). RLlib overhead, NOT the
  sim, is the ceiling. (Confirms ai-utopia-fast-sim-plan memory.)
- Step profile: `gatherer_nearest_columns` (topmost-column obs scan) = **65%** of step
  time; skill dynamics only 13%. So the obs build is the in-sim hot path.

## Landed this session
- `src/aiutopia/sim/vec_obs.py::gatherer_nearest_columns_batched` — batches the hot-path
  scan over B envs (scatter-max for topmost-per-column, integer composite key for the
  exact (distSq,dx,dz) top-8 sort). **Parity-locked** per-env vs the scalar fn
  (tests/unit/test_vec_obs_parity.py): grid, nearest8, nearest_dist, richness all equal.
- Measured: ~3x faster than scalar-loop, ~40k env-obs/s (flat across B=256..4096; bound by
  numpy argsort/scatter, not fixed overhead).

## Build sequence (next)
1. **VecGathererSim** (`vec_sim.py`): batched (B,...) SimWorld state (agent_pos (B,3),
   logs (B,n,3), log_alive (B,n), inventory (B,n_items) counts, tick (B,)). Vectorized
   `reset(seeds)` (reuse _JavaRandom layout per env), `step(actions: dict of (B,...))`:
   - vectorize apply_skill: NAVIGATE (B vectorized walk), HARVEST (batched nearest-alive +
     walk-into-reach + inventory increment, capped), SEARCH/WAIT/DEPOSIT no-ops.
   - build obs via vec_obs + batched scalar fields (constants broadcast).
   - batched reward (vectorize _compute_reward_stage_1_gatherer over B), term (goal 64),
     trunc (tick>=max OR arena OOB).
   - PARITY GATE: VecGathererSim B envs == scalar AiUtopiaSimEnv per-env over T steps
     (obs+reward+term+trunc), same seeds/actions. This is the correctness contract.
2. **Lean PPO loop** (`scripts/fast_train.py`): no Ray. B parallel envs, one RLModule
   forward over (B,obs) per step (reuse AiUtopiaRoleRLModule — already (B,T)-capable),
   collect (B,T) rollout, GAE, PPO epochs, torch optimizer. Target: ~40k env-steps/s
   (B=512), ~200x over RLlib's 190 — matches the memory's realistic projection.
3. **Validate**: fast-trained policy clears the same 3 M1 gate scenarios (scenario_runner)
   and, ultimately, the real-MC gate (the success criterion — speed is not the metric).

## Risks / notes
- sim<->real parity stays sacred: vec_obs is parity-tested; vec_sim must be too. Reward/
  spaces are reused unchanged.
- obs op is ~40k/s; if it becomes the loop bottleneck, swap argsort->argpartition(8) and
  replace maximum.at. Deferred (not the current ceiling).
- JAX end-to-end (path B) remains the higher ceiling for a later phase; numpy path first
  for parity-checkability + zero new deps.

---
## RESULT (measured 2026-05-31) — bottleneck chain fully worked out: 130 -> ~9,000 sps (69x)

Sequence of bottlenecks attacked + fixed, each measured + parity-locked:
1. RLlib framework tax (33x, ~65% of iter) -> lean no-Ray PPO loop. ~40-50x.
2. Throughput REGIME COLLAPSE (6.7k->1.5k as policy learns HARVEST; collect 1.7->11s)
   -> vectorized skill dynamics (vec_skills, closed-form walk + batched HARVEST chain,
   parity 8e-13). HARVEST skill-advance 20.9x; collapse gone, stable.
3. Per-step PYTHON overhead (list-of-SimWorld re-stacked ~6x/step; _autoreset double
   obs-build) -> batched-array state + subset autoreset + argsort->argpartition. 1.37x.
4. obs hot-path scatter (np.maximum.at) -> 7-level buffered scatter-max. obs op ~2x;
   END-TO-END FLAT (obs no longer binding).

FINAL: B=512 T=32 median ~9,000 env-steps/s = 69x over RLlib's 130/s. Per-iter
collect ~1.07s / update ~0.73s. Parity: 8 gate tests + 249 suite green; learning intact
(term_rate ->0.66 by iter 23, gate 2/3).

## NOW BALANCED REAL COMPUTE — no framework-tax hotspot left
- collect 1.07s: 32 SEQUENTIAL per-step LSTM forwards + action sampling + numpy<->GPU
  transfers. The sequential T-loop is INHERENT to on-policy recurrent RL (LSTM state +
  action feedback can't be batched over T during collection). Algorithmic floor.
- update 0.73s: backward (~0.34s real GPU) + .to transfers (~0.36s) + per-minibatch obs
  stacking listcomp (~0.44s, trainer-side).

## Remaining gains = diminishing micro-opts (not bottleneck removal)
- Trainer: pre-transfer the whole rollout to GPU once (kill per-step/per-mb .to); vectorize
  minibatch obs stacking. Est ~1.1-1.3x. Bounded.
- Bigger B amortizes the per-step forward (GPU underutilized at B=512) -> try B=1024/2048.
- Shorter seq-len or fp16 for the LSTM update. Diminishing.
- True ceiling jump = JAX end-to-end (path B): sim+policy+PPO jitted, 1000s envs on GPU,
  removes the per-step python dispatch entirely. Separate large effort; numpy path has
  delivered 69x and meets the measured "needed" (~3-5k for experiment iteration; ~10k for
  MARL is in reach via larger B).
