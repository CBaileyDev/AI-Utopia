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
