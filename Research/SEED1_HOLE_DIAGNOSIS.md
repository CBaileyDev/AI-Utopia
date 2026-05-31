# seed_1 Gate Hole — Root Cause: HARVEST-spam crutch

## Symptom
Converged sim gatherer (return 127, iter 200) scores gate 2/3:
seed_2=64, seed_3=64, **seed_1=0**. Persistent across checkpoints, not noise.

## Trace (greedy policy, per-step skill + oak)
- seed_2: **1 step**, skill=HARVEST(1) → oak=64, terminates. Sim HARVEST grabs ALL
  reachable logs in one env-step.
- seed_1: **120+ steps**, skill=SEARCH(3) EVERY step, never HARVEST, oak=0.

## Root cause (verified)
Initial skill masks at spawn:
- seed_1: `[1,0,0,1,1,1]` → **HARVEST (idx1) MASKED** (no log in perception reach)
- seed_2: `[1,1,0,1,1,1]` → HARVEST valid (a log spawned within reach)

The policy learned a **HARVEST-spam strategy**: if a log is in immediate reach, HARVEST
one-shots all 64. It NEVER learned NAVIGATE(0)→HARVEST. When seed_1 spawns with no log
in reach, HARVEST is masked, and the greedy policy falls back to SEARCH(3) in place
forever — it never navigates to the 16 logs that exist elsewhere in the arena.

The sim source even names this: `sim_env.py` comments reference "the old HARVEST-spam
path" and "preserves the proven HARVEST-spam M1B/survival path" (harvest_perception_mask
defaults off → HARVEST always valid → spam works).

## Implication
- The 127-return "convergence" is largely a sim artifact: a too-generous HARVEST (64 in
  one step) + logs often pre-placed in reach. It is NOT genuine gather behavior.
- Gate 0.667 is real but the failing seed fails *completely* (0), exposing the crutch.
- This matches the fast-sim memory's "real HARVEST non-determinism" fidelity flag.

## Fix experiment (running)
Train with `harvest_perception_mask=True`: HARVEST is gated on actually perceiving a
reachable log, so the policy MUST learn NAVIGATE→HARVEST instead of spamming. Hypothesis:
perception-masked training yields a policy that generalizes across all 3 seeds (no free
in-reach win), even if peak return is lower / convergence slower.

Success = gate 3/3 (or much closer) with non-degenerate skill usage (NAVIGATE present,
not SEARCH-only). Failure = still can't navigate → deeper reward/curriculum work needed.
