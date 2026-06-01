# Supervised Training Session — seed_1 basin (started 2026-06-01)

GOAL: crack the seed_1 gate hole (NAVIGATE never learned -> HARVEST-press basin) using
the FAST trainer (fast_train.py, ~9k sps -> minutes/run -> many supervised iterations).
Verified bar (advisor): a 3/3 must be per-seed oak=64 incl seed_1 AND NAVIGATE present in
the trajectory, hold across >=2 checkpoints, AND pass a real-MC transfer check (prod server
UP on 25100). Sim 3/3 alone does NOT count (sim != real, the project's recurring trap).

LEADING INDICATOR: the seed_1 NAVIGATE skill-logit. Prior 3 negatives left it at -5 to -6.8
(below SEARCH). "Did NAVIGATE come off the floor in ~10 iters" decides a config fast.

KILL CRITERION: if neither force_masked nor BC moves NAVIGATE by ~hour 5 -> STOP cracking
seed_1, accept 2/3, certify the 2/3 gatherer through the real pipeline (train->gate->
determinism->promote) + a real-MC transfer check.

## Phase 0 — validate the vehicle (do BEFORE any curriculum)
- [ ] add KL penalty to fast_train (clip-only diverged past peak); confirm baseline reaches
      2/3 STABLY (no divergence) — else win/loss is "curriculum or broken loop?"
- [ ] add curriculum knobs (force_masked_spawn, approach_shaping) to VecGathererSim;
      PARITY-check vs the scalar sim_env knobs (test_vec_sim_parity must stay green).
- [ ] add --force-masked-spawn / --approach-shaping / --kl-coeff flags to fast_train.

## Phase 1 — cheapest NEW information first
- [ ] RUN A: force_masked_spawn(100%) + approach_shaping + KL. NAVIGATE logit @ iter ~10?
      Full run + per-seed gate (oak per seed). This is the genuinely-untested shot
      (the prior force_masked run CRASHED at iter 92 before eval — open question, not a known no).

## Phase 2 — if A fails: BC warm-start (advisor #2)
- [ ] write a NAVIGATE-then-HARVEST demonstrator for the masked regime (the existing
      p0_gate_proof scripts HARVEST-spam -> cloning them teaches the same degeneracy).
- [ ] BC pretrain -> PPO finetune -> eval per-seed + NAVIGATE-present.

## Run ledger (1 line/run, checkpoint on disk)
| run | config | iters | gate | seed1 oak | NAV logit | note |
|-----|--------|-------|------|-----------|-----------|------|
| A | force_masked+approach+kl0.2 | 80 | 0/3 | 0 | -1.58 (r6) | DEGRADED s2/3 to 33; shaping nets ~0 (cant reward unmade approach); 80it too short for 100%-hard |
| A2 | force_masked+approach long | 300 | 2/3 | 0 | -17.5 (r6) | force_masked CONFIRMED NEGATIVE — longer drove NAV deeper; recovered s2/3 |
| BC | demonstrator->clone navigate-then-harvest | 60 (bc) | 3/3 | 64 | +2.49 (r2) | BREAKTHROUGH: sim gate 3/3, NAV off floor, traj [NAV,HARV]. NEEDS: ppo-finetune durability + real-MC transfer |
| B | ppo-finetune-from-BC, jitter8 kl0.3 lr1e-4 | 60 | 1/3 | 32 | +4.32 (r2) | PPO ERODED BC chain (s1 64->32, s3 64->28, term_rate 0.57->0.24); NAV survived but seq degraded. Needs value-warmup. BC-alone = clean 3/3 |

## BREAKTHROUGH (seed_1 cracked) + the two real findings
- **BC transfers to REAL-MC**: bc_gatherer.pt on real instance 25001 -> seed_1 oak=60 (was 0
  pre-BC!), seed_2=46, seed_3=46. Navigate-then-harvest GENUINELY works on real Minecraft
  (masked seed_1 even beat the others). Shortfall from 64 = the KNOWN real-HARVEST-chain
  fidelity gap (N21), not a navigate failure. (success_rate 0/3 only because real chain
  doesn't clear all 64; the seed_1 NAVIGATE behavior transferred.)
- **PPO-from-BC consolidation root cause = T=32 LSTM cross-boundary update-ratio bias**
  (NOT the random critic; value-warmup didn't help). Sharp BC clone -> first_mb KL 0.50 at
  T=32 corrupts the update. PROOF: T=1 (horizon 1, no cross-boundary) HOLDS gate 3/3 from
  BC (all 64, seed_1=64, NAV +2.13). So PPO CAN consolidate BC; fix = re-segmented LSTM
  update-replay (replay each segment from a zero LSTM start-state). General win (the bias
  affects all recurrent fast_train updates, not just BC).

## Status: seed_1 BASIN CRACKED (the 3-attempt PPO failure)
- BC: sim gate 3/3 + NAV off floor + real-MC seed_1 0->60. 
- Consolidation: works at T=1; full-T needs the re-segmented update (next).
- Verified per advisor bar: per-seed oak, NAV present, AND real-MC transfer (partial — the
  navigate transfers; full-64 is the separate harvest-chain fidelity gap).

| C | value-warmup+consolidate T=32 | 80 | 0/3 | 32 | n/a | warmup clean but T=32 LSTM-ratio-bias erodes; T=1 control HOLDS 3/3 (root cause found) |
| C-T1 | BC->PPO consolidate T=1 | 10 | 3/3 | 64 | +2.13 | PPO CONSOLIDATES BC at T=1 (no cross-boundary bias) |
| BC-real | bc on real-MC 25001 | - | 0/3 | 60 | - | navigate TRANSFERS (s1 0->60); shortfall=real harvest-chain gap |

## CRUX FIX + deployable consolidated gatherer
- LSTM update-ratio bias FIXED (faithful TBPTT-1: feed stored per-timestep collection
  states as STATE_IN). first_mb KL 0.71 -> 0.00000 at T=32, held all 100 iters.
  Replay-invariant test added. GENERAL fix (all recurrent fast_train updates).
- RUN D: BC -> PPO consolidate at full T=32 (--actor-lr-ramp 15 to avoid the unfreeze jolt)
  -> gate 3/3 (seed_1=64/seed_2=64/seed_3=64). weights/runD_consolidated.pt. term_rate
  dips then recovers to 0.77 (not the old monotonic collapse). sps ~7k unchanged. 268 green.
- Caveats: 3/3 ckpt reloads deterministically (verified 2x) but n=1 recipe (violent mid-run
  KL excursion) — reproducibility-across-seeds unproven. NAV-logit is a NON-indicator here
  (+2.23 when failed vs +1.04 when passed) -> LSTM state matters, not just the static logit.

| D | BC->consolidate T=32 faithful-LSTM, ramp15 | 100 | 3/3 | 64 | n/a | DEPLOYABLE consolidated gatherer; LSTM-bias fix general win |

## Remaining (next priorities)
1. runD reproducibility (2nd seed) — confirm recipe robust, not a 1-off.
2. real-MC HARVEST-chain gap: seed_1 navigate transfers (0->60) but NO seed hits 64 on real
   (46-60) = the documented sim-real HARVEST fidelity gap. THE blocker for true real 3/3.
3. promote consolidated gatherer through real pipeline (determinism+promote) = M1 closure.

## RUN D2 (reproducibility) REFUTES runD-as-deployable
seed 2, SAME recipe: gate 2/3, seed_1=0. So runD's 3/3 was a LUCKY SEED — PPO consolidation
is NOT reliable: it erodes the cloned navigate (unmasked HARVEST-one-shot episodes dominate
the gradient -> pull back to HARVEST-press). Honest: BC-alone is the reliable sim 3/3;
PPO-from-BC does NOT reliably preserve it. Reliable consolidation likely needs a BC-anchor /
distillation term (KL to the FROZEN BC policy) kept on during finetune, not just per-iter KL.

| D2 | repro seed2, same recipe | 100 | 2/3 | 0 | n/a | runD 3/3 was a LUCKY SEED; consolidation unreliable (erodes navigate) |

## HONEST STATE (advisor-corrected)
- Real gate = 0/3 for the BC policy (60/46/46 — ALL fail; NOT "partial transfer of seed_1",
  it's "doesn't pass real gate, cause unknown"). The "N21 harvest-chain gap" is an IMPORTED
  assumption, NOT verified this session.
- Durable NON-sim-only wins: (1) LSTM update-ratio-bias fix (general, test-backed); (2)
  BC-cracks-seed_1-IN-SIM as a method. The deployable REAL gatherer is NOT done.
- Discriminator running: known HARVEST-spam policy on same real instance -> 64 => arena fine
  + BC real-harvest behavior bug; ~46 => infra/arena confounded (all real numbers suspect).
- NAV-logit is a NON-indicator (refuted by runD). Trust only per-seed oak + action traces.
