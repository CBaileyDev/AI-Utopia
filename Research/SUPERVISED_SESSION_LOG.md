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
