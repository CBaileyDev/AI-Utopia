# Training Architecture Review — measured 2026-05-31

## Current speed (sim backend, 4 runners, Windows num_learners=0)
5.89 s/iter for 768 steps = **130 env-steps/s**.

## Per-iteration decomposition (avg last 50 iters of a real run)
| Component | Time | % | Real work vs framework tax |
|---|---|---|---|
| learner_update_timer | 3.96s | 67% | 0.84s REAL gradient (measured raw fwd+bwd 96x32 LSTM = 21ms x40) + 1.27s batch-prep connector (0.62s LSTM time-dim zero-pad) + ~1.85s RLlib minibatch machinery |
| env_runner_sampling_timer | 1.89s | 32% | env_step 0.77s (1ms x768) + inference ~0.5s + ~0.6s connector tax |
| sim env_step | ~1ms/step | - | negligible |

## Weakest point: RLlib framework tax (not model, not sim)
~3.5-4s of every 5.89s iter (~65%) is framework: connectors, time-dim zero-pad,
per-minibatch Python machinery, all single-process on the driver (Windows num_learners=0).
Actual ML work (sim + LSTM gradient) ~1.5-2s. GPU compute is only 0.84s but RLlib wraps
it in 3.96s.

## How fast needed
- M1 gatherer convergence ~30 iters ~23k steps -> ~3 min at 130/s. Already fine.
- Curriculum / experiment iteration (200-500 iter runs <2 min): ~3-5k steps/s (~30x).
- Future MARL (3-4 roles, harder tasks): ~10k+ steps/s.

## After fix (lean no-Ray loop, path A)
- Drop framework, same batch: ~500/s (~4x).
- Unlock cheap batching: B=512 vec envs + vec_obs (40k/s, done) + one fwd over 512 obs +
  direct torch PPO -> projected ~10-30k steps/s (~100-200x). Matches fast-sim memory.

## Caveats
- 100-200x is projection until VecGathererSim + lean loop measured. vec_obs (65% in-sim
  hot path) done + parity-locked.
- LSTM time-unroll (T=32 sequential) ~0.84s is real compute, stays in any loop = ceiling
  unless seq-len drops / batch amortizes.
- Real-MC backend: env_step ~1s (JVM) -> ~1 step/s, transfer-validation only. That's why
  sim is the trainer.
