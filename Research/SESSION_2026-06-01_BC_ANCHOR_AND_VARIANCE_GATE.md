# Session 2026-06-01 — BC-anchor + variance-controlled real-MC gate

Picks up the two top HANDOFF.md priorities: **P1.4** (reliable BC consolidation — the
"NOT yet built" lever) and **P0** (make the real gate measurable). Branch
`claude/handoff-review-1I8qK`, PR #1. Everything below was run on **CPU** in a cloud
container (no GPU; no live Fabric server) — see "Environment" for the implications.

## TL;DR

- **Built + unit-tested + committed (durable):** a BC-anchor in `scripts/fast_train.py`
  (`--bc-anchor-coeff`), a variance-controlled real-MC gate in
  `scripts/transfer_eval_bc.py` (rate over N repeats), a cross-hardware tolerance fix,
  and an RNG-isolation fix that makes the anchor cleanly A/B-able.
- **Honest finding — the anchor's net benefit is UNVERIFIED, do NOT claim it fixes
  erosion yet.** On CPU at reduced scale the consolidation control does **not** erode
  navigate (it holds the sim gate 3/3, seed_1=64), so this regime does **not reproduce
  the RUN D2 erosion** the anchor exists to correct. The anchor *mechanism* is confirmed
  (unit tests + it raises the real-run navigate logit +1.98→+2.73), but with no erosion
  to fix its extra gradient mildly perturbs the unmasked harvest (seed_2/3 64→58), so it
  nets *worse* on the aggregate gate here (1/3 vs control 3/3). The anchor's value can
  only be tested in a regime that actually erodes navigate — the full-scale RUN D2
  recipe (B=512, ~80+ iters, the eroding seed) on the user's GPU.

## What shipped (commits on the branch)

1. `test(fast_train)`: widen the LSTM-faithful-replay tolerance 1e-4 → 1e-3. The faithful
   update folds T sequential length-1 LSTM forwards into one `(B*T,1)` batch; the cell
   math is identical but float32 kernels accumulate differently batched-vs-sequential
   (~1e-4 GPU, ~3e-4 CPU/torch-2.12). 1e-3 still sits 10x below the legacy divergence
   (>1e-2) the control test asserts, so the regression guard keeps its teeth.
2. `feat(fast_train)`: **BC-anchor.** Each finetune iter samples a FRESH force-masked
   spawn batch (the erosion locus) and re-applies the scripted demonstrator's
   NAVIGATE-then-HARVEST supervision as an aux loss (mask-aware CE on skill + MSE on the
   nav bearing). Reference = the demonstrator (ground-truth optimum on masked spawns;
   needs no second frozen module). Separate per-iter opt step (PPO ratio math untouched);
   the loss reaches only actor params (CTDE critic byte-stable). Flags
   `--bc-anchor-coeff` (0=OFF, default; default path byte-unchanged), `--bc-anchor-envs`,
   `--bc-anchor-epochs`.
3. `feat(eval)`: **variance-controlled real-MC gate.** `transfer_eval_bc.py` runs each
   scenario `--repeats` (default 5) times and gates on a per-scenario success RATE
   (`--pass-threshold`, default 0.6 = ≥3/5); overall pass requires every seed to clear.
   Prints per-seed oak min/mean/max so the documented HARVEST non-determinism is visible.
   `--warmup` discards one seed_1 reset to absorb the cold-start spawn race. The decision
   (`summarize`) is a pure, server-free function, unit-tested.
4. `fix(fast_train)`: **RNG isolation.** The anchor drew its spawn seeds from the global
   numpy RNG, desyncing the main loop's minibatch shuffle (`np.random.shuffle`) — so
   turning the anchor on changed *every* subsequent shuffle, confounding control(coeff=0)
   vs treatment(coeff>0) with pure RNG drift (visible as a sign-flip across recipes). Now
   a dedicated `np.random.default_rng` keeps the global stream byte-identical regardless
   of the anchor, so the A/B is clean.

New tests (all green; full suite 275 passed): `test_fast_train_bc_anchor.py` (anchor
drives navigate on a real force-masked batch; critic byte-stable) and
`test_transfer_eval_variance.py` (rate gating, inclusive threshold, all-seeds-must-pass,
no vacuous pass).

## Experiments (CPU, sim only)

BC pretrain (`scripts/bc_pretrain.py`, 60 iters) → `weights/bc_gatherer.pt` clears the
**sim gate 3/3** (64/64/64) here — the pipeline reproduces end-to-end.

Consolidation A/B (gentle recipe: `--num-envs 256 --horizon 16 --iters 40
--value-warmup-iters 15 --actor-lr-ramp 4 --actor-lr 5e-5 --value-lr 2e-3 --kl-coeff 0.3`,
seed 0, RNG-isolated build):

| run                | seed_1 | seed_2 | seed_3 | gate | seed1 NAV logit |
|--------------------|:------:|:------:|:------:|:----:|:---------------:|
| control (coeff=0)  | 64 ✓   | 64 ✓   | 64 ✓   | 3/3  | +1.98           |
| anchor (coeff=0.5) | 64 ✓   | 58 ✗   | 58 ✗   | 1/3  | +2.73           |

Reading: the control already holds 3/3 (navigate did NOT erode at this scale), so there
is no erosion for the anchor to fix; the anchor raises the navigate logit (mechanism
works) but its perturbation costs ~6 logs on the unmasked seeds. Earlier confounded runs
(pre-RNG-fix, aggressive lr 1e-4) showed anchor 2/3 > control 0/3 — that "win" was the
RNG + instability confound, not a real anchor benefit; discount it.

Why no erosion on CPU: RUN D2 eroded at B=512 over ~80+ iters on a specific seed. At
B=256 / 22 finetune iters / lr 5e-5 the BC clone's navigate is robust and the anchor loss
stays ≈0 (nothing to correct). Reproducing erosion is what the anchor needs to be tested
against — it is GPU/scale work, not a CPU experiment.

## Environment (cloud container, CPU)

- No GPU; `torch` ran CPU-only. Deps were pip-installed fresh (`torch numpy scipy pydantic
  gymnasium pettingzoo ray[rllib] py4j chromadb ruff mypy`) — note `--ignore-installed
  packaging` was needed (debian-managed `packaging`). Ray 2.55.1.
- `fast_train.py` is "no-Ray trainer" but still imports the RLlib RLModule, so Ray IS
  required to run it.
- No live Fabric server here, so the variance gate was built + unit-tested but NOT run
  against real MC. That remains the user's Windows machine.
- CPU throughput ~600–9k env-steps/s depending on B/T (vs the ~9k GPU figure); fine for
  sim experiments at reduced scale.

## Next steps (prioritized)

**P0 — still the real blocker.** Run the new variance gate on the live server:
`py -3.11 scripts/transfer_eval_bc.py --weights weights/bc_gatherer.pt --port 25001
--repeats 5 --warmup`. Gate on the rate, not n=1. (And/or fix the Java HarvestSkill
non-determinism — the #1 fidelity item.)

**P1 — finish validating the anchor PROPERLY.** Reproduce the RUN D2 erosion first
(full-scale B=512, the eroding seed, no anchor → confirm it drops to 2/3), THEN turn the
anchor on at the same seed/RNG and check it holds 3/3 across ≥2 seeds. The RNG fix makes
this a clean A/B. If the anchor still costs unmasked-seed logs, try: (a) a smaller
`--bc-anchor-coeff`, (b) anchoring only the skill CE (drop the spatial MSE), or (c) a KL-
to-frozen-BC variant that is gentler than hard CE. Do NOT promote on the strength of the
CPU runs here — they did not exercise erosion.

**P3 — M1 closure** stays gated on a clean variance-controlled real-gate pass, unchanged.
