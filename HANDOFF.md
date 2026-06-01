# AI Utopia — Project Handoff (2026-06-01)

Detailed state-of-the-project + roadmap. Read alongside `PROJECT_CONTEXT.md` (architecture),
`CLAUDE.md` (gotchas), `Research/SUPERVISED_SESSION_LOG.md` (the latest training session),
`Research/TRAINING_ARCH_REVIEW_2026-05-31.md` (perf), `Research/FAST_TRAINER_PLAN.md`.

---

## 0. TL;DR — where we are

A persistent multi-agent Minecraft RL village. Mid-**M1** (single-role gatherer). This run delivered:
- A **fast trainer (~69× over RLlib)** — vectorized numpy sim + lean no-Ray PPO. Training is no longer the bottleneck.
- A **standalone desktop GUI** (Tauri v2 EXE) + a **FastAPI backend** wiring it to the real systems (agents, training, rewards, bridge).
- A **method that cracks the seed_1 "navigate" basin in sim** (behavior cloning) + the **general LSTM-update-bias fix** that lets PPO consolidate it.
- A hard, honest finding: **real-MC evaluation is high-variance** (real HARVEST non-determinism), so "does it pass the *real* gate" is currently **unmeasurable at n=1** and is the true blocker to calling M1 done.

Everything below is on GitHub (`CBaileyDev/AI-Utopia`, main).

---

## 1. Architecture (3 layers + the new GUI/API)

```
Desktop GUI (Tauri EXE, React)  ──HTTP──>  FastAPI backend  ──>  { identity.db, FabricBridge, train scripts, reward config }
Python RL  ──Py4J TCP──>  Java Fabric mod  ──>  Carpet fake players        (the live Minecraft path)
Fast sim   (pure numpy, no Java)  ──>  lean PPO loop                        (the training path)
```

- **Training path (NEW, fast):** `src/aiutopia/sim/` (VecGathererSim batched env) + `scripts/fast_train.py` (lean torch PPO, no Ray). ~9k env-steps/s.
- **Real-MC path:** `src/aiutopia/env/` (PettingZoo wrapper + Py4J `FabricBridge`) + `fabric_mod/` (Java). Used for transfer-validation, not bulk training (~1 step/s).
- **GUI:** `gui/` (Vite+React + Tauri v2). **API:** `src/aiutopia/api/` (FastAPI, port 8777).

---

## 2. What this session built (durable, tested, pushed)

### 2a. Fast trainer — 130 → ~9,000 env-steps/s (69×)
Bottleneck chain, each measured + parity-locked (`Research/TRAINING_ARCH_REVIEW_2026-05-31.md`, `FAST_TRAINER_PLAN.md`):
1. RLlib framework tax (~65% of an iter) → **lean no-Ray PPO loop** (`scripts/fast_train.py`).
2. Throughput regime-collapse (6.7k→1.5k as HARVEST chains grow) → **vectorized skill dynamics** (`src/aiutopia/sim/vec_skills.py`, closed-form walk + batched HARVEST chain, parity 8e-13).
3. Per-step Python overhead (list-of-SimWorld re-stacked 6×/step; autoreset double obs-build) → **batched-array state** in `VecGathererSim`.
4. obs hot-path scatter → **7-level buffered scatter-max** in `vec_obs.py`.
5. **LSTM update-ratio bias** (start-state-only T-segment replay diverged from collection → biased PPO ratio, first_mb KL 0.71 at T=32) → **faithful truncated-BPTT-1 replay** (feed stored per-timestep collection states) → KL 0.00000. *General fix — helps ALL recurrent training.*
Remaining cost is balanced real compute (collect = sequential per-step LSTM forwards; update = GPU). No framework hotspot left. JAX end-to-end is the only bigger ceiling (deferred).

Parity discipline: `tests/unit/test_vec_*` assert the fast sim is byte-identical to the scalar `AiUtopiaSimEnv` (which is byte-faithful to Java). 268 fast tests green.

### 2b. Desktop GUI + backend
- `gui/` — Tauri v2 desktop app (standalone EXE `gui/src-tauri/target/release/aiutopia-control-center.exe` ~3.1 MB + NSIS installer). Premium dark design (from the Claude Design handoff in `.design/`), 5 tabs (Dashboard/Bot Config/Training/Spectate/Settings), custom titlebar = real OS window (full-bleed).
- `src/aiutopia/api/` — FastAPI (port 8777) bridging GUI↔Python. Routes (all curl-verified): `/health` (FabricBridge), `/agents` (+spawn/kill via identity.db+Carpet), `/training/{runs,status,start,stop}` (parses `runs/*/progress.csv`, manages a subprocess), `/rewards` (live config GET/PUT), `/logs`. Heavy deps lazy-imported (boots <1s). Contract: `gui/API_CONTRACT.md`.
- **Reward externalization** (`src/aiutopia/env/reward.py`): LOG_VALUE/PBRS/allowlists now load from `config/rewards.json` (overlay over byte-identical defaults) → GUI can edit rewards without code. Hermetic (conftest pins it absent for tests).
- Launch both: `powershell -File scripts/launch-gui.ps1` (starts API with AIUTOPIA_ROOT, waits health, launches EXE).
- **GUI wiring status:** LIVE = health/bridge, agent roster+spawn/kill, training metrics/runs/start/stop, rewards editor, log feed. STILL MOCK (labelled "sample"): activity timeline, world-viewport animation, agent memory/plan tabs, chat. Offline-resilient (falls back to mock + "BRIDGE OFFLINE" pill).

### 2c. seed_1 basin — cracked in sim, via BC
The gatherer learned a degenerate **HARVEST-press** policy: it never learned NAVIGATE, so in sim it fails gate seed_1 (a spawn where no trunk is topmost-in-reach → HARVEST masked → must navigate first). 3 prior PPO curriculum attempts failed; this session **confirmed force_masked_spawn is a NEGATIVE** (300 iters drove NAVIGATE to −17.5).
- **Behavior cloning cracks it** (`src/aiutopia/sim/bc_demonstrator.py` + `scripts/bc_pretrain.py`): a NAVIGATE-then-HARVEST scripted demonstrator (proven 3/3 before cloning) → `weights/bc_gatherer.pt` clears **sim gate 3/3** (seed_1=64), NAVIGATE logit off the floor.
- **PPO can consolidate it** at full T=32 (after the 2a.5 LSTM fix) → `weights/runD_consolidated.pt` hit sim 3/3 — **BUT this is a lucky seed** (RUN D2, same recipe seed 2 → 2/3, seed_1=0). PPO erodes the cloned navigate because unmasked HARVEST-one-shot episodes dominate the gradient. Reliable consolidation needs a **BC-anchor** (KL-to-frozen-BC / distillation aux loss) — NOT yet built.

---

## 3. The honest blocker: real-MC evaluation is high-variance

This is the most important thing for the next session to internalize:

- The project's real success criterion is "**sim-trained policy clears the REAL-MC gate**", not the sim gate.
- Transfer evals this session (`scripts/transfer_eval_bc.py` on instance 25001) gave, per 3-seed run:
  - BC policy: 60 / 46 / 46
  - Known-good HARVEST-spam (stale instances): 64 / 46 / 58
  - Known-good HARVEST-spam (FRESH instances): 58 / 64 / … (numbers SHIFTED run-to-run)
- **Conclusion:** the per-seed oak count swings 46–64 across runs **regardless of policy or instance freshness**. This is the documented **real HARVEST non-determinism** (memory `ai-utopia-fast-sim-plan` "TWO MEASURED FIDELITY FINDINGS #2": real HARVEST collected 1/6/0 logs across back-to-back identical dispatches). A single greedy 3-seed eval therefore **cannot** declare a clean real-gate pass/fail.
- Secondary reframe: the sim's "seed_1 is the hard masked one" does NOT cleanly reproduce on real (real seed_1 often = 64 via pure HARVEST-spam). So part of the multi-session seed_1 saga was tuning against a **sim-specific geometry**. The navigate capability BC adds is still real + needed for harder/natural-world tasks, but it is NOT what the real M1 gate currently turns on.

**Implication:** you cannot trust real-MC numbers at n=1. Either (a) run N≥5 evals/seed and gate on a rate, or (b) fix the real HARVEST non-determinism (the #1 Phase-C fidelity item) first.

---

## 4. Verified state / how to run

- Tests: `pytest -q -m "not integration and not determinism"` → **268 passed** (sim/RL/api/reward all green).
- Fast train (sim): `PYTHONPATH=src AIUTOPIA_ROOT=<repo> py -3.11 scripts/fast_train.py --num-envs 512 --horizon 32 --iters 80 --kl-coeff 0.2 --gate-check` (--gate-check prints per-seed oak + the seed_1 NAVIGATE probe). Curriculum flags: `--spawn-jitter --approach-shaping --force-masked-spawn` (force_masked is a known negative). BC warm-start: `--load-weights weights/bc_gatherer.pt --value-warmup-iters N --actor-lr-ramp R`.
- GUI: `scripts/launch-gui.ps1` (or `npm run tauri:dev` in `gui/` + `py -3.11 -m aiutopia.api`).
- Real transfer: `py -3.11 scripts/transfer_eval_bc.py --weights weights/bc_gatherer.pt --port 25001` (needs live training instances; restart clean via `JDK_HOME=... bash scripts/launch-training-instances.sh`).
- Checkpoints (gitignored, local): `weights/bc_gatherer.pt` (sim 3/3), `weights/runD_consolidated.pt` (sim 3/3, lucky seed), `weights/fast_train_gatherer_peak.pt` (HARVEST-spam).

---

## 5. Where we're going — prioritized next steps

**P0 — make the real gate measurable (the actual blocker).**
1. Variance-control real eval: run the 3 gate scenarios N≥5× each on FRESH instances, gate on a success-rate (not n=1). Build into `transfer_eval_bc.py`.
2. OR/AND diagnose + fix the **real HARVEST non-determinism** (Java `HarvestSkill` — why back-to-back identical dispatches collect different counts; suspect findNearest/reach/tick-timing under Carpet). This is the #1 sim→real fidelity item and unblocks ALL real eval.
3. Re-establish a clean real baseline with the known-good HARVEST-spam policy before judging any new policy.

**P1 — reliable BC consolidation (sim, then transfer).**
4. Add a **BC-anchor** to `fast_train.py`: keep a frozen copy of the BC policy and add `beta * KL(current || BC_frozen)` (or a BC behavioral aux loss on a replay of demo states) during PPO, so consolidation can't erode navigate. Verify it holds sim 3/3 across ≥2 seeds (RUN D2 showed plain PPO does not).
5. Resolve the sim↔real seed_1 geometry discrepancy (why sim seed_1 is masked but real seed_1 isn't) — likely a spawn-position or arena-placement parity gap between `sim/world.py` and Java `WorldOps.resetEpisode`.

**P2 — GUI to production.**
6. Auto-start the backend from the EXE (Tauri sidecar or bundled launcher) so double-click "just works" (today needs `launch-gui.ps1`).
7. Wire the remaining mock views to real sources as they come online (world viewport ← live obs; memory/plan ← Chroma/planner when M5; chat ← `@agent` router).

**P3 — M1 closure + beyond (do NOT pre-declare).**
8. Only after a clean variance-controlled real-gate pass: run the real promotion pipeline (`determinism check` → `promote-weights`). Promotion is gated on real-gate + determinism — do not promote a policy that's unmeasured on real.
9. Then M2: second role (the stack is gatherer-hardcoded in spaces/encoder/actor_head — a real 2nd role is design-heavy; see CLAUDE.md specialist notes).

---

## 6. Key gotchas the next session must respect (hard-won)

- **Real-MC eval is non-deterministic — never trust n=1.** (This session's central finding.)
- **NAVIGATE skill-logit is a NON-indicator** (RUN D: +2.23 when seed_1 FAILED, +1.04 when it PASSED). Trust per-seed oak + action traces, not the static logit.
- **force_masked_spawn is a confirmed NEGATIVE** for learning navigate (drives NAVIGATE deeper). approach_shaping alone can't bootstrap the first navigate. BC is the lever that works.
- **PPO erodes a sharp cloned policy** unless (a) the LSTM update is faithful (fixed) AND (b) there's a BC-anchor (not yet built) — and even then unmasked-episode gradient dominance is a risk.
- **sim↔real parity is sacred:** the fast sim + vec_obs are parity-locked to the scalar env (byte-faithful to Java). Any change to obs/mask/reward must keep `test_vec_*`/`test_sim_*` green.
- Run instances fresh before real eval (stale arenas after ~24h add noise). Restart: `JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 bash scripts/launch-training-instances.sh`.
- Operational Python is **3.11** via `PYTHONPATH=src` (the 3.12 gate is why). Windows forces `num_learners=0` on Ray.
