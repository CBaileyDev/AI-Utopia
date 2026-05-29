# AI Utopia — Project Context (snapshot 2026-05-28)

This file is the single-source overview of the project. Read top-to-bottom; everything else (NEXT_SESSION.md, M0_PROGRESS.md, M1A_PIPELINE_PLAN.md, M1B_TRAINING_PLAN.md, IMPLEMENTATION_PLAN.md) is sub-spec detail.

---

## 1. Vision

**A persistent, multi-agent Minecraft AI village.** Four specialized agents — gatherer, builder, farmer, defender — cooperatively grow and operate a village on a private Minecraft Java 1.21.1 server. Human friends join the same server, get recognized as friendly entities, and can natural-language chat with agents via `@<agent_name>` mentions. The system mixes:

- **Per-role reinforcement learning** (PPO + LSTM RLModule per role) for low-level skill execution
- **LLM planner** at the top of the stack producing goal specs that flow down to per-role policies
- **Persistent identity** with ULID Crockford base32 IDs across server restarts
- **Episodic memory** (Chroma) and a skill library that grows with each agent's experience

End-state target: 22–28 weeks to M6 (full village operation). Currently working through M1 (single-role training pipeline).

## 2. Hardware available (yours)

- **AMD Ryzen 9 9950X3D** — 16 physical cores / 32 threads
- **64 GB DDR5** RAM
- **NVIDIA RTX 4080** — 9728 CUDA cores
- All currently idle during training because the bottleneck is the JVM Minecraft simulation, not the policy net (see §9).

## 3. Current state — top-line

| Aspect | Status |
|---|---|
| **Milestone reached** | M0 (infrastructure) fully verified; M1A (single-role pipeline) code-complete; M1B (training) IN PROGRESS |
| **Active branch** | `main` |
| **HEAD** | `14b32dc` |
| **Tags** | `m0` (`6d693d9`), `m0-source` (`cc0cd9d`), `m0-verified` (`ec5a66d`) |
| **Active training run** | v21, PID file `Research/train-v21.log.pid`, started 17:24 |
| **v21 latest** | iter 4 of 200 max, t=4488s, NaN-free, episode_return_max +19.83 (first time we've seen positive reward at the iter level) |
| **Real-MC infrastructure** | 4 Fabric instances on ports 25001-25004, MC ports 25566-25569 |

## 4. Architecture (three layers)

```
┌────────────────────────────────────────────────────────────────┐
│  Python — RL stack                                             │
│  ┌────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │ Ray RLlib  │  │ Ray Tune      │  │ PPO + LSTM RLModule   │  │
│  │ EnvRunners │──│ trial driver  │──│ (256-hidden, 512-256  │  │
│  │ (4 procs)  │  │ scripts/train │  │  encoder, 256 backbone│  │
│  └─────┬──────┘  └───────────────┘  └───────────────────────┘  │
│        │                                                       │
│  ┌─────▼──────────────────────────────────────────────────┐    │
│  │  AiUtopiaPettingZooEnv (PettingZoo Parallel API)       │    │
│  │  src/aiutopia/env/wrapper.py                           │    │
│  │  - reset() → bridge.reset_world + bridge.reset_episode │    │
│  │  - step() → bridge.dispatch_skill + advance + observ.. │    │
│  │  - reward shaping (§5.4) + exploit detector            │    │
│  └─────┬──────────────────────────────────────────────────┘    │
│        │                                                       │
│  ┌─────▼──────────────────────────────────────────────────┐    │
│  │  FabricBridge (Py4J client)                            │    │
│  │  src/aiutopia/env/bridge.py                            │    │
│  │  - JavaGateway → entry_point                           │    │
│  │  - dispatch_skill, observations_all, reset_episode...  │    │
│  └─────┬──────────────────────────────────────────────────┘    │
└────────┼─────────────────────────────────────────────────────  ┘
         │  Py4J TCP socket (ports 25001-25004)
┌────────▼───────────────────────────────────────────────────┐
│  Java/JVM — Minecraft 1.21.1 Fabric server                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AiUtopiaMod (ModInitializer)                        │  │
│  │  fabric_mod/src/main/java/dev/aiutopia/mod/          │  │
│  │   ├─ AiUtopiaMod.java          (mod entry)           │  │
│  │   ├─ Py4JEntryPoint.java       (Py4J surface)        │  │
│  │   ├─ bridge/                                         │  │
│  │   │   ├─ MotorBridge.java      (skill dispatch +     │  │
│  │   │   │                         per-tick driver)     │  │
│  │   │   ├─ WorldOps.java         (arena setup +        │  │
│  │   │   │                         reset_episode +      │  │
│  │   │   │                         carpetSpawn)         │  │
│  │   │   ├─ CommBus.java          (agent-agent msgs)    │  │
│  │   │   └─ skill/                                      │  │
│  │   │       ├─ NavigateSkill.java                      │  │
│  │   │       ├─ HarvestSkill.java   ← N16 fixes here    │  │
│  │   │       ├─ DepositChestSkill.java                  │  │
│  │   │       ├─ SearchSkill.java                        │  │
│  │   │       └─ WaitSkill.java                          │  │
│  │   ├─ obs/      (observation builder + ItemIdTable)   │  │
│  │   ├─ agent/    (AgentRegistry — name→role mapping)   │  │
│  │   ├─ chat/     (player-mention router)               │  │
│  │   └─ mixin/    (KickPlayerMixin and friends)         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Carpet fake players (one per instance)              │  │
│  │  - gatherer_0 lives on each instance                 │  │
│  │  - tick rate warped 20 → 60 TPS                      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## 5. Codebase layout

```
AiUtopia/
├── src/aiutopia/                  Python package
│   ├── cli/                       Typer CLI surface (§7.5)
│   │   ├── agent.py               `aiutopia agent spawn|list`
│   │   ├── determinism.py         `aiutopia determinism check`
│   │   ├── memory.py              `aiutopia memory inspect`
│   │   └── promote.py             `aiutopia promote-weights` (T16/T17)
│   ├── common/                    IDs, paths, config, logging
│   ├── determinism/               Weights-replay harness (§7.8)
│   ├── env/                       PettingZoo Parallel env + Py4J bridge (§7.3)
│   │   ├── wrapper.py             AiUtopiaPettingZooEnv (511 lines)
│   │   ├── bridge.py              FabricBridge / encode_action (139)
│   │   ├── reward.py              §5.7 LOG_VALUE table + PBRS (299)
│   │   ├── spaces.py              build_role_observation_space etc (113)
│   │   ├── exploit.py             ExploitDetector (§5.5 penalties)
│   │   └── goal.py                Goal embedding adapter
│   ├── identity/                  SQLite + migrations (§3.5)
│   ├── memory/                    Chroma episodic memory (§4.9, §5.6)
│   ├── planner/                   LLM planner stubs (real impl M5)
│   ├── promotion/                 Weight promotion (§5.10)
│   ├── rl_module/                 RLModule layer (§7.2) — LSTM, encoders, heads
│   ├── schemas/                   Pydantic v2 schemas (§6)
│   └── train/                     PPO training stack (§7.1)
│       ├── config.py              m1_gatherer_config PPOConfig builder
│       ├── env_factory.py         RLlib env registration
│       ├── callbacks.py           AiUtopiaMetricsCallback, ExploitHuntCallback,
│       │                          EvalGateStopCallback, M1EvalScenarioCallback
│       └── scenario_runner.py     run_scenario for eval (LSTM-threaded greedy)
│
├── fabric_mod/                    Java Fabric 1.21.1 mod (Java 21)
│   ├── build.gradle               Fabric-loom 1.7-SNAPSHOT, yarn mappings
│   ├── gradle.properties          mod_version = 0.0.0-m1c-n16c
│   └── src/main/java/dev/aiutopia/mod/  (see §4 diagram)
│
├── scripts/                       Shell + Python utilities
│   ├── train.py                   Ray Tune entry point
│   ├── launch-training-instances.sh  Bootstrap 4 Fabric servers
│   ├── m1a-smoke.sh / m1b-evaluation-gate.sh
│   ├── n14_reward_probe.py ... n18_eval_e2e_test.py  ← today's diagnostics
│   └── rllib_smoke.py             1-iter synthetic-env smoke (T7.5)
│
├── server-runtime/                gitignored — actual server installs
│   └── training/
│       ├── instance-1..4/         Each has mods/, world/, server.properties
│       │   ├── mods/aiutopia-mod-0.0.0-m1c-n16c.jar
│       │   ├── mods/fabric-api / fabric-carpet / ferritecore
│       │   ├── logs/latest.log
│       │   ├── instance-N.log     (java stdout)
│       │   └── instance-N.pid
│
├── runs/                          Ray Tune trial outputs
│   └── aiutopia_M1_seed1/         PPO_aiutopia_minecraft_<id>/{progress.csv,result.json,events.out.*}
│
├── docs/                          Spec history (committed)
│   └── superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md
│
├── tests/                         {unit, integration, determinism}
│
├── IMPLEMENTATION_PLAN.md         5774 lines — M0-M6 master plan
├── M0_PROGRESS.md                  M0 verification log
├── M1A_PIPELINE_PLAN.md           4035 lines — single-role pipeline tasks T1-T22
├── M1B_TRAINING_PLAN.md           3107 lines — actually run training to gate
├── NEXT_SESSION.md                Post-v20 handoff (resumes from v21 status)
├── PROJECT_CONTEXT.md             ← you are here
├── README.md                      Public-facing quickstart
├── pyproject.toml                 Python 3.12, deps include ray[rllib,tune], pettingzoo, gym, torch, chromadb, py4j, typer
└── ruff.toml / .pre-commit-config.yaml / .python-version
```

## 6. Milestone history

### M0 — Infrastructure foundation (DONE)
- `aiutopia agent spawn --role gatherer` produces a Carpet fake player on a live MC 1.21.1 server
- Java mod compiles green via Gradle wrapper; jar deployed to `server-runtime/`
- Python identity service + SQLite migrations + Chroma collections all wire end-to-end
- Tag `m0-verified` at `ec5a66d`

### M1A — Single-role RL pipeline code (DONE)
Tasks T1-T22 completed. Highlights:
- T1-T6 — CoreEncoder, SharedBackbone, CTDECritic, GathererRoleEncoder, GathererActorHead, AiUtopiaRoleRLModule with LSTM time-dim
- T7-T8 — `m1_gatherer_config` PPOConfig builder (new API stack), `env_factory.make_aiutopia_env_wrapped`
- T9-T11 — three callbacks (metrics, exploit, eval-gate-stop)
- T12-T13 — `WorldOps.resetEpisode` (Java) + wrapper.reset wiring
- T14-T15 — `scenario_runner` with LSTM state threading + `M1EvalScenarioCallback`
- T16-T17 — `promote-weights` CLI + §5.10 5-gate checklist
- T18 — `scripts/train.py` Ray Tune driver
- T19 — determinism harness real-weights replay
- T20 — `launch-training-instances.sh` + `m1b-evaluation-gate.sh`
- T22 — jar version bump + `m1b-verified` tag (target)

### M1B — Actually train to gate (IN PROGRESS) ← we are here

The eval gate: **80% success on "collect 64 oak_log within 1000 env steps" over 3 consecutive evaluation runs.**

This milestone has consumed nearly a full day of integration debugging — N1 through N19. The pipeline now works end-to-end but training convergence is slow due to MC-pipeline throughput limits (see §9).

## 7. The 19+ integration fixes (N1-N19, today + prior sessions)

Numbered chronologically. **N14 onwards is today's work.**

| # | Commit | Issue | Fix |
|---|---|---|---|
| **N1** | (prior) | pre-flight sanity for the training session | env probes |
| **N2** | `db51a33` | manual Fabric server boot was tedious | `launch-training-instances.sh` waits for `Done (`, then runs `setup_training_scene()` on each port |
| **N3** | (multi-commit) | training run not converging | series of fixes (timeout, batch size, tick rate) |
| **N7.5** | `5e0a374` | `custom_metrics/M1/gate_passed` doesn't surface as a Tune stop key in Ray 2.55 | docs memo + drop the stop-key, watch via post-hoc script |
| **N8** | `db51a33` | bake bootstrap into launch script | done |
| **N9** | `c79afc5` | `_inventory_from_obs` fell back to `item_{N}` because Java ItemIdTable wasn't shipped to Python | `Py4JEntryPoint.getItemIdNameTable()` exposed; wrapper imports it at env init |
| **N10** | `3b5244a` | Java skill `timeout_ticks` default was 6000 (20s wall at tick warp 300) — random-policy NAVIGATEs burned the whole budget | wrapper injects `timeout_ticks=400` per dispatch |
| **N11** | `eeae98b` | Lithium's `forEachInBox` → CME with Carpet fake players under tick warp | removed Lithium from training instances |
| **N12** | `27ff6cd` | tick rate 300 → fastutil `LongOpenHashSet.rehash` AIOOBE during chunk tick | dropped tick rate to 60 |
| **N13** | `8364c4b` | at tick 60, per-step latency ~5-7s → batch fill exceeded `sample_timeout_s` | shrank `train_batch_size` 2048→256, `rollout_fragment_length` 128→32, `sample_timeout_s` 600→1800 |
| **N14** | `694b44d` | (initially suspected ItemId table empty; really wasn't — just log-routing) | upgrade log to WARNING + add defensive eager seed in reward.py |
| **N15** | `694b44d` | skill dispatch path looked broken (no completion events for 30s) | confirmed dispatch path fine via standalone Py4J probe; root cause was N16 |
| **N16** | `694b44d` | **HARVEST silently failed forever** — REACH_RADIUS=2.0 + `step = min(WALK, dist - REACH)` formula meant when collision pinned the agent at `dist=2.0+1 ULP`, step shrunk to 1e-18 and agent never moved. Compounded: after fixing REACH→4.5, agent stopped 3-4 blocks away from log, OUTSIDE vanilla's 1.5b auto-pickup radius → drops despawned uncaught | (a) REACH 2.0→4.5, (b) always full WALK_PER_TICK, (c) bypass entity-form drops with `Block.getDroppedStacks` + `agent.getInventory().offerOrDrop` + `setBlockState(AIR)` |
| **N16-followup** | `4a9bff0` | `max_episode_ticks=12_000` meant episodes never ended | 12_000 → 2_000 |
| **N17** | `abe1c12` | eval scenarios crashed every 10 iters with `numpy.object_ → torch.as_tensor` | `action_mask` is a **nested dict** (not an ndarray); added recursive batching in `scenario_runner.py` |
| **N17b** | `abe1c12` | `max_episode_ticks=2000` still wasteful (ring depletes in ~150 steps) | 2_000 → 300 |
| **N18** | `0c52bd4` | eval iter took 75 min wall (3 scenarios × 1000 ticks × 1.5s) | scenario `max_ticks` 1000 → 300 |
| **N19** | `14b32dc` | three pre-staged improvements | (a) wrapper.py truncates when agent strays ±24 horizontally or y<60 — keeps on-policy buffer oak_log-dominated; (b) launch script `-Xms1g -Xmx2g → -Xms2g -Xmx4g` (G1 GC pressure); (c) config `metrics_num_episodes_for_smoothing` 200→20 so `episode_return_mean` populates within first iters |

Verified end-to-end via `scripts/n14_reward_probe.py`: 6 oak_log per 6 env_steps, reward **+11.78** in isolation.

## 8. Training run history (today)

| Run | Start | End | Iters | Notable |
|---|---|---|---|---|
| v17 | 11:25 (prev sess.) | killed | 6 | First run that completed multiple iters but with broken HARVEST (zero reward signal) |
| v18 | ~05:46 today | killed at 56 | 56 | First run post-N14-N16 fixes; reward signal worked but only 1 of 4 envs collected oak_log; slow degradation |
| v19 | ~11:56 | killed at 9 | 9 | First N17 attempt; iter 10 eval hung for 1h+ (eval too expensive) |
| v20 | ~13:58 | killed at 16 | 16 | Eval disabled; step time degraded 1.5s→16s by iter 16 |
| **v21** | 17:24 (active) | running | 4 (now) | All N19 fixes; episode metrics actually populating; first time `episode_return_max > 0` |

## 9. The fundamental throughput problem

> **⚠️ Note (N21, 2026-05-29):** everything in §9 below was written *before* the sim
> existed — it is the accurate diagnosis of the **real-MC** env. The sim now exists and
> has been measured + profiled. See **§9.1** for the corrected, measured reality: the
> bottleneck moved off the JVM (it is now RLlib framework + policy overhead on one CPU
> core, not the sim and not the GPU), and the "1000–50,000×" JAX figure in the table
> below is largely a unit artifact (realistic ≈ 200–500× over the current sim).

Your hardware is built for parallel compute; Minecraft is built for single-threaded simulation. The mismatch creates the loop:

```
Python step:      < 1 ms   (LSTM forward + JSON serialize)
Py4J round trip:    1-2 ms (TCP socket)
JAVA SKILL RUN:  1000-15000 ms  ← bottleneck
Python obs read:  10-50 ms (JSON deserialize)
```

**Result you're seeing now:**
- ~5% CPU utilization (4 single-threaded MC server threads on a 32-thread CPU)
- 0% GPU utilization (LSTM is microseconds; sits idle 99.9% of step)
- 32 GB / 64 GB RAM (4 instances × 2-4 GB each + Python/Ray overhead)

**Practical options:**

| Approach | Throughput multiplier | Engineering cost | Sim-to-real risk |
|---|---|---|---|
| Run **16 Fabric instances** instead of 4 | 4× | Low (just bash + config) | None |
| **Multiple agents per instance** via `carpetSpawn` extra players | 1.5-2× (shared server tick) | Medium | None |
| **Raise tick rate** 60 → 100-120 | 2× | Medium (crash risk; had to drop from 300) | None |
| **Bigger model** (LSTM 256→1024) | actually uses GPU | Low | None |
| **Custom Python/JAX sim of M1B task** | ~~1000-50000×~~ → **~250× DONE** (scalar NumPy sim built); **~200-500×** more via full JAX rewrite (§9.1) | Medium (~500-1000 lines) | Real — policy may exploit sim quirks |
| **Hybrid: train in fast sim, fine-tune in real MC** | best of both | High (need sim + fine-tune loop + transfer eval) | Mitigated by fine-tune phase |

The **honest** answer to "are we training with all resources available?" is **no**, and the **honest** answer to "should we simulate it ourselves?" is **yes for M1B**, **maybe for M2+**, **no for deployment**.

### 9.1 Measured reality after the sim was built (N21, 2026-05-29)

The fast sim now exists and has been **measured + adversarially profiled** (workflow +
the reusable `scripts/sim_step_microbench.py`). Three of §9's projections are wrong in
instructive ways.

**Measured throughput.** The sim does **~222 macro-steps/s** (verified across several
`PPO_aiutopia_sim_*` 200-iter runs) vs **0.73–0.9 macro-steps/s** for real MC (aggregate
over the 4 JVM instances; the docs' "0.9" is rounded slightly optimistic — best measured
real run is 0.726). That is **~250× raw** (≈306× vs the measured real rate) — the sim
converges in **~90 s** vs ~6–8 h of real-MC for the same step budget. A "macro-step" =
one skill dispatch on both sides (verified: `sim_env.py` step() and `wrapper.py` step()
each issue exactly one dispatch), so the ratio is apples-to-apples. Caveat: real HARVEST
*chains* (clears many logs per dispatch) while the scalar sim does one, so real does more
work per macro-step — 250× in dispatches/s modestly overstates true task-progress speedup.
The old **"~50 days → ~48,000×"** figure is a **category error**: that estimate was against
the structurally-broken, unwinnable task (see memory `m1b-blocked-by-broken-task`), not a
fair baseline.

**The bottleneck moved — it is NOT the sim, and NOT the GPU.** Profile of one ~4.56 ms
training step (`num_env_runners=0`, single in-process env):

| Component | % of step |
|---|---|
| PPO learner update (CPU kernel-launch dispatch; **GPU ~84% idle**, 56 W of 320 W) | 33.7% |
| LSTM policy inference (tiny model, batch=1) | 32.5% |
| RLlib connector / sampling pipeline (pure new-API-stack framework) | 27.8% |
| **The sim env itself** (apply_skill incl. walk chain + obs build + reward) | **5.7%** |

We traded the *JVM single-thread* bottleneck for an *RLlib single-env single-thread* one:
~94% of every step is framework + policy overhead on **one CPU core of 16**; the RTX 4080
and the other 15 cores sit idle. **Optimizing the scalar NumPy sim does almost nothing**
(Amdahl-capped to ~6%). Within the env, the *typical* step is `build_gatherer_obs`-bound
(the 64-log Python grid loop), not walk-loop-bound; the walk chain only spikes on a rare
full-field cap=64 harvest.

**The "1000–50,000×" JAX projection is largely a UNIT ARTIFACT.** Craftax's "~280k steps/s
on a 4090" counts *primitive ticks*; this sim reports *macro-steps*, and one macro-step ≈
50–509 primitive ticks. In consistent units a full GPU rewrite (env + policy + learner
batched under one `jit`, thousands of worlds via `vmap`, policy resident on the idle GPU)
realistically reaches **~40–50k macro-steps/s ≈ 200–500× over the current sim** — not
1000×+. M1B *does* vectorize cleanly (fixed 64 logs / 6144 grid, no terrain gen) via
fixed-depth masked `lax.scan` + `jnp.where` skill-select; the single biggest code win is
replacing the ~445-iteration Python walk-loop with the closed-form
`ticks = ceil((dist − REACH)/WALK_PER_TICK)`. Fidelity catch: precompute the Java-faithful
arena layouts on CPU and load them as device constants (JAX PRNG ≠ `java.util.Random`
would break the golden-trace obs parity).

**Scaling levers, in ROI order (corrected — §9's table over-promised runner count):**
1. **`num_env_runners` — free, but small.** Currently 0. PPO here is *synchronous*
   (sample → learner update, no overlap) with a **fixed ~1.29 s/batch learner**, giving a
   hard **~595 steps/s asymptote reached by N≈2**. So more runners buy **~2.5×, not §9's
   "4×."** Recommended ~8 (≈80% of the ceiling, fits the 16 physical cores). **Gotcha:**
   set `torch.set_num_threads(1)` / `OMP_NUM_THREADS=1` per runner or N×16 threads thrash.
   *(This N-table is model-based — its adversarial verifier failed to return; the ~595/s
   learner asymptote is the robust part, derived from the measured 1.29 s/batch.)*
2. **Batching — the real lever on this stack.** Bigger `train_batch`/`minibatch` (GPU 84%
   idle → bigger updates are nearly free and *raise* the 595/s ceiling) +
   `num_envs_per_env_runner > 1` (amortize the 1.48 ms LSTM forward across k envs). These
   *break* the ceiling; runners only approach it.
3. **Vectorized JAX rewrite (Phase D)** — ~200–500× as above; the big multiplier, but only
   worth building when there are enough roles/agents to consume the throughput (M2+).

**The true ceiling is PPO sample-efficiency, not steps/s.** Past a useful batch width, more
throughput stops cutting wall-clock-to-gate (oversized batches hurt per-sample learning);
for single-agent M1B that saturates well before even the 4080 does. So **stronger hardware
buys ≈ nothing for M1B** — the GPU is already 84% idle and the learner is launch-bound on a
tiny model. A bigger GPU (4090/5090/H100) or a high-core cloud VM earns its keep only for
the Phase-D JAX path (wider world batch) and **M2+ multi-role MARL / parameter sweeps**,
where the extra width is actually consumed.

**Measured hardware (this box):** AMD Ryzen 9 9950X3D (16c/32t, 3D V-Cache), 61 GB RAM,
NVIDIA RTX 4080 (16 GB, 320 W).

## 10. How to operate the system

### Launch Fabric servers (4 instances)

```bash
cd /c/Users/Carte/OneDrive/Desktop/AiUtopia
JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 bash scripts/launch-training-instances.sh
```

Checks each Py4J port (25001-25004), skips already-running instances, waits for `Done (` in each instance log, then runs `setup_training_scene()` on each port. Idempotent.

### Launch training

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
  nohup py -3.11 -u scripts/train.py --milestone M1 --max-iters 200 --evaluation-interval 10 \
    > Research/train-vN.log 2>&1 &
```

Args:
- `--max-iters` cap PPO iters (default 2000)
- `--evaluation-interval` how often `M1EvalScenarioCallback` runs (set to 999 to disable)
- `--num-env-runners` defaults to 4 (one per Fabric instance)
- `--seed` PPO + env seed

### Probe agent state

```bash
PYTHONPATH=src py -3.11 scripts/n14_reward_probe.py    # standalone HARVEST + reward probe
PYTHONPATH=src py -3.11 scripts/n15_py4j_diag.py       # raw Py4J dispatch health
PYTHONPATH=src py -3.11 scripts/n16_verify.py          # 5x HARVEST sequential verify
PYTHONPATH=src py -3.11 scripts/n17_eval_repro.py      # find numpy.object_ obs keys
PYTHONPATH=src py -3.11 scripts/n18_eval_e2e_test.py   # full scenario_runner against random RLModule
```

All take ~1-5 minutes. They compete with the live training run for instance-1, so kill the training first.

### Watch v21 progress

```bash
PROGRESS=$(ls -t runs/aiutopia_M1_seed1/PPO*/progress.csv | head -1)
python -c "import csv; rows=list(csv.DictReader(open('$PROGRESS'))); print(rows[-1])"
```

Or the smoke summary:

```bash
tail -50 Research/train-v21.log | grep -E "iter|Total running"
```

### Stop training cleanly + force a checkpoint

```bash
DRIVER_PID=$(cat Research/train-v21.log.pid)
powershell -Command "Stop-Process -Id $DRIVER_PID"
# Tune signal handler should fire checkpoint_at_end=True
# Checkpoint lands at runs/aiutopia_M1_seed1/.../checkpoint_*/
```

### Manually evaluate a saved policy

```python
from ray.rllib.algorithms.algorithm import Algorithm
algo = Algorithm.from_checkpoint("runs/aiutopia_M1_seed1/PPO_.../checkpoint_xxx")
module = algo.get_module("gatherer_policy")
# pass module=module into scripts/n18_eval_e2e_test.py main()
```

Or use the existing `aiutopia.cli.promote_weights` CLI (T16/T17).

### Rebuild the Java mod after editing skills

```bash
cd fabric_mod
export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10
./gradlew build
# new jar at build/libs/aiutopia-mod-0.0.0-<version>.jar
# copy to each instance's mods/ and restart all 4 Java processes
```

## 11. Key constraints

- **Python 3.12** declared in pyproject.toml — but **the codebase actually runs under 3.11** via `PYTHONPATH=src` (no editable install). 3.12 is the eventual target.
- **Minecraft Java 1.21.1** pinned (UnionClef baseline; see IMPLEMENTATION_PLAN T19 if bumping)
- **Fabric API 0.116.12 / Carpet 1.4.147 / FerriteCore 7.0.3**. **Lithium is NOT installed on training instances** (N11)
- **Java 21** (Microsoft OpenJDK portable at `/c/Users/Carte/jdk/jdk-21.0.11+10/`)
- **Identifiers:** ULID Crockford base32 (`python-ulid` package)
- **Stop the world on commit hooks** — `ruff`, `mypy`, `pytest` must pass via `pre-commit`

## 12. Known broken / disabled

- **Eval gate `M1/gate_passed`** — Ray 2.55 doesn't expose `custom_metrics/*` as a Tune stop-key path (N7.5). The callback still emits the metric to `result.json` and `progress.csv` once eval runs; use `scripts/m1b-evaluation-gate.sh` post-hoc to check gate passage.
- **v21 eval disabled** — launched with `--evaluation-interval 999` to maximize training time. N17 fix is in place; just need to re-enable on next run.
- **Per-step time degradation** — even with 4g JVM heap, step time grows ~3× over the first 5-10 iters of a run. Suspected G1 GC pressure or Carpet fake-player state accumulation. Not yet root-caused.
- **3 of 4 instances bias toward cobblestone** despite N19 arena bounds. The bounds help but the policy still needs many more iters before the gradient cleanly favors oak_log.
- **No automatic checkpoint until iter 50** (`train.py:101`) or graceful shutdown.

## 13. Important files modified today

```
fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java    # N16, N16c
fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/DepositChestSkill.java # N16b (same precision fix)
fabric_mod/gradle.properties                                               # version → 0.0.0-m1c-n16c
src/aiutopia/env/wrapper.py                                                # N14 logging + N19 arena bounds
src/aiutopia/env/reward.py                                                 # N14 eager ItemId seed
src/aiutopia/train/scenario_runner.py                                      # N17 dict batching + N18 max_ticks
src/aiutopia/train/config.py                                               # max_episode_ticks 12000→2000→300, N19 smoothing 200→20
scripts/launch-training-instances.sh                                        # N19 -Xms2g -Xmx4g
```

## 14. New diagnostic scripts added today

- `scripts/n14_reward_probe.py` — 6-step HARVEST against reset arena; verifies reward path; **was the smoking gun for N16**
- `scripts/n15_py4j_diag.py` — raw Py4J health (health, runCommand, dispatch); proved dispatch path was fine
- `scripts/n15b_payload_diag.py` — full-fat action dict vs minimal
- `scripts/n15c_bisect.py` — bisect which action_dict field broke dispatch
- `scripts/n16_pos_trace.py` — agent position trace exposing the float-precision attractor
- `scripts/n16_verify.py` — 5 consecutive HARVEST dispatches (was the "did we fix it?" gate)
- `scripts/n16b_arena_check.py` — confirm log placement at y=66
- `scripts/n16b_probe_state.py` — agent pos + inventory + block state inspector
- `scripts/n17_eval_repro.py` — walks each obs key and reports which produce dtype=object
- `scripts/n18_eval_e2e_test.py` — standalone scenario_runner against a random RLModule

## 15. Pending tasks (per TaskList)

```
N3   in_progress  Launch training run + monitor first 30 min
N4   pending      Intervene if non-convergent (hyperparameter tuning)
N5   pending      Eval gate verification + weight promotion
N6   pending      Final T22 — M0_PROGRESS + m1b-verified tag
N7-followup pending  3 brainstorm caveats for M2 plan
T21  in_progress  Empirical training run to evaluation gate
```

## 16. The big open question — ✅ RESOLVED (N21, 2026-05-29): we built it

**Should we switch to a custom Python/JAX simulator for M1B?** → **Yes, and we did** — a
scalar-NumPy sim (not yet JAX) behind the same obs/action/reward contracts. Measured **~250×**
over real MC, converges in ~90 s, transfer-validated. The decision below played out as
predicted; see **§9.1** for the measured numbers and the corrected JAX outlook. The framing
below is preserved for the record (one correction inline).

**Should we switch to a custom Python/JAX simulator for M1B?**

Arguments **for**:
- ~~1000-50000× throughput~~ → **~250× delivered** by the scalar-NumPy sim; a *further*
  ~200–500× via a full JAX rewrite (the "1000-50000×" was a macro-step-vs-primitive-tick
  unit artifact — §9.1). The GPU + threads are **still mostly idle** even in the sim, because
  the bottleneck is now RLlib framework + policy overhead, not the env.
- M1B task is simple: agent position, oak_log grid, walk + break
- ~500-1000 lines of code
- Same `build_role_observation_space` / `build_role_action_space` contracts; policy drops into real-MC env unchanged
- Bug surface dramatically smaller (no JVM, Carpet, chunk loading, AABB precision)

Arguments **against**:
- Sim-to-real gap (policy may exploit sim quirks)
- Today's pipeline still has value for M2+ (where real MC complexity matters)
- Engineering investment

**Recommended path**: build a fast sim alongside the real-MC env; train PPO in sim 99% of the time; reuse `M1EvalScenarioCallback` to validate transfer in real MC every N iters; deploy in real MC. This preserves today's infrastructure while removing the throughput ceiling.

## 17. Useful absolute paths

```
Project root:           /c/Users/Carte/OneDrive/Desktop/AiUtopia
Working directory:      /c/Users/Carte/OneDrive/Desktop/AiUtopia/Research (Claude session)
Java home:              /c/Users/Carte/jdk/jdk-21.0.11+10
v21 training log:       Research/train-v21.log
v21 driver PID file:    Research/train-v21.log.pid
v21 Tune trial dir:     runs/aiutopia_M1_seed1/PPO_aiutopia_minecraft_9c764_00000_0_2026-05-28_17-24-29/
Active Fabric jars:     server-runtime/training/instance-{1,2,3,4}/mods/aiutopia-mod-0.0.0-m1c-n16c.jar
Active player name:     gatherer_0 (on all 4 instances)
Spawn coords:           (64.5, 66, -47.5)  — log ring radius ~7 blocks
```

## 18. References inside the repo

- `IMPLEMENTATION_PLAN.md` (5774 lines) — Master plan M0-M6, every section numbered (§1-§7.8)
- `M1A_PIPELINE_PLAN.md` (4035 lines) — Tasks T1-T22 with checkbox-level detail
- `M1B_TRAINING_PLAN.md` (3107 lines) — How to actually train
- `M0_PROGRESS.md` — M0 verification with live smoke logs
- `NEXT_SESSION.md` — Per-session handoff (currently post-v20)
- `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md` — Original design spec
- `docs/superpowers/specs/2026-05-27-m2-builder-design.md` — M2 builder brainstorm (N7)
- `Kimi_Agent_Minecraft AI Village Research/` — Reference research (sections 00-11)
- `GeminiResearch.md` — Additional research notes

---

**Commit history summary (this session):**

```
14b32dc  fix(train): N19 — arena bounds + JVM heap + smoothing window
55789e2  docs(NEXT_SESSION): post-v20 handoff — N14-N18 fixes verified
0c52bd4  fix(eval): N18 — scenario max_ticks 1000 → 300
abe1c12  fix(train): N17 — recursive obs batching for nested dicts + max_episode_ticks 2000→300
4a9bff0  fix(train): N16-followup — max_episode_ticks 12_000 → 2_000
694b44d  fix(motor): N14+N15+N16 — restore HARVEST reward signal end-to-end
```

End of context. Read `NEXT_SESSION.md` for actionable next steps.
