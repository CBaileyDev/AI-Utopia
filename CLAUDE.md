# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A persistent, multi-agent **Minecraft AI village**: four specialized RL agents
(gatherer, builder, farmer, defender) cooperatively run a village on a private
Minecraft Java **1.21.1** server. Python drives reinforcement learning
(Ray RLlib PPO + per-role LSTM RLModule); a Java Fabric mod exposes the live
Minecraft world over a Py4J socket. Currently mid-**M1** (single-role gatherer
training pipeline).

**Canonical living docs — read these before deep work, don't duplicate them here:**
- `PROJECT_CONTEXT.md` — full project overview, architecture diagram, hardware, throughput analysis.
- `NEXT_SESSION.md` — current run state, active PIDs, immediate next steps (this is where *volatile* state lives, not CLAUDE.md).
- `IMPLEMENTATION_PLAN.md` (M0–M6 master), `M1A_PIPELINE_PLAN.md` (tasks T1–T22), `M1B_TRAINING_PLAN.md` (training to gate) — large checkbox specs.
- `docs/superpowers/specs/` — committed design specs.

## Shell

Commands below assume **git-bash via the Bash tool** (sources use `cd /c/...`,
`nohup … &`, `export VAR=`). The default session shell is **PowerShell** — for
the few commands that differ, PowerShell equivalents are noted. Env-var prefix
`VAR=x cmd` is bash-only; in PowerShell use `$env:VAR='x'; cmd`.

## Commands

### Python: runtime vs. declared (important gotcha)

`pyproject.toml` declares `requires-python >=3.12,<3.13`, but the code is run
operationally under **Python 3.11** via `PYTHONPATH=src` with **no editable
install** (`pip install -e .` fails the 3.12 gate — that's *why* `PYTHONPATH` is
used). Prefer the operational form for anything that runs training/scripts:

```bash
PYTHONPATH=src py -3.11 scripts/<script>.py        # bash
# PowerShell:  $env:PYTHONPATH='src'; py -3.11 scripts/<script>.py
```

The 3.12 editable install (`python -m pip install -e ".[dev]"`) is the eventual
target and is what CI/tooling configs (ruff `py312`, mypy `3.12`) assume.

### Tests (pytest, marker-gated)

`pytest.ini_options` sets `pythonpath=["src"]`, so plain `pytest` resolves the
package without an install.

```bash
pytest -v -m "not integration and not determinism"   # default fast unit suite
pytest tests/unit/test_reward.py                       # one file
pytest tests/unit/test_reward.py::test_oak_log_value   # one test
pytest -m integration    # needs a live Fabric server (skips gracefully if unreachable)
pytest -m determinism    # needs GPU + saved weights
```

Markers: `integration` (external services: Chroma, Fabric), `determinism`
(GPU weights-replay), `slow`.

### Lint / type-check (must pass — enforced by pre-commit)

```bash
ruff check .          # add --fix to auto-fix
ruff format .
mypy                  # strict; checks src/aiutopia only (config in pyproject.toml)
pre-commit run --all-files
```

### Build the Fabric mod (Java 21)

Server-side mod only. Requires **JDK 21** (portable at `/c/Users/Carte/jdk/jdk-21.0.11+10`).

```bash
cd fabric_mod
export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10
./gradlew build       # → build/libs/aiutopia-mod-<mod_version>.jar
```

After editing skills/obs, the new jar must be **copied into each running
instance's `mods/` and all 4 Java processes restarted** — Fabric does not
hot-reload. `mod_version` lives in `fabric_mod/gradle.properties`.

### Run / operate the system

```bash
# Launch 4 Fabric training servers (idempotent; waits for "Done (", then bootstraps each arena)
JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 bash scripts/launch-training-instances.sh

# Train (Ray Tune driver). Writes a PID file next to the log.
PYTHONPATH=src PYTHONUNBUFFERED=1 AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
  nohup py -3.11 -u scripts/train.py --milestone M1 --max-iters 200 --evaluation-interval 10 \
    > Research/train-vN.log 2>&1 &
#   --evaluation-interval 999 disables eval; --num-env-runners defaults to 4 (one per instance); --seed

# End-to-end smoke test (needs a live server). The port MUST match the port the
# server was launched on (-Daiutopia.py4j.port=…). The README quickstart uses 25099.
PY4J_PRODUCTION_PORT=<server-launch-port> scripts/smoke-test.sh

# CLI (entry point: aiutopia = aiutopia.cli.app:app)
PYTHONPATH=src py -3.11 -m aiutopia.cli.app agent spawn --role gatherer
#   subcommands: agent (spawn/list), memory (inspect), determinism (check), promote-weights
```

Standalone diagnostics live in `scripts/n14_reward_probe.py` … `n18_eval_e2e_test.py`
(reward path, raw Py4J health, HARVEST verify, eval repro). They contend for
instance-1 — **stop the training run first**.

## Architecture (three layers, top to bottom)

```
Python RL  ──Py4J TCP──>  Java Fabric mod  ──>  Carpet fake players
```

**1. Python RL stack** (`src/aiutopia/`)
- `train/` — `m1_gatherer_config` builds the PPOConfig (new RLlib API stack); Ray Tune driver is `scripts/train.py`; `callbacks.py` composes metrics / exploit-hunt / eval-scenario / eval-gate callbacks; `scenario_runner.py` runs greedy LSTM-threaded eval.
- `rl_module/` — the policy is assembled from layers: `CoreEncoder` → `SharedBackbone` → role-specific `RoleEncoder` → `ActorHead`, plus a CTDE `ctde_critic` (centralized training, decentralized execution). `role_rl_module.py` ties them together with the LSTM time dimension.
- `env/wrapper.py` — `AiUtopiaPettingZooEnv` (PettingZoo **Parallel** API): `reset()` → bridge reset; `step()` → dispatch skill, advance ticks, read obs, apply reward shaping + exploit penalties.
- `env/reward.py` — `LOG_VALUE` per-item table + potential-based reward shaping (PBRS); `env/exploit.py` — `ExploitDetector` penalties; `env/spaces.py` — `build_role_observation_space` / action space (the contract a policy must satisfy in *any* env, real or future-sim).
- `env/bridge.py` — `FabricBridge`, the Py4J client. Owns the JavaGateway lifecycle.

**2. Java Fabric mod** (`fabric_mod/src/main/java/dev/aiutopia/mod/`)
- `Py4JEntryPoint` — the Python-facing surface (`observationsAll`, `motorBridge`, `resetWorld`, `advanceTickAwaitEvents`, `getItemIdNameTable`, `setupTrainingScene`). Forked from UnionClef.
- `bridge/MotorBridge` (skill dispatch + per-tick driver), `bridge/WorldOps` (arena setup + `resetEpisode` + Carpet spawn), `bridge/skill/*` (Navigate/Harvest/DepositChest/Search/Wait).
- `obs/` (observation builder + `ItemIdTable`), `agent/` (name→role registry), `chat/` (`@<agent_name>` router), `mixin/` (e.g. `KickPlayerMixin`).

**3. Carpet fake players** — one bot per instance, server tick warped above 20 TPS.

**Two-world topology:** 4 *training* instances on Py4J ports **25001–25004** (MC
25566–25569); a single *production* server on Py4J port **25100** (config default;
note the smoke-test docs reference `25099` — `config.py` is the source of truth,
overridable via `PY4J_PRODUCTION_PORT`). Ports/paths come from
`src/aiutopia/common/config.py` (`Paths.from_env` keyed on `AIUTOPIA_ROOT`,
`Py4JConfig` for ports).

**Supporting subsystems:** `identity/` (SQLite + SQL migrations, ULID Crockford
base32 IDs, skin pool), `memory/` (Chroma episodic memory + retriever/writer),
`planner/` (LLM planner — stubs until M5), `promotion/` (the §5.10 weight-promotion
checklist), `schemas/` (Pydantic v2).

## Critical constraints & gotchas

These bite across files; they are not obvious from any single source:

- **Batched observations only.** Read all agents via `observationsAll()` once per
  tick — never `observation(agent)` per-agent. Per-agent Py4J roundtrips cap
  throughput (~300 agent-steps/s). This is a hard performance invariant.
- **`FabricBridge.close()` is mandatory** but must *not* call `JavaGateway.shutdown()`
  — that kills the shared server-side gateway for every other env worker. Close the
  client connection only.
- **`num_learners=0` on Windows.** PyTorch lacks libuv there, so the Learner must
  run in the driver process (`train.py` forces this).
- **Lithium is NOT installed on training instances** (N11): its `forEachInBox`
  optimization throws CME with Carpet fake players mutating entity lists under tick
  warp. Stack is Fabric API + Carpet + FerriteCore only for training.
- **Server tick rate ≤ ~60.** Higher (300 was tried) crashes the server
  (fastutil rehash AIOOBE / GC pressure).
- **Skill movement uses float-precision-sensitive reach math** (`HarvestSkill`,
  `DepositChestSkill`): an over-tight `REACH_RADIUS` + `step = min(WALK, dist-REACH)`
  can collapse the per-tick step to ~0 and silently stall the agent forever. Keep
  reach generous (≥4.5) and always apply full `WALK_PER_TICK`.
- **The eval gate is not a Tune stop-key.** Ray ≥2.55 doesn't expose
  `custom_metrics/*` to Tune's stop criteria; `EvalGateStopCallback` emits the
  metric for observability, but gate passage is checked **post-hoc** via
  `scripts/m1b-evaluation-gate.sh`. Tune stops only on `training_iteration`.
- **GC differs by environment:** training instances use **G1GC**; the M6
  production deploy (`docker-compose.production.yml`) targets **Generational ZGC**.
- **MC pinned to 1.21.1** (UnionClef baseline). Bumping requires re-verifying the
  UnionClef fork compiles — see `gradle.properties` / IMPLEMENTATION_PLAN T19.
- **Throughput bottleneck is the single-threaded JVM sim,** not the policy net.
  A training step is dominated by the Java skill run (1–15 s); the LSTM forward is
  microseconds. GPU/CPU sit ~idle. Don't chase Python-side perf for training speed.
- **Keep the volatile SQLite stores off OneDrive.** `identity.db`,
  `planner_state.db`, and Chroma (`chroma/chroma.sqlite3`) are WAL-mode SQLite; a
  file-sync client touching their `-wal`/`-shm` sidecars can corrupt the database.
  They resolve under `AIUTOPIA_ROOT`, which the operational commands currently point
  at this **OneDrive-synced repo** — a live corruption risk. Set **`AIUTOPIA_DATA_DIR`**
  to a local-disk base (e.g. `C:\Users\Carte\aiutopia-data`) to relocate ONLY those
  three stores there; repo-committed assets and `weights/` / `runs/` stay under
  `AIUTOPIA_ROOT`. Unset = unchanged behaviour. The per-worker root suffix is
  preserved as `<AIUTOPIA_DATA_DIR>/<root-name>_w{N}` (see `env/wrapper.py`), so
  concurrent EnvRunners stay isolated. **One-time migration (with training stopped):**
  the persistent `identity.db` / `planner_state.db` live under the **un-suffixed** CLI
  root — copy those into `<AIUTOPIA_DATA_DIR>/<root-name>/` so agent ULIDs carry over.
  Training `chroma/` is currently **scattered across the `_w*` worker dirs** (see the
  bug note below), so decide which episodic memory is worth keeping before copying it.
- **Related bug — per-worker root compounds.** `env/wrapper.py` re-suffixes
  `AIUTOPIA_ROOT` *in place* on every `__init__`, so it accumulates across
  re-instantiation: each re-init writes a fresh `chroma/` under a new `…_w0_w0_w0…`
  path, scattering per-worker episodic memory instead of accumulating it in one place.
  Low impact at M1; a real correctness issue by M2+ (the premise is "memory grows with
  experience"). Fix is a one-line idempotent guard: only append `_w{widx}` when `base`
  doesn't already end with it.

## Conventions

- **Commit per plan checkbox.** Each change is a focused commit advancing one task
  in the relevant plan file. Integration fixes during M1B are numbered `N<n>`
  (e.g. `fix(motor): N16 — …`); the running ledger is in `PROJECT_CONTEXT.md` / git
  log, so don't restate the chronology elsewhere.
- **`pre-commit` is the gate:** `ruff`, `ruff-format`, and whole-package `mypy`
  (strict, `src/aiutopia` only) must pass before commit.
- **All identifiers are ULID** (Crockford base32, `python-ulid`).

## Specialist subagents

These domain experts are installed at the user level (`~/.claude/agents`, symlinked
to the agent collection at `C:\Users\Carte\agents`) and have been tuned with this
project's gotchas (the N-fix ledger, Ray-2.55 quirks, the JVM-sim bottleneck). Reach
for them via the Task/Agent tool when the work matches:

- **RL training dynamics, PPO config, Ray/RLlib new-API-stack drift, NaN-KL / entropy / value diagnosis** → `deep-rl-training-specialist`
- **MARL: MAPPO/CTDE, `policy_mapping_fn` / MultiRLModule, cross-policy weight sharing, role collapse** → `multi-agent-rl-specialist`
- **Reward shaping (PBRS), eval-gate discipline, imitation/BC (latent until a pixel stack exists)** → `reward-design-and-imitation-learning-specialist`
- **Env wrapper, obs/action spaces, Py4J bridge boundary, train-vs-deploy parity, determinism/reset** → `minecraft-rl-environment-specialist`
- **Fabric mod, skills (HARVEST / place_block / …), Carpet, Yarn mappings, world ops, JVM/GC server tuning** → `minecraft-modding-and-server-specialist`
- **LLM planner architecture (M5 DAG runtime, memory salience, planner↔policy interface)** → `agent-behavior-architecture-specialist`
- **Anthropic SDK plumbing: cost caps, retry/backoff, prompt caching, Qwen fallback, RAG** → `llm-application-builder`
- **JVM/GC + step-time profiling (profile the JVM sim, not the GPU)** → `performance-and-profiling-engineer` / `java-kotlin-specialist`
- **SQLite schema / migrations / WAL, Chroma-on-SQLite reality** → `sql-and-database-specialist`
- **Numeric Python: NumPy/Polars vectorization, numpy↔JSON/SQLite serialization** → `data-science-numerics-specialist`

Each carries this project's specific gotchas (N16 float-precision skill stall, N17
nested-Dict obs batching, the Ray-2.55 eval-gate-not-a-stop-key, `num_learners=0` on
Windows). Background: `C:\Users\Carte\agents\docs\AIUTOPIA_AGENT_GAP_ANALYSIS.md`.
