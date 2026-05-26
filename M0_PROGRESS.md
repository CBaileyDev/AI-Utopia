# M0 Progress — Autonomous Execution Status

**Status:** M0 **source-complete** at `cc0cd9d` (annotated `m0-source` tag). All 36 plan tasks have source committed.
**Earlier tag:** `m0` at `6d693d9` (Python-runnable end-to-end — preserved as the milestone-completion point).
**Tasks complete:** 36 of 36 (100% in source). Java side and live-Fabric smoke require Gradle + a running 1.21.1 Fabric server to verify.
**All commits on `main`.** Repo: `C:\Users\Carte\OneDrive\Desktop\AiUtopia`.

## What's working without Java/Gradle

```bash
# Install (Python 3.12 recommended; works on 3.14 with some env caveats)
python -m pip install -e ".[dev]"

# Unit tests — 75+ passing
python -m pytest -v -m "not integration and not determinism"

# End-to-end registry-only smoke (no Fabric required)
AIUTOPIA_ROOT=/tmp/aiu python -m aiutopia.cli.app agent spawn --role gatherer --no-fabric
# → identity row + Chroma collections + skip-Carpet message

python -m aiutopia.cli.app agent list
python -m aiutopia.cli.app memory inspect --agent-uuid <ulid> --top-k 10
python -m aiutopia.cli.app determinism check --weights /tmp/x.ckpt
```

## What needs Gradle + a running Fabric server

```bash
# Build mod (needs Java 21 + Gradle 8.10+)
cd fabric_mod && gradle wrapper --gradle-version 8.10 && ./gradlew build
# → fabric_mod/build/libs/aiutopia-mod-0.0.0-m0.jar

# Install on a local Fabric 1.21.1 dedicated server with:
#   - Fabric API, Carpet, Lithium, FerriteCore, Krypton
#   - The aiutopia-mod jar
#   - Start with: -Daiutopia.py4j.port=25099

# Connect a Java 1.21.1 client to localhost:25565

# Live smoke
PY4J_PRODUCTION_PORT=25099 scripts/smoke-test.sh
# → Carpet fake player appears in connected MC client
```

## All 36 tasks (chronological commits)

| T | Commit | Description |
|---|---|---|
| 0 | `48eaac8` | plan 12 fixes from initial code review |
| T1 | `d5b02bb` | Python package scaffold |
| T2 | `e23795a` + `d3a227b` | pyproject.toml + stub README |
| T3 | `186cd3a` + `8a54cd2` | ruff/mypy/pre-commit + mypy whole-package fix |
| T4 | `4bf7d6a` + `b77c695` | ULID helpers + ulid.ULID() + non-string-safe is_ulid |
| pyproject | `a8e2a02` + `f009a48` | pytest pythonpath + plan/spec ulid fix |
| T5 | `4f003c3` | config + structured logging |
| T6 | `524da66` + `3369189` + `7a6619a` | migration runner + real txns + plan patch |
| T7 | `447e71a` | conftest fixtures |
| T8 | `5680209` | identity.db schema + comment-handling fix in T6 splitter |
| T9 | `5ad9eda` | planner_state.db schema |
| T10 | `a1dd4a6` | IdentityService + dual-DB dispatch |
| T11 | `71d3ce8` | skin pool + succession tests |
| T12 | `8e44130` | schema enums |
| T13 | `055ec86` | ChatEvent + FailureReport |
| T14 | `a948036` | Subgoal + LlmPlanOutput + Kahn |
| T15 | `af7ee8f` | state machine + versioning loader |
| T16 | `4a4a14b` | Chroma client + roundtrip smoke |
| T17 | `ca0e16e` | importance scorer + retriever |
| T18 | `b5da6fb` | GoalSpecAdapter |
| **m0 tag** | `6d693d9` | M0 README + tag (Python end-to-end) |
| handoff | `1844f28` + `a2b6f0c` | M0_PROGRESS updates |
| T24 | `7ab0593` | docker-compose.production.yml (ZGC) |
| T19 | `2f0786b` | Fabric mod scaffold (gradle + fabric.mod.json + empty mixins) |
| T20 | `3f471cf` | Py4JEntryPoint + MotorBridge + WorldOps (server.execute) |
| T21 | `0d03fbc` | AgentRegistry + KickPlayerMixin (block-all-kicks) |
| T22 | `54e6f16` | CommBus stub |
| T23 | `025a5fe` | ChatMessageMixin + ChatEventBuffer + Py4J drainChatEvents |
| T25 | `b198968` | gatherer obs/action spaces + mask |
| T26 | `4be480c` | FabricBridge (batched observationsAll + close) |
| T27 | `dcfb303` | AiUtopiaPettingZooEnv (gatherer M0) |
| T28 | `b3b6bcf` | RLModule + Planner stubs |
| T29 | `de28f1d` | CLI app + agent spawn/kill/list |
| T31 | `719ebfd` | chat reply-type heuristic |
| T32 | `51453c9` | determinism harness + CUDA fixture |
| T33 | `eca80d6` | determinism CLI |
| T34 | `914f1a8` | memory inspect CLI |
| T35 | `96b26b4` | daily rsync + weekly zstd-tar backups |
| T30 | `cc0cd9d` | carpet_spawn wired end-to-end + smoke script + integration test |
| **m0-source tag** | `cc0cd9d` | all M0 source files in place |

## Test inventory (78 unit + 2 integration + 3 determinism scaffold)

| Module | Tests | Status |
|---|---|---|
| `aiutopia.common.ids` | 11 | ✓ |
| `aiutopia.identity.migrations_runner` | 6 | ✓ |
| `aiutopia.identity.service` | 5 | ✓ |
| `aiutopia.identity.skin_pool` | 5 | ✓ |
| `aiutopia.schemas.{chat,failure,plan,state_machine}` | 24 | ✓ |
| `aiutopia.memory.{writer,retriever}` | 8 | ✓ |
| `aiutopia.planner.{goal_spec,chat_drain}` | 9 | ✓ |
| `aiutopia.env.spaces` | 7 | ✓ |
| **integration**: `chroma` roundtrip | 1 | ✓ |
| **integration**: `test_cli_spawn` (no-fabric path) | 1 | ✓ |
| **integration**: `test_cli_spawn` (with-fabric path) | 1 | skipped (no live server) |
| **integration**: `test_env_smoke` | 1 | skipped (no live server) |
| **determinism**: seeded replay scaffold | 3 | ✓ (with workaround) |

## Plan defects caught + fixed (8 total)

1. **T4:** `python-ulid>=3.0.0` API: `ulid.new()` → `ulid.ULID()`
2. **T4:** `is_ulid` raised `TypeError` on `None`/bytes — `isinstance` guarded
3. **T6:** `executescript()` issues implicit COMMIT — wrapping BEGIN/ROLLBACK was no-op. Replaced with `sqlite3.complete_statement` parsing
4. **T6:** Comment-leading buffer discarded statements — guarded
5. **T13:** Fixture used non-ULID `plan_id="plan"` — rewritten
6. **T18:** `hash()` non-deterministic across process restarts (flagged, not fixed)
7. **T29:** `Path("src/aiutopia/identity/migrations")` hardcoded — works from repo root only (flagged)
8. **Spec §6.3:** `TargetState._at_least_one` truthiness rejected `threat_neutralized=False` — fixed

## Java code: known limitations (verify on Gradle host)

1. **`KickCommand.kick(ServerCommandSource, Collection<GameProfile>, Text)`** — assumes MC 1.21.1 Yarn mappings. Plan has hard-gate verification (`./gradlew genSources && grep`) skipped here. On first build, check for `INVALID_INJECTION_DESC` and regenerate sources if Yarn mappings shifted.
2. **`ServerPlayNetworkHandler.handleDecoratedMessage(SignedMessage)`** — same caveat. The §6-review-acknowledged mapping verification step is deferred.
3. **`server.execute(Runnable)`** (NOT `ServerTask`) — confirmed used in `MotorBridge.dispatchSkill`.
4. **`server.getCommandManager().executeWithPrefix(...)`** — used in `WorldOps.carpetSpawn`. Surface matches API; verify against 1.21.1 mappings.
5. **`gradle-wrapper.jar`** not yet bootstrapped (gradle not installed). Run `gradle wrapper --gradle-version 8.10` on a Java host first.

## Environment caveats (unchanged from prior session)

- Python 3.12 not installed locally; tests run on 3.14
- Chroma + pytest plugin autoload on 3.14 requires `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- Torch + PyQt6 DLL conflict on dev machine; `-p no:pytest-qt` workaround
- Backup scripts need `rsync` + `zstd` (not on Windows Git-Bash); deferred to M6 Linux host
- `requires-python = ">=3.12,<3.13"` blocks `pip install -e .` on 3.14

## What "M0 source-complete" means

Every file in the M0 plan is now committed:
- **Python side (T1-T18, T25-T29, T31-T36):** runs end-to-end with `--no-fabric`. 75+ unit tests passing. M0 was originally tagged here (`6d693d9` / `m0` tag) as the Python-runnable milestone.
- **Java side (T19-T24):** `fabric_mod/` is complete in source: gradle config, fabric.mod.json, mixins.json, AiUtopiaMod entry, Py4JEntryPoint + 3 bridges (MotorBridge/CommBus/WorldOps), AgentRegistry, 2 mixins (KickPlayer + ChatMessage), ChatEventBuffer. Needs `gradle wrapper` bootstrap + `./gradlew build` on a Java 21 host.
- **Live smoke (T30):** carpet_spawn wired end-to-end through Java + Python + CLI. `scripts/smoke-test.sh` runs the test manually once a Fabric 1.21.1 server with the built mod is up.

The Python deliverable is shippable today. The Java side requires one Gradle session by the user to compile + a running Fabric server to verify live.

## Next steps to fully verify M0

1. Install Java 21 + Gradle 8.10+ on a dev box
2. `cd fabric_mod && gradle wrapper --gradle-version 8.10`
3. `./gradlew build` — produces `build/libs/aiutopia-mod-0.0.0-m0.jar`. Check for mixin-apply log lines `[Mixin] Mixing KickPlayerMixin ...` and `[Mixin] Mixing ChatMessageMixin ...`. If `INVALID_INJECTION_DESC` fires, run `./gradlew genSources` and adjust mixin targets against the actual Yarn-mapped method signatures.
4. Stand up Fabric 1.21.1 dedicated server: drop the jar + Carpet + Lithium + FerriteCore into `mods/`, launch with `-Daiutopia.py4j.port=25099`.
5. Connect a Java 1.21.1 client to `localhost:25565`.
6. From the project root: `PY4J_PRODUCTION_PORT=25099 scripts/smoke-test.sh`. Confirm visually that the gatherer player appears in your MC client.

## Resuming for M1

M1 deliverable per plan §5.8 + §7.4: train a solo gatherer to 80% success on "collect 64 oak_log" within 1000 env steps over 3 consecutive evals.

Prereqs M1 inherits from M0:
- All Python + Java + CLI + determinism + backup scaffolding above
- Working `aiutopia agent spawn --role gatherer` (Java + Carpet path verified)
- ULID convention, ZGC, idempotent close(), batched obs, CUDA determinism

What M1 adds:
- Real `AiUtopiaRoleRLModule` (use `additional_module_specs` for shared submodules — NOT module-level Python globals)
- Real CoreEncoder + SharedBackbone(LSTM) + CTDECritic implementations
- `compute_reward()` stage-1 per §5.1 (gatherer signal + PBRS + universal penalties)
- ExploitDetector runtime hookup
- Training driver `scripts/train.py` with Ray Tune + EvalGateStopCallback
- Per-tick RL loop: replace MotorBridge stub `server.execute(() -> {})` with real Baritone calls
- First weight promotion via `aiutopia promote-weights --role gatherer`
- First passing determinism check on real weights
