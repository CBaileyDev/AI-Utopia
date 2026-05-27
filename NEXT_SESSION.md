# Next session runbook — M1B training to `m1b-verified` tag

**Resume point:** repo HEAD `4ca6bd8`. M1B code-complete + 10 integration fixes applied + 3 "likely to break" pre-emptive fixes. 142/142 unit tests pass. `aiutopia-mod-0.0.0-m1b.jar` deployed to 5 mod directories. No live processes — last session cleaned up.

**Goal this session:** Run T21 to convergence, hit the 80% eval gate, promote weights, tag `m1b-verified`. Then optionally start M2 brainstorming.

---

## Phase 1 — Pre-flight (5 min) — Task N1

```powershell
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
git log --oneline -1                           # expect: 4ca6bd8
$env:PYTHONPATH = "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
py -3.11 -m pytest tests/unit -q | Select-Object -Last 3   # expect: 142 passed
nvidia-smi | Select-Object -First 8            # expect: RTX 4080, CUDA OK
& "C:\Users\Carte\jdk\jdk-21.0.11+10\bin\java.exe" --version   # expect: openjdk 21.0.11
Get-ChildItem server-runtime\mods\aiutopia-mod-*.jar | Select-Object Name, Length
```

Stop here if any check fails.

---

## Phase 2 — Bootstrap 4 training instances (3 min) — Task N2

```bash
# Git Bash
cd "/c/Users/Carte/OneDrive/Desktop/AiUtopia"
JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 ./scripts/launch-training-instances.sh
# Wait ~5s. Verify all 4 servers booted:
for i in 1 2 3 4; do tail -1 "server-runtime/training/instance-$i/instance-$i.log"; done
# Expect "Done (X.Xs)!" on each line.
```

Then run setup (forceloads chunks + flattens the grass arena per commit 95c1d55):

```powershell
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
$env:PYTHONPATH = "src"
py -3.11 -c "from aiutopia.env.bridge import FabricBridge
for port in (25001, 25002, 25003, 25004):
    with FabricBridge(port=port) as b:
        ok = b.setup_training_scene()
        print(f'port {port}: setup={ok}')"
```

Expect: 4 lines, all `setup=True`.

**Note:** the wrapper's `auto_spawn_agents` flag (commit eeee2de) means train.py spawns its own `gatherer_0` agents per worker. You do **not** need to run `aiutopia agent spawn` manually anymore.

---

## Phase 3 — Train (hours, mostly hands-off) — Task N3

```powershell
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
$env:AIUTOPIA_ROOT = "C:\tmp\aiu-m1b-train"
$env:PYTHONPATH = "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
py -3.11 scripts\train.py --milestone M1 --max-iters 2000 `
    --evaluation-interval 50 --num-env-runners 4 --num-envs-per-runner 1 `
    --num-learners 0 --seed 1
```

In a second shell:

```powershell
tensorboard --logdir C:\tmp\aiu-m1b-train\runs --port 6006
# open http://localhost:6006 in browser
```

**Healthy run signals (first 30 min):**
- log line `Trial PPO_aiutopia_minecraft_XXX started with configuration:` within 2 min
- `training_iteration: 1` completes within ~5 min of launch
- `iter rate >= 1/min` thereafter (each iter samples 4096 env steps across 4 workers at tick rate 300)
- `episode_return_mean` is non-zero (positive **or** negative) by iter ~50
- Fabric `instance-N.log` shows `Changed the block at` lines on each `env.reset()` — not "Could not set"

**Stop criterion:** the configured Tune stop is `training_iteration == 2000`. The `EvalGateStopCallback` still writes `M1/gate_passed` into `result["custom_metrics"]` but Ray 2.55 doesn't pipe that to Tune's stop-dict key path (T21 fix #9). Either: (a) watch TensorBoard for `eval_m1_oak_log_success_rate >= 0.80` for 3 consecutive evals, then Ctrl-C, (b) let it run all 2000 iters and pick the best checkpoint at the end.

---

## Phase 4 — Conditional: intervene if non-convergent — Task N4

If `episode_return_mean` stays at 0 (or strongly negative) past iter 200, apply diagnostics in this order:

| Symptom | Likely cause | Fix |
|---|---|---|
| Fabric log shows oak_logs placed each reset, but obs `g_resource_grid` is all zeros | Obs builder doesn't see logs at Y=66 — grid origin mismatch | Probe with `bridge.observations_all()` and inspect `g_resource_grid[6][6]`. Adjust obs builder Y window or place logs at Y=65 instead. |
| Logs placed, obs sees them, but agent never moves | NAVIGATE skill broken or action mask blocking | Check `skill_counters` in worker logs. Run `aiutopia agent drive --agent gatherer_0 --skill 0 ... --py4j-port 25001` manually. |
| Agent moves a bit then stalls forever | Canopy-stall on remaining log debris (M1A T3 known) | Widen log ring radius to 6-10 in `WorldOps.java:resetEpisode` (`r = 6 + epRand.nextInt(5)`), rebuild, redeploy, restart Fabric. |
| `entropy` collapsing to ~0 by iter 50 | Policy over-confident too early | In `train/config.py` bump `entropy_coeff` 0.01 → 0.05 and `kl_coeff` 0.2 → 0.5. |
| `vf_loss` exploding (>1e4) | PBRS overflow on large inventories | Cap PBRS contribution; lower `lr` 3e-4 → 1e-4. |

Each intervention requires: Ctrl-C → fix → relaunch Phase 3. Java fixes require rebuild + Fabric restart (Phase 2 again).

---

## Phase 5 — Eval gate + promotion (15 min) — Task N5

When training stops (auto at iter 2000, or manual Ctrl-C after convergence):

```powershell
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
$BEST_CKPT = (Get-ChildItem "C:\tmp\aiu-m1b-train\runs\aiutopia_M1_seed1\PPO_*\checkpoint_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "Best checkpoint: $BEST_CKPT"
bash scripts/m1b-evaluation-gate.sh "$BEST_CKPT"
```

Expected output: 3 determinism replays each marked PASS, then `section 5.10 checklist: PASS (5 gates)`, then `promoted gatherer: v0 -> v1`.

If determinism replays raise a `RuntimeError: determinism replay: rl_module._forward_inference failed ...` with shape diagnostic (from likely-to-break fix #3) — read the obs/state shapes in the error, locate the mismatch, patch `replay_with_rlmodule` to match.

If checklist Gate 2 (scenario success < 80%) fails: training did not converge. Either continue training (`scripts/train.py --max-iters 4000 ...`) or revisit Phase 4.

---

## Phase 6 — Tag `m1b-verified` (5 min) — Task N6

Append M1B section to `M0_PROGRESS.md` per plan T22 step 3:

```markdown
## M1-Training Progress
**Status:** Complete. Tag `m1b-verified`.
### Empirical results (T21)
- Env steps to gate: <FILL>
- Wall-clock: <FILL>
- Final entropy: <FILL>, vf_loss: <FILL>, kl: <FILL>
- Gatherer weights at: `<paths.weights_dir>/gatherer/v1/`
- Deployment ID: <FILL>
- Determinism: argmax_div=<FILL>, L2=<FILL>
### Integration fixes applied during T21 (vs plan v2)
1. CheckpointConfig.checkpoint_frequency deprecated → removed
2. CheckpointConfig.checkpoint_at_end deprecated → removed
3. ray.train.RunConfig strips Tune kwargs → use ray.tune.RunConfig
4. num_learners=1 crashes on Windows (libuv) → --num-learners 0
5. flush_comm_batch missing _to_python recursion
6. Py4J auto_convert=True needed on GatewayParameters
7. Wrapper auto-spawns own Carpet agents (manual CLI agents broke gym-id mapping)
8. resetEpisode requires forceloaded chunks + flat-grass arena
9. ConnectorV2 flattens Dict obs → role_encoder uses unflatten_role_obs helper
10. custom_metrics/M1/gate_passed not a Tune stop key path in Ray 2.55 → drop from stop dict
```

Then:

```bash
git add M0_PROGRESS.md
git commit -m "docs(M0_PROGRESS): M1-Training complete — first trained gatherer + 10 integration fixes (M1-Training T22)"
git tag -a m1b-verified -m "M1-Training: PPO config + RLModule + training driver + 4-instance Fabric topology + promotion CLI + determinism on real weights; first gatherer weights at weights_dir/gatherer/v1/"
git tag -l
```

Expected tags: `m0`, `m0-source`, `m0-verified`, `m1a-verified`, `m1b-verified`.

---

## Phase 7 — M2 brainstorming (open-ended) — Task N7

Once M1B is tagged, brainstorm M2 scope. Key open questions:

1. **Cross-policy weight sharing.** Plan v2 explicitly punted on this — `MultiRLModuleSpec.additional_module_specs` doesn't exist in Ray 2.55. Options: (a) accept that each role policy has its own copy of CoreEncoder/SharedBackbone/CTDECritic (cheap memory-wise: ~3M params × 4 roles = 12M, still tiny), (b) implement a custom MultiRLModule subclass that shares weights at the learner update step, (c) defer to M3+.
2. **BuilderRoleEncoder pixel patch.** Per spec §4.4 the builder needs a pixel view of the construction site. Approaches: (a) Iris/Sodium offscreen framebuffer (Java mod work — complex), (b) software raycaster that reads chunk data and produces a synthetic 64×64×3 RGB patch (no GPU rendering, deterministic).
3. **Stage-2 reward + curriculum.** Spec §5.1 has a curriculum decay schedule. Test it on solo gatherer first to validate the curriculum mechanism, then add builder.
4. **BULK_FARMING exploit.** Cross-agent inventory transfer is the M2-new exploit. Add to ExploitDetector at env-wrapper level (it currently has per-agent rules only).

Output of Phase 7: a M2 brainstorm document at `docs/superpowers/specs/<date>-m2-builder-design.md`. Not yet a plan — brainstorm first, plan after the design is approved.

---

## Known caveats carried into next session

- **Forceload no-op edge case.** `/forceload add` reported "No chunks were marked" on a fresh boot — chunks at the arena boundary may not be held resident. In practice the agent's spawn-area chunks are kept loaded by player proximity, but if an episode lets the agent wander to the arena edge it could find a chunk hole. If this bites: add explicit `world.getChunkManager().getChunk(cx, cz, ChunkStatus.FULL, true)` calls in `setupTrainingScene` Java to force-generate the chunks at boot.
- **First-reset /clear race.** The wrapper's auto-spawn fires before the player has joined the server, so the first `/clear gatherer_0` after auto-spawn reports "No player was found". Subsequent resets work. Currently not blocking (rewards begin from the second reset). Fix: add 200ms sleep in `FabricBridge.carpet_spawn` or have the bridge await the `PlayerJoinEvent`.
- **`RLModule(config=...)` deprecation warning.** Ray 2.55 prefers `RLModule(observation_space=, action_space=, ...)`. Our config currently uses the older path via `RLModuleSpec`. Cosmetic — will become a hard error in a future Ray version.

---

## Estimated time for the full session
- Pre-flight: 5 min
- Bootstrap: 3 min
- Training to convergence: **1-12 hours** depending on hyperparameters + canopy issues
- Eval gate: 15 min
- Tag: 5 min
- M2 brainstorming: open-ended (1-3 hours for first design pass)

**Critical-path total to `m1b-verified`:** roughly half a day to a day of wall-clock, mostly hands-off training time. The interactive parts only take ~30 minutes spread across the run.
