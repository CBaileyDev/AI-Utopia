# Next session runbook — M1B training to `m1b-verified` tag

**Resume point:** repo HEAD `e466063`. M1B code-complete + **13** integration fixes (12 + slow-env timeout) + 3 "likely to break" preemptive + 2 pre-N3 safety + N9 (Java ItemId table) + N7.5 stop-key fix recipe. 142/142 unit tests pass. `aiutopia-mod-0.0.0-m1b.jar` deployed to 5 mod directories. No live processes — last session cleaned up.

**Goal this session:** Run T21 to convergence, hit the 80% eval gate, promote weights, tag `m1b-verified`. Then optionally start M2 brainstorming.

**Supervision mode:** USER BABYSITS the run (1-2 hours of TensorBoard watching, Ctrl-C on convergence). No custom auto-stopper installed; training will otherwise run to `training_iteration == 2000`. See N7.5 if you want to fix the Tune stop-key path for M2.

---

## ⚠️ N10 BLOCKER — known v13 hang to debug FIRST

The last training attempt (v13) silently hung on the first `env.step()`. Symptoms:
- Workers spawned cleanly
- env.reset() executed (oak_logs placed at 16:18:27 across all 4 Fabric instances)
- Then **complete silence** — no skill execution, no errors, no progress for 5+ minutes
- Python worker CPU went idle (53s total across 26 procs)

**Most likely cause:** The N9 obs builder refactor at commit `c79afc5` — `CoreObsBuilder.maskedItemId()` now delegates to the new `ItemIdTable.idOf()` singleton. Possible failure modes:
1. `ItemIdTable` singleton init blocks indefinitely on first call from obs path
2. `idOf()` throws on a specific item type silently swallowed by the obs builder
3. The new mapping produces an obs the Python wrapper's `_decode_obs` can't parse

**Debug recipe (do this BEFORE relaunching training):**

```bash
# 1. Launch ONE Fabric instance + one agent + try a single skill manually
JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 ./scripts/launch-training-instances.sh
# (the script needs only 1 instance for the probe, but easier to launch all 4)
```

```powershell
# 2. Probe: run a single HARVEST skill via the existing aiutopia agent drive CLI
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
$env:PYTHONPATH = "src"
$env:AIUTOPIA_ROOT = "C:\tmp\aiu-n10-probe"
py -3.11 -m aiutopia.cli.app agent spawn --role gatherer --py4j-port 25001
py -3.11 -m aiutopia.cli.app agent drive --agent-name <NAME_FROM_SPAWN> `
    --skill 1 --target 0 --dx 0.5 --dy 0.0 --dz 0.5 --scalar 0.5 `
    --py4j-port 25001 --timeout-ms 30000
```
Expected: completion event JSON printed within ~5s. If hangs >30s: skill dispatch path is broken (probably ItemIdTable.init or the obs builder).

```powershell
# 3. If skill drive works manually, the issue is in the WRAPPER's env.step path.
# Run the N2.5 reward sanity probe (same script from NEXT_SESSION.md Phase 2.5)
# but watch which step it stalls on. The first call should be env.reset().
```

```powershell
# 4. If both probes hang, the issue is in observationsAll() (the obs builder is
# blocking). Inspect with:
py -3.11 -c "from aiutopia.env.bridge import FabricBridge
import time
with FabricBridge(port=25001) as b:
    t = time.time()
    obs = b.observations_all()
    print(f'observationsAll took {time.time()-t:.2f}s, keys={list(obs.keys())[:5]}')"
# Expect: <1s. If >5s: ItemIdTable lookup is slow.
```

**Fix candidates (in priority order):**
- Add debug log in `ItemIdTable.idOf()` to count calls + time
- If `ItemIdTable` builds its mapping lazily on first call, switch to eager init at mod boot
- If `idOf()` does a linear scan, add a HashMap cache
- If the regression is real but unclear, `git revert c79afc5` and reintroduce N9 fix differently (keep the obs builder unchanged, just expose the existing table via Py4J)

**Once N10 is fixed**, resume below from Phase 1.

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

## Phase 2.5 — Reward sanity check (5 min) — Task N2.5

Quick insurance before committing to a long training run. Drive one agent through a scripted skill sequence and verify the reward histogram is sane (non-zero variance, no NaN, magnitudes in `[-2, +5]` ballpark).

```powershell
cd C:\Users\Carte\OneDrive\Desktop\AiUtopia
$env:PYTHONPATH = "src"
$env:AIUTOPIA_ROOT = "C:\tmp\aiu-m1b-probe"
py -3.11 -c @"
import json, numpy as np
from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
env = AiUtopiaPettingZooEnv({
    'stage': 1, 'active_roles': ['gatherer'],
    'seed_strategy': 'fixed_easy', 'py4j_ports': [25001],
    'tick_warp': True, 'max_episode_ticks': 50,
    'per_worker_seed_offset': False, 'enable_memory_writes': False,
    'aiutopia_root_per_worker': False,
})
obs, info = env.reset(seed=1)
rewards = []
for tick in range(30):
    actions = {}
    for agent_id in obs:
        a = env.action_space(agent_id).sample()
        actions[agent_id] = a
    obs, rew, term, trunc, info = env.step(actions)
    for r in rew.values():
        rewards.append(float(r))
    if all(term.values()) or all(trunc.values()):
        break
env.close()
r = np.asarray(rewards)
print(f'reward stats: n={len(r)} mean={r.mean():.3f} std={r.std():.3f} min={r.min():.3f} max={r.max():.3f}')
print(f'histogram (5 bins):')
hist, edges = np.histogram(r, bins=5)
for i, h in enumerate(hist):
    print(f'  [{edges[i]:.2f}, {edges[i+1]:.2f}]: {"#"*h} ({h})')
print('NaN check:', np.isnan(r).any())
print('Inf check:', np.isinf(r).any())
"@
```

**Pass criteria:**
- `n == 30` (or close — episode terminations OK)
- `std > 0` (not a constant signal)
- `NaN check: False`, `Inf check: False`
- `min >= -10` and `max <= +20` (sanity bounds)

If any check fails, fix `compute_reward_stage_1` in `src/aiutopia/env/reward.py` before launching N3. A silent `0` mean+std with no NaN means the reward is computing but is constant — usually a sign the agent isn't actually triggering the events that produce reward (check that the obs delta logic in reward.py sees the obs changing tick-to-tick).

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

**Verify the 4 per-review-fix invariants in the first 5 min:**

```powershell
# (1) Per-worker AIUTOPIA_ROOT actually created (4 separate dirs):
Get-ChildItem C:\tmp\aiu-m1b-train_w* -Directory | Select-Object Name, LastWriteTime
# Expect: aiu-m1b-train_w0, _w1, _w2, _w3 — each with its own identity.db
Get-ChildItem C:\tmp\aiu-m1b-train_w0\identity.db -ErrorAction SilentlyContinue

# (2) Checkpoints actually being written (cadence verification):
Get-ChildItem "C:\tmp\aiu-m1b-train\runs\aiutopia_M1_seed1\PPO_*\checkpoint_*" -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime
# Expect: at least one checkpoint_NNNNNN/ directory by ~iter 10. If nothing
# appears past iter 50, Ray 2.55's default cadence may be "end only" —
# pass --checkpoint-frequency to train.py or hack RunConfig.checkpoint_config.

# (3) Custom metrics actually in TensorBoard:
# Open http://localhost:6006/, expand "custom_metrics" scope —
# should see gatherer_policy/entropy, gatherer_policy/vf_loss, M1/gate_history.
# If the scope is empty, the RLlibCallback isn't piping to the new metrics_logger
# in Ray 2.55 — see N7.5.

# (4) /clear race fixed — Fabric instance-1.log should NOT have
# "No player was found" lines after the initial spawn await:
Select-String "No player was found" C:\Users\Carte\OneDrive\Desktop\AiUtopia\server-runtime\training\instance-1\instance-1.log | Measure-Object | Select-Object Count
# Expect: 0 (or very few — at most one per worker on auto-spawn race)
```

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

## Phase 6.5 — Resume training after a crash (5 min) — failure path

If training dies mid-run (Ray worker OOM, machine reboot, Ctrl-C without convergence, electricity glitch), you can pick up from the last checkpoint instead of restarting from scratch.

**Locate the last good checkpoint:**
```powershell
$LAST_CKPT = (Get-ChildItem "C:\tmp\aiu-m1b-train\runs\aiutopia_M1_seed1\PPO_*\checkpoint_*" `
              -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "Resume from: $LAST_CKPT"
```

**Restore the Tuner and continue:**
```powershell
$env:AIUTOPIA_ROOT = "C:\tmp\aiu-m1b-train"; $env:PYTHONPATH = "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
py -3.11 -c @"
from ray import tune
tuner = tune.Tuner.restore(
    path=r'C:\tmp\aiu-m1b-train\runs\aiutopia_M1_seed1',
    trainable='PPO',
    restart_errored=True,
)
results = tuner.fit()
"@
```

**Caveats:**
- Restore requires the original `trainable='PPO'` + the same env registration to still exist. Re-run Phase 2 (launch Fabric + bootstrap) first so the 4 instances are up before resuming.
- Restore preserves the random seed sequence but not the per-EnvRunner random state — minor divergence is expected.
- If the restore itself errors with "checkpoint format mismatch," the Ray version may have changed since the checkpoint was written — fall back to launching fresh and accept the lost iters.

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
- **Pre-flight (N1):** 5 min
- **Bootstrap (N2):** 3 min
- **Reward sanity check (N2.5):** 5 min
- **Training to convergence (N3):** **1-2 hours of babysitting** + however long the wall-clock takes (no auto-stopper, you Ctrl-C on convergence)
- **Eval gate (N5):** 15 min
- **Tag (N6):** 5 min
- **M2 brainstorming (N7):** open-ended (1-3 hours for first design pass)
- **Tune stop-key path investigation (N7.5):** 30 min (post-tag, defer if M1B done)

**Critical-path total to `m1b-verified`:** ~half a day of wall-clock with focused 1-2h babysitting window. If reward sanity check (N2.5) fails OR canopy issue triggers an N4 intervention, add 1-2 hours per iteration.

## Re-review delta vs the v1 chain

Per the user's pre-implementation review:
- Added N2.5 (reward sanity check) as a 5-min insurance step before N3.
- Pre-N3 code fixes committed at `e979748` for per-worker AIUTOPIA_ROOT (SQLite contention prevention) and 200ms join-await in `carpet_spawn` (silences /clear race).
- N3 now has explicit invariant-verification checks (per-worker dirs exist, checkpoints write, custom_metrics in TB, no /clear races).
- N4 reframed from "sequel of N3" to "conditional interrupt during N3".
- Added Phase 6.5 (resume-from-crash via `tune.Tuner.restore()`).
- Added N7.5 (post-M1B Tune stop-key path investigation) — feeds M2's auto-stopper.
