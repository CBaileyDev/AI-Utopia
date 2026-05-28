# Next Session Handoff — Post v20 Training Day

## Status as of 17:20

- **v20 PPO training is RUNNING** in the background — iter ~20, started 14:30 (~2h 50m elapsed in v20)
- 4 Fabric servers (PIDs visible via `Get-Process java`) on ports 25001-25004
- v20 log: `Research/train-v20.log` — driver PID file: `Research/train-v20.log.pid`
- Ray Tune trial dir: `runs/aiutopia_M1_seed1/PPO_aiutopia_minecraft_2d8eb_00000_0_2026-05-28_14-29-35/`

## What got fixed this session (commit chain)

| Commit | Issue | Fix |
|--------|-------|-----|
| `694b44d` | N15+N16 — HARVEST silently fails | float-precision attractor in `HarvestSkill.java`, REACH 2.0→4.5, full WALK_PER_TICK, direct inventory insertion via `getDroppedStacks` + `offerOrDrop` |
| `4a9bff0` | episodes never end | `max_episode_ticks` 12_000 → 2_000 |
| `abe1c12` | N17 — eval scenarios crash | nested-dict obs batching in `scenario_runner.py` (action_mask is a dict not an ndarray); `max_episode_ticks` 2_000 → 300 |
| `0c52bd4` | N18 — eval too expensive | `M1_SCENARIOS` max_ticks 1000 → 300 |

End-to-end probe (`scripts/n14_reward_probe.py`) confirms reward signal: 6 oak_log per 6 env_steps, reward +11.78 in isolation.

## Why training hasn't converged

**Empirically observed**: across 3 training runs (v18, v19, v20), only 1 of 4 instances accumulates oak_log; the other 3 wander off and mine cobblestone. Inventory probes at iter ~10-15:

```
inst-1: 64 cobblestone, 0 oak_log
inst-2: 39 cobblestone, 0 oak_log
inst-3:  1 cobblestone + 2 coal, 0 oak_log
inst-4:  8 oak_log + 9 cobblestone  ← only one on-task
```

PPO averages gradients across all rollouts. With 3/4 envs producing cobblestone-mining trajectories (LOG_VALUE 0.091/block) and 1/4 producing oak_log trajectories (LOG_VALUE 1.000/block), the policy mixes the two attractors. Per-iter gradient updates are dilute and slow to converge on oak_log specifically.

**Root mechanism**: the seeded 8-log ring depletes within ~50-150 env steps. After that, the agent has zero oak_log signal until `reset_episode` re-places the ring. Meanwhile NAVIGATE can take the agent 32 blocks away per skill call, so the agent often ends up underground (`y=40-65` from spawn `y=66`) mining cobblestone before the ring resets.

## Training rate / instability issues seen

- **Per-step time degrades over ~5-9 iters**: starts at 1-2s, grows to 8-15s. Cause unknown — possibly Java GC pressure (instances hit ~1.5GB of 2GB heap), possibly Carpet fake-player state, possibly chunk-load fragmentation as agents wander to new chunks.
- **NaN KL ~40-50% of iters**: training continues but ~half of gradient steps are no-ops. Iter 13 KL = 0.01 (converging) but iter 14 NaN. Pattern oscillates throughout.
- **Iter pace ~500-600s after warmup**: at this rate, 200 iters = ~33h. We got ~20 iters in 3h of v20.

## Recommended next-session priorities

### High value, low risk

1. **Constrain agents to the arena.** The 32-block NAVIGATE range lets agents escape the seeded ring. Two cheap options:
   - In `WorldOps.resetEpisode`, add `/forceload add` for the arena chunks only, so unloaded chunks elsewhere don't load (won't stop movement but reduces load cost).
   - In `wrapper.py step()`, after each step check if `agent.position.y < 60` or distance from origin > 32, and force a `reset_episode` if so. This pulls agents back to the ring quickly and dominates the on-policy buffer with oak_log trajectories.

2. **Bump Java heap.** Edit `scripts/launch-training-instances.sh:67` from `-Xmx2g` to `-Xmx4g`. May fix the per-step time degradation.

3. **Run a full N18 e2e eval test.** `scripts/n18_eval_e2e_test.py` (added this session) tests scenario_runner against a random AiUtopiaRoleRLModule on instance-1. **Stop v20 first** (it competes for instance-1) and run:
   ```
   PYTHONPATH=src py -3.11 -u scripts/n18_eval_e2e_test.py
   ```
   Expected: all 3 scenarios complete in ~22min total, success_rate=0 (random policy). Proves the N17 fix end-to-end.

### Medium value

4. **Stop v20 cleanly to trigger `checkpoint_at_end`.** `train.py` line 102 has `checkpoint_at_end=True` — sending SIGTERM to the driver PID should save the current weights. Then load + run scenarios manually.

5. **Investigate why `metrics_num_episodes_for_smoothing=200`** in `config.py:122`. With max_episode_ticks=300 and 64 steps/env/iter, we'd need 200×300/64 = ~940 iters of training before the smoothing window fills — way longer than M1B targets. Lower to 20.

### Low value, high effort

6. Custom MetricsCallback that publishes `episode_return_mean` to a top-level CSV column (we couldn't find it directly in 106 columns of `progress.csv` — only buried in `result.json`).

## Files added this session

- `scripts/n14_reward_probe.py` — HARVEST reward sanity probe (6 oak_log, +11.78 reward verified)
- `scripts/n15_py4j_diag.py`, `scripts/n15b_payload_diag.py`, `scripts/n15c_bisect.py` — dispatch path diagnostics (proved dispatch works)
- `scripts/n16_verify.py`, `scripts/n16b_probe_state.py` — REACH_RADIUS attractor reproduction
- `scripts/n17_eval_repro.py` — pinpoints `action_mask` as the numpy.object_ source
- `scripts/n18_eval_e2e_test.py` — standalone scenario_runner test against random policy

## Files modified this session

- `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java` (N16+N16c)
- `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/DepositChestSkill.java` (N16b)
- `src/aiutopia/env/wrapper.py` (N16, N14 logging upgrade)
- `src/aiutopia/env/reward.py` (N14 eager seed)
- `src/aiutopia/train/scenario_runner.py` (N17 dict batching, N18 max_ticks)
- `src/aiutopia/train/config.py` (N17 max_episode_ticks)
- `fabric_mod/gradle.properties` (version → 0.0.0-m1c-n16c)

## How to stop v20 cleanly when ready

```bash
DRIVER_PID=$(cat /c/Users/Carte/OneDrive/Desktop/AiUtopia/Research/train-v20.log.pid)
# Tune's signal handler should trigger checkpoint_at_end on SIGTERM
powershell -Command "Stop-Process -Id $DRIVER_PID"
```

Final v20 checkpoint will be under `runs/aiutopia_M1_seed1/.../checkpoint_*/`.

## How to manually evaluate the trained policy

After v20 stops with a saved checkpoint:

```python
from ray.rllib.algorithms.algorithm import Algorithm
algo = Algorithm.from_checkpoint("runs/aiutopia_M1_seed1/PPO_.../checkpoint_xxx")
module = algo.get_module("gatherer_policy")
# then run scripts/n18_eval_e2e_test.py but pass module= from the checkpoint
```

Or use the existing `aiutopia.cli.promote_weights` CLI (T16/T17) once the checkpoint exists.
