# Gatherer Survival-Forest Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Validation here is **empirical** (determinism probe + real-MC transfer gate), not only unit tests — treat the gate as the acceptance test.

**Goal:** Harden the Lumberjack from the M1B toy arena (flat grid + instant break) to survival-forest fidelity — survival break-*timing* with a stone axe, then real capped-height vertical oak trees — each validated by the sim→real transfer gate.

**Architecture:** Two staged increments. Inc 1 = Java `HarvestSkill` breaks each log over tool-derived ticks + the fake player holds a stone axe; the policy is unchanged (still HARVEST-spam), so **try the existing checkpoint first — likely no retrain** (same as the `MAX_SEARCH_RADIUS` fix). Inc 2 = replace the flat 8×8 log grid with scattered 4-tall **bare** oak trunks (≤ reach, no leaves) in both `WorldOps.java` and `sim/world.py`, with byte-faithful `_JavaRandom` layout parity.

**Tech Stack:** Java 21 / Fabric mod, Python 3.11 sim, Ray RLlib, Py4J bridge. Build: `cd fabric_mod && export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 && ./gradlew build`. Deploy: copy jar → `server-runtime/training/instance-*/mods/` → restart. Tests: `PYTHONPATH=src py -3.11 -m pytest`. Gate: `scripts/transfer_eval.py` (instance-1, Py4J 25001). Determinism: `scripts/n21_breaktiming_determinism.py`.

---

## File Structure

- `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java` — add break-timing (Inc 1).
- `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java` — equip axe (Inc 1); tree arena (Inc 2).
- `src/aiutopia/sim/skills.py` — (Inc 1, only if retrain needed) break-tick cost; (Inc 2) verify trunk selection.
- `src/aiutopia/sim/world.py` — tree arena, `_JavaRandom`-parity (Inc 2).
- `src/aiutopia/sim/obs_adapter.py` — verify trunk projection; regenerate golden trace (Inc 2).
- `tests/unit/test_sim_world.py` (or existing) — tree-arena layout-parity test (Inc 2).
- `scripts/n21_breaktiming_determinism.py` — reuse as the determinism acceptance check.

---

## INCREMENT 1 — Survival break-timing + stone axe (flat grid)

### Task 1: Java — break each log over tool-derived ticks

**Files:** Modify `fabric_mod/.../skill/HarvestSkill.java`

- [ ] **Step 1: Add the break-timing constant + field.** After `STALL_DIST_EPSILON_SQ` (line ~76) add:

```java
    // N21: survival break-timing. Oak log hardness 2.0; with a stone axe ~0.75 s
    // ≈ 15 ticks (vanilla). Each log is "mined" for this many ticks while in
    // reach before it breaks (was instant/creative). Deterministic tick counter;
    // the N16c direct-inventory-insert is unchanged (no item-drop/pickup race).
    private static final int    BREAK_TICKS_PER_LOG = 15;
```

After `private int stuckTicks;` (line ~84) add:

```java
    private int    breakProgress = 0;  // N21: ticks spent mining current target
```

- [ ] **Step 2: Reset the mining timer on new target.** In `tick(...)`, in the block that sets `currentTarget = cand;` (line ~143-145), add `breakProgress = 0;` as the last line before the closing `}`.

- [ ] **Step 3: Gate the break behind the timer.** In `tick(...)`, immediately AFTER the "target changed under us" check block (the one that does `currentTarget = null; return SkillResult.RUNNING;`) and BEFORE the `// 1. Compute drops` comment, insert:

```java
        // N21 survival break-timing: "mine" the in-reach block for
        // BREAK_TICKS_PER_LOG ticks before it breaks. Deterministic; the
        // direct-inventory insert below is unchanged.
        if (++breakProgress < BREAK_TICKS_PER_LOG) {
            return SkillResult.RUNNING;
        }
        breakProgress = 0;
```

- [ ] **Step 4: Equip a stone axe so the break time reflects a tool.** In `WorldOps.resetEpisode(...)`, after the `/clear ` + before the `/fill` line (line ~111-113), add a give-axe command so the agent holds it each episode:

```java
            cm.executeWithPrefix(src, "/give " + playerName + " minecraft:stone_axe");
            cm.executeWithPrefix(src, "/item replace entity " + playerName
                + " weapon.mainhand with minecraft:stone_axe");
```

(`/clear` then re-give keeps it idempotent and in the main hand. The break-time constant above is hardcoded to the stone-axe value, so the skill and the equipped tool agree.)

- [ ] **Step 5: Commit the Java change.**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java \
        fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java
git commit -m "feat(skill): N21 Inc1 — survival break-timing (15 ticks/log) + stone axe"
```

### Task 2: Build, redeploy, and validate with the EXISTING policy (no retrain)

**Files:** none (build + ops)

- [ ] **Step 1: Build the mod.**

```bash
cd fabric_mod && export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 \
  && export PATH="$JAVA_HOME/bin:$PATH" && ./gradlew build
```
Expected: `BUILD SUCCESSFUL`, jar at `build/libs/aiutopia-mod-0.0.0-m1c-p0.jar`.

- [ ] **Step 2: Redeploy to instance-1 + restart.** Copy the jar into `server-runtime/training/instance-1/mods/` (overwrite), stop the PID owning port 25001 (`Get-NetTCPConnection -LocalPort 25001 ... | Stop-Process`), then `NUM_INSTANCES=1 JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 bash scripts/launch-training-instances.sh`. Expected: `READY`, `setup=True`.

- [ ] **Step 3: Determinism probe (acceptance check).**

```bash
PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
  py -3.11 scripts/n21_breaktiming_determinism.py
```
Expected: `DETERMINISTIC` on seeds 1/2/3, all clear 64/64, ~28 s/clear. (Spike already showed this for 20 ticks; 15 ticks → slightly faster + more logs/dispatch.)

- [ ] **Step 4: Transfer gate with the CURRENT checkpoint (no retrain).**

```bash
PYTHONPATH=src AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
  AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data TRANSFER_WALL_CAP_S=240 \
  py -3.11 scripts/transfer_eval.py
```
Expected: `SIM->REAL M1B GATE: PASS` (3/3). The first run may show seed_1=0/64 (cold-start spawn race) — **re-run once** on the warm server. If 3/3 → **Increment 1 done, skip Task 3.**

### Task 3: (ONLY IF Task 2 transfer fails) model break-ticks in sim + retrain

**Files:** Modify `src/aiutopia/sim/skills.py`

- [ ] **Step 1: Add the break-tick cost to the sim.** Add `BREAK_TICKS_PER_LOG = 15` near the other constants; in `_apply_harvest`, charge `BREAK_TICKS_PER_LOG` (plus the walk ticks) against a per-dispatch budget so a `cap=64` dispatch yields ~the same logs/dispatch as real (the spike's ~17 at 20 ticks → recompute for 15). NOTE: the reverted HEAD sim has no budget loop — re-introduce a bounded budget *only* if real diverges; mirror `_walk_into_reach(world, target, budget)` from the v4 implementation (git history `624da53`), but keep `MAX_SEARCH_RADIUS=48`.

- [ ] **Step 2: Sim test for the budget.** In `tests/unit/test_sim_skills.py`:

```python
def test_harvest_break_timing_is_budget_bounded():
    w = SimWorld(); w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(scalar=1.0))  # cap=64, default budget
    n = w.inventory.get("oak_log", 0)
    assert 0 < n < 64, f"break-timing should bound one dispatch, got {n}"
```
Run: `PYTHONPATH=src py -3.11 -m pytest tests/unit/test_sim_skills.py -q` → PASS.

- [ ] **Step 3: Retrain in sim + re-validate.** `PYTHONPATH=src AIUTOPIA_DATA_DIR=... py -3.11 scripts/train.py --milestone M1 --max-iters 200 --evaluation-interval 999 --num-env-runners 0 --backend sim`; then `scripts/sim_rollout_check.py` (clears the field) and re-run Task 2 Step 4. Expect 3/3.

- [ ] **Step 4: Commit.** `git add -A && git commit -m "feat(sim): N21 Inc1 — sim break-tick budget + retrain"`

---

## INCREMENT 2 — Real capped-height bare oak trees

### Task 4: Sim tree arena (`world.py`) with `_JavaRandom` parity

**Files:** Modify `src/aiutopia/sim/world.py`; Test `tests/unit/test_sim_world.py`

**Design (fixed to keep the 64-log gate exact + parity simple):** 16 trees on a 4×4 spaced grid, each a **4-tall** vertical oak_log trunk (Y=66..69, all ≤ 4.5 reach) at a `_JavaRandom` ±1-jittered (x,z) base. Exactly 16×4 = 64 logs. **No height draw** (fixed 4) ⇒ the per-tree RNG draw pattern is the SAME 2× `next_int(3)` (x then z) as today — **no new draws**, so parity is the existing surface. Bare trunks (no leaves).

- [ ] **Step 1: Write the layout-parity test FIRST.** In `tests/unit/test_sim_world.py`:

```python
import numpy as np
from aiutopia.sim.world import SimWorld, LOG_Y

def test_tree_arena_is_64_logs_in_16_trunks_of_height_4():
    w = SimWorld(); w.reset(seed=1)
    assert w.logs.shape == (64, 3)
    assert int(w.log_alive.sum()) == 64
    # 16 distinct (x,z) columns, each with 4 logs stacked Y=66..69
    cols = {}
    for (x, y, z) in w.logs.tolist():
        cols.setdefault((x, z), []).append(y)
    assert len(cols) == 16, f"expected 16 trunks, got {len(cols)}"
    for (x, z), ys in cols.items():
        assert sorted(ys) == [LOG_Y, LOG_Y+1, LOG_Y+2, LOG_Y+3]

def test_tree_arena_is_seed_deterministic():
    a = SimWorld(); a.reset(seed=2)
    b = SimWorld(); b.reset(seed=2)
    assert np.array_equal(a.logs, b.logs)
```

- [ ] **Step 2: Run the test — verify it FAILS** (`PYTHONPATH=src py -3.11 -m pytest tests/unit/test_sim_world.py -q`). Expected: fails (current arena is flat 64 single-log cells, 64 trunks not 16).

- [ ] **Step 3: Implement the tree arena in `world.py reset()`.** Replace the 8×8 single-log loop (lines ~114-136) with a 4×4 trunk loop. Trunk base grid spaced so trunks don't overlap and the agent can pass between; per trunk draw x-jitter then z-jitter (same order as Java will use), then stack 4 logs:

```python
        rng = _JavaRandom(seed)
        logs = np.zeros((TREES * TRUNK_H, 3), dtype=np.int64)  # 16*4 = 64
        used: set[tuple[int, int]] = set()
        idx = 0
        for row in range(TREE_GRID):          # 4
            for col in range(TREE_GRID):      # 4
                base_x = 52 + 7 * col         # 52,59,66,73  (≤80)
                base_z = -61 + 7 * row        # -61,-54,-47,-40  (≥-64)
                x = base_x + (rng.next_int(3) - 1)
                z = base_z + (rng.next_int(3) - 1)
                x = max(MIN_X, min(MAX_X, x)); z = max(MIN_Z, min(MAX_Z, z))
                while (x == SPAWN_X and z == SPAWN_Z) or (x, z) in used:
                    if x < MAX_X: x += 1
                    elif z < MAX_Z: z += 1
                    elif x > MIN_X: x -= 1
                    else: z -= 1
                used.add((x, z))
                for dy in range(TRUNK_H):     # 4 logs stacked
                    logs[idx] = (x, LOG_Y + dy, z); idx += 1
```
Add constants near the others: `TREE_GRID = 4`, `TREES = TREE_GRID * TREE_GRID`, `TRUNK_H = 4`. Update the module docstring to say "16 4-tall oak trunks (bare)". Adjust the `logs`/`log_alive` default factories if they hardcode `GRID*GRID` (they already equal 64, so `TREES*TRUNK_H == 64` keeps shapes).

- [ ] **Step 4: Run the test — verify it PASSES.**

- [ ] **Step 5: Run the FULL sim suite** (`PYTHONPATH=src py -3.11 -m pytest -m "not integration and not determinism" -q`). Fix any obs/skill test that assumed flat single-logs. Expected: green.

- [ ] **Step 6: Commit.** `git add src/aiutopia/sim/world.py tests/unit/test_sim_world.py && git commit -m "feat(sim): N21 Inc2 — 16 4-tall bare oak trunks (64 logs), _JavaRandom parity"`

### Task 5: Java tree arena (`WorldOps.java`) — IDENTICAL draw order

**Files:** Modify `fabric_mod/.../bridge/WorldOps.java`

- [ ] **Step 1: Replace the 8×8 log loop with the 4×4 trunk loop** (lines ~118-146), matching `world.py` byte-for-byte (same `base_x/base_z` formulas, same `nextInt(3)` x-then-z order, same clamp + dedup-nudge), stacking 4 `oak_log` per trunk:

```java
            epRand.setSeed(seed);
            final int SPAWN_X = 64, SPAWN_Z = -48;
            final int MIN_X = 48, MAX_X = 80, MIN_Z = -64, MAX_Z = -32;
            final int TREE_GRID = 4, TRUNK_H = 4;
            java.util.HashSet<Long> used = new java.util.HashSet<>();
            for (int row = 0; row < TREE_GRID; row++) {
                for (int col = 0; col < TREE_GRID; col++) {
                    int baseX = 52 + 7 * col;
                    int baseZ = -61 + 7 * row;
                    int x = baseX + (epRand.nextInt(3) - 1);
                    int z = baseZ + (epRand.nextInt(3) - 1);
                    x = Math.max(MIN_X, Math.min(MAX_X, x));
                    z = Math.max(MIN_Z, Math.min(MAX_Z, z));
                    while ((x == SPAWN_X && z == SPAWN_Z) || used.contains(key(x, z))) {
                        if (x < MAX_X)      x += 1;
                        else if (z < MAX_Z) z += 1;
                        else if (x > MIN_X) x -= 1;
                        else                z -= 1;
                    }
                    used.add(key(x, z));
                    for (int dy = 0; dy < TRUNK_H; dy++) {
                        cm.executeWithPrefix(src,
                            "/setblock " + x + " " + (66 + dy) + " " + z + " oak_log");
                    }
                }
            }
```
Update the `/fill ... air replace` Y-range (line ~113) to cover Y=66..69 (e.g. `/fill 48 66 -64 80 73 -32 air replace`) so old logs clear. Update the method docstring.

- [ ] **Step 2: Build the mod** (Task 2 Step 1 command). Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit.** `git add fabric_mod/.../WorldOps.java && git commit -m "feat(arena): N21 Inc2 — WorldOps 16 4-tall oak trunks, matches sim draw order"`

### Task 6: Obs parity + regenerate golden trace

**Files:** `src/aiutopia/sim/obs_adapter.py` (verify), `tests/fixtures/gatherer_obs_trace_seed1.json` (regenerate)

- [ ] **Step 1: Verify the obs adapter handles trunks.** Read `obs_adapter.py`: the grid marks each log's `(dx+16, dz+16, 0)=1` and `g_nearest_resources` reports nearest-8 with `dy`. A trunk = 4 logs at the same `(x,z)` different `y` → the grid cell is marked (idempotently); nearest-resources will include trunk logs with `dy` 1..4. Confirm no code assumes one-log-per-(x,z) or `dy==+1` only. If any constant assumes flat `dy`, note it (don't enrich — just ensure it doesn't crash/mis-rank).

- [ ] **Step 2: Redeploy the tree jar to instance-1 + restart** (Task 2 Step 2).

- [ ] **Step 3: Regenerate the golden trace** against the new tree arena: `PYTHONPATH=src py -3.11 scripts/capture_gatherer_obs_fixture.py` (the Phase-A capture script; confirm its path). Then run the golden-trace parity test: `PYTHONPATH=src py -3.11 -m pytest tests/unit/test_sim_obs_parity.py -q` → PASS (sim obs == real obs on the deterministic prefix for the tree arena).

- [ ] **Step 4: Commit.** `git add src/aiutopia/sim/obs_adapter.py tests/fixtures/gatherer_obs_trace_seed1.json && git commit -m "test(sim): N21 Inc2 — regenerate golden trace for tree arena"`

### Task 7: Transfer gate for the tree forest (retrain if needed)

**Files:** none (validate); maybe retrain

- [ ] **Step 1: Try the EXISTING checkpoint first** — `scripts/sim_rollout_check.py` then `scripts/transfer_eval.py` (warm instance-1). If 3/3 → done. The policy is still HARVEST-spam; trees may transfer without retrain.

- [ ] **Step 2: If sim rollout doesn't clear the forest, retrain** in sim (Task 3 Step 3 command) and re-run the gate. Expect 3/3.

- [ ] **Step 3: Determinism re-check** on the tree arena (`scripts/n21_breaktiming_determinism.py`). Expect DETERMINISTIC.

### Task 8: Finalize

- [ ] **Step 1: Advisor review** of the completed increments (call advisor before declaring done).
- [ ] **Step 2: Update docs** — `NEXT_SESSION.md` (survival-forest milestone status), memory `ai-utopia-fast-sim-plan`. `git add -A && git commit && git push origin main`.

---

## Risks / watch-items (from the advisor + the session's lessons)

- **Sim must never out-reach real.** Trunk height is fixed at 4 (≤ 4.5 reach) precisely so this can't recur (the seed-3 flat-y trap). Do NOT raise trunk height without a climbing model.
- **No leaves** — MC oak leaves are collidable and would block the straight-line `agent.move` walk in real MC while the leafless sim sails through. Bare trunks only.
- **Draw-order parity is load-bearing** — `WorldOps` and `world.py` must issue the SAME `nextInt(3)` calls in the SAME order; the Task-4 determinism test + Task-6 golden trace guard it. Fixed height (no height draw) keeps this simple.
- **Cold-start spawn race** — first `transfer_eval` reset on a fresh server can strand the agent (seed_1=0/64); re-run on the warm server.
- **Honest scope** — the gate validates skill+sim parity, not policy spatial learning (the skill does the chaining). Fine for a fidelity milestone.
