# Gatherer Fast-Sim — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast, headless, single-process Python sim of the gatherer M1B task that satisfies the *existing* obs/action/reward contract, so a policy can be trained in it and (Phase C) validated against real MC unchanged.

**Architecture:** A new `src/aiutopia/sim/` package mirrors the `AiUtopiaPettingZooEnv` interface behind the same contract (`env/spaces.py`, `env/reward.py`, `env/action_mask.py`). The sim **reuses the pure functions** — `reward.compute_reward_stage_1`, the action-mask builder, and the wrapper's constant embeds — and only re-implements (a) world state, (b) macro-skill dynamics matching the Java skills, and (c) the *dynamic* obs fields (`position`, inventory, `g_resource_grid`, `g_nearest_resources`, distances). This makes reward/mask parity free and keeps fidelity risk concentrated in dynamics + the spatial obs.

**Tech Stack:** Python 3.11, NumPy (cheap path — single env first, vectorize later), gymnasium/PettingZoo spaces, pytest. No JAX in Phase A.

**Parity sources (authoritative — replicate byte-faithfully):**
- Obs/action shapes: `src/aiutopia/env/spaces.py`
- Spatial obs build: `fabric_mod/.../obs/GathererOverlayBuilder.java`
- Skill dynamics: `fabric_mod/.../bridge/skill/{HarvestSkill,NavigateSkill,DepositChestSkill,SearchSkill,WaitSkill}.java`
- Arena: `fabric_mod/.../bridge/WorldOps.java` `resetEpisode` (64 oak_log 8×8 grid, y=66, spawn (64.5,65,-47.5))
- Reward + inv decode + action_mask: `src/aiutopia/env/{reward.py,action_mask.py}`
- Env lifecycle to mirror: `src/aiutopia/env/wrapper.py` (`reset`, `step`, termination at goal / out_of_bounds / max_ticks)

---

## File Structure

```
src/aiutopia/sim/
  __init__.py
  world.py          # SimWorld: dataclass state + reset(seed) + low-level move()/break_block()
  skills.py         # apply_skill(world, action) -> (world', skill_completion) — macro-action dynamics
  obs_adapter.py    # build_gatherer_obs(world) -> obs dict (reuses uuid/goal embeds + action_mask builder)
  reward_adapter.py # step_reward(obs_prev, obs_curr, action, env_meta) -> float (calls compute_reward_stage_1)
  sim_env.py        # AiUtopiaSimEnv: PettingZoo-Parallel-shaped env (reset/step) drop-in for scenario_runner
tests/unit/
  test_sim_world.py
  test_sim_skills.py
  test_sim_obs_parity.py
  test_sim_reward_parity.py
  test_sim_env.py
```

**Reuse, do NOT reimplement:** `reward.compute_reward_stage_1`, `reward._inventory_from_obs`, `reward._ITEM_ID_TO_NAME`, the action-mask builder in `env/action_mask.py`, and the wrapper's `_uuid_embed` + goal-embedding stub (import them; if a helper is private to the wrapper, lift it into a shared `env/_embeds.py` only if needed — note in the task).

**Conventions:** run tests with `py -3.11 -m pytest <file> -x -q` (pyproject sets `pythonpath=src`; project runs under 3.11). Commit after each green task. Match existing code style (the repo's auto-formatter normalizes on save).

**Execution note (RAM):** the 12-instance M1B run (12×3 GB JVM + Ray) may still be live on the box. The sim code (T1–T2, T4–T5) is pure Python and light, but parallel torch-importing pytest subagents contend for RAM — keep the verify step lean/sequential and check headroom before fanning out.

---

## Task 1: SimWorld state + reset (the 64-log arena)

**Files:**
- Create: `src/aiutopia/sim/__init__.py` (empty), `src/aiutopia/sim/world.py`
- Test: `tests/unit/test_sim_world.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sim_world.py
import numpy as np
from aiutopia.sim.world import SimWorld

def test_reset_places_64_logs_flat_at_y66_in_bounds():
    w = SimWorld()
    w.reset(seed=1)
    assert w.logs.shape == (64, 3)                     # (x, y, z) per log
    assert np.all(w.logs[:, 1] == 66)                  # all flat at y=66
    xs, zs = w.logs[:, 0], w.logs[:, 2]
    assert np.all((xs >= 48) & (xs <= 80))             # arena x-bounds
    assert np.all((zs >= -64) & (zs <= -32))           # arena z-bounds
    assert len({(int(x), int(z)) for x, z in zip(xs, zs)}) == 64   # all distinct (x,z)
    assert not np.any((xs == 64) & (zs == -48))        # none on spawn tile

def test_reset_agent_at_spawn_and_empty_inventory():
    w = SimWorld(); w.reset(seed=1)
    assert np.allclose(w.agent_pos, [64.5, 65.0, -47.5])   # matches real spawn obs
    assert w.inventory.get("oak_log", 0) == 0

def test_reset_is_seed_deterministic_and_seed_varies_layout():
    a = SimWorld(); a.reset(seed=1)
    b = SimWorld(); b.reset(seed=1)
    c = SimWorld(); c.reset(seed=2)
    assert np.array_equal(a.logs, b.logs)              # same seed -> same layout
    assert not np.array_equal(a.logs, c.logs)          # different seed -> different
```

- [ ] **Step 2: Run to verify it fails** — `py -3.11 -m pytest tests/unit/test_sim_world.py -x -q` → FAIL (no module `aiutopia.sim.world`).

- [ ] **Step 3: Implement `SimWorld`**

Mirror `WorldOps.resetEpisode` exactly. Use a `numpy.random.Generator(PCG64(seed))` (NOT global random). Build the 8×8 grid: `base_x = 50 + 3*col` (col 0..7), `base_z = -62 + 3*row` (row 0..7); per-cell seeded jitter in {-1,0,+1} on each axis; clamp to `[48,80]×[-64,-32]`; resolve collisions with the spawn tile `(64,-48)` or an already-placed cell by nudging deterministically through neighbors (replicate `WorldOps.java` lines 122-146). State fields:
```python
@dataclass
class SimWorld:
    agent_pos: np.ndarray   # float (3,), init [64.5, 65.0, -47.5]
    logs: np.ndarray        # int (64,3) present-log positions (x,y,z); rows removed on harvest -> track via a bool mask `log_alive: np.ndarray (64,)`
    inventory: dict[str,int]
    tick: int               # env-step counter
    def reset(self, seed: int) -> None: ...
```
Keep a `log_alive` boolean mask rather than deleting rows, so indices stay stable.

- [ ] **Step 4: Run to verify pass** — same command → PASS.

- [ ] **Step 5: Commit** — `git add src/aiutopia/sim/__init__.py src/aiutopia/sim/world.py tests/unit/test_sim_world.py && git commit -m "feat(sim): SimWorld state + 64-log arena reset (Phase A task 1)"`

---

## Task 2: Skill dynamics (HARVEST first, then NAVIGATE/DEPOSIT/SEARCH/WAIT)

**Files:**
- Create: `src/aiutopia/sim/skills.py`
- Test: `tests/unit/test_sim_skills.py`

Constants (from the Java skills): `WALK_PER_TICK=0.215`, `HARVEST_REACH=4.5`, `MAX_SEARCH_RADIUS=16.0`, `NAV_ARRIVAL=1.0`, `MAX_NAV_RANGE=32`, ground-preference `dy∈[-2,1]`, `cap=round(scalar_param*64)`. Skill enum order (Discrete(6)): `0=navigate, 1=harvest, 2=deposit_chest, 3=search, 4=wait, 5=noop_broadcast` (confirm against `motor`/`AgentRegistry`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sim_skills.py
import numpy as np
from aiutopia.sim.world import SimWorld
from aiutopia.sim.skills import apply_skill

def _harvest(target_class=0, scalar=1/64):
    return {"skill_type": 1, "target_class": target_class,
            "spatial_param": np.zeros(3, np.float32),
            "scalar_param": np.array([scalar], np.float32),
            "comm_payload": np.zeros(128, np.float32),
            "should_broadcast": 0, "comm_target_mask": np.zeros(4, np.int8)}

def test_harvest_collects_one_log_and_moves_into_reach():
    w = SimWorld(); w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(scalar=1/64))     # cap=1
    assert w.inventory.get("oak_log", 0) == 1
    assert comp["resultCode"] == "COMPLETED"
    assert int(w.log_alive.sum()) == 63                 # exactly one removed

def test_harvest_cap_collects_up_to_cap():
    w = SimWorld(); w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(scalar=1.0))      # cap=64
    assert w.inventory.get("oak_log", 0) == 64
    assert int(w.log_alive.sum()) == 0

def test_harvest_fails_when_no_logs_left():
    w = SimWorld(); w.reset(seed=1)
    w.log_alive[:] = False
    w, comp = apply_skill(w, _harvest())
    assert comp["resultCode"] in ("FAILED_TIMEOUT", "IMMEDIATE_FAILURE")
    assert w.inventory.get("oak_log", 0) == 0
```

- [ ] **Step 2: Run to verify fails** → FAIL (no `apply_skill`).

- [ ] **Step 3: Implement `apply_skill`**

`apply_skill(world, action) -> (world, skill_completion: dict)`. For HARVEST (skill_type=1): find the nearest *alive* log to `agent_pos` (Euclidean; honor ground-preference by matching the real `findNearest` which only matters when logs differ in y — here all y=66, so plain nearest), walk toward it `WALK_PER_TICK` per simulated tick until within `HARVEST_REACH`, mark it harvested (`log_alive[i]=False`), `inventory["oak_log"] += 1`; repeat up to `cap=round(scalar*64)` times or until no logs remain. Return `skill_completion` dict with keys matching the real one: `resultCode` ("COMPLETED"/"FAILED_TIMEOUT"/"IMMEDIATE_FAILURE"), `failureReason` (str), `clippedAxesBitset` (int, 0 here). Update `agent_pos` to the last harvested log's position. NAVIGATE (0): walk toward `origin(64.5,?,-47.5) + spatial*[32,8,32]` until within `NAV_ARRIVAL`; DEPOSIT (2)/SEARCH (3)/WAIT (4)/noop (5): minimal effects (WAIT no-op COMPLETED; DEPOSIT with no chest in sim → COMPLETED no-op for Phase A; SEARCH → COMPLETED). Compute `n_clipped_param_axes` for the reward env_meta by counting action params clipped to their Box bounds.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(sim): macro-skill dynamics (harvest/navigate/...) matching Java skills (Phase A task 2)"`

---

## Task 3: Obs adapter (byte-faithful gatherer obs)

**Files:**
- Create: `src/aiutopia/sim/obs_adapter.py`
- Test: `tests/unit/test_sim_obs_parity.py`

- [ ] **Step 1: Write the failing test (shape + range parity against the real space)**

```python
# tests/unit/test_sim_obs_parity.py
import numpy as np
from aiutopia.env.spaces import build_role_observation_space
from aiutopia.sim.world import SimWorld
from aiutopia.sim.obs_adapter import build_gatherer_obs

def test_obs_matches_contract_keys_and_is_contained():
    space = build_role_observation_space("gatherer", stage=1)
    w = SimWorld(); w.reset(seed=1)
    obs = build_gatherer_obs(w)
    assert set(obs.keys()) == set(space.spaces.keys())          # exact key parity
    assert space.contains(obs), "sim obs not contained in the declared space"

def test_resource_grid_lights_a_log_cell():
    w = SimWorld(); w.reset(seed=1)
    obs = build_gatherer_obs(w)
    grid = obs["g_resource_grid"]                                # (32,32,6)
    assert grid.shape == (32, 32, 6)
    assert grid[..., 0].sum() >= 1                               # >=1 log cell lit (channel 0)
    assert set(np.unique(grid)).issubset({0.0, 1.0})

def test_nearest_resources_top8_normalized():
    w = SimWorld(); w.reset(seed=1)
    obs = build_gatherer_obs(w)
    nr = obs["g_nearest_resources"]                              # (8,6)
    assert nr.shape == (8, 6)
    assert np.all(np.abs(nr[:, 0]) <= 1.0) and np.all(np.abs(nr[:, 2]) <= 1.0)  # dx/16, dz/16
```

- [ ] **Step 2: Run to verify fails** → FAIL.

- [ ] **Step 3: Implement `build_gatherer_obs`**

Produce every key in `build_role_observation_space("gatherer",1)`. **Dynamic fields** from world state, replicating `GathererOverlayBuilder.java`:
- **Coordinate convention (parity-critical — the #1 bug class):** the grid/nearest origin is the agent's **BlockPos = floor(agent_pos)**. The agent settles on the grass surface at obs-y **65** (`/tp` to y=66 drops to the y=64 grass top → feet y=65; confirmed by `n14_reward_probe`: `pos=[64.5,65.0,-47.5]`), while logs are placed at y **66**. So relative to the agent a log's **`dy = 66 - 65 = +1`** (NOT 0). Get this exactly right or transfer silently fails; the golden-trace test (Task 3b) is what catches it.
- `g_resource_grid` (32,32,6) float32: for each alive log, compute `dx,dz` relative to `floor(agent_pos)`; the log is within the ±3 dy scan band (dy=+1), so if `-16<=dx<16 and -16<=dz<16` set `grid[dx+16][dz+16][0]=1.0` (channel 0 = log). Other channels 0 in M1B (no stone/ore/food in arena).
- `g_nearest_resources` (8,6): alive logs within `SCAN_RADIUS=16` sorted by distance, top 8; row `[dx/16, dy/8, dz/16, 0(=log channel), 1.0, 1.0]` with **`dy=+1` → `dy/8=0.125`** for these logs; pad missing rows with zeros.
- `g_richness_score` = `min(1.0, count/64.0)`.
- `g_hostiles_nearby` (4,4) = zeros (no mobs in M1B).
- `position` = `agent_pos.astype(float32)`; `inv_slot_item_ids`/`inv_slot_counts` packed from `inventory` using the *same* item-id mapping as `reward._ITEM_ID_TO_NAME` (reverse it: name→id; e.g. oak_log→132). Unfilled slots: id 0, count 0.
- Also emit `nearest_resource_distance` (vec1) and `nearest_chest_distance` (vec1, sentinel 999.0) — the wrapper/`action_mask` consume these.
**Constant/derived fields — REUSE existing code:** `agent_uuid_embed` (wrapper's `_uuid_embed` for "gatherer_0"), `role_one_hot=[1,0,0,0]`, `goal_embedding` (the wrapper's hardcoded "collect 64 oak_log" stub), `comm_payloads`/`comm_metadata` zeros, `time_of_day`/`weather`/`biome_id`/`light_level` constants matching the training arena, health/hunger/saturation/armor at full (20). Build `action_mask` by calling `env/action_mask.py`'s builder with the in-range booleans derived from `nearest_resource_distance <= HARVEST_REACH` and `nearest_chest_distance <= reach` (replicate `wrapper._normalize_raw`). If `_uuid_embed`/goal-stub are wrapper-private, import them; if not importable cleanly, lift into `src/aiutopia/env/_embeds.py` and have both wrapper and sim import from there (note the refactor in the commit).

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(sim): byte-faithful gatherer obs adapter (Phase A task 3)"`

---

## Task 3b: Golden-trace obs parity (THE fidelity gate — tests against the *real* env, not the sim's own assumptions)

**Why:** Tasks 3/4/6 only check *internal validity* (legal-in-space, function-equals-itself, sim-is-winnable). None verifies the sim matches *Minecraft*. This task captures real-MC obs once and asserts the sim reproduces it field-by-field — the only test that catches byte-parity bugs (channel-id map, `oak_log→132` inventory id, the `dy=+1` off-by-one) before they cost hours/eval in Phase C.

**Honesty note:** this task requires a **one-time ~seconds touch of a live Fabric instance** to capture the fixture (so Phase A is *not* purely server-independent). Use a free instance — the 12-instance M1B run occupies 25001–25012, so either capture after it finishes or launch a dedicated `NUM_INSTANCES=13` instance on port 25013 for the capture. The committed fixture then makes the test run offline forever.

**Files:**
- Create: `scripts/capture_gatherer_obs_fixture.py` (capture), `tests/fixtures/gatherer_obs_trace_seed1.json` (committed fixture), and a test in `tests/unit/test_sim_obs_parity.py`.

- [ ] **Step 1: Write the capture script** — model on `scripts/n14_reward_probe.py`: build the real `AiUtopiaPettingZooEnv` on a free port, `reset(seed=1)`, record `obs["gatherer_0"]` after reset and after each of a fixed scripted action sequence (e.g. 1× WAIT, then 3× HARVEST(oak_log, cap=1)). Serialize each obs to JSON (numpy→list) into the fixture. Keep the action sequence as a constant shared with the test.

- [ ] **Step 2: Capture the fixture** — on a free instance: `PYTHONPATH=src py -3.11 scripts/capture_gatherer_obs_fixture.py --port <free_port> --out tests/fixtures/gatherer_obs_trace_seed1.json`. Commit the fixture.

- [ ] **Step 3: Write the parity test**

```python
# add to tests/unit/test_sim_obs_parity.py
import json, numpy as np, pathlib, pytest
from aiutopia.sim.sim_env import AiUtopiaSimEnv   # (after Task 5) or drive world+obs_adapter directly

FIXTURE = pathlib.Path("tests/fixtures/gatherer_obs_trace_seed1.json")

# Fields the sim cannot/should-not match the real env on (constant embeds aside):
# none expected to differ for the DYNAMIC set we assert below.
DYNAMIC_FIELDS = ["position", "inv_slot_item_ids", "inv_slot_counts",
                  "g_resource_grid", "g_nearest_resources", "g_richness_score",
                  "nearest_resource_distance"]

@pytest.mark.skipif(not FIXTURE.exists(), reason="golden fixture not captured yet (needs a live instance once)")
def test_sim_obs_matches_real_golden_trace():
    trace = json.loads(FIXTURE.read_text())            # list of per-step real obs
    env = AiUtopiaSimEnv({"active_roles":["gatherer"], "max_episode_ticks":1000})
    obs, _ = env.reset(seed=1)
    _assert_step(obs["gatherer_0"], trace[0])
    for i, act in enumerate(_SCRIPTED_ACTIONS):        # same sequence the capture used
        obs, *_ = env.step({"gatherer_0": act})
        _assert_step(obs["gatherer_0"], trace[i+1])

def _assert_step(sim_obs, real_obs):
    for k in DYNAMIC_FIELDS:
        s = np.asarray(sim_obs[k], np.float64); r = np.asarray(real_obs[k], np.float64)
        assert s.shape == r.shape, f"{k}: shape {s.shape} != real {r.shape}"
        assert np.allclose(s, r, atol=1e-3), f"{k}: sim != real (max |Δ|={np.abs(s-r).max():.4f})"
```

- [ ] **Step 4: Run** — `py -3.11 -m pytest tests/unit/test_sim_obs_parity.py -x -q`. With the fixture present it must PASS; **a failure here (esp. on `g_nearest_resources` dy or `g_resource_grid`) is the real fidelity bug to fix in the obs adapter.** Without the fixture it SKIPS (so the suite stays green pre-capture, but the skip is visible).

- [ ] **Step 5: Commit** — `git add scripts/capture_gatherer_obs_fixture.py tests/fixtures/gatherer_obs_trace_seed1.json tests/unit/test_sim_obs_parity.py && git commit -m "test(sim): golden-trace obs parity vs real MC (Phase A task 3b)"`

---

## Task 4: Reward adapter (reuse the real reward fn)

**Files:**
- Create: `src/aiutopia/sim/reward_adapter.py`
- Test: `tests/unit/test_sim_reward_parity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sim_reward_parity.py
import numpy as np
from aiutopia.sim.world import SimWorld
from aiutopia.sim.obs_adapter import build_gatherer_obs
from aiutopia.sim.skills import apply_skill
from aiutopia.sim.reward_adapter import step_reward
from aiutopia.env.reward import compute_reward_stage_1

# NOTE: this is a FORWARDING test (step_reward must just call compute_reward_stage_1),
# NOT a parity test — reward correctness rides entirely on obs/inventory faithfulness,
# which is covered by the golden-trace test in Task 3b. Don't mistake green here for fidelity.
def test_step_reward_forwards_to_compute_reward_stage_1():
    w = SimWorld(); w.reset(seed=1)
    obs_prev = build_gatherer_obs(w)
    action = {"skill_type":1,"target_class":0,"spatial_param":np.zeros(3,np.float32),
              "scalar_param":np.array([1/64],np.float32),"comm_payload":np.zeros(128,np.float32),
              "should_broadcast":0,"comm_target_mask":np.zeros(4,np.int8)}
    w, comp = apply_skill(w, action)
    obs_curr = build_gatherer_obs(w)
    env_meta = {"died_this_tick": False, "n_clipped_param_axes": comp["clippedAxesBitset"].bit_count() if isinstance(comp["clippedAxesBitset"],int) else 0, "exploit_penalties": []}
    got = step_reward(obs_prev, obs_curr, action, env_meta)
    want = compute_reward_stage_1(role="gatherer", obs_prev=obs_prev, obs_curr=obs_curr, action=action, env_meta=env_meta)
    assert got == want                          # adapter must be a faithful pass-through
    assert got > 1.5                            # +1 oak_log primary + PBRS, matches real probe
```

- [ ] **Step 2: Run to verify fails** → FAIL.

- [ ] **Step 3: Implement `step_reward`** — thin wrapper that assembles `env_meta` (from the `skill_completion`: `died_this_tick=False` in M1B, `n_clipped_param_axes` from the clip bitset, `exploit_penalties=[]` for Phase A) and returns `compute_reward_stage_1(role="gatherer", obs_prev=..., obs_curr=..., action=..., env_meta=...)`. No reward math is reimplemented.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(sim): reward adapter reusing compute_reward_stage_1 (Phase A task 4)"`

---

## Task 5: SimEnv (PettingZoo-Parallel-shaped, drop-in for scenario_runner)

**Files:**
- Create: `src/aiutopia/sim/sim_env.py`
- Test: `tests/unit/test_sim_env.py`

- [ ] **Step 1: Write the failing test (lifecycle + termination parity)**

```python
# tests/unit/test_sim_env.py
import numpy as np
from aiutopia.sim.sim_env import AiUtopiaSimEnv

def _harvest(scalar=1/64):
    return {"gatherer_0": {"skill_type":1,"target_class":0,"spatial_param":np.zeros(3,np.float32),
            "scalar_param":np.array([scalar],np.float32),"comm_payload":np.zeros(128,np.float32),
            "should_broadcast":0,"comm_target_mask":np.zeros(4,np.int8)}}

def test_reset_returns_obs_for_agent():
    env = AiUtopiaSimEnv({"active_roles":["gatherer"],"max_episode_ticks":1000})
    obs, info = env.reset(seed=1)
    assert "gatherer_0" in obs

def test_success_terminates_at_64_oak_log():
    env = AiUtopiaSimEnv({"active_roles":["gatherer"],"max_episode_ticks":1000})
    obs, _ = env.reset(seed=1)
    term = {"gatherer_0": False}
    steps = 0
    while not term["gatherer_0"] and steps < 70:
        obs, rew, term, trunc, info = env.step(_harvest(scalar=1/64)); steps += 1
    assert term["gatherer_0"] is True
    assert info["gatherer_0"]["goal_success"] is True
    assert steps == 64                                   # 64 logs, cap=1 -> exactly 64 steps

def test_truncates_at_max_ticks():
    env = AiUtopiaSimEnv({"active_roles":["gatherer"],"max_episode_ticks":5})
    env.reset(seed=1)
    for _ in range(5):
        obs, rew, term, trunc, info = env.step({"gatherer_0": {"skill_type":4,  # WAIT
            "target_class":0,"spatial_param":np.zeros(3,np.float32),"scalar_param":np.zeros(1,np.float32),
            "comm_payload":np.zeros(128,np.float32),"should_broadcast":0,"comm_target_mask":np.zeros(4,np.int8)}})
    assert trunc["gatherer_0"] is True
```

- [ ] **Step 2: Run to verify fails** → FAIL.

- [ ] **Step 3: Implement `AiUtopiaSimEnv`** — same surface as `AiUtopiaPettingZooEnv`: `__init__(config: dict)`, `reset(seed) -> (obs, infos)`, `step(action_dict) -> (obs, rew, term, trunc, info)`, `observation_space(agent)`/`action_space(agent)` from `spaces.py`, `agents`/`possible_agents=["gatherer_0"]`, `close()`. Internally: hold a `SimWorld`; `step` → `apply_skill` → build `obs_curr` → `step_reward` → set `term[agent]` via the **same `_goal_success` check the wrapper uses** (import it from `env/wrapper.py` or the shared helper) against the goal spec target {oak_log:64}; `trunc[agent]` on `world.tick >= max_episode_ticks` OR out-of-bounds (`abs(x-64)>24 or abs(z+48)>24 or y<60`, matching wrapper N19); `info[agent]["goal_success"]` + `info[agent]["skill_completion"]`. Prune finished agents like the wrapper.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(sim): AiUtopiaSimEnv lifecycle + termination parity (Phase A task 5)"`

---

## Task 6: Full-suite + the scripted gate proof, in sim

**Files:** Test: reuse `scripts/p0_gate_proof.py` logic against the sim.

- [ ] **Step 1** Add `tests/unit/test_sim_env.py::test_scripted_policy_solves_gate_in_sim`: drive `AiUtopiaSimEnv` with scripted HARVEST(cap=1) and assert `scenario_runner._gatherer_collected_64_oak_log(final_obs)` is True and the episode terminated with `goal_success` — the sim analogue of the real gate proof.
- [ ] **Step 2** Run the **whole** unit suite: `py -3.11 -m pytest tests -m "not integration and not determinism" -q` → all pass (sim adds tests, breaks nothing).
- [ ] **Step 3: Commit** — `git commit -m "test(sim): scripted gate proof in sim + full-suite green (Phase A task 6)"`

---

## Out of scope for Phase A (follow-on plans)

- **Phase B** — vectorize `SimWorld`/`apply_skill` (batched NumPy), wrap as an RLlib vector env, register in `env_factory`, train the gatherer in sim, confirm the learning slope. (Separate plan.)
- **Phase C** — transfer validation: run the sim-trained RLModule through `scenario_runner.run_scenario` in real MC via `M1EvalScenarioCallback`; close fidelity gaps until the real-MC gate clears; **record the fidelity-debug cost.** (Separate plan.)
- **Phase D** — JAX end-to-end port for the 10⁵–10⁶ steps/s regime, with a weight-bridge to the PyTorch RLModule for real-MC deploy. (Separate plan, conditional.)

---

## Self-Review

**Spec coverage:** Components 1–6 of the spec map to Tasks 1–5 (SimWorld→T1, SkillModel→T2, ObsAdapter→T3, RewardAdapter→T4, SimEnv→T5; TransferValidator is Phase C, explicitly deferred). **Spec §7's real parity test — "drive identical actions through sim and recorded real-MC traces, assert obs match within tolerance" — is Task 3b (golden trace); this is the one test that validates against Minecraft rather than the sim's own assumptions.** Be clear-eyed that T3 (`space.contains`), T4 (forwarding), and T6 (sim-is-winnable) are *internal-validity* checks — necessary but not fidelity proof; T3b is the fidelity gate. Full transfer (sim-trained policy clears the real-MC gate) remains Phase C. ✓
**Placeholder scan:** No TBD/TODO; each task has real test code + concrete constants/formulas; byte-faithful obs internals reference the authoritative `GathererOverlayBuilder.java` with the exact channel/scaling rules reproduced inline. ✓
**Type consistency:** `SimWorld` fields (`agent_pos`, `logs`, `log_alive`, `inventory`, `tick`) used consistently across T1–T5; `apply_skill(world,action)->(world,completion)` and `build_gatherer_obs(world)->dict` and `step_reward(...)->float` signatures stable across tasks. Skill enum (0..5) fixed in T2 and reused in T5/T6. ✓
**Open item to confirm during T2/T3 (flagged, not a placeholder):** the exact skill-enum integer order and whether `_uuid_embed`/goal-stub are importable from `wrapper.py` or need lifting to `env/_embeds.py` — the implementer resolves by reading `wrapper.py` and refactors minimally if needed.
