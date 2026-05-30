# M2 Farmer Role — Obs/Action/Reward Contract Design

**Date:** 2026-05-30  
**Author:** Reward-design-and-imitation-learning-specialist  
**Milestone:** M2 (Farmer is M3 per IMPLEMENTATION_PLAN; this spec is M2 planning work)  
**Status:** Design spec for M2 integration (not yet code)

---

## Executive Summary

The **Farmer role** tests **temporal credit assignment and delayed rewards** — a fundamentally different learning axis from the Gatherer's immediate-harvest model. The task: plant seeds on tilled soil, wait for crops to mature (8 stages, 64–80 ticks total), harvest ripe crops, repeat. This spec defines the obs/action/reward contract, grounded in Minecraft crop mechanics and AiUtopia's architecture, and proposes test infrastructure for validating transfer from sim to real MC.

**Key design constraints:**
- Temporal delay (50–100 ticks between plant and harvest) creates late reward signal
- Sparse principal reward (harvest only) must be paired with carefully-decayed shaping to guide exploration
- Exploit-resistance: agent must not re-farm the same cell, not stand idle while waiting, not harvest unripe crops
- Integration with MAPPO/CTDE: Farmer's delayed rewards should not degrade Gatherer's faster convergence

**TL;DR** — The contract is ready for M2 implementation. The three open questions are empirical: (1) does PPO converge on delayed rewards within the M1B wall-clock budget (~4 hours real MC, ~1–2 minutes sim)? (2) does the policy learn to wait flexibly for ripeness, or does it collapse to a fixed timer? (3) what is the true sim-to-real transfer gap for this temporal structure?

---

## 1. Observation Space Contract

### 1.1 Core obs (shared with all roles)

Reuse the existing `_core_space()` from `src/aiutopia/env/spaces.py`:
- `position`, `velocity`, `yaw_pitch` — agent state
- `goal_embedding`, `goal_ticks_left` — LLM planner interface (M5; M2 static for now)
- `time_of_day` — optional (crops grow on Minecraft day-night cycles; defer to M2.2 if needed)
- `health`, `hunger`, `saturation` — agent state
- `inv_slot_item_ids`, `inv_slot_counts` — inventory (tracks seeds, crops, harvestable food)
- `agent_uuid_embed`, `role_one_hot`, `tick_in_episode` — identity

See `spaces.py::_core_space()` for full spec; Farmer uses the same 512-dim GOAL_EMBED_DIM as Gatherer.

### 1.2 Farmer-specific obs: crop grid + ripeness signals

Analogous to `_gatherer_overlay()`, add a `_farmer_overlay()` dict with:

```python
def _farmer_overlay() -> dict:
    return {
        # Per-cell crop age: 0=empty, 1-7=growing stages, 8=ripe.
        # Shape (32, 32) horizontal scan, single y-layer (farmland).
        # Updated every step via Java-side crop grid scan.
        "f_crop_grid": Box(0, 8, (32, 32), np.uint8),
        
        # Fraction of visible cells at stage 8 (ripe).
        # Used for progress monitoring and reward shaping.
        # Scalar, range [0, 1].
        "f_ripeness": Box(0, 1, (1,), np.float32),
        
        # Count of unique cells planted this episode (first-plant only).
        # Resets on reset_episode. Used to detect re-planting the same cell.
        # Scalar, range [0, 1024] (max cells in 32x32).
        "f_planted_count": Box(0, 1024, (1,), np.int32),
        
        # Count of cells harvested this episode (ripe crops broken).
        # Scalar, cumulative. Used for the sparse principal reward and eval.
        "f_harvested_count": Box(0, 1024, (1,), np.int32),
        
        # Per-cell harvest history (binary): has this cell been harvested
        # this episode? Shape (32, 32). Used to enforce "harvest once per
        # plant" and to detect exploits (re-harvest the same cell).
        "f_harvested_mask": MultiBinary((32, 32)),
        
        # Time since crop[i, j] entered stage 8 (ripeness window).
        # Shape (32, 32). Value is ticks since ripeness; 0 = not ripe yet.
        # Capped at 100 (max window before despawn). Used for reward shaping
        # to guide "harvest ripe crops in a timely manner, not too late."
        "f_time_at_ripeness": Box(0, 100, (32, 32), np.int32),
    }
```

**Justification for each:**
- **f_crop_grid** — Analogous to g_resource_grid; agent needs to know which cells have crops and at what stage. Stages 1-7 are "not yet ripe" (policy should wait or plant elsewhere); stage 8 is "harvest now."
- **f_ripeness** — Scalar progress signal for the PBRS potential-based shaping (see §3.3). Peaks when many crops are ripe.
- **f_planted_count** — Tracks novelty (new cells planted). Decays the shaping reward for planting: only the *first* plant per cell yields r_plant bonus.
- **f_harvested_count** — The *actual outcome* we care about. Not used in reward shaping (that would create an attractor); instead it is the sparse principal reward and the eval metric.
- **f_harvested_mask** — Anti-exploit: prevents "harvest the same cell twice" by masking out harvested cells. Cleared on reset_episode.
- **f_time_at_ripeness** — Temporal shaping signal. If a crop has been ripe for 50+ ticks without being harvested, the agent may be idle or the policy may have learned a suboptimal ordering. This term can decay to push harvest timeliness.

**Implementation notes:**
- Java-side: extend `GathererOverlayBuilder` → `FarmerOverlayBuilder` (or refactor as role-agnostic overlay builder). Scan farmland (block type = `minecraft:farmland` or similar), iterate (x, z) in a 32×32 grid at agent.y (same y-layer scan as Gatherer), read the crop age from NBT or from a custom Fabric-side tracker.
- Crop age in real MC: crops use a block-state property like `age:0..7` (varies by crop type — wheat, carrots, potatoes, etc.). Stage 8 in this spec maps to "fully grown" (age=7 in vanilla, or a custom "ripe" flag if we track it ourselves). We can unify different crop types by normalizing age to 0-7 and adding a "ripe" flag for stage 8.
- Python-side: if `time_of_day` is included (crops grow only during day; night growth is slow), pass it to the policy so it learns to wait during daytime and harvest at dawn. M2.1 simplification: ignore day-night cycles, assume 24/7 growth for faster training.

---

## 2. Action Space Contract

### 2.1 Discrete skill types for Farmer

Extend the skill enum. Currently Gatherer has 6 skills (navigate, harvest, deposit_chest, search, wait, noop_broadcast). **Farmer has 7:**

```python
N_FARMER_SKILLS = 7

# Enum-like:
# 0 = NAVIGATE (move toward (x, z) target)
# 1 = PLOW (till soil at agent's position)
# 2 = PLANT (place seed on tilled soil at agent's position)
# 3 = HARVEST (break fully-grown crop at agent's position)
# 4 = WAIT (do nothing for 1 tick; allows temporal waiting)
# 5 = NOOP_BROADCAST (communicate without action; Carpet idle tick)
# 6 = noop or SEARCH (not used in M2.1; defer to M2.2 if needed)
```

The action dict remains the same structure as Gatherer:

```python
DictSpace({
    "skill_type": Discrete(N_FARMER_SKILLS),        # which skill to dispatch
    "target_class": Discrete(N_TARGET_CLASSES_PER_ROLE),  # 64 target options (block position, direction bias, etc.)
    "spatial_param": Box(-1, 1, (3,), np.float32),  # e.g., normalized offset for next plant cell
    "scalar_param": Box(0, 1, (1,), np.float32),    # e.g., timeout factor (crops rarely time out; mostly unused)
    "comm_payload": Box(-1, 1, (COMM_PAYLOAD_DIM,), np.float32),  # future: inter-agent message
    "should_broadcast": Discrete(2),                # broadcast message or not
    "comm_target_mask": MultiBinary(4),             # which agents see the message (gatherer, builder, farmer, defender)
})
```

**Why these three skills are new:**
- **PLOW** — Minecraft farmland requires two steps: (1) hoe a dirt/grass block to create `minecraft:farmland`, (2) place seed on the farmland. Farmer's first action when reaching a cell must be PLOW if the cell is bare dirt, then PLANT on the next tick (or same tick if we chain). Real MC crops work this way; the sim can simplify (auto-till on PLANT), but must align for transfer.
- **PLANT** — Place a seed (wheat_seeds, carrot, potato, etc.) on tilled farmland. Consumes a seed from inventory; farmland transitions to stage 1. Success requires: (a) agent within reach (≤4.5 blocks, learned from HARVEST N16 fix), (b) target cell has farmland, (c) inventory has a seed.
- **HARVEST** — Break a fully-grown crop (stage 8). Yields seeds + crop items (wheat, carrots, etc.). Analogous to HARVEST for wood, but must **check ripeness before breaking**: agent must confirm the crop is at stage 8. Breaking stage 7 yields nothing (waste). No skill will output harvest on stage <8 if the policy learns well, but the action space allows it; the reward signal must not reinforce it.

### 2.2 Action dispatch in Motor

Java-side `MotorBridge.dispatchSkill` adds three branches:

```java
case "PLOW":
    // Input: target_class -> block position (x, z) within 32x32 grid
    // Walk to (x, z, farmland_level)
    // If not farmland: use a hoe (if in inventory) to till it
    // If already farmland: no-op (already tilled)
    // Timeout: 400 ticks (same as HARVEST)
    break;
case "PLANT":
    // Input: target_class -> block position
    // Walk to (x, z, farmland_level)
    // Find a seed in inventory (wheat_seeds, carrot, potato, etc.)
    // Place it on the farmland
    // Consume 1 seed from inventory
    // Farmland transitions to stage 1 (crop age = 0)
    // Timeout: 400 ticks
    break;
case "HARVEST":
    // Input: target_class -> block position
    // Walk to (x, z)
    // Read the crop block's age property from NBT
    // If age == 7 (or ripe flag set): break the block, collect drops
    // If age < 7: do nothing (silent fail; policy learns not to harvest unripe)
    // Timeout: 400 ticks
    break;
```

All three reuse the same reach/walk math as Gatherer's HARVEST (HarvestSkill.java):
- `REACH_RADIUS = 4.5` (blocks away before auto-break)
- `WALK_PER_TICK = 0.3` (Minecraft-faithful movement)
- `ticks_to_target = ceil((distance - REACH) / WALK_PER_TICK)`

---

## 3. Reward Contract

### 3.1 Sparse principal reward

```
r_principal = +1.0 per unique cell successfully harvested (stage 8 → break → items received)
```

**Key:**
- Increments only when a crop at stage 8 is harvested and items enter inventory.
- Once per cell per episode (same cell harvested twice yields +1.0 only on the first harvest).
- Non-zero only when the policy completes the full plant → wait → harvest cycle.
- This is the true objective; all shaping decays to zero by episode end.

### 3.2 Potential-based reward shaping (PBRS, decaying)

Let φ(s) = potential function. We define three potential terms:

```python
def farmer_potential(
    obs: dict, 
    decay_coeff: float = 1.0  # 1.0 → no decay; 0.0 → fully decayed by episode end
) -> float:
    """Φ(s) for Farmer PBRS. Three terms, all scaled by decay_coeff."""
    
    # Term 1: Progress toward planting a full field.
    # Φ₁ = min(1.0, f_planted_count / 64)
    # Rationale: early in training, encourages exploration and planting.
    # By mid-episode, as crops ripen, this term is nearly constant; PBRS ≈ 0.
    # Decayed: yes, via decay_coeff.
    planted_progress = min(1.0, obs["f_planted_count"][0] / 64.0)
    
    # Term 2: Ripeness progress.
    # Φ₂ = f_ripeness (fraction of visible cells at stage 8)
    # Rationale: guides the policy to wait for crops to mature.
    # Peaks when many crops are ripe at once (temporal coordination).
    # Decayed: yes, via decay_coeff.
    ripeness_progress = obs["f_ripeness"][0]
    
    # Term 3: Harvest timeliness (inverse of staleness).
    # Φ₃ = mean(1.0 - min(1.0, f_time_at_ripeness[i,j] / 50.0))
    #      over all ripe cells (f_crop_grid[i,j] == 8)
    # Rationale: penalizes leaving ripe crops un-harvested for >50 ticks.
    # If a crop is harvested quickly after ripening, Φ₃ is high.
    # If a crop sits ripe for 100+ ticks, Φ₃ approaches 0.
    # Decayed: yes, via decay_coeff.
    ripe_cells = obs["f_crop_grid"] == 8
    if ripe_cells.any():
        staleness = np.minimum(1.0, obs["f_time_at_ripeness"][ripe_cells] / 50.0)
        timeliness_progress = np.mean(1.0 - staleness)
    else:
        timeliness_progress = 0.0
    
    # Composite potential (all three terms, with per-term decay).
    phi = (
        0.15 * planted_progress +  # weight: early-game exploration
        0.50 * ripeness_progress +  # weight: main signal (waiting + ripeness)
        0.35 * timeliness_progress  # weight: harvest discipline
    ) * decay_coeff
    
    return phi
```

**Decay schedule:**
```python
def decay_coeff_for_tick(tick: int, max_ticks: int = 1000) -> float:
    """Linear decay: 1.0 at tick 0, 0.0 at tick max_ticks."""
    return max(0.0, 1.0 - tick / max_ticks)
```

The r_pbrs term in `compute_reward()` becomes:

```python
gamma = 0.99
decay_t = decay_coeff_for_tick(current_tick, max_episode_ticks)
decay_t_next = decay_coeff_for_tick(current_tick + 1, max_episode_ticks)
phi_curr = farmer_potential(obs_curr, decay_coeff=decay_t)
phi_next = farmer_potential(obs_next, decay_coeff=decay_t_next)
r_pbrs = gamma * phi_next - phi_curr
```

**Justification:**
- **All three terms are decay-mandatory.** If we kept decay_coeff = 1.0 throughout the episode, the policy might learn to plant and wait forever (infinite ripeness reward). By decaying, we force the policy to harvest as ripeness appears (the r_pbrs gradient pushes toward harvest when decay is high).
- **Why three separate terms?** (1) Planted count pushes early exploration; (2) ripeness is the main temporal signal (waits for crops); (3) timeliness prevents the policy from planting once and then ignoring ripe crops for 100 ticks.
- **Weights (0.15 / 0.50 / 0.35):** Ripeness is the primary signal; timeliness is a secondary discipline term; planted count is early-game only. Tunable in M2.2 if convergence is slow.

### 3.3 Anti-exploit penalties (per-tick deductions)

```python
def farmer_exploit_penalties(env_meta: dict) -> list[tuple[str, float]]:
    """Return a list of (exploit_name, penalty_value) tuples.
    These are summed in compute_reward as: r_exploits = Σ penalties.
    """
    penalties = []
    
    # Penalty 1: Re-plant detection.
    # If the policy tries to PLANT on a cell already in f_planted_mask, penalize.
    if env_meta.get("tried_to_plant_harvested_cell", False):
        penalties.append(("re_plant_same_cell", 0.1))  # small but noticeable
    
    # Penalty 2: Harvest unripe.
    # If HARVEST is dispatched on a cell with age < 8, penalize.
    if env_meta.get("tried_to_harvest_unripe", False):
        penalties.append(("harvest_unripe", 0.05))
    
    # Penalty 3: Idle detection (similar to Gatherer's lazy penalty).
    # If no skill changes state (inventory, position, crop grid), penalize lightly.
    if env_meta.get("no_state_change_this_tick", False):
        penalties.append(("idle_tick", 0.01))
    
    return penalties
```

**Why these penalties exist:**
- **Re-plant:** Prevents the policy from planting the same cell multiple times, which would earn multiple +1.0 rewards and waste seeds.
- **Harvest unripe:** Prevents premature harvesting (wastes ticks waiting for that broken crop to regrow).
- **Idle:** Prevents standing still while waiting for crops (subtle but important — the policy should use the time to plant more cells or prepare the next area).

**Magnitude notes:**
- Each penalty is small (0.01–0.1) compared to r_principal = 1.0 per harvest. This ensures they don't dominate the reward signal; they are guardrails, not the main gradient.
- All penalties are *independent* of decay_coeff; they apply throughout the episode. This is correct: we always want to discourage these behaviors, even late in the episode.

### 3.4 Stage-1 reward composition for Farmer

```python
def compute_reward_stage_1_farmer(
    *, 
    role: str,  # must be "farmer"
    obs_prev: dict, 
    obs_curr: dict, 
    action: dict, 
    env_meta: dict
) -> float:
    """Farmer stage-1 reward (M2 implementation).
    
    r_farmer = r_principal + r_pbrs - r_death - r_time - r_exploits - r_clip
    
    See reward.py compute_reward_stage_1 for gatherer. Farmer extends with:
    - r_principal: +1.0 per unique harvested cell
    - r_pbrs: potential-based shaping with decay (farmer_potential φ)
    - r_death, r_time, r_exploits: same as gatherer
    """
    assert role == "farmer", f"use compute_reward_stage_1 for {role}"
    
    # Sparse principal: did a new crop get harvested?
    harvested_delta = (
        obs_curr["f_harvested_count"][0] - obs_prev["f_harvested_count"][0]
    )
    r_principal = float(harvested_delta)  # +1.0 per harvest, else 0.0
    
    # PBRS potential with decay.
    tick_curr = obs_curr["tick_in_episode"][0]
    tick_next = obs_curr["tick_in_episode"][0]  # next obs has +1 tick
    max_ticks = env_meta.get("max_episode_ticks", 1000)
    
    decay_curr = max(0.0, 1.0 - tick_curr / max_ticks)
    decay_next = max(0.0, 1.0 - (tick_curr + 1) / max_ticks)
    
    phi_curr = farmer_potential(obs_prev, decay_coeff=decay_curr)
    phi_next = farmer_potential(obs_curr, decay_coeff=decay_next)
    gamma = 0.99
    r_pbrs = gamma * phi_next - phi_curr
    
    # Universal penalties.
    r_death = 10.0 if env_meta.get("died_this_tick", False) else 0.0
    r_time = 0.001  # per-tick cost (tiny, mostly cosmetic)
    r_clip = 0.05 * int(env_meta.get("n_clipped_param_axes", 0))
    
    # Farmer-specific exploit penalties.
    exploit_penalties = farmer_exploit_penalties(env_meta)
    r_exploits = sum(p for _, p in exploit_penalties)
    
    return r_principal + r_pbrs - r_death - r_time - r_exploits - r_clip
```

---

## 4. Episode Mechanics

### 4.1 Duration and reset

- **Episode length:** 1000 ticks (50 seconds real-time at 20 TPS; ~3–5 seconds sim-time at ~222 macro-steps/s)
- **Crop growth timer (per-stage):** Fixed 8 ticks per stage (8 stages = 64 ticks total from plant to harvest). Real MC's growth is probabilistic (random ticks); sim should match this (deterministic growth is sim-only; real MC validation will show variance). See §5.2 for sim-fidelity discussion.
- **Reset on episode end:** f_planted_count, f_harvested_count, f_harvested_mask all reset. Crop grid clears (farmland reverts to empty).

### 4.2 Initial conditions

```
Agent spawn: (64.5, 66, -47.5)  [same ring as Gatherer M1B]
Farmland arena: cleared dirt area, pre-tilled farmland, seed chest nearby.
Seed inventory: start with ~128 wheat_seeds in inventory (or chest nearby).
Horizon: 1000 ticks; if agent reaches 64 harvests before tick 1000, episode ends early (opt-in gate).
```

### 4.3 Eval gate proposal (M2.2)

```
Farmer must achieve 32 harvested crops within 1000 ticks over 3 consecutive eval runs.
Success rate: 80%+ (2 of 3 evals pass 32-harvest threshold).
Timing: eval runs every 10 training iters (same as M1B Gatherer gate).
```

Rationale:
- 32 harvests ≈ 2–3 minutes of continuous efficient planting/waiting/harvesting (if the policy learns well).
- 80% is a meaningful success rate (not 100%, allowing for some variance in seed availability and crop timing).
- 3 evals reduces noise from a single lucky/unlucky run.

---

## 5. Integration with MAPPO/CTDE

### 5.1 Multi-role training

Farmer will be trained **in parallel with Gatherer** under MAPPO (per-role PPO, shared centralized critic). In M2, we skip Builder/Defender and focus on Gatherer + Farmer, testing:
- Do they learn stable roles (Gatherer collects wood, Farmer grows food)?
- Does delayed Farmer reward degrade Gatherer's convergence?
- Can the shared critic handle the two different reward scales/timescales?

### 5.2 Shared critic weight-sharing

The CTDE critic remains centralized:

```python
class CTDECritic:
    def forward(self, state_dict):
        # All agents' obs → shared features → value head
        # Farmer's slower reward convergence should not destabilize this.
        # Key: the critic sees *all* agents' obs, so it learns to predict
        # both fast (Gatherer harvest) and slow (Farmer ripeness) rewards.
        ...
```

**Concern:** If Gatherer converges quickly (by iter 20) and Farmer is still exploring (iter 30+), the shared critic might over-fit to Gatherer's policy and give weak gradients to Farmer. **Mitigation:** use value-function clipping + KL penalties per role (already in m1_gatherer_config; reuse for both).

### 5.3 Potential-based PBRS under CTDE

The PBRS shaping term `r_pbrs = γ Φ(s') - Φ(s)` is **policy-invariant per single agent** (Ng et al. 1999). In CTDE, we sum the per-agent PBRS terms:

```
r_total = r_gatherer_principal + r_gatherer_pbrs + r_farmer_principal + r_farmer_pbrs - r_shared_penalties
```

**Key constraint:** Φ_gatherer and Φ_farmer must be on **comparable magnitude scales** so the shared critic doesn't get dominated by one role's potential. Currently:
- `tech_tree_potential` (Gatherer's Φ) sums LOG_VALUE over capped inventory: 0–200 typical.
- `farmer_potential` (proposed Φ_farmer) is a weighted sum of three progress terms: 0–1.0 typical.

**Fix:** Scale Farmer's φ by ~100 so both roles' Φ deltas feed gradients at similar scale:

```python
def farmer_potential(...) -> float:
    # ... compute as above ...
    phi = (0.15 * planted + 0.50 * ripeness + 0.35 * timeliness) * decay_coeff
    return 100.0 * phi  # scale to match gatherer_potential's ~0-200 range
```

This is **not** a policy-invariance issue (PBRS remains policy-invariant per agent), just a **numerical stability** issue in CTDE. Verify during M2.2 training: if Farmer's gradient is 100× smaller than Gatherer's, the critic learns a weird interpolation; scaling fixes it.

---

## 6. Sim Expressibility

### 6.1 Required sim extensions

The AiUtopiaSim (fast NumPy scalar sim; see PROJECT_CONTEXT §9.1) must be extended to support Farmer:

```python
class SimSkillDispatcher:
    def dispatch_skill(self, skill_type: int, target_class: int, ...) -> dict:
        """Extend to handle PLOW / PLANT / HARVEST."""
        if skill_type == SkillEnum.PLOW:
            # Find cell at target position
            # Mark it as "tilled farmland"
            # No cost (instant)
            pass
        elif skill_type == SkillEnum.PLANT:
            # Check: cell is tilled farmland
            # Check: agent has seeds in inventory
            # Place a crop (age=0) at the cell
            # Consume 1 seed
            pass
        elif skill_type == SkillEnum.HARVEST:
            # Check: crop at target is age==7 (ripe)
            # If yes: break it, add items to inventory, set cell to empty
            # If no: silent fail (no penalty, just no items)
            pass
```

**Crop aging:** Deterministic (not random-tick based like MC). Each tick, age += 1 if the crop is planted:

```python
for i, j in planted_cells:
    crop_age[i, j] += 1
    if crop_age[i, j] > 8:
        crop_age[i, j] = 8  # cap at "ripe"
```

**Inventory tracking:** Same as real MC — PlantSkill consumes seeds; HarvestSkill adds items.

**Observable:** Obs contract (f_crop_grid, f_ripeness, etc.) updated from `crop_age` array.

### 6.2 Real MC implementation

On real Fabric servers (with Py4J):

1. **Farmland arena:** A pre-tilled, seed-stocked area (64×64 blocks, all farmland, seed chest at corner).
2. **PLOW skill:** If target cell is `minecraft:dirt` or `minecraft:grass_block`, till it (hoe + right-click). If already farmland, no-op.
3. **PLANT skill:** Place seed on farmland; consume from inventory. Vanilla crop mechanics handle growth (random ticks).
4. **HARVEST skill:** Right-click a fully-grown crop. Check NBT `age` property or a custom Fabric-side tag. If ripe: break, collect drops. If not ripe: no-op.
5. **Crop growth:** Vanilla Minecraft — random-tick based, day-night dependent, biome-dependent. Policy must learn to be patient (crops take 50–150 ticks typically, not fixed 64).

---

## 7. Temporal Credit Assignment & Learning Challenges

### 7.1 The delayed reward problem

**Challenge:** Harvests happen ~64 ticks after planting. On-policy PPO must wait that long to see a reward signal. In the worst case:

- Tick 0–10: plant 10 seeds
- Tick 10–64: wait (exploration, or plant more)
- Tick 64–80: crops ripen, policy gets +1.0 per harvest
- Tick 80–1000: policy learns the pattern and repeats

The 64-tick delay means the policy must **learn to correlate its tick-0–10 actions with tick-64 outcomes**, a gap that simple TD bootstrapping may struggle with.

**Mitigations (M2.2 tuning):**
1. **Multi-step returns:** Use n-step TD with n ≥ 64 so the value function can see across the delay. Ray/RLlib's `lambda=0.99` (GAE) already does this; may need to increase `lambda` (trade bias for variance).
2. **Reward shaping (decay):** The r_pbrs term with decay helps by providing intermediate signals (ripeness progress, timeliness) that appear earlier than harvest.
3. **Curriculum learning:** Phase 1 (ticks 0–200): harvest only (crops pre-planted, agent just learns to harvest). Phase 2 (ticks 0–1000): full plant→wait→harvest. This is deferred to M2.3 but is a known good lever.

### 7.2 Exploration and patience

**Challenge:** The policy must learn to *wait*. If it takes action every tick (PLANT or HARVEST), but crops ripen on a fixed timer, the policy might learn to:
- Plant, wait for 3 ticks (impatience), plant again (same cell) → re-plant exploit
- Plant, wait for 60 ticks, harvest (good)
- OR: plant once, then stand idle for 60 ticks (passive patience, but wastes exploration)

**The honest truth:** r_pbrs shaping with decay should encourage planting more cells (spread exploration) rather than passivity. But there is a risk the policy learns "plant once, wait 64 ticks, harvest" and ignores all other cells. Test metric: harvest efficiency = harvested_count / planted_count. If it's <0.5 (harvest only 50% of planted crops), the policy is abandoning cells.

**Mitigation:** The f_time_at_ripeness term in φ is designed to address this — if a crop has been ripe for >50 ticks, the PBRS gradient pushes toward harvesting it soon. If the policy ignores it, φ → 0 and r_pbrs becomes negative, providing a small penalty.

### 7.3 Sim-to-real transfer gap

**Primary risk:** Crop growth is deterministic in sim (always 64 ticks), probabilistic in real MC (50–150 ticks, varies by biome and light level). A policy trained in sim might learn a fixed "wait 64 ticks" timer. On real MC, some crops ripen early (40 ticks), others late (150 ticks). The policy could:
- Harvest too early (< 50 ticks) on some crops → fail
- Idle waiting for a late crop → inefficiency

**Transfer validation (M2.3):** After training in sim to convergence (~1–2 minutes), run a short eval in real MC (~5 minutes wall, ~3 iter equivalents) on a subset of the Fabric servers. Compare:
- Sim eval: harvested_count over 300 ticks
- Real-MC eval: harvested_count over 300 ticks

If real-MC is ≥80% of sim, transfer is successful. If <50%, the timing assumptions broke and we need to fine-tune on real MC.

---

## 8. Test Proposals

### 8.1 Sim Phase (M2.2a)

**Setup:**
- 16×16 farmland (256 cells, vs real 32×32 = 1024)
- Agent spawns at center
- 1000 seed chest nearby (unlimited supply)
- Goal: maximize harvested_count in 1000 ticks

**Baselines:**
1. **Scripted upper bound:** Tile-scan pattern (walk in rows, PLOW, PLANT on every cell, wait 64 ticks, HARVEST all). Expected: ~40–50 harvested (seed depletion or time limit).
2. **Greedy random:** PLANT at random empty cells; HARVEST when ripe. Expected: ~15–25 harvested (inefficient planting pattern).
3. **Policy trained in sim (M2.2 PPO):** Expected: 30–40 harvested if policy learns the task.

**Metric:** harvested_count at episode end.

**Failure modes to watch for:**
- Policy learns re-planting (same cell twice) → hit re-plant penalty.
- Policy learns unripe harvest → no items, but slow down.
- Policy learns idle waiting → wastes ticks.
- Policy converges to "plant few, harvest all" instead of "plant many, harvest many" (low coverage).

### 8.2 Real-MC Phase (M2.2b → M2.3)

**Setup:**
- 32×32 farmland arena (already used for Gatherer; can be reused or adjacent)
- 1 Fabric instance (port 25001)
- Real Minecraft 1.21.1 crop mechanics (random ticks, day-night cycle)

**Test 1: Cold-start policy transfer**
- Load a weights checkpoint from sim-trained Farmer (iter 100+)
- Run 5 eval episodes on real MC
- Metric: harvested_count (3 of 5 eps should meet 32-harvest gate)
- Expected outcome: 60–80% success (some crops fail due to timing variance, but policy adapts quickly)

**Test 2: Fine-tune on real MC**
- If Test 1 < 60%: continue training on real MC for 5–10 iters
- Metric: gate passage
- Expected outcome: >80% success after fine-tune

**Test 3: Generalization to new farmland layout**
- Re-arrange farmland (different seed placements, different shape)
- Run 3 evals with frozen fine-tuned policy
- Metric: harvested_count
- Expected outcome: ≥80% of original layout (policy learned the task, not memorized the layout)

---

## 9. Integration Checklist (M2 Implementation)

### 9.1 Python-side changes

- [ ] `src/aiutopia/env/spaces.py`: Add `_farmer_overlay()` + `build_role_observation_space` branch for "farmer"
- [ ] `src/aiutopia/env/reward.py`: Add `farmer_potential()`, `farmer_exploit_penalties()`, `compute_reward_stage_1_farmer()`, extend LOG_VALUE for farmer items (wheat, bread, etc.)
- [ ] `src/aiutopia/rl_module/farmer_encoders.py`: FarmerCoreEncoder (extends CoreEncoder with farm-specific preprocessing), FarmerActorHead (7 skills)
- [ ] `src/aiutopia/rl_module/role_rl_module.py`: Add FarmerRoleRLModule (LSTM time-stepping like GathererRoleRLModule)
- [ ] `src/aiutopia/train/config.py`: Add `m2_marl_config` builder (MAPPO + Gatherer + Farmer)
- [ ] `src/aiutopia/env/bridge.py`: Update action encoder for 7-skill Farmer; decode target_class → (x, z) on farmland
- [ ] `src/aiutopia/env/wrapper.py`: f_crop_grid, f_ripeness, etc. observation extraction from Java-side obs

### 9.2 Java-side changes

- [ ] `fabric_mod/src/main/java/dev/aiutopia/mod/obs/FarmerOverlayBuilder.java`: Scan farmland, crop age per-cell
- [ ] `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/PlowSkill.java`: New skill
- [ ] `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/PlantSkill.java`: New skill
- [ ] `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java`: Extend to check crop ripeness (age==7), fail silently if unripe
- [ ] `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java`: Dispatch PLOW, PLANT, HARVEST skills
- [ ] `fabric_mod/gradle.properties`: Bump mod_version to reflect M2 changes

### 9.3 Sim-side changes (if building full JAX vectorized sim in Phase D)

- [ ] `scripts/sim_env.py` (or new `src/aiutopia/sim/`): Add crop age array + PLOW/PLANT/HARVEST logic
- [ ] Same obs contract (f_crop_grid, f_ripeness, etc.)
- [ ] Deterministic crop growth (age += 1 per tick)
- [ ] Integration with existing RLlib env factory

### 9.4 Tests

- [ ] Unit test: `farmer_potential()` against known inputs (e.g., 32 planted, 8 ripe → φ should be mid-range)
- [ ] Integration test: Farmer obs space builds correctly; `build_role_observation_space("farmer")` returns the right shape dict
- [ ] Smoke test: Random policy rolls out 100 steps on sim; obs values are in range, no NaN
- [ ] Scenario test: Scripted Farmer (tile-walk, PLOW, PLANT, WAIT 64, HARVEST) reaches 40+ harvests in sim
- [ ] M2.2a eval: PPO-trained Farmer in sim reaches 30+ harvests
- [ ] M2.2b/M2.3 transfer: Real-MC eval shows >60% gate passage on first transfer attempt

---

## 10. Open Questions & Honest Caveats

### 10.1 Will PPO converge on delayed rewards?

**Question:** Does PPO's on-policy learning + GAE bootstrapping handle a 64-tick delay well?

**Honest answer:** We don't know. M1B Gatherer converged in ~200 iters (on real MC, ~1 iter ≈ 10–15 min wall, so ~40 hours total; on sim, ~100 s). Farmer might take 2–10× longer due to the delay, or it might converge faster with good shaping. The only way to know is to train.

**Mitigation:** M2.2 is a short exploratory run (100 iters in sim, ~2 minutes wall). If convergence is obvious by iter 50, proceed to M2.3 transfer. If not, switch to curriculum (Phase 1 = pre-planted crops, Phase 2 = full task) or increase n-step GAE.

### 10.2 Will the policy learn patience or passivity?

**Question:** Does the policy learn to *wait actively* (plant more cells) or *wait passively* (stand still for 64 ticks)?

**Honest answer:** The r_pbrs shaping gradient should push toward active waiting (more ripeness = higher φ = higher r_pbrs), but this is an empirical question. If skill histograms show high WAIT and low PLANT by mid-episode, the policy is being passive. Then we need to either:
- Increase weight of f_planted_count term in φ (encourage more planting)
- Add a small negative reward for WAIT (active penalty)
- Use curriculum (Phase 1) to force planting

### 10.3 What is the true transfer gap?

**Question:** How much does the sim-to-real gap hurt? Is it 5%, 20%, 50%?

**Honest answer:** We'll know after M2.3 real-MC eval. The primary risk is crop-growth variance (sim: 64 ticks always; real: 50–150 ticks). If the policy learns a fixed timer (bad), transfer is poor. If it learns robust patience (good), transfer is fine.

### 10.4 Multi-role credit assignment

**Question:** In MAPPO with shared critic, does Farmer's slow convergence degrade Gatherer's fast convergence?

**Honest answer:** Unlikely if both roles get equal update weight in the learner. But if Gatherer's gradient dominates (because it converges early and stops changing), the critic might overfit to Gatherer's reward scale. The fix (scaling farmer_potential to match gatherer_potential's magnitude) should mitigate this. Test: compare M1B Gatherer solo vs M2 Gatherer+Farmer. If M2 Gatherer's return is <90% of M1B, credit assignment is broken.

---

## 11. Handoff & Collaboration

### 11.1 To: deep-rl-training-specialist

**Task:** Tune PPO for delayed rewards in M2.2a.

**Input:** This spec (obs/action/reward contract).

**Deliverables:**
1. M2.2a config (PPO hyperparams, n-step GAE, lambda, learning rate for temporal credit assignment)
2. Convergence curve (iter vs episode_return_mean; target >30 harvests by iter 50 in sim)
3. Recommendation: proceed to M2.2b transfer, or switch to curriculum

**Known constraints:**
- Same LSTM size (256-hidden) as Gatherer (no GPU benefit from bigger model at M1B scale)
- num_learners=0 on Windows (Learner in driver process)
- num_env_runners≥2 for better batching (but not >4, returns diminish)

### 11.2 To: minecraft-rl-environment-specialist

**Task:** Extend sim & real-MC env for Farmer (M2.2 + M2.3).

**Input:** This spec (obs contract, skill dispatch, farmland arena).

**Deliverables:**
1. Sim env: crop_age array, PLOW/PLANT/HARVEST logic, obs extraction
2. Real-MC arena: 32×32 farmland, seed chest, spawn point
3. Py4J bridge: FarmerOverlayBuilder, PlowSkill, PlantSkill, HarvestSkill dispatch
4. Validation: scripted Farmer reaches 40+ harvests in sim
5. Transfer test harness (real-MC eval runner)

**Known constraints:**
- Real MC crops are random-tick (not fixed-timer sim)
- Reach radius consistent with Gatherer (4.5 blocks)
- Inventory seed tracking (PlantSkill consumes seeds)

### 11.3 To: multi-agent-rl-specialist

**Task:** Validate MAPPO/CTDE with two roles (M2 planning).

**Input:** This spec + Gatherer spec + CTDE critic architecture.

**Deliverables:**
1. Multi-role config (policy_mapping_fn for "gatherer" and "farmer")
2. Weight-sharing strategy (critic sees all agents; per-agent PBRS stays per-agent)
3. Potential magnitude calibration (farmer_potential scale = ~100 to match tech_tree_potential)
4. Convergence comparison: M1B solo Gatherer vs M2 Gatherer+Farmer (goal: >90% return parity)

**Known constraints:**
- Shared critic in CTDE_critic (no separate per-role critics)
- Potential-based shaping is policy-invariant per agent; sums are not, so magnitude matters

---

## 12. Summary of Key Design Decisions

| Decision | Rationale | M2.2 validation |
|---|---|---|
| **PLOW + PLANT separate skills** | Real MC requires two steps; policy learns the sequence | Scripted baseline reaches PLANT |
| **f_crop_grid (0–8 age)** | Policy needs per-cell ripeness; guides waiting | Policy learns to distinguish ripe vs unripe |
| **r_principal = +1.0 per harvest only** | Sparse reward avoids hacking; r_pbrs guides exploration | No reward-up/eval-flat bug; harvests are real |
| **PBRS with decay (φ → 0 by episode end)** | Forces policy to harvest as crops ripen, not farm forever | r_pbrs ≤ 0 after decay ends; policy doesn't exploit shaping |
| **Three shaping terms (planted/ripeness/timeliness)** | Addresses three learning objectives: exploration, patience, discipline | Policy learns all three behaviors (not just ripeness) |
| **Skill timeout = 400 ticks (same as Gatherer)** | Limits exploration; forces a decision | Policy doesn't waste budget on a single cell |
| **Eval gate = 32 harvests in 1000 ticks** | ~3× Gatherer's oak_log gate (more complex task) | Achievable in sim (~2 min), transferable to real MC |
| **Sim crop growth = 64 ticks fixed** | Deterministic for reproducibility; transfer will show variance | M2.3 transfer reveals sim-to-real timing gap |

---

## 13. References & Context

- **IMPLEMENTATION_PLAN.md § Milestone section:** Farmer is M3, but we design for M2 integration
- **PROJECT_CONTEXT.md § §5 Reward architecture:** Stage-1 composition (§5.1), PBRS (§5.4), potential-based shaping (§5.7), exploit detection (§5.3)
- **NEXT_SESSION.md:** M1B training state (v21 running, gate in progress)
- **Ng et al. 1999:** "Policy Invariance Under Reward Transformations" (PBRS foundation)
- **M1B Gatherer tests (§8 above):** Scenarios, eval gate, transfer validation patterns (Farmer reuses these for M2.2b/M2.3)

---

## 14. Appendix: Psuedocode Reference

### 14.1 farmer_potential (full code)

```python
def farmer_potential(obs: dict, decay_coeff: float = 1.0) -> float:
    """Φ(s) for Farmer PBRS. Returns a scalar potential in range [0, 100]."""
    import numpy as np
    
    # Term 1: Planting progress
    planted_count = obs["f_planted_count"][0]
    planted_progress = min(1.0, planted_count / 64.0)
    
    # Term 2: Ripeness progress
    ripeness = obs["f_ripeness"][0]
    ripeness_progress = float(ripeness)
    
    # Term 3: Timeliness (inverse staleness)
    crop_grid = obs["f_crop_grid"]
    time_at_ripeness = obs["f_time_at_ripeness"]
    ripe_cells = crop_grid == 8
    
    if ripe_cells.any():
        staleness = np.minimum(1.0, time_at_ripeness[ripe_cells] / 50.0)
        timeliness = np.mean(1.0 - staleness)
    else:
        timeliness = 0.0
    
    # Composite with decay
    phi_unscaled = (
        0.15 * planted_progress +
        0.50 * ripeness_progress +
        0.35 * timeliness
    ) * decay_coeff
    
    # Scale to match gatherer_potential magnitude (0–100 range)
    return 100.0 * phi_unscaled
```

### 14.2 decay_coeff_for_tick

```python
def decay_coeff_for_tick(tick: int, max_ticks: int = 1000) -> float:
    """Linear decay: 1.0 at tick 0, 0.0 at tick max_ticks."""
    return max(0.0, 1.0 - tick / max_ticks)
```

### 14.3 farmer_exploit_penalties

```python
def farmer_exploit_penalties(env_meta: dict) -> list[tuple[str, float]]:
    """Return list of (exploit_name, penalty) tuples."""
    penalties = []
    if env_meta.get("tried_to_plant_harvested_cell", False):
        penalties.append(("re_plant_same_cell", 0.1))
    if env_meta.get("tried_to_harvest_unripe", False):
        penalties.append(("harvest_unripe", 0.05))
    if env_meta.get("no_state_change_this_tick", False):
        penalties.append(("idle_tick", 0.01))
    return penalties
```

---

**End of Spec**

---

### Reviewer Notes

**Spec Status:** Ready for M2 implementation. Three open empirical questions (§10):

1. **PPO on delayed rewards:** Will converge in M2.2a? (1–2 min sim time to answer)
2. **Patience vs passivity:** Will policy learn active waiting? (Skill histograms + behavior video in M2.2a)
3. **Transfer gap:** How much does sim→real timing variance hurt? (M2.3 eval, ~5 min real-MC time)

**Blockers:** None. The spec is self-contained and aligns with AiUtopia's MAPPO/CTDE architecture. Farmer can train alongside Gatherer in M2 with no infrastructure changes beyond the checklist (§9).

**Success metrics (M2.2 + M2.3):**
- M2.2a: Farmer converges to >30 harvests in sim by iter 50
- M2.2b/M2.3: >60% real-MC gate passage on first transfer (32 harvests in 1000 ticks)
- Bonus: Policy learns active waiting (high PLANT, low idle WAIT) by mid-episode
