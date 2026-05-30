# Explorer Role — Observation / Action / Reward Contract

**Date:** 2026-05-30  
**Scope:** D1a, "find-the-forest" partial-info problem in real peaceful Minecraft  
**Status:** DESIGN SPEC (not yet implemented)

---

## Context & Assumptions

**Why the Explorer exists:**
- The Lumberjack (current Gatherer) is **perception-limited**: obs range = 16 blocks. Oak forests are sparse and often >16 blocks away.
- The Lumberjack's HARVEST skill internally chains within visible logs. Once the visible cluster is cleared, the agent is **blind** to the tail.
- **Solution:** A separate role (Explorer) that infers forest direction from partial observation, then emits a **bearing** that the Lumberjack can follow.

**Design philosophy:**
- Explorer is a **producer**: reads the 16-block window and outputs a compass direction.
- Lumberjack is a **consumer**: interprets the bearing as "which way to navigate?"
- This is the **thin reactive controller + smart producers** pattern (N24 audit, fork A phase 2).
- Test: **real-MC transfer**. Forest found faster than greedy spiral.

**Known constraints from N21–N24 audits:**
- **Flat toy arena has no exploitable signal**: the old `clusters` arena (B always south) made metrics untrustworthy. **This contract only makes sense in structure-rich environments** (real MC). Test will be real-MC transfer.
- **Perception mask is load-bearing**: N23 oracle-follower test showed a scripted bearing tied the "trained" policy. **Hard work = producing bearing from partial info, NOT learning to navigate.**
- **Obs format is flat/scalar**: `g_resource_grid` arrives as flat `(6144,)`, not `(32, 32, 6)`.

---

## 1. Observation Contract

### Core signals (from `spaces.py:_core_space()`)

| Signal | Shape | Semantics |
|--------|-------|-----------|
| `position` | (3,) | Origin for 16-block window |
| `yaw_pitch` | (2,) | Agent facing direction |
| `biome_id` | scalar | Current biome (forests = 25–35) |
| `light_level` | (1,) | Sky light (forests shadier) |
| `time_of_day` | (1,) | Optional: dusk/dawn search timing |

### Spatial signals (Gatherer overlay, repurposed)

| Signal | Shape | Semantics |
|--------|-------|-----------|
| `g_resource_grid` | flat (6144,) | 32×32 grid (C-order x,z,ch). Reshape to (32,32,6) to index. |
| `g_nearest_resources` | (8, 6) | Top 8 logs: [dx/16, dy/8, dz/16, ch0, ch1, ch2] |
| `g_richness_score` | scalar | min(1, count_in_16b_radius / 64) |

### Proposed extension (if Phase 2a insufficient)

```
g_biome_grid: Box(0, 64, (32, 32), np.int32)  # biome_id per (x,z) cell
```

**Java PR:** Extend `GathererOverlayBuilder`. Est. ~30 min. Propose if richness alone is insufficient.

---

## 2. Action Contract

### Discrete 8-way bearing (recommended for M2)

```
target_class: Discrete(8)  # Reinterpreted as compass direction
  0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
```

**Reuses existing `target_class` action channel.** No RLModule changes.

**Phase 3 alternative:** Continuous `spatial_param` Box(-1, 1, (3,)) for refinement.

---

## 3. Reward Contract

### Primary: Sparse discovery

```
r_discovery = 1.0  when g_richness_score >= threshold (8 logs = 0.125)
            = 0.0  all other steps
```

### Optional shaping (with decay)

```
r_progress(t) = +0.05 if richness improved, −0.02 if worsened
                × (1 − t / max_steps)  # decay to 0 by episode end
r_coverage(t) = +0.01 for new (x,z) column visited
                × (1 − t / max_steps)
r_stuck      = −0.1 if no movement >2 blocks in 10 steps
r_timeout    = −1.0 at episode termination (max_steps, no discovery)
```

### Anti-hacking constraints

1. **No standing-still reward.** Only reward if agent moved into forest, not just adjacent.
2. **Richness progress:** Emit only if Δ > 0.05 (avoid noise from local grid shifts).
3. **Coverage decays.** Prevents agent circling endlessly to farm bonus.

---

## 4. Integration: Separate MARL Agent (Recommended M2)

Explorer is a **separate policy with its own RLModule**. Trained jointly with Lumberjack via **MAPPO**.

```
Env state → [Explorer policy → bearing] ⊕ [Lumberjack policy → action]
  |                                        ↓
  |          bearing feeds NAVIGATE skill
  └────────────────────────────────────────┘
```

**Phase 3:** Freeze Explorer, BC-pretrain, swap to sub-skill for deployment efficiency.

---

## 5. Test Proposal

### Sim Phase 2a: 30 seeds, 100 episodes

**World:** Procedurally generated (biome-provider) with 5–10 scattered oak forests.

**Metric:** Steps-to-first-oak-log discovery.

**Baselines:**
1. **Greedy spiral:** Fixed 8-direction rotation (open-loop).
2. **Scripted greedy:** If richness > 0, navigate nearest log; else spiral.
3. **Trained Explorer:** MAPPO-trained discrete 8-way policy.

**Success:** Trained ≤ spiral (beats pure coverage); Trained ≥ scripted (policy adds value).

**Output:** Percentile steps-to-discovery across 30 seeds.

### Real-MC Phase 2b: 10 episodes

**World:** Peaceful 128×128 arena with 3–5 natural oak forests.

**Metric:** Time-to-first-oak-log harvest (capped 5 min).

**Gate:** ≥7 episodes reach discovery (70% success).

**Discriminator:** If slow, visualize Explorer's bearing actions over trajectory. Sensible directions?

### Known risks

1. **Procedural world must be non-degenerate.** Use biome-provider, test 5+ independent seeds.
2. **Perception mask is load-bearing.** Run oracle-follower test (N23 pattern): freeze Explorer, replace with scripted bearing. Does Lumberjack's success change? If not, mask is doing the work.
3. **Skill chaining hides policy failures.** During Phase 2b, limit `HarvestSkill.MAX_SEARCH_RADIUS` to force genuine navigation.

---

## 6. Known Unknowns & Decisions

| Item | Status | Owner |
|------|--------|-------|
| Is richness sufficient, or need biome_grid? | Test Phase 2a | User / Phase 2 training |
| Discrete 8-way or continuous bearing? | **M2: Discrete. Phase 3: Continuous.** | User design |
| Separate policy or sub-skill? | **M2: Separate (MAPPO). Phase 3: Freeze→sub-skill.** | User design |
| What triggers Explorer→Lumberjack mode switch? | Deferred to M3 (LLM planner integration) | Agent-behavior-architecture-specialist |
| Real-MC biome_grid in obs? | Propose if Phase 2a insufficient | minecraft-modding-and-server-specialist |

---

## 7. Deliverables Checklist (D1a → Phase 2 handoff)

- [x] **Obs schema:** position, yaw_pitch, biome_id, light_level, time_of_day + Gatherer overlay (g_resource_grid flat, g_nearest_resources, g_richness_score). Proposed: g_biome_grid.

- [x] **Action schema:** Discrete 8-way bearing via reinterpreted target_class.

- [x] **Reward formula + constraint:** Sparse +1.0 discovery; optional decayed progress/coverage; penalties (stuck, timeout). Anti-hacking: no standing-still, threshold-crossing, decay.

- [x] **Integration:** Separate MARL agent, MAPPO with Lumberjack.

- [x] **Test proposal:** Sim baselines (spiral, scripted, trained) on 30 seeds. Real-MC 10 episodes, 70% discovery gate + bearing discriminator.

---

**Prepared for:** User design review and Phase 2 implementation handoff to `deep-rl-training-specialist` (MAPPO pipeline) and `minecraft-rl-environment-specialist` (real-MC transfer testing).
