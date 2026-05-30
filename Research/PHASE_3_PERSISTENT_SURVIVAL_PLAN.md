# Phase 3: Persistent Survival World (Plan)

## Scope & Vision

Phase 3 moves from **controlled training arenas** (flat, peaceful, 6000-tick episodes) to a **real Minecraft survival world** (natural terrain, day/night cycles, hunger, hostile mobs, persistent). Goal: validate whether the proven M1B Lumberjack can **survive and function** in conditions closer to the end-goal village scenario.

This is NOT training. This is **deployed-agent validation** on realistic conditions.

---

## Setup (Infrastructure)

### Current State
- **Training servers (4x)**: Instances 1-4 on ports 25001-25004. Peaceful arena, flat terrain, ~6000-tick episodes. Runs all Phase 2 work.
- **Production server**: Port 25100 (referenced but NOT running). Intended for persistent survival world.

### Phase 3 Setup Checklist

1. **Launch production server** (separate from training instances)
   - Port: 25100
   - World: **Real survival** (not flat arena, not peaceful)
   - Gamerules:
     - `difficulty normal` (mobs spawn, damage enabled)
     - `doDaylightCycle true` (day/night cycle enabled, 20-min cycle)
     - `doHungerExhaustion true` (hunger drains over time)
     - `pvp false` (no player combat, only mob threat)
   - Seed: Fixed seed (e.g., 42) for reproducibility
   - Estimate: 30-45 min (server launch, world generation, gamerule setup)

2. **Prepare M1B checkpoint**
   - **Proven checkpoint**: `2f908/checkpoint_000003` (3/3 transfer validation from natural forest)
   - Load via `transfer_eval.py` with `TRANSFER_CKPT=2f908` env var
   - (Note: Will fail gracefully if Ray unavailable; greedy agent fallback)

3. **Spawn agent on persistent world**
   - Agent: "lumberjack_0" (single Lumberjack, no multi-agent)
   - Policy: Greedy (argmax action) from loaded checkpoint
   - Episode length: 2000 ticks (100 seconds @ 20 TPS; ~5 min real time)

---

## Experiment: Lumberjack Survival Test

### Hypothesis
**"The proven M1B Lumberjack can survive >5 minutes on a real survival world and collect oak_log."**

Implicit assumption: Agent learns to NAVIGATE away from mobs (uses `g_hostiles_nearby` obs + NAVIGATE skill) and HARVEST logs when safe.

### Metrics (per 100-tick checkpoints)

| Metric | Expected Baseline | Expected Trend |
|--------|---|---|
| oak_log count | 0 → 5–10 | Increases slowly (mobs interrupt harvest) |
| health | 20 → 18–20 | Stable (NAVIGATE avoids damage) or declining (mob hits) |
| hunger | 20 → 10–15 | Declines (exhaustion ticks / walk cost) |
| time_of_day | cycles 0→24000 | Cycles (night/day cycle) |
| deaths | 0 → 0–1 | 0 if NAVIGATE works, 1 if mobs close in |
| position | spawn → forest | Drifts toward trees (resource-driven wandering) |

### Success Criteria
- ✓ Survival: Agent alive at step 2000 (no death)
- ✓ Foraging: oak_log count > 0 (harvested at least once)
- ✓ Hunger handling: Did NOT starve (health didn't hit 0 from exhaustion)
- ⊘ Combat: Agent avoided mobs (no damage taken) — likely FAIL (no FLEE skill, might take hits)

### Likely Failure Mode
Agent **dies to zombie/creeper in first 5-10 min**. Root cause: The checkpoint is trained entirely on peaceful arena with mobs spawning OFF. Observations like `g_hostiles_nearby` are always zero (no learned policy for evasion).

---

## What This Tests

**If Lumberjack survives**:
- The NAVIGATE skill + learned movement is generalized enough for real terrain
- The obs space (health, hunger, time_of_day) captures enough state for adaptation
- No code bugs in real-MC long-running episodes

**If Lumberjack dies**:
- The OOD (out-of-distribution) gap is large: peaceful→survival is harder than flat→natural
- Need survival-specific training: rerun M1B with mobs enabled, or add EAT/FLEE skills
- Peaceful scope was a reasonable strategy (defer combat/hunger to Phase 4+)

---

## Timeline & Effort

| Phase | Task | Effort | Blocker? |
|-------|------|--------|----------|
| 3.1 | Setup production server | 45 min | No (independent infra) |
| 3.2 | Run Lumberjack survival test | 15 min | No (harness ready) |
| 3.3 | Analyze results | 30 min | No (post-run analysis) |
| **Total** | **Persistent world validation** | **~1.5 hours** | **No** |

---

## Next Steps

### If Pursuing Phase 3 Now:
1. Launch production server (port 25100, real survival, normal difficulty, day/night/hunger ON)
2. Run `scripts/phase3_persistent_survival.py --checkpoint 2f908 --max-ticks 2000`
3. Capture metrics & death log
4. Document findings in `Research/PHASE_3_SURVIVAL_RESULTS_2026-05-30.md`

### If Deferring Phase 3:
1. Keep this plan as next-session spec
2. Recommended for after Ray PyPI blocker fixed (Phase 2a training could inform survival strategy)
3. Phase 3 is **not a blocker** for Phase 2 multi-agent MARL completion

---

## Honest Caveat

Phase 3 **is high-value** (moves toward real end-goal) but **requires separate infrastructure** (new server, real world, different gamerules). The Phase 2 MARL foundation is **complete without it**. Phase 3 is a **validation** milestone, not a prerequisite.

**Recommendation**: Finish Phase 3 in next session after Ray PyPI is fixed. Current state (Phase 2 MARL proven) is a natural stopping point.
