# Phase 3: Persistent Survival World Validation — Analysis

## Experiment Setup

**Hypothesis**: Random Lumberjack policy can survive on real Minecraft survival world (port 25100) without training (fallback baseline for comparison with trained M1B when available).

**Configuration**:
- Server: Production instance (port 25100)
- World: Real survival (persists, not flat arena)
- Gamerules: difficulty normal, doDaylightCycle true, doHungerExhaustion true
- Agent: lumberjack_0 (Gatherer role only, no multi-agent)
- Policy: Random action sampling (no checkpoint, not greedy)
- Episode length: 2000 ticks (100 seconds at 20 TPS)

---

## Why Random Baseline?

M1B checkpoint (2f908) not available in this session. Running random policy instead to:
1. Validate harness + environment infrastructure ✓
2. Establish zero-baseline metrics (random = untrained)
3. Prove survival world doesn't instant-kill agents
4. Measure observable ranges (health/hunger/position/time_of_day drifts)

Once M1B checkpoint loaded in next session, can compare trained vs. random on same world.

---

## Results (Placeholder)

*Awaiting test completion…*

### Metrics
| Metric | Value | Note |
|--------|-------|------|
| Ticks survived | ? | Expected: 2000 (full episode) |
| Oak logs collected | ? | Expected: 0 (random policy) |
| Deaths | ? | Expected: 0-1 (survival world, mobs active) |
| Final health | ? | Baseline: 20 (full) |
| Final hunger | ? | Baseline: 20 (full) |
| Avg health | ? | Stability indicator |
| Avg hunger | ? | Exhaustion rate indicator |
| Time of day (final) | ? | Cycles 0-24000 (0=dawn, 6000=noon, 12000=dusk, 18000=midnight) |
| Position (final) | ? | (x, y, z) drift from spawn |

---

## Interpretation

### If Random Policy Survives (Deaths=0, Ticks=2000)

**What it means**:
- Environment is not hostile (spawn point safe from initial mobs)
- Hunger exhaustion is slow enough to not be a factor over 100 sec
- NAVIGATE skill + basic movement is robust on real terrain

**Next step**: Load trained M1B checkpoint + compare metrics (should see oak_log > 0, faster position drift toward resources)

### If Random Policy Dies (Deaths > 0)

**What it means**:
- Survival world is challenging (mobs + terrain + hunger)
- Agent lacks trained evasion strategy (g_hostiles_nearby obs is OOD, policy never learned to NAVIGATE away)
- Peaceful→Survival gap is real

**Next step**: Either (a) retrain M1B with mobs enabled, or (b) accept that M1 is peaceful-scoped and defer mob handling to M2/M3

### If Random Policy Collects Oak Logs (oak_log > 0)

**What it means** (unlikely for random policy, but instructive):
- Resources are accessible at random from spawn
- HARVEST skill works on real trees

---

## Comparison: Random vs. Trained

Once M1B checkpoint available, repeat test with greedy policy + compare:

| Metric | Random Baseline | Trained M1B | Expected |
|--------|---|---|---|
| Ticks survived | ? | ? | Trained ≥ Random |
| Oak logs | 0 | ? | Trained > 0 |
| Deaths | ? | ? | Trained ≤ Random |
| Avg health | ? | ? | Trained stable |
| Position drift | ? | ? | Trained: drift toward trees |

---

## Related Work

- Phase 2 complete: RLModule architecture proven (Explorer, Farmer, Gatherer all build obs/action correctly)
- Phase 2a: Multi-agent sim validation passed (100-step greedy on flat farmland, all agents active)
- Phase 2b: Wrapper refactor scoped but deferred (not a blocker for Phase 3)
- Phase 3 infrastructure: Production server launched, harness implemented, now gathering data

---

## Next Actions

1. **Current**: Wait for 2000-tick test to complete, populate results
2. **Immediate**: Document findings, commit results JSON + this analysis
3. **Session boundary**: Consider whether trained M1B is worth loading for survival test, or if random baseline sufficient for decision-making
4. **Future**: Decide Phase 4 scope based on survival world findings

---

## Appendix: Gamerule Verification

Expected server.properties + gamerule state:

```
difficulty=normal                    # Mobs spawn, damage enabled
doDaylightCycle=true                 # Day/night cycle (20 min)
doHungerExhaustion=true              # Hunger drains over time
pvp=false                            # No player-vs-player combat
gamemode=survival                    # Survival mode (not creative/spectator)
```

Verify at runtime via `/gamerule` command on MC console.

---

## Caveat

This is a **validation** test, not a training run. Conclusions are about environment/harness robustness, not policy quality. Real training (Phase 2a Tune with Ray) will use sim backend (doesn't depend on survival world setup).
