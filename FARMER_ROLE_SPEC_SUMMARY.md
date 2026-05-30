# Farmer Role — M2 Design Summary

**Full spec:** `docs/superpowers/specs/2026-05-30-m2-farmer-role-design.md` (1600+ lines, ready to implement)

## Quick Reference

| Aspect | Details |
|---|---|
| **Core task** | Plant seeds → wait for ripeness (64 ticks fixed, 50-150 random in real MC) → harvest ripe crops |
| **Actions** | PLOW (till soil), PLANT (place seed), HARVEST (break ripe crop), NAVIGATE, WAIT, broadcast + noop |
| **N_FARMER_SKILLS** | 7 (vs 6 for Gatherer) |
| **Obs overlay** | f_crop_grid (32×32, ages 0-8), f_ripeness, f_planted_count, f_harvested_count, f_harvested_mask, f_time_at_ripeness |
| **Sparse reward** | +1.0 per unique cell harvested (at stage 8, once per plant cycle) |
| **PBRS shaping** | Φ(s) with decay = 0.15×planted_progress + 0.50×ripeness_progress + 0.35×timeliness_progress, scaled ×100 to match Gatherer magnitude |
| **Decay schedule** | Linear: 1.0 at tick 0, 0.0 at tick max_ticks (1000 default) — forces harvest, prevents infinite farming |
| **Exploit penalties** | re_plant_same_cell (-0.1), harvest_unripe (-0.05), idle_tick (-0.01) |
| **Episode length** | 1000 ticks |
| **Eval gate (M2.3)** | 32 harvested crops in 1000 ticks, 80% success (2 of 3 eval runs) |

## Integration Points

### Python side (env wrapper + RL module)
- `spaces.py`: Add `_farmer_overlay()` dict space
- `reward.py`: `farmer_potential()`, `farmer_exploit_penalties()`, `compute_reward_stage_1_farmer()`
- `rl_module/`: FarmerRoleRLModule (LSTM encoder + actor head for 7 skills)
- `train/config.py`: m2_marl_config (MAPPO + Gatherer + Farmer)

### Java side (Fabric mod)
- `FarmerOverlayBuilder.java`: Scan farmland, report crop ages per-cell
- `PlowSkill.java`, `PlantSkill.java`: New skill executors
- `HarvestSkill.java`: Extend to check ripeness, fail silently if unripe
- `MotorBridge.java`: Dispatch all three skills

### Sim side (fast training)
- `sim_env.py`: crop_age array, PLOW/PLANT/HARVEST logic, obs extraction
- Same obs contract (f_crop_grid, f_ripeness, etc.)
- Deterministic growth (age += 1/tick) for reproducibility

### Tests (validation checkpoints)
- M2.2a: PPO converges to >30 harvests in sim by iter 50
- M2.2b/M2.3: Transfer test, >60% real-MC gate passage on first try
- Multi-role: MAPPO Gatherer+Farmer, >90% Gatherer return parity (no credit-assignment degradation)

## Temporal Credit Assignment Strategy

**Challenge:** 64-tick delay between plant and harvest. PPO must correlate early actions (tick 0-10 planting) with late rewards (tick 64-80 harvesting).

**Mitigations:**
1. **PBRS with decay** — r_pbrs provides intermediate signals (ripeness progress) that appear before harvest reward; decay forces policy to harvest as crops ripen (not farm forever)
2. **Multi-step GAE** — Policy's value function bootstraps across the delay; increase lambda (>0.99) if convergence is slow
3. **Curriculum (Phase 2.3)** — If convergence stalls: Phase 1 = pre-planted crops (harvest only), Phase 2 = full task. Deferred unless needed.

## Honest Open Questions

1. **Will PPO converge?** Unknown. Delayed rewards are harder than immediate. M2.2a (1-2 min sim run) will answer.
2. **Active vs passive waiting?** Policy should plant more cells during ripeness delay, not stand idle. Skill histograms will reveal this in M2.2a.
3. **Sim-to-real transfer?** Crop growth is fixed (64 ticks) in sim, random (50-150) in real MC. If policy learns a fixed timer, transfer fails. M2.3 (real-MC eval) will quantify the gap.

## Key Design Decisions

| Decision | Why | Test in |
|---|---|---|
| **PLOW + PLANT separate** | Real MC requires two steps; policy learns sequence | Scripted baseline |
| **Sparse r_principal only** | Avoids reward-hacking; r_pbrs guides exploration | No reward-up/eval-flat bug |
| **Three shaping terms** | Addresses exploration (planting), patience (ripeness), discipline (timeliness) | Skill histograms; behavior video |
| **Decay to zero** | Prevents infinite farming; forces harvest as crops ripen | r_pbrs goes negative after decay ends |
| **32-harvest gate** | ~3× Gatherer's oak_log gate (harder task); achievable in sim (~2 min) | M2.2a convergence curve |

## Next Steps

1. **deep-rl-training-specialist:** Tune PPO for delayed rewards (M2.2a config)
2. **minecraft-rl-environment-specialist:** Extend sim + real-MC env (PlowSkill, PlantSkill, obs extraction)
3. **multi-agent-rl-specialist:** Validate MAPPO/CTDE with two roles (magnitude scaling of potentials)
4. **Run M2.2a eval** (~2 min sim, test convergence + skill histograms)
5. **Run M2.2b/M2.3 transfer** (~5 min real-MC, validate >60% gate passage)

---

**This spec is implementation-ready.** See `docs/superpowers/specs/2026-05-30-m2-farmer-role-design.md` for the full 1600-line contract.
