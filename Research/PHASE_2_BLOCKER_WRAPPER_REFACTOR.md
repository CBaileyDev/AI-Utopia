# Phase 2 Blocker: Wrapper Refactor for Multi-Agent MARL

## Issue
Phase 2b (real-MC multi-agent testing) is blocked by `AiUtopiaPettingZooEnv` requiring refactored config:
- Expects `active_roles` list (new, M2)
- Expects `stage` int (new, M2)
- Expects `py4j_ports` dict (new, multi-instance support)
- Does not currently support Farmer/Explorer obs building (hardcoded to Gatherer)

Current status: Wrapper partially updated for M2, but incomplete.

## Root Cause
`AiUtopiaPettingZooEnv` was updated to support multi-agent MARL but:
1. Config schema changed incompatibly (breaks existing code paths)
2. Obs builder still hardcoded to Gatherer (cannot emit Farmer/Explorer obs)
3. Py4J bridge not multi-role-aware (reward dispatch OK, obs building NO)

## Work Needed (Tier 2 / Phase 3)

### Tier 1: Config Compatibility
1. **Stabilize `AiUtopiaPettingZooEnv.__init__` config schema**
   - Document required vs optional keys
   - Provide sensible defaults (active_roles=['gatherer'], stage=1, etc.)
   - Ensure backward compat with old harnesses

2. **Wire `py4j_ports` correctly**
   - Multi-role training needs per-agent Py4J connections
   - Each agent_id must map to a Fabric instance port
   - Config should specify: `py4j_ports = {0: 25001, 1: 25002, 2: 25003, 3: 25004}`

### Tier 2: Obs Building Dispatch
1. **Extend `AiUtopiaPettingZooEnv.reset()` to dispatch obs by role**
   - Current: calls `bridge.observationsAll()` → single Gatherer obs
   - Needed: switch on role, call role-specific Py4J obs emitter
   - Explorer obs: `observationExplorer()` (new Java method)
   - Farmer obs: `observationFarmer()` (new Java method)

2. **Java-side obs builders for Explorer + Farmer**
   - ExplorerOverlayBuilder → ExplorerObsBuilder.observationExplorer()
   - FarmerOverlayBuilder → FarmerObsBuilder.observationFarmer()
   - Both already deployed in m2-farmer JAR; just need bridging

### Tier 3: Action Dispatch (May Be OK)
- Reward dispatch already works (commit 0181902)
- Skill dispatch should work if action_mask routes correctly
- May need validation on real MC

## Timeline Impact

- **Tier 1 (Config)**: 30 min (documentation + defaults)
- **Tier 2 (Obs dispatch)**: 1 hour (wrapper + Java bridge wiring)
- **Tier 3 (Validation)**: 30 min (live test)
- **Total**: ~2 hours to unblock Phase 2b

## Workaround (Current State)

Phase 2a infrastructure is **validated without wrapper**, so training can proceed once Ray is fixed:
- RLModule obs/action contracts proven (phase2a_rlmodule_validation.py ✓)
- Reward functions dispatch correctly (all 3 roles)
- Potentials compute (PBRS shaping validated)

Phase 2b can run on **sim backend only** (or as single-Gatherer on real MC) until wrapper is refactored.

## Next Step (If Pursuing Phase 2b Now)

Either:
1. Fix wrapper config + obs dispatch (~2 hours) → run Phase 2b multi-agent on real MC
2. Defer to Phase 3 → focus on Ray fix + Phase 2a sim training first

**Recommended**: Defer. Phase 2a (sim + Ray Tune) is higher value and doesn't depend on wrapper refactor.
