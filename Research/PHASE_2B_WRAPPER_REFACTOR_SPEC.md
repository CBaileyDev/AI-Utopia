# Phase 2b Wrapper Refactor Spec (Detailed)

## Problem

Real-MC multi-agent testing (Phase 2b) is blocked because:
1. `AiUtopiaPettingZooEnv.__init__()` expects config keys with no defaults: `active_roles`, `stage`, `py4j_ports`
2. `_read_all_obs()` calls `self.bridge.observations_all()` which returns only Gatherer obs
3. Java-side Py4J entry point has no role-specific obs methods (ExplorerObservationBuilder, FarmerObservationBuilder not exposed)

## Solution Layers

### Layer 1: Python Wrapper (1 hour)

**File: `src/aiutopia/env/wrapper.py`**

#### 1.1 Fix config defaults

```python
def __init__(self, config: dict[str, Any]):
    self.cfg = config
    # Add defaults for M2 multi-role setup
    self.active_roles = list(config.get("active_roles", ["gatherer"]))  # Default: solo Gatherer
    self.agents_init = [f"{r}_0" for r in self.active_roles]
    self.possible_agents = list(self.agents_init)
    self.agents: list[str] = []
    self.stage = int(config.get("stage", 1))  # Default: stage 1
    # ... rest of __init__ ...
    
    # Py4J ports: must be provided OR default to production port
    ports = config.get("py4j_ports", [25100])  # Single production port
    widx = int(getattr(config, "worker_index", config.get("worker_index", 0))) % len(ports)
    self.bridge = FabricBridge(port=ports[widx])
```

#### 1.2 Refactor `_read_all_obs()` with role dispatch

```python
def _read_all_obs(self) -> dict[str, dict]:
    """Read observations for all active agents, dispatching by role."""
    out: dict[str, dict] = {}
    for agent in self.agents:
        role = _role_of(agent)
        player_name = self.agent_id_to_player_name.get(agent, agent)
        
        # Dispatch to role-specific Java obs builder
        if role == "gatherer":
            raw = self.bridge.observations_gatherer(player_name)
        elif role == "explorer":
            raw = self.bridge.observations_explorer(player_name)
        elif role == "farmer":
            raw = self.bridge.observations_farmer(player_name)
        else:
            raw = {}  # Unknown role, empty obs
        
        raw = _normalize_raw(raw)
        
        # Compute action mask (Gatherer only; others empty)
        mask = compute_gatherer_action_mask(raw) if role == "gatherer" else {}
        
        # Decode obs with role-specific schema
        out[agent] = _decode_obs(raw, role, self.stage, mask, self._stub_goal_embed)
    
    return out
```

### Layer 2: Java Py4J Entry Point (45 min)

**File: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`**

Add three new methods (delegates to role-specific builders):

```java
public Map<String, Object> observationsGatherer(String playerName) {
    // Return obs as built by GathererOverlayBuilder
    // Existing path: GathererOverlayBuilder.observationAll(player)
}

public Map<String, Object> observationsExplorer(String playerName) {
    // Return obs as built by ExplorerOverlayBuilder (committed but not exposed)
    // New call: ExplorerOverlayBuilder.observationAll(player)
}

public Map<String, Object> observationsFarmer(String playerName) {
    // Return obs as built by FarmerOverlayBuilder (committed but not exposed)
    // New call: FarmerOverlayBuilder.observationAll(player)
}
```

**Also extend `observations_all()` to multi-role:**

```java
public Map<String, Map<String, Object>> observationsAll() {
    Map<String, Map<String, Object>> out = new HashMap<>();
    for (ServerPlayer player : server.getPlayers()) {
        String role = getRoleForPlayer(player);  // Query agent registry
        Map<String, Object> obs;
        if ("explorer".equals(role)) {
            obs = ExplorerOverlayBuilder.observationAll(player);
        } else if ("farmer".equals(role)) {
            obs = FarmerOverlayBuilder.observationAll(player);
        } else {
            obs = GathererOverlayBuilder.observationAll(player);  // Default
        }
        out.put(player.getName(), obs);
    }
    return out;
}
```

### Layer 3: Obs Builders (Already Done)

ExplorerOverlayBuilder and FarmerOverlayBuilder are **already implemented** (m2-farmer JAR). Just need to:
- Verify they have `observationAll(player)` method (not just `buildObs()`)
- Wire them into Py4J entry point

If missing the public method, add:
```java
public static Map<String, Object> observationAll(ServerPlayer player) {
    // Call buildExplorerObs(player) or similar, return as dict
}
```

---

## Implementation Checklist

### Phase 2b.1: Python Wrapper (1 hour)

- [ ] Add config defaults to `__init__()` (active_roles, stage, py4j_ports)
- [ ] Refactor `_read_all_obs()` with role dispatch
- [ ] Add role checks to action_mask computation (Gatherer only)
- [ ] Test wrapper instantiation with config.get() defaults
- [ ] Commit: "refactor(wrapper): multi-role obs dispatch + config defaults"

### Phase 2b.2: Java Py4J Entry Point (45 min)

- [ ] Add `observationsGatherer(playerName)` method
- [ ] Add `observationsExplorer(playerName)` method
- [ ] Add `observationsFarmer(playerName)` method
- [ ] Extend `observationsAll()` with role-based dispatch
- [ ] Verify ExplorerOverlayBuilder + FarmerOverlayBuilder have `observationAll()` public method
- [ ] Rebuild JAR: `cd fabric_mod && ./gradlew build`
- [ ] Deploy to all 4 instances
- [ ] Commit: "feat(fabric-mod): multi-role Py4J obs methods"

### Phase 2b.3: Validation (30 min)

- [ ] Run Phase 2b real-MC greedy test on 3-agent (gather,explore,farm)
- [ ] Verify all 3 agents return valid obs dicts
- [ ] Verify rewards dispatch correctly per role
- [ ] Commit: "test(phase2b): multi-agent real-MC validation"

---

## Timeline & Effort

| Component | Effort | Blocker? | Note |
|-----------|--------|----------|------|
| Python wrapper | 1 hour | No | Straightforward refactor |
| Java entry point | 45 min | No | Delegates to existing builders |
| Obs builder integration | 15 min | No | If methods already exist |
| JAR rebuild + deploy | 15 min | No | Familiar process (m2-farmer) |
| Validation test | 30 min | No | Existing harness |
| **Total** | **~2.5 hours** | **No** | Parallelizable (Python + Java) |

---

## Honest Assessment

This is **well-scoped and implementable** but requires both Python and Java work. **NOT a blocker** for Phase 2a or Phase 3 because:
- Phase 2a sim training works with sim env (doesn't use real wrapper)
- Phase 3 persistent world can run single-agent (only needs Lumberjack)
- Phase 2b real-MC multi-agent is a **validation milestone**, not a prerequisite

**Recommendation**: Tackle after Phase 3 persistent survival test (gives real-world validation data that informs whether Phase 2b is even necessary). Or punt to Phase 4 if Phase 3 results suggest focus should be elsewhere.

---

## Related Files

- `src/aiutopia/env/wrapper.py` — Python wrapper (edit `__init__`, `_read_all_obs`)
- `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java` — Java entry point (add 3 methods)
- `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ExplorerOverlayBuilder.java` — Explorer obs (already exists)
- `fabric_mod/src/main/java/dev/aiutopia/mod/obs/FarmerOverlayBuilder.java` — Farmer obs (already exists)
- `scripts/phase2b_realmc_transfer_test.py` — Validation harness (ready to run)
