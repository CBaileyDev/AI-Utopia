# Phase 2b: Java Implementation (Multi-Role Obs Dispatch)

## Status

**Python wrapper**: ✅ DONE (config defaults + role dispatch ready)
**Java Py4J entry point**: ⏳ PENDING (add 3 new methods + update observations_all)

---

## What Python Is Waiting For

The Python wrapper now calls role-specific Java methods if they exist:

```python
# In _read_all_obs():
if role == "gatherer" and hasattr(self.bridge.entry_point, "observationsGatherer"):
    raw = self.bridge.entry_point.observationsGatherer(player_name)
elif role == "explorer" and hasattr(self.bridge.entry_point, "observationsExplorer"):
    raw = self.bridge.entry_point.observationsExplorer(player_name)
elif role == "farmer" and hasattr(self.bridge.entry_point, "observationsFarmer"):
    raw = self.bridge.entry_point.observationsFarmer(player_name)
```

Python falls back to `observations_all()` if these don't exist (backward-compatible).

---

## Java Changes Required

### File: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`

#### 1. Add Three New Methods

```java
public Map<String, Object> observationsGatherer(String playerName) {
    try {
        ServerPlayer player = server.getPlayerList().getPlayer(playerName);
        if (player == null) return new HashMap<>();
        return GathererOverlayBuilder.observationAll(player);
    } catch (Exception e) {
        log.error("observationsGatherer failed: {}", e.getMessage());
        return new HashMap<>();
    }
}

public Map<String, Object> observationsExplorer(String playerName) {
    try {
        ServerPlayer player = server.getPlayerList().getPlayer(playerName);
        if (player == null) return new HashMap<>();
        return ExplorerOverlayBuilder.observationAll(player);
    } catch (Exception e) {
        log.error("observationsExplorer failed: {}", e.getMessage());
        return new HashMap<>();
    }
}

public Map<String, Object> observationsFarmer(String playerName) {
    try {
        ServerPlayer player = server.getPlayerList().getPlayer(playerName);
        if (player == null) return new HashMap<>();
        return FarmerOverlayBuilder.observationAll(player);
    } catch (Exception e) {
        log.error("observationsFarmer failed: {}", e.getMessage());
        return new HashMap<>();
    }
}
```

#### 2. Update `observationsAll()` to Be Multi-Role

Replace current (Gatherer-only) implementation:

```java
public Map<String, Map<String, Object>> observationsAll() {
    Map<String, Map<String, Object>> out = new HashMap<>();
    
    for (ServerPlayer player : server.getPlayerList().getPlayers()) {
        String playerName = player.getName().getString();
        String role = getRoleForPlayer(player);  // Query agent registry
        
        Map<String, Object> obs;
        switch (role) {
            case "explorer":
                obs = ExplorerOverlayBuilder.observationAll(player);
                break;
            case "farmer":
                obs = FarmerOverlayBuilder.observationAll(player);
                break;
            default: // "gatherer"
                obs = GathererOverlayBuilder.observationAll(player);
        }
        
        if (obs != null) {
            out.put(playerName, obs);
        }
    }
    
    return out;
}
```

**Note**: `getRoleForPlayer(player)` is assumed to exist in the agent registry. If it doesn't, add:

```java
private String getRoleForPlayer(ServerPlayer player) {
    // Query the agent registry to find this player's role
    // Existing pattern in bridge/MotorBridge or agent/AgentRegistry
    String playerName = player.getName().getString();
    // Placeholder — update with actual registry lookup
    return "gatherer"; // Default
}
```

---

## Verification Checklist

After implementing:

1. Verify classes exist:
   - ✓ `GathererOverlayBuilder` (already exists, used by current observations_all)
   - ✓ `ExplorerOverlayBuilder` (committed in Phase 2a, m2-farmer JAR)
   - ✓ `FarmerOverlayBuilder` (committed in Phase 2a, m2-farmer JAR)

2. Verify public methods exist:
   - [ ] `GathererOverlayBuilder.observationAll(player)` — if not, rename from `buildObs()` or add wrapper
   - [ ] `ExplorerOverlayBuilder.observationAll(player)` — if not, rename or add wrapper
   - [ ] `FarmerOverlayBuilder.observationAll(player)` — if not, rename or add wrapper

3. Build & test:
   ```bash
   cd fabric_mod
   export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10
   ./gradlew build   # → build/libs/aiutopia-mod-0.0.0-m2b-multiagent.jar
   ```

4. Deploy to all 4 training instances:
   ```bash
   for i in 1 2 3 4; do
     cp fabric_mod/build/libs/aiutopia-mod-*.jar server-runtime/training/instance-$i/mods/
   done
   ```

5. Restart training instances (Java processes only, world data persists)

6. Run Phase 2b validation:
   ```bash
   PYTHONPATH=src py -3.11 scripts/phase2b_realmc_transfer_test.py
   ```

---

## Timeline

- **This session**: Python wrapper ready (✓ done)
- **Next session**: Java implementation + rebuild + deploy (45 min) + validation (30 min)

---

## Backward Compatibility

The Python wrapper gracefully falls back to `observations_all()` if role-specific methods don't exist. This means:

- **Before Java update**: Uses `observations_all()`, works but all agents get Gatherer obs
- **After Java update**: Uses role-specific methods, each agent gets appropriate obs

No breaking changes. Existing code continues to work.

---

## Related Files

- `src/aiutopia/env/wrapper.py` (Python, ✓ updated)
- `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java` (Java, pending)
- `fabric_mod/src/main/java/dev/aiutopia/mod/obs/GathererOverlayBuilder.java` (exists, used)
- `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ExplorerOverlayBuilder.java` (exists, committed)
- `fabric_mod/src/main/java/dev/aiutopia/mod/obs/FarmerOverlayBuilder.java` (exists, committed)
- `scripts/phase2b_realmc_transfer_test.py` (validation harness, ready to run)

---

## Summary

Phase 2b Python wrapper is production-ready. Java changes are ~50 lines (3 new methods + 1 updated method). After Java rebuild + deploy, Phase 2b real-MC multi-agent validation can proceed.
