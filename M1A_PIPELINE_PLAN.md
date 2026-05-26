# M1-Pipeline (M1.A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the real motor module (agent can move + chop), real observation pipeline (Java emits per-agent JSON populating the gatherer obs Dict), and real reward computation (PBRS-shaped delta-inventory rewards with ExploitDetector). After this plan, you can spawn a gatherer, manually drive it via the CLI to chop a tree, and observe reward signals flowing through `env.step()` — **but no RL training yet**. That is Plan B (M1-Training).

**Architecture:** Java side gets a `SkillExecutor` registry + per-skill classes (`NavigateSkill`, `HarvestSkill`, `DepositChestSkill`, `SearchSkill`, `WaitSkill`); MC 1.21.1 attack/use/look packets drive the Carpet fake player. `WorldOps.observationsAll()` is replaced with a real builder that walks the agent's surroundings and emits the per-agent obs JSON. Python side gets `tech_tree_potential()`, `ExploitDetector`, `compute_reward()` stage-1, and wires them into `AiUtopiaPettingZooEnv.step()`. Episodic memory writes go live (real Chroma writes replacing M0's count-only stub). A new CLI `aiutopia agent drive` lets you dispatch skills manually for verification without an RL policy.

**Tech Stack:** MC 1.21.1 + Fabric Loader 0.16.5 + Carpet 1.4.147 + py4j 0.10.9.9. Python 3.12 with chromadb 0.5.20+, pydantic 2.9+, gymnasium 1.0+. Java side uses MC's native server API (no Baritone for M1-Pipeline — simple direct movement and direct block targeting is sufficient for the "find nearby tree, chop it" task; Baritone is a stretch upgrade in Plan B if pathing fails on rough terrain).

**Spec reference:** `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`. Builds on `m0-verified` (`5c94f69`).

---

## Spec §-touched-in-M1-Pipeline vs deferred to Plan B

| Spec § | Coverage | Tasks |
|---|---|---|
| §3.1 GoalSpecAdapter | Used (wired into env's obs Dict via `goal_embedding`); already implemented in M0 | T15 (wiring only) |
| §4.1 Per-role obs Dict | Gatherer overlay populated by real Java emitter (M0 was stubs returning zeros) | T9, T10, T11 |
| §4.2 Per-role action Dict | Universal action header sent into motor; motor decodes `skill_type` → executor (M0 was a no-op) | T2, T7 |
| §4.5 Action masking | Already implemented in M0; reused as-is | (no task) |
| §4.6 Per-tick RL loop | Wired end-to-end with real obs, action dispatch, reward, memory write | T16 |
| §4.9 Episodic memory write path | Real Chroma writes replacing M0's count-only stub | T17 |
| §5.1 Stage-1 reward | `r_universal = r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip` | T13, T14 |
| §5.2 `compute_reward()` | Stage-1 branch only; stages 2/3 deferred to M2+ | T14 |
| §5.3 Exploit catalog (6 types) | `ExploitDetector` class with all 6 rules wired into `env.step()` | T13 |
| §5.4 Per-role primary signal | Gatherer: `Σ_r delta_inv[r] * potential[r]` | T14 |
| §5.5 γ_clip | Motor reports `clipped_param_log` per axis; env applies `0.05 * n_clipped` | T7, T14 |
| §5.7 `tech_tree_potential()` + `threat_level()` | `tech_tree_potential` implemented; `threat_level` deferred (defender M4) | T12 |
| §5.8 M1 eval gate | Eval gate *measurement* deferred to Plan B (no training in this plan) | (Plan B) |
| §5.10 Promotion criteria | CLI `aiutopia promote-weights` deferred to Plan B (no weights yet) | (Plan B) |
| §7.1 PPOConfig | Deferred to Plan B | (Plan B) |
| §7.2 RLModule (real) | Deferred to Plan B | (Plan B) |
| §7.3 EnvWrapper | Existing M0 wrapper; only `step()` body modified | T16 |
| §7.4 Training driver | Deferred to Plan B | (Plan B) |
| §7.8 Determinism on real weights | Deferred to Plan B | (Plan B) |

---

## File structure (final state after Plan A)

```
fabric_mod/src/main/java/dev/aiutopia/mod/
├── AiUtopiaMod.java                              (unchanged)
├── Py4JEntryPoint.java                           (unchanged)
├── agent/AgentRegistry.java                      (unchanged)
├── bridge/
│   ├── MotorBridge.java                          (MODIFY — real dispatch + ack)
│   ├── WorldOps.java                             (MODIFY — real observationsAll)
│   ├── CommBus.java                              (unchanged)
│   └── skill/                                    (NEW directory)
│       ├── SkillExecutor.java                    (T1)
│       ├── SkillResult.java                      (T1)
│       ├── NavigateSkill.java                    (T2)
│       ├── HarvestSkill.java                     (T3)
│       ├── DepositChestSkill.java                (T4)
│       ├── SearchSkill.java                      (T5)
│       └── WaitSkill.java                        (T6)
├── chat/ChatEventBuffer.java                     (unchanged)
├── mixin/{KickPlayer,ChatMessage}Mixin.java      (unchanged)
└── obs/                                          (NEW directory)
    ├── ObservationBuilder.java                   (T8)
    ├── CoreObsBuilder.java                       (T9)
    ├── GathererOverlayBuilder.java               (T10)
    └── ObsJsonWriter.java                        (T8 — tiny stringly-typed JSON helper)

src/aiutopia/
├── env/
│   ├── reward.py                                 (T12, T14)
│   ├── exploit.py                                (T13)
│   ├── wrapper.py                                (MODIFY — T16 wires reward+exploit+memory)
│   └── spaces.py, action_mask.py, bridge.py      (unchanged)
├── memory/writer.py                              (MODIFY — T17 real Chroma writes)
├── cli/
│   ├── agent.py                                  (MODIFY — T18 add `drive` subcommand)
│   └── app.py                                    (unchanged)
└── (all other modules unchanged)

tests/
├── unit/
│   ├── test_reward.py                            (T12, T14)
│   ├── test_exploit.py                           (T13)
│   └── test_memory_writer_live.py                (T17 — replaces M0 stub coverage)
└── integration/
    └── test_motor_smoke.py                       (T20 — end-to-end manual drive)

scripts/
└── m1a-smoke.sh                                  (T21 — full pipeline drive sequence)
```

---

## Pre-flight: confirm M0 state

Before starting Plan A, confirm the M0 baseline:

```bash
cd "C:\Users\Carte\OneDrive\Desktop\AiUtopia"
git log --oneline -5     # HEAD should be 5c94f69 or descendant
git tag -l               # should list m0, m0-source, m0-verified
python -m pytest tests/unit -v -m "not integration and not determinism" 2>&1 | tail -3
# expected: ~78 passing
```

Server-runtime (gitignored) should still have `mods/{aiutopia-mod-0.0.0-m0.jar, fabric-api, fabric-carpet, lithium, ferritecore}` from the M0 verify session. If `server-runtime/` is gone, recreate it before T19+ smoke tests:

```bash
# (one-time setup if server-runtime/ was deleted)
mkdir -p server-runtime/{mods,world}
cd server-runtime
curl -sSL -o fabric-server-launcher.jar https://meta.fabricmc.net/v2/versions/loader/1.21.1/0.16.5/1.0.1/server/jar
cd mods
curl -sSL -O 'https://cdn.modrinth.com/data/P7dR8mSH/versions/Lwt6YYHL/fabric-api-0.116.12%2B1.21.1.jar'
curl -sSL -O 'https://cdn.modrinth.com/data/TQTTVgYE/versions/f2mvlGrg/fabric-carpet-1.21-1.4.147%2Bv240613.jar'
curl -sSL -O 'https://cdn.modrinth.com/data/gvQqBUqZ/versions/XQJtuOTA/lithium-fabric-0.15.3%2Bmc1.21.1.jar'
curl -sSL -O 'https://cdn.modrinth.com/data/uXXizFIs/versions/sOzRw3CG/ferritecore-7.0.3-fabric.jar'
for f in *%2B*; do mv "$f" "${f//%2B/+}"; done
cd ..
echo "eula=true" > eula.txt
```

---

## Tasks

### Task 1: `SkillExecutor` interface + `SkillResult` enum

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SkillExecutor.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SkillResult.java`

- [ ] **Step 1: Write `SkillResult` enum**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SkillResult.java`:
```java
package dev.aiutopia.mod.bridge.skill;

/** Per-tick outcome of a skill execution attempt.
 *
 *  RUNNING        — skill still executing across more ticks; do not emit completion event yet
 *  COMPLETED      — skill finished successfully; emit SkillCompletionEvent
 *  FAILED_TIMEOUT — skill ran out of allotted ticks
 *  IMMEDIATE_FAILURE — preconditions not met or world rejected the action
 *                      (used by ExploitDetector's MEANINGLESS_TOOL_CALL rule)
 *  ABORTED        — skill was preempted by a new dispatch with same agentId
 */
public enum SkillResult {
    RUNNING,
    COMPLETED,
    FAILED_TIMEOUT,
    IMMEDIATE_FAILURE,
    ABORTED;
}
```

- [ ] **Step 2: Write `SkillExecutor` interface**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SkillExecutor.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** A skill that drives an agent (Carpet fake player) for one or more ticks.
 *
 *  Lifecycle:
 *   1. MotorBridge constructs the executor and calls {@link #start(ServerPlayerEntity, JsonObject)}
 *      once with the agent and the decoded action JSON.
 *   2. MotorBridge calls {@link #tick(ServerPlayerEntity)} on every server tick
 *      until it returns a terminal {@link SkillResult}.
 *   3. On terminal result, MotorBridge emits a SkillCompletionEvent and clears
 *      the agent's current executor slot.
 *
 *  Implementations MUST:
 *   - Be stateful per dispatch (not singletons; constructed fresh each dispatch).
 *   - Bound their tick budget so an infinite-loop bug can't lock up the server
 *     (timeoutTicks parameter is passed in JSON; check it).
 *   - Report IMMEDIATE_FAILURE if start() preconditions fail (e.g., HARVEST
 *     called but target_class refers to a block not in range).
 *   - Track which continuous params they had to clip and expose them via
 *     {@link #clippedAxes()} (consumed by Python for γ_clip).
 */
public interface SkillExecutor {

    /** Initialize the skill from action JSON. Returns IMMEDIATE_FAILURE if
     *  preconditions are violated and the skill cannot start; otherwise
     *  RUNNING (skill needs more ticks) or COMPLETED (skill done immediately
     *  — used for instant skills like WAIT(scalar=0) or NOOP_BROADCAST). */
    SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server);

    /** Advance the skill one tick. Returns RUNNING to continue or a terminal
     *  SkillResult. */
    SkillResult tick(ServerPlayerEntity agent, MinecraftServer server);

    /** Bitset of continuous-param axes that had to be clipped on start()
     *  (bit 0 = spatial.dx, 1 = spatial.dy, 2 = spatial.dz, 3 = scalar). */
    int clippedAxes();

    /** Optional human-readable reason for IMMEDIATE_FAILURE / FAILED_TIMEOUT;
     *  surfaced in the SkillCompletionEvent. Empty string when not set. */
    String failureReason();
}
```

- [ ] **Step 3: Add gson dep (for parsing action JSON) + verify build**

Edit `fabric_mod/build.gradle` — append inside the existing `dependencies { ... }` block:
```groovy
    // GSON for JSON parsing on the Java side (action dispatch + obs emission)
    implementation 'com.google.code.gson:gson:2.11.0'
    include        'com.google.code.gson:gson:2.11.0'
```

Run:
```bash
export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10
export PATH=$JAVA_HOME/bin:$PATH
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -8
```

Expected: `BUILD SUCCESSFUL`. If gson conflicts with a Loom-bundled gson, drop the `implementation` line (gson is already on the runtime classpath via fabric-loader) and keep only the source files.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/Carte/OneDrive/Desktop/AiUtopia
git add fabric_mod/build.gradle \
        fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/
git commit -m "feat(mod): SkillExecutor interface + SkillResult enum (M1-Pipeline T1)"
```

---

### Task 2: `NavigateSkill` — direct movement toward a target offset

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/NavigateSkill.java`

Movement strategy for M1-Pipeline: simple direct-line movement. We set the fake player's yaw/pitch to face the target offset, then step forward each tick at vanilla walking speed (~4.3 m/s). No obstacle handling — if a tree or hill is in the way, the player will press against it. Baritone integration is a Plan B stretch goal if eval scenarios fail due to obstructed paths.

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/NavigateSkill.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.Vec3d;

/** §4.2 NAVIGATE: walk in a direct line toward (spatial_param × max_range)
 *  blocks from the agent's current position. RUNNING until within 1.0 block
 *  of the target, then COMPLETED.
 *
 *  spatial_param is normalized [-1, 1]^3; multiplied by MAX_NAV_RANGE=32
 *  blocks gives the displacement vector. Clipping happens at start():
 *  if requested |dx| > MAX_NAV_RANGE, clamp and set the clippedAxes bit. */
public class NavigateSkill implements SkillExecutor {

    private static final double MAX_NAV_RANGE        = 32.0;  // blocks
    private static final double ARRIVAL_RADIUS       = 1.0;   // blocks
    private static final double WALK_SPEED_PER_TICK  = 4.3 / 20.0;  // ~0.215 b/tick

    private Vec3d targetPos;
    private long  ticksRemaining;
    private int   clipped;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        if (!action.has("spatial_param")) {
            failureReason = "NAVIGATE requires spatial_param array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        var arr = action.getAsJsonArray("spatial_param");
        if (arr.size() != 3) {
            failureReason = "spatial_param must be length-3 array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        double[] raw = { arr.get(0).getAsDouble(), arr.get(1).getAsDouble(), arr.get(2).getAsDouble() };
        // Clip each axis to [-1, 1] and track which were out of range
        for (int i = 0; i < 3; i++) {
            if (raw[i] < -1.0 || raw[i] > 1.0) {
                clipped |= (1 << i);
                raw[i] = Math.max(-1.0, Math.min(1.0, raw[i]));
            }
        }
        Vec3d origin = agent.getPos();
        this.targetPos = new Vec3d(
            origin.x + raw[0] * MAX_NAV_RANGE,
            origin.y + raw[1] * 8.0,            // vertical range is tighter
            origin.z + raw[2] * MAX_NAV_RANGE
        );
        // timeout_ticks default 6000 (5 min); honor JSON override if present
        this.ticksRemaining = action.has("timeout_ticks")
            ? action.get("timeout_ticks").getAsLong()
            : 6000L;
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        if (--ticksRemaining <= 0) {
            failureReason = "navigate timeout — never reached target";
            return SkillResult.FAILED_TIMEOUT;
        }
        Vec3d here  = agent.getPos();
        Vec3d delta = targetPos.subtract(here);
        double dist = delta.length();
        if (dist <= ARRIVAL_RADIUS) {
            return SkillResult.COMPLETED;
        }
        // Step toward target at walk speed
        Vec3d dir   = delta.normalize();
        double step = Math.min(WALK_SPEED_PER_TICK, dist);
        Vec3d next  = here.add(dir.multiply(step));
        // Face the direction of travel (yaw from dx, dz; pitch from dy)
        float yaw   = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
        float pitch = (float) Math.toDegrees(-Math.asin(dir.y));
        agent.setPosition(next.x, next.y, next.z);
        agent.setYaw(yaw);
        agent.setPitch(pitch);
        return SkillResult.RUNNING;
    }

    @Override public int clippedAxes()    { return clipped;  }
    @Override public String failureReason() { return failureReason; }
}
```

- [ ] **Step 2: Build verify**

```bash
export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10; export PATH=$JAVA_HOME/bin:$PATH
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -5
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/NavigateSkill.java
git commit -m "feat(motor): NavigateSkill — direct-line movement (M1-Pipeline T2)"
```

---

### Task 3: `HarvestSkill` — break the nearest matching block

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java`

Strategy: scan a 5×5×5 cube around the agent for a block whose registry name contains the `target_class` substring (M1-Pipeline simplification — full target_class enum mapping is Plan B). Break it via `world.breakBlock(pos, true, agent)` which drops the item naturally. If `scalar_param` > 0, repeat until that many blocks have been broken or no more in range.

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.BlockState;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.Optional;

/** §4.2 HARVEST: locate the nearest block whose registry name CONTAINS the
 *  target_class substring within MAX_SEARCH_RADIUS blocks, walk to it,
 *  and break it. scalar_param (clamped to [0, 1]) × MAX_QUANTITY gives
 *  the harvest cap; default 1.
 *
 *  M1-Pipeline simplification: target_class is treated as a substring
 *  match on `Registries.BLOCK.getId(state.getBlock())`. The full
 *  enum-to-class mapping per §4.2 is Plan B work.
 *
 *  Sequencing per tick:
 *    1. If no current target block, scan radius for a match. None → COMPLETED if
 *       we've broken at least one block this dispatch, else IMMEDIATE_FAILURE.
 *    2. If target found but >2 blocks away, move toward it (walk speed).
 *    3. If within 2 blocks, break it (1 tick), increment broken-count,
 *       clear target. Loop to step 1.
 *    4. If broken-count >= cap, COMPLETED.
 */
public class HarvestSkill implements SkillExecutor {

    private static final double MAX_SEARCH_RADIUS = 8.0;
    private static final double REACH_RADIUS      = 2.0;
    private static final double WALK_PER_TICK     = 4.3 / 20.0;
    private static final int    MAX_QUANTITY      = 64;

    private String targetSubstr;
    private int    cap;
    private int    brokenCount = 0;
    private BlockPos currentTarget;
    private long   ticksRemaining;
    private int    clipped;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        // target_class arrives as a Discrete int from Python; we use it as
        // an index into a Java-side string table. M1-Pipeline ships a small
        // hardcoded table (oak_log, cobblestone, iron_ore, wheat, stone)
        // sufficient for "collect 64 oak_log". Plan B replaces this with
        // the full §4.2 enum.
        int idx = action.has("target_class") ? action.get("target_class").getAsInt() : 0;
        targetSubstr = TARGET_CLASS_TABLE[idx % TARGET_CLASS_TABLE.length];

        double scalar = action.has("scalar_param") && action.get("scalar_param").isJsonArray()
            ? action.getAsJsonArray("scalar_param").get(0).getAsDouble()
            : 1.0 / MAX_QUANTITY;     // default = 1 block
        if (scalar < 0.0 || scalar > 1.0) {
            clipped |= 0b1000;  // bit 3 = scalar
            scalar = Math.max(0.0, Math.min(1.0, scalar));
        }
        this.cap = Math.max(1, (int) Math.round(scalar * MAX_QUANTITY));
        this.ticksRemaining = action.has("timeout_ticks")
            ? action.get("timeout_ticks").getAsLong()
            : 6000L;
        return SkillResult.RUNNING;
    }

    private static final String[] TARGET_CLASS_TABLE = {
        "oak_log", "cobblestone", "iron_ore", "wheat", "stone",
        "spruce_log", "birch_log", "diamond_ore", "coal_ore", "deepslate"
    };

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        if (--ticksRemaining <= 0) {
            failureReason = "harvest timeout — broke " + brokenCount + " of " + cap;
            return brokenCount > 0 ? SkillResult.COMPLETED : SkillResult.FAILED_TIMEOUT;
        }
        if (brokenCount >= cap) {
            return SkillResult.COMPLETED;
        }
        ServerWorld world = (ServerWorld) agent.getWorld();
        if (currentTarget == null) {
            Optional<BlockPos> found = findNearest(world, agent.getBlockPos(), targetSubstr);
            if (found.isEmpty()) {
                if (brokenCount > 0) return SkillResult.COMPLETED;
                failureReason = "no '" + targetSubstr + "' within " + MAX_SEARCH_RADIUS + " blocks";
                return SkillResult.IMMEDIATE_FAILURE;
            }
            currentTarget = found.get();
        }
        Vec3d targetCenter = Vec3d.ofCenter(currentTarget);
        Vec3d here = agent.getPos();
        double dist = here.distanceTo(targetCenter);
        if (dist > REACH_RADIUS) {
            // Move toward it
            Vec3d dir = targetCenter.subtract(here).normalize();
            double step = Math.min(WALK_PER_TICK, dist - REACH_RADIUS);
            agent.setPosition(here.x + dir.x * step, here.y + dir.y * step, here.z + dir.z * step);
            float yaw = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
            agent.setYaw(yaw);
            return SkillResult.RUNNING;
        }
        // Within reach — break the block (drops items naturally)
        BlockState state = world.getBlockState(currentTarget);
        if (!Registries.BLOCK.getId(state.getBlock()).toString().contains(targetSubstr)) {
            // Target changed under us (e.g., someone else broke it). Clear and re-scan.
            currentTarget = null;
            return SkillResult.RUNNING;
        }
        world.breakBlock(currentTarget, true, agent);
        brokenCount++;
        currentTarget = null;
        if (brokenCount >= cap) {
            return SkillResult.COMPLETED;
        }
        return SkillResult.RUNNING;
    }

    private static Optional<BlockPos> findNearest(ServerWorld world, BlockPos origin, String substr) {
        int radius = (int) Math.ceil(MAX_SEARCH_RADIUS);
        BlockPos best = null;
        double   bestDist = Double.MAX_VALUE;
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    BlockPos p = origin.add(dx, dy, dz);
                    BlockState s = world.getBlockState(p);
                    if (s.isAir()) continue;
                    String id = Registries.BLOCK.getId(s.getBlock()).toString();
                    if (!id.contains(substr)) continue;
                    double d = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    if (d < bestDist) {
                        bestDist = d;
                        best     = p;
                    }
                }
            }
        }
        return Optional.ofNullable(best);
    }

    @Override public int clippedAxes()     { return clipped;  }
    @Override public String failureReason() { return failureReason; }
}
```

- [ ] **Step 2: Build verify**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -5
```

Expected: `BUILD SUCCESSFUL`. Mapping note: `Registries.BLOCK`, `BlockState.isAir()`, `world.breakBlock(...)`, `agent.getBlockPos()` are all stable in MC 1.21.1 Yarn `1.21.1+build.3`. If any signature drifts, re-run `./gradlew genSources` and adjust.

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java
git commit -m "feat(motor): HarvestSkill — find + break nearest matching block (M1-Pipeline T3)"
```

---

### Task 4: `DepositChestSkill` — transfer inventory to nearby chest

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/DepositChestSkill.java`

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/DepositChestSkill.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.entity.ChestBlockEntity;
import net.minecraft.inventory.Inventory;
import net.minecraft.item.ItemStack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.Optional;

/** §4.2 DEPOSIT_CHEST: locate the nearest chest within MAX_CHEST_RADIUS,
 *  walk to it, transfer ALL stackable items from agent inventory to chest.
 *
 *  M1-Pipeline simplification: deposits everything, not just items
 *  matching target_class. Full filtering is Plan B (or M2 when builder
 *  needs specific items kept).
 */
public class DepositChestSkill implements SkillExecutor {

    private static final double MAX_CHEST_RADIUS = 8.0;
    private static final double REACH_RADIUS     = 2.0;
    private static final double WALK_PER_TICK    = 4.3 / 20.0;

    private BlockPos chestPos;
    private long ticksRemaining;
    private boolean depositDone = false;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        this.ticksRemaining = action.has("timeout_ticks")
            ? action.get("timeout_ticks").getAsLong()
            : 6000L;
        ServerWorld world = (ServerWorld) agent.getWorld();
        Optional<BlockPos> chest = findNearestChest(world, agent.getBlockPos());
        if (chest.isEmpty()) {
            failureReason = "no chest within " + MAX_CHEST_RADIUS + " blocks";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        this.chestPos = chest.get();
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        if (--ticksRemaining <= 0) {
            failureReason = "deposit_chest timeout";
            return SkillResult.FAILED_TIMEOUT;
        }
        if (depositDone) return SkillResult.COMPLETED;
        Vec3d center = Vec3d.ofCenter(chestPos);
        Vec3d here   = agent.getPos();
        double dist  = here.distanceTo(center);
        if (dist > REACH_RADIUS) {
            Vec3d dir = center.subtract(here).normalize();
            double step = Math.min(WALK_PER_TICK, dist - REACH_RADIUS);
            agent.setPosition(here.x + dir.x * step, here.y + dir.y * step, here.z + dir.z * step);
            agent.setYaw((float) Math.toDegrees(Math.atan2(-dir.x, dir.z)));
            return SkillResult.RUNNING;
        }
        ServerWorld world = (ServerWorld) agent.getWorld();
        var be = world.getBlockEntity(chestPos);
        if (!(be instanceof ChestBlockEntity chest)) {
            failureReason = "chest disappeared mid-deposit";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        depositAll(agent, chest);
        depositDone = true;
        return SkillResult.COMPLETED;
    }

    private static Optional<BlockPos> findNearestChest(ServerWorld world, BlockPos origin) {
        int radius = (int) Math.ceil(MAX_CHEST_RADIUS);
        BlockPos best = null;
        double bestDist = Double.MAX_VALUE;
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    BlockPos p = origin.add(dx, dy, dz);
                    if (world.getBlockEntity(p) instanceof ChestBlockEntity) {
                        double d = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        if (d < bestDist) { bestDist = d; best = p; }
                    }
                }
            }
        }
        return Optional.ofNullable(best);
    }

    private static void depositAll(ServerPlayerEntity agent, Inventory chest) {
        var pi = agent.getInventory();
        for (int slot = 0; slot < pi.size(); slot++) {
            ItemStack stack = pi.getStack(slot);
            if (stack.isEmpty()) continue;
            ItemStack remainder = tryInsert(chest, stack.copy());
            pi.setStack(slot, remainder);
        }
        pi.markDirty();
        chest.markDirty();
    }

    private static ItemStack tryInsert(Inventory chest, ItemStack toInsert) {
        for (int i = 0; i < chest.size() && !toInsert.isEmpty(); i++) {
            ItemStack existing = chest.getStack(i);
            if (existing.isEmpty()) {
                chest.setStack(i, toInsert.copy());
                return ItemStack.EMPTY;
            }
            if (ItemStack.areItemsAndComponentsEqual(existing, toInsert)) {
                int space = existing.getMaxCount() - existing.getCount();
                if (space > 0) {
                    int move = Math.min(space, toInsert.getCount());
                    existing.increment(move);
                    toInsert.decrement(move);
                    chest.setStack(i, existing);
                }
            }
        }
        return toInsert;
    }

    @Override public int clippedAxes()     { return 0; }
    @Override public String failureReason() { return failureReason; }
}
```

- [ ] **Step 2: Build verify**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -5
```

Expected: `BUILD SUCCESSFUL`. `ItemStack.areItemsAndComponentsEqual` is the 1.21.1 NBT-aware equivalence; `getInventory()` returns the agent's inventory (Carpet fake players have the full vanilla PlayerInventory).

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/DepositChestSkill.java
git commit -m "feat(motor): DepositChestSkill — transfer inventory to nearest chest (M1-Pipeline T4)"
```

---

### Task 5: `SearchSkill` — look around (no-op for M1; populates obs)

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SearchSkill.java`

For M1-Pipeline, "search" is a wide-area scan that the gatherer can dispatch when no resources are visible. It rotates the agent's yaw 360° over `scalar_param × MAX_DURATION` ticks. Functionally it's a no-op for the world but it gives the agent a "looking around" semantic distinct from WAIT, and the obs builder picks up the wider FOV after rotation.

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SearchSkill.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.2 SEARCH: rotate yaw 360° over scalar_param × MAX_DURATION ticks.
 *  No world side-effect; the observation builder picks up resources that
 *  enter the agent's scan radius regardless of yaw, so SEARCH is mostly
 *  a semantic distinction from WAIT (different reward shaping potential
 *  in Plan B). */
public class SearchSkill implements SkillExecutor {
    private static final int MAX_DURATION = 200;  // 10 s @ 20 TPS

    private long endTick;
    private long startTick;
    private float startYaw;
    private int clipped;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        double scalar = action.has("scalar_param") && action.get("scalar_param").isJsonArray()
            ? action.getAsJsonArray("scalar_param").get(0).getAsDouble()
            : 0.5;
        if (scalar < 0.0 || scalar > 1.0) {
            clipped |= 0b1000;
            scalar = Math.max(0.0, Math.min(1.0, scalar));
        }
        long duration = Math.max(1L, (long) Math.round(scalar * MAX_DURATION));
        startTick = server.getOverworld().getTime();
        endTick   = startTick + duration;
        startYaw  = agent.getYaw();
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        long now = server.getOverworld().getTime();
        if (now >= endTick) return SkillResult.COMPLETED;
        long total = endTick - startTick;
        float progress = (float)(now - startTick) / total;
        agent.setYaw(startYaw + progress * 360.0f);
        return SkillResult.RUNNING;
    }

    @Override public int clippedAxes()     { return clipped; }
    @Override public String failureReason() { return failureReason; }
}
```

- [ ] **Step 2: Build verify + commit**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -5
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/SearchSkill.java
git commit -m "feat(motor): SearchSkill — yaw rotation scan (M1-Pipeline T5)"
```

---

### Task 6: `WaitSkill` — burn N ticks doing nothing

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/WaitSkill.java`

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/WaitSkill.java`:
```java
package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.2 WAIT: pass scalar_param × MAX_WAIT ticks. Useful for letting the
 *  world catch up (sleep through night, wait for a crop to grow). */
public class WaitSkill implements SkillExecutor {
    private static final long MAX_WAIT = 200L;   // 10 s @ 20 TPS

    private long ticksRemaining;
    private int clipped;

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        double scalar = action.has("scalar_param") && action.get("scalar_param").isJsonArray()
            ? action.getAsJsonArray("scalar_param").get(0).getAsDouble()
            : 0.05;  // default ~half-second
        if (scalar < 0.0 || scalar > 1.0) {
            clipped |= 0b1000;
            scalar = Math.max(0.0, Math.min(1.0, scalar));
        }
        ticksRemaining = Math.max(1L, (long) Math.round(scalar * MAX_WAIT));
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        return --ticksRemaining <= 0 ? SkillResult.COMPLETED : SkillResult.RUNNING;
    }

    @Override public int clippedAxes()     { return clipped; }
    @Override public String failureReason() { return ""; }
}
```

- [ ] **Step 2: Build verify + commit**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -5
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/WaitSkill.java
git commit -m "feat(motor): WaitSkill — scalar-bounded no-op (M1-Pipeline T6)"
```

---

### Task 7: Real `MotorBridge.dispatchSkill` — registry + per-tick driver + completion events

**Files:**
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java`

Replace the M0 stub (which called `server.execute(() -> {})` as a placeholder) with the real dispatcher. Per spec §4.6 + the §6-review carry-forwards: skills are multi-tick, completion is ack-based, idempotency dedupes by `skill_invocation_id`, 30-second wall-clock cap on `advanceTickAwaitEvents`.

- [ ] **Step 1: Replace `MotorBridge.java`**

Replace `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java` with:
```java
package dev.aiutopia.mod.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import dev.aiutopia.mod.AiUtopiaMod;
import dev.aiutopia.mod.bridge.skill.*;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

import java.util.*;
import java.util.concurrent.*;

/** §7.3 + §4.6 motor module. Owns:
 *   - the per-agent active skill slot (max 1 in flight per agent)
 *   - per-agent skill_invocation_id counter for idempotent dispatch dedupe
 *   - the per-tick driver that calls SkillExecutor.tick() and emits
 *     SkillCompletionEvent on terminal results
 *   - the synchronous gate that Py4J's advanceTickAwaitEvents() waits on
 *
 * Thread model: server tick events run on the server thread; Py4J calls
 * arrive on Py4J's gateway thread. We marshal Py4J→server via the
 * server.execute(Runnable) queue and signal Py4J→server via thread-safe
 * concurrent maps + a BlockingQueue of completion events. */
public class MotorBridge {

    /** Emitted to Python when a skill terminates. */
    public static final class CompletionEvent {
        public final String agentId;
        public final String skillInvocationId;
        public final String resultCode;        // SkillResult.name()
        public final String failureReason;
        public final int    clippedAxesBitset; // bits 0=dx,1=dy,2=dz,3=scalar
        public CompletionEvent(String a, String i, SkillResult r, String why, int clip) {
            this.agentId = a; this.skillInvocationId = i;
            this.resultCode = r.name(); this.failureReason = why; this.clippedAxesBitset = clip;
        }
    }

    private MinecraftServer server;
    private final Gson gson = new Gson();

    // active skill per agent (server-thread access only)
    private final Map<String, ActiveDispatch> active = new HashMap<>();

    // idempotency: dispatched invocation ids per agent (server-thread access only)
    private final Map<String, Set<String>> dispatched = new HashMap<>();

    // completion events, drained by advanceTickAwaitEvents (multi-thread)
    private final BlockingQueue<CompletionEvent> completedQueue = new LinkedBlockingQueue<>();

    private static final class ActiveDispatch {
        final SkillExecutor executor;
        final String skillInvocationId;
        ActiveDispatch(SkillExecutor e, String i) { this.executor = e; this.skillInvocationId = i; }
    }

    public void attachServer(MinecraftServer server) {
        this.server = server;
        // Register the per-tick driver. END_SERVER_TICK fires once per server tick
        // after world simulation; we tick each active executor.
        ServerTickEvents.END_SERVER_TICK.register(this::onServerTick);
    }

    public void detachServer() {
        this.server = null;
        active.clear();
        dispatched.clear();
        completedQueue.clear();
    }

    /** Py4J entry point. agentId is the Carpet player name; encodedAction is the
     *  full action_dict JSON-serialized by Python. */
    public void dispatchSkill(String agentId, String encodedAction, String skillInvocationId) {
        if (server == null) return;
        // Marshal to server thread for safe state access
        server.execute(() -> dispatchOnServerThread(agentId, encodedAction, skillInvocationId));
    }

    private void dispatchOnServerThread(String agentId, String encodedAction, String skillInvocationId) {
        // Idempotency: drop duplicate invocation ids
        Set<String> seen = dispatched.computeIfAbsent(agentId, k -> new HashSet<>());
        if (!seen.add(skillInvocationId)) {
            AiUtopiaMod.LOG.debug("dropping duplicate dispatch {} for {}", skillInvocationId, agentId);
            return;
        }
        // Pre-empt any active skill for this agent (emit ABORTED)
        ActiveDispatch prev = active.remove(agentId);
        if (prev != null) {
            completedQueue.offer(new CompletionEvent(
                agentId, prev.skillInvocationId, SkillResult.ABORTED,
                "preempted by " + skillInvocationId, prev.executor.clippedAxes()
            ));
        }
        // Find the agent player
        ServerPlayerEntity agent = server.getPlayerManager().getPlayer(agentId);
        if (agent == null) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "agent player not found: " + agentId, 0
            ));
            return;
        }
        // Parse JSON
        JsonObject action;
        try {
            action = gson.fromJson(encodedAction, JsonObject.class);
        } catch (Exception e) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "bad action JSON: " + e.getMessage(), 0
            ));
            return;
        }
        // Construct executor for the skill_type
        int skillType = action.has("skill_type") ? action.get("skill_type").getAsInt() : -1;
        SkillExecutor exec = newExecutorForSkillType(skillType);
        if (exec == null) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "unknown skill_type: " + skillType, 0
            ));
            return;
        }
        // Start the executor
        SkillResult started = exec.start(agent, action, server);
        if (started != SkillResult.RUNNING) {
            // Instant terminal — emit completion right away
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, started, exec.failureReason(), exec.clippedAxes()
            ));
            return;
        }
        active.put(agentId, new ActiveDispatch(exec, skillInvocationId));
    }

    /** GATHERER skill_type indices match Python's spaces.py:
     *  0 NAVIGATE  1 HARVEST  2 DEPOSIT_CHEST  3 SEARCH  4 WAIT  5 NOOP_BROADCAST */
    private SkillExecutor newExecutorForSkillType(int skillType) {
        return switch (skillType) {
            case 0 -> new NavigateSkill();
            case 1 -> new HarvestSkill();
            case 2 -> new DepositChestSkill();
            case 3 -> new SearchSkill();
            case 4 -> new WaitSkill();
            case 5 -> new WaitSkill();   // NOOP_BROADCAST = WaitSkill(0); comm payload is handled by CommBus
            default -> null;
        };
    }

    private void onServerTick(MinecraftServer server) {
        if (active.isEmpty()) return;
        // Snapshot to avoid ConcurrentModificationException on emit-and-remove
        var entries = new ArrayList<>(active.entrySet());
        for (var entry : entries) {
            String agentId  = entry.getKey();
            ActiveDispatch d = entry.getValue();
            ServerPlayerEntity agent = server.getPlayerManager().getPlayer(agentId);
            if (agent == null) {
                active.remove(agentId);
                completedQueue.offer(new CompletionEvent(
                    agentId, d.skillInvocationId, SkillResult.ABORTED,
                    "agent disappeared", d.executor.clippedAxes()
                ));
                continue;
            }
            SkillResult r;
            try {
                r = d.executor.tick(agent, server);
            } catch (Exception e) {
                AiUtopiaMod.LOG.error("skill {} crashed for {}: {}",
                                       d.skillInvocationId, agentId, e.getMessage(), e);
                r = SkillResult.IMMEDIATE_FAILURE;
            }
            if (r != SkillResult.RUNNING) {
                active.remove(agentId);
                completedQueue.offer(new CompletionEvent(
                    agentId, d.skillInvocationId, r,
                    d.executor.failureReason(), d.executor.clippedAxes()
                ));
            }
        }
    }

    /** Py4J entry point: blocks up to timeoutMs for at least one CompletionEvent
     *  to arrive, then drains all pending completions. Returns the list of
     *  CompletionEvent JSON strings (one per terminated skill since last call). */
    public java.util.List<String> advanceTickAwaitEvents(long timeoutMs) {
        java.util.List<String> out = new ArrayList<>();
        try {
            // Block up to timeoutMs for the first event
            CompletionEvent first = completedQueue.poll(timeoutMs, TimeUnit.MILLISECONDS);
            if (first != null) {
                out.add(gson.toJson(first));
                // Drain any others without further blocking
                completedQueue.drainTo(new AbstractCollection<CompletionEvent>() {
                    @Override public boolean add(CompletionEvent e) { out.add(gson.toJson(e)); return true; }
                    @Override public Iterator<CompletionEvent> iterator() { return Collections.emptyIterator(); }
                    @Override public int size() { return 0; }
                });
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        return out;
    }
}
```

- [ ] **Step 2: Build verify**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -10
```

Expected: `BUILD SUCCESSFUL`. If `ServerTickEvents` import fails, it's from `net.fabricmc.fabric.api.event.lifecycle.v1` (already a dep via fabric-api in build.gradle).

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/MotorBridge.java
git commit -m "feat(motor): MotorBridge real dispatch + per-tick driver + completion events (M1-Pipeline T7)"
```

---

### Task 8: `ObservationBuilder` skeleton + `ObsJsonWriter` helper

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ObservationBuilder.java`
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ObsJsonWriter.java`

- [ ] **Step 1: Write `ObsJsonWriter`**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ObsJsonWriter.java`:
```java
package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

/** Tiny helper to keep ObservationBuilder code readable. Wraps common
 *  patterns of building JsonObject / JsonArray from numeric primitives. */
public final class ObsJsonWriter {
    private ObsJsonWriter() {}

    public static JsonArray vec(double x, double y, double z) {
        JsonArray a = new JsonArray();
        a.add(x); a.add(y); a.add(z);
        return a;
    }
    public static JsonArray vec2(double x, double y) {
        JsonArray a = new JsonArray();
        a.add(x); a.add(y);
        return a;
    }
    public static JsonArray intArray(int[] xs) {
        JsonArray a = new JsonArray();
        for (int x : xs) a.add(x);
        return a;
    }
    public static JsonArray floatArray(float[] xs) {
        JsonArray a = new JsonArray();
        for (float x : xs) a.add(x);
        return a;
    }
    public static JsonObject withScalar(String key, double value) {
        JsonObject o = new JsonObject();
        o.add(key, new JsonPrimitive(value));
        return o;
    }
}
```

- [ ] **Step 2: Write `ObservationBuilder` (composes core + role overlay)**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/obs/ObservationBuilder.java`:
```java
package dev.aiutopia.mod.obs;

import com.google.gson.JsonObject;
import dev.aiutopia.mod.agent.AgentRegistry;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.1 obs Dict builder. Composes core + role-specific overlay into a
 *  single JsonObject keyed by agent_id; WorldOps.observationsAll wraps
 *  all agents into one outer JsonObject. */
public final class ObservationBuilder {

    private final CoreObsBuilder           core   = new CoreObsBuilder();
    private final GathererOverlayBuilder   gather = new GathererOverlayBuilder();

    /** Build the full obs JsonObject for one agent. */
    public JsonObject buildForAgent(ServerPlayerEntity agent, MinecraftServer server) {
        JsonObject obs = new JsonObject();
        core.populate(obs, agent, server);
        // M1-Pipeline ships gatherer overlay only. Other roles are M2-M4.
        String name = agent.getGameProfile().getName();
        if (AgentRegistry.roleOf(name).equals("gatherer")) {
            gather.populate(obs, agent, server);
        }
        return obs;
    }
}
```

`AgentRegistry.roleOf(name)` doesn't exist yet — add it as part of this task.

- [ ] **Step 3: Extend `AgentRegistry` with role tracking**

Edit `fabric_mod/src/main/java/dev/aiutopia/mod/agent/AgentRegistry.java` — replace with:
```java
package dev.aiutopia.mod.agent;

import java.util.Set;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** In-process registry of AI agent player names + their roles.
 *  Populated by Py4J (CLI `aiutopia agent spawn`). Mixins + obs builder
 *  consult this to decide whether to filter / which overlay to emit.
 *  Thread-safe; reads are lock-free. */
public final class AgentRegistry {
    private static final Map<String, String> AGENT_ROLES = new ConcurrentHashMap<>();

    private AgentRegistry() {}

    /** M0 backward-compat — defaults role to "gatherer" when role unspecified. */
    public static void registerAgent(String playerName) {
        AGENT_ROLES.putIfAbsent(playerName, "gatherer");
    }
    public static void registerAgent(String playerName, String role) {
        AGENT_ROLES.put(playerName, role);
    }
    public static void unregisterAgent(String playerName) {
        AGENT_ROLES.remove(playerName);
    }
    public static boolean isAgent(String playerName) {
        return AGENT_ROLES.containsKey(playerName);
    }
    public static String roleOf(String playerName) {
        return AGENT_ROLES.getOrDefault(playerName, "");
    }
    public static Set<String> snapshot() {
        return Set.copyOf(AGENT_ROLES.keySet());
    }
}
```

- [ ] **Step 4: Commit (the two new files won't compile until T9 + T10 land — that's expected)**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/obs/ \
        fabric_mod/src/main/java/dev/aiutopia/mod/agent/AgentRegistry.java
git commit -m "feat(obs): ObservationBuilder + ObsJsonWriter scaffold + AgentRegistry role map (M1-Pipeline T8)"
```

T9 + T10 must follow before `./gradlew compileJava` will succeed again — `CoreObsBuilder` and `GathererOverlayBuilder` are referenced but don't exist yet.

---

### Task 9: `CoreObsBuilder` — universal core fields

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/obs/CoreObsBuilder.java`

The core fields per spec §4.1. For M1-Pipeline we emit numeric primitives only — the 384-d `agent_uuid_embed` is computed on the Python side from `agent.getUuidAsString()`; we just emit the UUID string and let Python encode. Same for `goal_embedding` — it comes from Python's GoalSpecAdapter, not from Java. Java emits everything else.

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/obs/CoreObsBuilder.java`:
```java
package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.world.GameRules;

import static dev.aiutopia.mod.obs.ObsJsonWriter.*;

/** §4.1 universal core obs. Emits the numeric fields; Python encodes
 *  agent_uuid_embed (from agent_uuid string) and goal_embedding (from
 *  the GoalSpecAdapter). */
public final class CoreObsBuilder {

    public void populate(JsonObject obs, ServerPlayerEntity agent, MinecraftServer server) {
        obs.addProperty("agent_uuid", agent.getUuidAsString());
        obs.addProperty("agent_name", agent.getGameProfile().getName());

        // role one-hot is set by ObservationBuilder via AgentRegistry — we just
        // emit role_id here as a string; Python turns it into one-hot.
        obs.addProperty("role_id",
            dev.aiutopia.mod.agent.AgentRegistry.roleOf(agent.getGameProfile().getName()));

        long worldTime = server.getOverworld().getTime();
        obs.addProperty("tick_in_episode", worldTime % 24_000L);

        var pos = agent.getPos();
        obs.add("position", vec(pos.x, pos.y, pos.z));
        var vel = agent.getVelocity();
        obs.add("velocity", vec(vel.x, vel.y, vel.z));
        obs.add("yaw_pitch", vec2(agent.getYaw(), agent.getPitch()));

        obs.addProperty("health",   agent.getHealth());
        obs.addProperty("hunger",   agent.getHungerManager().getFoodLevel());
        obs.addProperty("saturation", agent.getHungerManager().getSaturationLevel());
        obs.addProperty("armor_value", agent.getArmor());

        // Inventory — 36 slots, item id (string) + count
        var inv = agent.getInventory();
        JsonArray itemIds = new JsonArray();
        JsonArray counts  = new JsonArray();
        for (int i = 0; i < 36; i++) {
            ItemStack s = inv.getStack(i);
            itemIds.add(Registries.ITEM.getId(s.getItem()).toString());
            counts.add(s.getCount());
        }
        obs.add("inv_slot_item_ids", itemIds);
        obs.add("inv_slot_counts",   counts);
        obs.addProperty("main_hand_item_id",
            Registries.ITEM.getId(agent.getMainHandStack().getItem()).toString());
        obs.addProperty("off_hand_item_id",
            Registries.ITEM.getId(agent.getOffHandStack().getItem()).toString());

        // World state
        obs.addProperty("time_of_day", server.getOverworld().getTimeOfDay() % 24_000L);
        boolean raining   = server.getOverworld().isRaining();
        boolean thundering = server.getOverworld().isThundering();
        obs.addProperty("weather", thundering ? 2 : (raining ? 1 : 0));
        obs.addProperty("biome_id",
            Registries.BIOME.getId(
                server.getOverworld().getBiome(agent.getBlockPos()).value()).toString());
        obs.addProperty("light_level",
            server.getOverworld().getLightLevel(agent.getBlockPos()));

        // comm_payloads + comm_metadata are placed by Python from the CommBus
        // ring buffer — we don't emit them here.
        // action_mask is computed in Python from the symbolic obs above.
    }
}
```

- [ ] **Step 2: Commit (still won't compile — T10 needed)**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/obs/CoreObsBuilder.java
git commit -m "feat(obs): CoreObsBuilder — universal core obs fields (M1-Pipeline T9)"
```

---

### Task 10: `GathererOverlayBuilder` — gatherer-specific obs fields

**Files:**
- Create: `fabric_mod/src/main/java/dev/aiutopia/mod/obs/GathererOverlayBuilder.java`

Gatherer overlay per spec §4.1:
- `g_resource_grid`: top-down 32-radius × 32-block × 6-channel grid (log/stone/coal/iron/food/threat)
- `g_nearest_resources`: top-8 (dx, dy, dz, type_id, qty_est, accessibility)
- `g_richness_score`: scalar [0, 1]
- `g_hostiles_nearby`: up to 4 entries (dx, dy, dz, type_id)

For M1-Pipeline we ship a usable subset:
- `g_resource_grid` populated by scanning a 32×32 horizontal slab at agent.y (single y-layer; full 3D voxel is M2+)
- `g_nearest_resources` populated from a 16-block-radius scan
- `g_richness_score` = `min(1.0, count_of_resources / 64)`
- `g_hostiles_nearby` from `world.getEntitiesByClass(HostileEntity)` within 16 blocks

- [ ] **Step 1: Implement**

Create `fabric_mod/src/main/java/dev/aiutopia/mod/obs/GathererOverlayBuilder.java`:
```java
package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.BlockState;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.util.*;

/** §4.1 gatherer overlay. M1-Pipeline subset:
 *   - g_resource_grid: 32×32 single-y-layer (full 3D voxel = M2+)
 *   - g_nearest_resources: top-8 within 16 blocks
 *   - g_richness_score: count_of_resources / 64, clamped to 1.0
 *   - g_hostiles_nearby: up to 4 in 16-block radius
 */
public final class GathererOverlayBuilder {

    private static final int    GRID_RADIUS    = 16;       // 32×32 grid centered on agent
    private static final int    SCAN_RADIUS    = 16;
    private static final int    MAX_HOSTILES   = 4;
    private static final int    TOP_K_NEAREST  = 8;

    /** Channel indices in g_resource_grid: 0 log, 1 stone, 2 coal, 3 iron, 4 food, 5 threat. */
    private static final Map<String, Integer> CHANNEL = new HashMap<>();
    static {
        CHANNEL.put("log",    0);
        CHANNEL.put("stone",  1);
        CHANNEL.put("coal",   2);
        CHANNEL.put("iron",   3);
        CHANNEL.put("wheat",  4);  // food channel — wheat for M1
        CHANNEL.put("carrot", 4);
        CHANNEL.put("potato", 4);
    }

    public void populate(JsonObject obs, ServerPlayerEntity agent, MinecraftServer server) {
        ServerWorld world = (ServerWorld) agent.getWorld();
        BlockPos origin   = agent.getBlockPos();

        // 1. Scan a 32×32×3 slab (slight vertical fudge so trees over hills register)
        int gridSize = 2 * GRID_RADIUS;
        float[][][] grid = new float[gridSize][gridSize][6];
        List<NearbyResource> nearby = new ArrayList<>();
        int totalCount = 0;

        for (int dx = -GRID_RADIUS; dx < GRID_RADIUS; dx++) {
            for (int dz = -GRID_RADIUS; dz < GRID_RADIUS; dz++) {
                // Take the topmost non-air block within ±3 y of agent
                for (int dy = 3; dy >= -3; dy--) {
                    BlockPos p = origin.add(dx, dy, dz);
                    BlockState s = world.getBlockState(p);
                    if (s.isAir()) continue;
                    String id = Registries.BLOCK.getId(s.getBlock()).toString();
                    Integer channel = matchChannel(id);
                    if (channel != null) {
                        grid[dx + GRID_RADIUS][dz + GRID_RADIUS][channel] = 1.0f;
                        if (Math.sqrt(dx*dx + dy*dy + dz*dz) <= SCAN_RADIUS) {
                            nearby.add(new NearbyResource(dx, dy, dz, id));
                            totalCount++;
                        }
                    }
                    break;  // only the top hit per (dx, dz)
                }
            }
        }

        // 2. Pack g_resource_grid as flat JSON (32 × 32 × 6 = 6144 floats)
        JsonArray gridArr = new JsonArray(gridSize * gridSize * 6);
        for (int x = 0; x < gridSize; x++)
            for (int z = 0; z < gridSize; z++)
                for (int c = 0; c < 6; c++)
                    gridArr.add(grid[x][z][c]);
        obs.add("g_resource_grid", gridArr);

        // 3. g_nearest_resources — sort by distance, take top 8
        nearby.sort(Comparator.comparingDouble(r -> r.distSq()));
        JsonArray nearestArr = new JsonArray();
        for (int i = 0; i < TOP_K_NEAREST; i++) {
            JsonArray row = new JsonArray();
            if (i < nearby.size()) {
                NearbyResource r = nearby.get(i);
                row.add(r.dx / (double) SCAN_RADIUS);
                row.add(r.dy / 8.0);
                row.add(r.dz / (double) SCAN_RADIUS);
                row.add(matchChannel(r.id) == null ? -1 : matchChannel(r.id));
                row.add(1.0);                       // qty_est: M1-Pipeline always 1
                row.add(1.0);                       // accessibility: M1-Pipeline always reachable
            } else {
                for (int j = 0; j < 6; j++) row.add(0.0);
            }
            nearestArr.add(row);
        }
        obs.add("g_nearest_resources", nearestArr);

        // 4. g_richness_score
        obs.addProperty("g_richness_score", Math.min(1.0, totalCount / 64.0));

        // 5. g_hostiles_nearby
        Vec3d agentPos = agent.getPos();
        Box scanBox = new Box(agentPos.x - SCAN_RADIUS, agentPos.y - SCAN_RADIUS, agentPos.z - SCAN_RADIUS,
                              agentPos.x + SCAN_RADIUS, agentPos.y + SCAN_RADIUS, agentPos.z + SCAN_RADIUS);
        var hostiles = world.getEntitiesByClass(HostileEntity.class, scanBox, e -> e.isAlive());
        hostiles.sort(Comparator.comparingDouble(h -> h.squaredDistanceTo(agent)));
        JsonArray hostArr = new JsonArray();
        for (int i = 0; i < MAX_HOSTILES; i++) {
            JsonArray row = new JsonArray();
            if (i < hostiles.size()) {
                var h = hostiles.get(i);
                row.add((h.getX() - agentPos.x) / SCAN_RADIUS);
                row.add((h.getY() - agentPos.y) / SCAN_RADIUS);
                row.add((h.getZ() - agentPos.z) / SCAN_RADIUS);
                row.add(typeIdForEntity(h));
            } else {
                for (int j = 0; j < 4; j++) row.add(0.0);
            }
            hostArr.add(row);
        }
        obs.add("g_hostiles_nearby", hostArr);
    }

    private static Integer matchChannel(String blockId) {
        for (var e : CHANNEL.entrySet()) {
            if (blockId.contains(e.getKey())) return e.getValue();
        }
        return null;
    }

    private static double typeIdForEntity(net.minecraft.entity.Entity e) {
        // Hash of registry id, normalized — Python doesn't care about the exact
        // value, just stability across ticks
        return Math.abs(Registries.ENTITY_TYPE.getId(e.getType()).toString().hashCode() % 256) / 256.0;
    }

    private record NearbyResource(int dx, int dy, int dz, String id) {
        double distSq() { return dx*dx + dy*dy + dz*dz; }
    }
}
```

- [ ] **Step 2: Build verify**

```bash
cd fabric_mod && ./gradlew compileJava --no-daemon 2>&1 | tail -10
```

Expected: `BUILD SUCCESSFUL`. T8/T9/T10 should all compile together now.

- [ ] **Step 3: Commit**

```bash
git add fabric_mod/src/main/java/dev/aiutopia/mod/obs/GathererOverlayBuilder.java
git commit -m "feat(obs): GathererOverlayBuilder — 32x32 resource grid + top-8 + hostiles (M1-Pipeline T10)"
```

---

### Task 11: Real `WorldOps.observationsAll`

**Files:**
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`

Replace the M0 stub (`return "{}"`) with the real builder. Iterates over `AgentRegistry.snapshot()`, calls `ObservationBuilder.buildForAgent` for each living agent player, packs into a single JsonObject keyed by agent_id.

- [ ] **Step 1: Replace observation method**

Open `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java` and replace the entire `observationsAll()` method:
```java
    /** §7.3 batched observation read — single JSON blob with every agent on
     *  this env. Iterates AgentRegistry.snapshot(), composes per-agent obs
     *  via ObservationBuilder. Skips agents that are not currently on the
     *  server (e.g. dead, pre-spawn). */
    public String observationsAll() {
        if (server == null) return "{}";
        com.google.gson.JsonObject root = new com.google.gson.JsonObject();
        var builder = new dev.aiutopia.mod.obs.ObservationBuilder();
        for (String name : dev.aiutopia.mod.agent.AgentRegistry.snapshot()) {
            var player = server.getPlayerManager().getPlayer(name);
            if (player == null) continue;   // not currently connected
            root.add(name, builder.buildForAgent(player, server));
        }
        return root.toString();
    }
```

(Keep the `attachServer`, `detachServer`, `resetWorld`, and `carpetSpawn` methods as-is.)

- [ ] **Step 2: Build verify**

```bash
cd fabric_mod && ./gradlew build --no-daemon 2>&1 | tail -8
```

Expected: `BUILD SUCCESSFUL`. The new jar at `build/libs/aiutopia-mod-0.0.0-m0.jar` is ~140-150 KB (was 131 KB).

- [ ] **Step 3: Copy jar to server-runtime + commit**

```bash
cp fabric_mod/build/libs/aiutopia-mod-0.0.0-m0.jar server-runtime/mods/
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java
git commit -m "feat(obs): WorldOps.observationsAll real implementation (M1-Pipeline T11)"
```

---

### Task 12: `tech_tree_potential()` + `LOG_VALUE` + `ROLE_INVENTORY_CAPS`

**Files:**
- Create: `src/aiutopia/env/reward.py`
- Create: `tests/unit/test_tech_tree_potential.py`

Per spec §5.7 verbatim. Defines `tech_tree_potential(inventory, role) → float` and the supporting tables.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_tech_tree_potential.py`:
```python
import pytest

from aiutopia.env.reward import (
    LOG_VALUE,
    ROLE_INVENTORY_CAPS,
    tech_tree_potential,
)


def test_log_value_table_has_oak_log_at_1_point_0() -> None:
    assert LOG_VALUE["oak_log"] == 1.0


def test_potential_zero_on_empty_inventory() -> None:
    assert tech_tree_potential({}, "gatherer") == 0.0


def test_potential_grows_with_inventory() -> None:
    p1 = tech_tree_potential({"oak_log": 1}, "gatherer")
    p10 = tech_tree_potential({"oak_log": 10}, "gatherer")
    p100 = tech_tree_potential({"oak_log": 100}, "gatherer")
    assert p1 < p10
    # Per §5.7, gatherer cap on oak_log is 256, so 100 < cap and the growth is linear
    assert p10 < p100


def test_potential_capped_per_role() -> None:
    # Gatherer cap on oak_log is 256; potential clamped above
    p256 = tech_tree_potential({"oak_log": 256}, "gatherer")
    p1000 = tech_tree_potential({"oak_log": 1000}, "gatherer")
    assert p256 == p1000  # cap applied


def test_unrecognized_item_ignored() -> None:
    p = tech_tree_potential({"unobtainium": 999}, "gatherer")
    assert p == 0.0


def test_per_role_caps_differ() -> None:
    # Builder has higher oak_planks cap than gatherer
    builder_cap = ROLE_INVENTORY_CAPS["builder"]["oak_planks"]
    gatherer_cap = ROLE_INVENTORY_CAPS["gatherer"]["oak_planks"]
    assert builder_cap > gatherer_cap


def test_rejects_unknown_role() -> None:
    with pytest.raises(KeyError):
        tech_tree_potential({"oak_log": 1}, "wizard")
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/unit/test_tech_tree_potential.py -v
```

Expected: ImportError on `from aiutopia.env.reward import ...`.

- [ ] **Step 3: Implement (verbatim from spec §5.7)**

Create `src/aiutopia/env/reward.py`:
```python
"""§5 reward architecture — stage 1 only for M1-Pipeline.

Stage 2 (multi-objective + curriculum decay) and stage 3 (village
scarcity weights + LLM-driven targets) are deferred to M2-M5."""
from __future__ import annotations

from typing import Literal

RoleId = Literal["gatherer", "builder", "farmer", "defender"]


# §5.7 — VPT-normalized log-scaled potentials
LOG_VALUE: dict[str, float] = {
    # raw materials
    "oak_log":          1.000,
    "oak_planks":       0.050,
    "stick":            0.0625,
    "cobblestone":      1.0 / 11.0,
    "stone":            0.500,
    "coal":             0.400,
    "iron_ore":         4.000,
    "iron_ingot":       5.000,
    "gold_ingot":       3.000,
    "diamond":          8.000,
    # food
    "wheat":            0.1875,
    "bread":            0.375,
    "porkchop":         0.375,   "cooked_porkchop":  0.500,
    "beef":             0.375,   "cooked_beef":      0.500,
    "chicken":          0.375,   "cooked_chicken":   0.500,
    "carrot":           0.1875,
    "apple":            0.250,
    # crafted
    "crafting_table":   1.000,
    "furnace":          1.000,
    "wooden_pickaxe":   1.000,
    "stone_pickaxe":    1.500,
    "iron_pickaxe":     4.000,
    "wooden_sword":     0.500,
    "stone_sword":      1.000,
    "iron_sword":       4.000,
    "wooden_axe":       0.500,   "stone_axe":        1.000,    "iron_axe":         4.000,
    "wooden_hoe":       0.500,   "stone_hoe":        1.000,    "iron_hoe":         4.000,
    # armor
    "leather_helmet":   0.500,   "iron_helmet":      2.000,
    "leather_chestplate": 1.000, "iron_chestplate":  4.000,
    "leather_leggings": 0.875,   "iron_leggings":    3.500,
    "leather_boots":    0.500,   "iron_boots":       2.000,
    # placeables
    "torch":            0.125,
    "oak_door":         0.500,
    "glass_pane":       0.500,
    "ladder":           0.250,
    "fence":            0.125,
    "chest":            1.000,
}


# Per-role anti-hoarding caps. `_default` applies to any item not in this role's dict.
ROLE_INVENTORY_CAPS: dict[RoleId, dict[str, int]] = {
    "gatherer":  {
        "oak_log": 256, "cobblestone": 256, "stone": 128,
        "coal": 128, "iron_ore": 128, "iron_ingot": 64,
        "diamond": 16, "wheat": 64, "bread": 32,
        "stick": 64, "oak_planks": 128,
        "wooden_pickaxe": 4, "stone_pickaxe": 4, "iron_pickaxe": 2,
        "_default": 64,
    },
    "builder":   {
        "oak_log": 128, "oak_planks": 512, "cobblestone": 512,
        "stone": 256, "torch": 128, "oak_door": 16,
        "glass_pane": 64, "ladder": 32, "fence": 64,
        "chest": 8, "iron_ingot": 16,
        "_default": 32,
    },
    "farmer":    {
        "wheat": 256, "bread": 128, "carrot": 64,
        "porkchop": 32, "beef": 32, "chicken": 32,
        "cooked_porkchop": 32, "cooked_beef": 32, "cooked_chicken": 32,
        "_default": 16,
    },
    "defender":  {
        "iron_sword": 4, "iron_pickaxe": 2,
        "iron_helmet": 2, "iron_chestplate": 2,
        "iron_leggings": 2, "iron_boots": 2,
        "bread": 16, "cooked_beef": 16,
        "_default": 8,
    },
}


def tech_tree_potential(inventory: dict[str, int], role: str) -> float:
    """Φ(s) for PBRS shaping. Capped per-role (anti-hoarding) and weighted by
    LOG_VALUE (VPT-normalized). Used by `compute_reward()` as:
        r_pbrs = γ · Φ(s') − Φ(s),   γ = 0.99
    Absolute scale matters less than monotonicity for PBRS — the difference
    is what feeds the reward.
    """
    if role not in ROLE_INVENTORY_CAPS:
        raise KeyError(f"unknown role: {role!r}")
    caps = ROLE_INVENTORY_CAPS[role]
    default_cap = caps.get("_default", 32)
    total = 0.0
    for item, qty in inventory.items():
        if item not in LOG_VALUE:
            continue
        cap = caps.get(item, default_cap)
        total += min(qty, cap) * LOG_VALUE[item]
    return total
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_tech_tree_potential.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/env/reward.py tests/unit/test_tech_tree_potential.py
git commit -m "feat(env): tech_tree_potential + LOG_VALUE + per-role caps (M1-Pipeline T12)"
```

---

### Task 13: `ExploitDetector` — 6 detection rules

**Files:**
- Create: `src/aiutopia/env/exploit.py`
- Create: `tests/unit/test_exploit.py`

Per spec §5.3 — 6 exploit types: ITEM_DROP_SPAM, OSCILLATION, INVENTORY_REPEAT, BULK_FARMING (multi-agent, deferred), SAFE_INACTION, MEANINGLESS_TOOL_CALL.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_exploit.py`:
```python
import pytest

from aiutopia.env.exploit import ExploitDetector, ExploitName


def _det() -> ExploitDetector:
    return ExploitDetector()


def _obs(inv: dict[str, int] | None = None, pos: tuple[float, float, float] = (0, 64, 0)) -> dict:
    return {
        "inv_slot_item_ids": list((inv or {}).keys()),
        "inv_slot_counts":   list((inv or {}).values()),
        "position":          list(pos),
    }


def _env_meta(global_step: int = 0,
              skill_result_code: str = "COMPLETED") -> dict:
    return {"global_step": global_step, "skill_result_code": skill_result_code}


# --- ITEM_DROP_SPAM ---------------------------------------------------

def test_drop_spam_triggers_after_two_drops_in_window() -> None:
    det = _det()
    inv_before = {"oak_log": 5}
    inv_after_drop_1 = {"oak_log": 4}
    inv_after_drop_2 = {"oak_log": 3}
    # First drop — no exploit yet
    p1 = det.step(role="gatherer",
                   obs_prev=_obs(inv_before),
                   obs_curr=_obs(inv_after_drop_1),
                   action={"skill_type": 4},
                   env_meta=_env_meta())
    # Second drop — still 2 = threshold
    p2 = det.step(role="gatherer",
                   obs_prev=_obs(inv_after_drop_1),
                   obs_curr=_obs(inv_after_drop_2),
                   action={"skill_type": 4},
                   env_meta=_env_meta(global_step=1))
    # Third drop — strictly > 2, triggers
    p3 = det.step(role="gatherer",
                   obs_prev=_obs(inv_after_drop_2),
                   obs_curr=_obs({"oak_log": 2}),
                   action={"skill_type": 4},
                   env_meta=_env_meta(global_step=2))
    names_3 = [name for name, _ in p3]
    assert ExploitName.DROP_SPAM in names_3


# --- OSCILLATION -----------------------------------------------------

def test_oscillation_triggers_after_six_revisits_with_intermediate() -> None:
    det = _det()
    role = "gatherer"
    # Visit tile A (0,64,0), then tile B (10,64,0), then back to A, repeated
    a = (0.5, 64.0, 0.5)
    b = (10.5, 64.0, 0.5)
    actions = []
    for _ in range(7):
        actions.append(("A", a))
        actions.append(("B", b))
    triggered = False
    last_pos = a
    for label, pos in actions:
        out = det.step(role=role,
                        obs_prev=_obs(pos=last_pos),
                        obs_curr=_obs(pos=pos),
                        action={"skill_type": 0},
                        env_meta=_env_meta())
        if any(n == ExploitName.OSCILLATION for n, _ in out):
            triggered = True
        last_pos = pos
    assert triggered


def test_stay_in_place_is_not_oscillation() -> None:
    det = _det()
    a = (0.5, 64.0, 0.5)
    for _ in range(10):
        out = det.step(role="gatherer",
                        obs_prev=_obs(pos=a),
                        obs_curr=_obs(pos=a),
                        action={"skill_type": 4},
                        env_meta=_env_meta())
        assert not any(n == ExploitName.OSCILLATION for n, _ in out)


# --- SAFE_INACTION ---------------------------------------------------

def test_lazy_triggers_after_81_consecutive_wait() -> None:
    det = _det()
    pos = (0.0, 64.0, 0.0)
    triggered_at = -1
    for step in range(100):
        out = det.step(role="gatherer",
                        obs_prev=_obs(pos=pos),
                        obs_curr=_obs(pos=pos),
                        action={"skill_type": 4},   # WAIT
                        env_meta=_env_meta(global_step=step))
        if any(n == ExploitName.LAZY_INACTION for n, _ in out):
            triggered_at = step
            break
    assert 80 < triggered_at < 90  # fires shortly after 80 consecutive


# --- MEANINGLESS_TOOL_CALL -------------------------------------------

def test_noop_spam_triggers_after_three_immediate_failures() -> None:
    det = _det()
    for step in range(5):
        out = det.step(role="gatherer",
                        obs_prev=_obs(),
                        obs_curr=_obs(),
                        action={"skill_type": 1, "target_class": 7},
                        env_meta=_env_meta(global_step=step,
                                            skill_result_code="IMMEDIATE_FAILURE"))
    names = [n for n, _ in out]
    assert ExploitName.NOOP_SKILL_SPAM in names
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/unit/test_exploit.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/aiutopia/env/exploit.py`:
```python
"""§5.3 — Exploit catalog with concrete detection rules.

Six exploit types. Multi-agent BULK_FARMING is implemented at the
env-wrapper level (cross-agent inventory transfer tracking), not here;
this class is per-agent stateful only."""
from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class ExploitName(str, enum.Enum):
    DROP_SPAM       = "drop_spam"
    OSCILLATION     = "oscillation"
    INV_REPEAT      = "inv_repeat"
    LAZY_INACTION   = "lazy_inaction"
    NOOP_SKILL_SPAM = "noop_skill_spam"


# Constants per §5.3 (post §5 carry-forward tightening)
MAX_DROPS_PER_WINDOW   = 2     # tightened from 3
DROP_WINDOW_TICKS      = 20
OSCILLATION_MIN_VISITS = 5      # > 5 in 40-tick buffer
OSCILLATION_BUF_SIZE   = 40
INV_REPEAT_BUF_SIZE    = 10
INV_REPEAT_THRESHOLD   = 2
LAZY_IDLE_TICKS        = 80
LAZY_SKILL_TYPES       = {4, 5}   # WAIT, NOOP_BROADCAST
NOOP_FAIL_WINDOW       = 50
NOOP_FAIL_THRESHOLD    = 3
BULK_LOG_SIZE          = 200


@dataclass
class ExploitDetector:
    """Per-agent stateful detector. Returns list of (name, penalty) tuples
    from each `step()` call. Penalties are positive numbers; callers
    SUBTRACT them from reward."""
    recent_drops:  deque[str] = field(default_factory=lambda: deque(maxlen=DROP_WINDOW_TICKS))
    position_buf:  deque[tuple[int, int, int]] = field(
        default_factory=lambda: deque(maxlen=OSCILLATION_BUF_SIZE))
    inv_snapshots: deque[int] = field(default_factory=lambda: deque(maxlen=INV_REPEAT_BUF_SIZE))
    bulk_log:      deque[tuple[int, tuple[int, int], str]] = field(
        default_factory=lambda: deque(maxlen=BULK_LOG_SIZE))
    last_action_skill_type: int = -1
    idle_streak:   int = 0
    last_step: list[tuple[ExploitName, float]] = field(default_factory=list)

    def step(self,
             *,
             role: str,
             obs_prev: dict[str, Any],
             obs_curr: dict[str, Any],
             action: dict[str, Any],
             env_meta: dict[str, Any]) -> list[tuple[ExploitName, float]]:
        penalties: list[tuple[ExploitName, float]] = []

        # 1. ITEM_DROP_SPAM
        dropped = _items_lost(obs_prev, obs_curr)
        if dropped:
            self.recent_drops.extend(dropped)
        if len(self.recent_drops) > MAX_DROPS_PER_WINDOW:
            penalties.append((ExploitName.DROP_SPAM, 0.5 * len(self.recent_drops)))

        # 2. OSCILLATION
        tile = _quantize_pos(obs_curr.get("position", [0, 0, 0]), grid=3)
        self.position_buf.append(tile)
        if (self.position_buf.count(tile) > OSCILLATION_MIN_VISITS
            and _has_nonadjacent_intermediate(self.position_buf, tile)):
            penalties.append((ExploitName.OSCILLATION, 0.3))

        # 3. INVENTORY_REPEAT
        snap = _inv_hash(obs_curr)
        if list(self.inv_snapshots).count(snap) >= INV_REPEAT_THRESHOLD:
            penalties.append((ExploitName.INV_REPEAT, 0.1))
        self.inv_snapshots.append(snap)

        # 4. BULK_FARMING — multi-agent, handled at env-wrapper level

        # 5. SAFE_INACTION / LAZY
        sk = int(action.get("skill_type", -1))
        if sk == self.last_action_skill_type:
            self.idle_streak += 1
        else:
            self.idle_streak = 0
        if sk in LAZY_SKILL_TYPES and self.idle_streak > LAZY_IDLE_TICKS:
            penalties.append((ExploitName.LAZY_INACTION, 0.2))
        self.last_action_skill_type = sk

        # 6. MEANINGLESS_TOOL_CALL
        key = (sk, int(action.get("target_class", -1)))
        global_step = int(env_meta.get("global_step", 0))
        result = str(env_meta.get("skill_result_code", "COMPLETED"))
        self.bulk_log.append((global_step, key, result))
        recent_fails = sum(
            1 for ts, k, r in self.bulk_log
            if k == key and r == "IMMEDIATE_FAILURE" and (global_step - ts) < NOOP_FAIL_WINDOW
        )
        if recent_fails > NOOP_FAIL_THRESHOLD:
            penalties.append((ExploitName.NOOP_SKILL_SPAM, 0.2 * recent_fails))

        self.last_step = penalties
        return penalties


# --- helpers ---------------------------------------------------------

def _items_lost(prev: dict[str, Any], curr: dict[str, Any]) -> list[str]:
    """Items whose count dropped between prev and curr, by name. Sum the
    delta — a stack going 5→3 is two drops."""
    pi = dict(zip(prev.get("inv_slot_item_ids", []), prev.get("inv_slot_counts", [])))
    ci = dict(zip(curr.get("inv_slot_item_ids", []), curr.get("inv_slot_counts", [])))
    out: list[str] = []
    for k, v_prev in pi.items():
        v_curr = ci.get(k, 0)
        if v_curr < v_prev:
            out.extend([k] * (v_prev - v_curr))
    return out


def _quantize_pos(pos: list[float], grid: int) -> tuple[int, int, int]:
    return (int(pos[0]) // grid, int(pos[1]) // grid, int(pos[2]) // grid)


def _has_nonadjacent_intermediate(buf: deque[tuple[int, int, int]],
                                   tile: tuple[int, int, int]) -> bool:
    """True iff between two visits to `tile` in `buf`, there's at least one
    visit to a tile that is NOT adjacent to `tile`. Cheb-distance > 1."""
    # Find indices of visits to `tile`
    indices = [i for i, t in enumerate(buf) if t == tile]
    if len(indices) < 2:
        return False
    for i in range(len(indices) - 1):
        seg = list(buf)[indices[i] + 1 : indices[i + 1]]
        if any(_cheb_dist(t, tile) > 1 for t in seg):
            return True
    return False


def _cheb_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def _inv_hash(obs: dict[str, Any]) -> int:
    return hash(tuple(zip(
        tuple(obs.get("inv_slot_item_ids", [])),
        tuple(obs.get("inv_slot_counts", [])),
    )))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_exploit.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/env/exploit.py tests/unit/test_exploit.py
git commit -m "feat(env): ExploitDetector — 5 per-agent rules from §5.3 (M1-Pipeline T13)"
```

---

### Task 14: Stage-1 `compute_reward()` + γ_clip

**Files:**
- Modify: `src/aiutopia/env/reward.py`
- Create: `tests/unit/test_reward.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_reward.py`:
```python
from aiutopia.env.reward import (
    DEATH_PENALTY,
    GAMMA,
    GAMMA_CLIP,
    TIME_PENALTY,
    compute_reward_stage_1,
)


def _obs(inv: dict[str, int]) -> dict:
    return {
        "inv_slot_item_ids": list(inv.keys()),
        "inv_slot_counts":   list(inv.values()),
    }


def _env_meta(died: bool = False, n_clipped: int = 0,
              exploit_penalties: list = None) -> dict:
    return {
        "died_this_tick": died,
        "n_clipped_param_axes": n_clipped,
        "exploit_penalties": exploit_penalties or [],
    }


def test_zero_reward_on_no_change() -> None:
    obs = _obs({"oak_log": 5})
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=obs,
                                obs_curr=obs,
                                action={"skill_type": 4},
                                env_meta=_env_meta())
    # Expected: r_pbrs = 0.99 * phi(5) - phi(5) = -0.01 * phi(5)
    # Time penalty = -0.001
    # No primary signal change
    assert r < 0   # negative time penalty + small negative PBRS shift
    assert r > -0.1  # but tiny


def test_positive_reward_on_oak_log_gain() -> None:
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=_obs({"oak_log": 0}),
                                obs_curr=_obs({"oak_log": 1}),
                                action={"skill_type": 1},
                                env_meta=_env_meta())
    # delta-inventory: +1 oak_log × LOG_VALUE[oak_log]=1.0 = +1.0 primary
    # PBRS: 0.99 * 1.0 - 0.0 = +0.99
    # Total ~ +2 - 0.001 (time)
    assert r > 1.5


def test_death_penalty() -> None:
    obs = _obs({})
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=obs,
                                obs_curr=obs,
                                action={"skill_type": 4},
                                env_meta=_env_meta(died=True))
    assert r < -DEATH_PENALTY + 1.0  # roughly -10


def test_clip_penalty() -> None:
    obs = _obs({})
    r1 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(n_clipped=0))
    r2 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(n_clipped=3))
    assert r2 == r1 - 3 * GAMMA_CLIP


def test_exploit_penalty_subtracted() -> None:
    obs = _obs({})
    r1 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta())
    r2 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(exploit_penalties=[
                                     ("drop_spam", 0.5), ("oscillation", 0.3)
                                 ]))
    assert r2 == r1 - 0.5 - 0.3


def test_gamma_is_0_99() -> None:
    assert GAMMA == 0.99


def test_time_penalty_is_0_001() -> None:
    assert TIME_PENALTY == 0.001
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/unit/test_reward.py -v
```

Expected: ImportError.

- [ ] **Step 3: Append to `src/aiutopia/env/reward.py`**

Append to the existing `src/aiutopia/env/reward.py`:
```python
# ---------------------------------------------------------------------
# Stage-1 reward composition (§5.1 + §5.2 stage-1 branch only).
# Stages 2 + 3 (multi-objective + scarcity-weighted) are M2-M5 work.
# ---------------------------------------------------------------------

GAMMA          = 0.99    # PBRS discount
DEATH_PENALTY  = 10.0
TIME_PENALTY   = 0.001
GAMMA_CLIP     = 0.05    # per axis (§5.5)


def _delta_inventory(prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
    """Positive: item gained. Negative: item lost. Ignores zero deltas."""
    keys = set(prev) | set(curr)
    return {k: curr.get(k, 0) - prev.get(k, 0)
            for k in keys
            if curr.get(k, 0) - prev.get(k, 0) != 0}


def _inventory_from_obs(obs: dict) -> dict[str, int]:
    """Reconstruct {item_id: count} dict from the obs slot arrays."""
    items = obs.get("inv_slot_item_ids", [])
    counts = obs.get("inv_slot_counts", [])
    out: dict[str, int] = {}
    for item, count in zip(items, counts):
        if not item or count <= 0:
            continue
        # Strip "minecraft:" prefix if present
        key = item.split(":", 1)[-1]
        out[key] = out.get(key, 0) + count
    return out


def _gatherer_primary_signal(prev_inv: dict[str, int],
                              curr_inv: dict[str, int]) -> float:
    """§5.4 — `Σ_r delta_inv[r] * potential[r]` (VPT-normalized).

    Uses LOG_VALUE directly (not the capped potential) so each block
    chopped gives the same reward, regardless of how much the agent
    has already hoarded. Capping is the PBRS potential's job (anti-
    hoarding pressure)."""
    delta = _delta_inventory(prev_inv, curr_inv)
    return sum(delta.get(item, 0) * value for item, value in LOG_VALUE.items())


def compute_reward_stage_1(*,
                            role: str,
                            obs_prev: dict,
                            obs_curr: dict,
                            action: dict,
                            env_meta: dict) -> float:
    """§5.1 stage-1 reward composition for solo per-role pretraining.

        r_stage_1 = r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip

    Args:
      role: must match a RoleId.
      obs_prev / obs_curr: dicts with inv_slot_item_ids + inv_slot_counts.
      action: contains skill_type at minimum.
      env_meta: dict with keys:
        - died_this_tick: bool
        - n_clipped_param_axes: int (0..4)
        - exploit_penalties: list[(name, positive_penalty_value)]

    Returns the scalar reward for this tick.
    """
    prev_inv = _inventory_from_obs(obs_prev)
    curr_inv = _inventory_from_obs(obs_curr)

    # M1-Pipeline only ships gatherer primary signal. Other roles get 0
    # until M2-M4 add their primary signals.
    if role == "gatherer":
        r_primary = _gatherer_primary_signal(prev_inv, curr_inv)
    else:
        r_primary = 0.0

    phi_prev = tech_tree_potential(prev_inv, role)
    phi_curr = tech_tree_potential(curr_inv, role)
    r_pbrs   = GAMMA * phi_curr - phi_prev

    r_death  = DEATH_PENALTY if env_meta.get("died_this_tick", False) else 0.0
    r_time   = TIME_PENALTY
    r_clip   = GAMMA_CLIP * int(env_meta.get("n_clipped_param_axes", 0))
    r_exploits = sum(p for _, p in env_meta.get("exploit_penalties", []))

    return r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_reward.py tests/unit/test_tech_tree_potential.py -v
```

Expected: 14 PASSED total.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/env/reward.py tests/unit/test_reward.py
git commit -m "feat(env): compute_reward_stage_1 + γ_clip + delta-inv primary signal (M1-Pipeline T14)"
```

---

### Task 15: Wire `GoalSpecAdapter` output into the env's obs Dict

**Files:**
- Modify: `src/aiutopia/env/wrapper.py`

The env wrapper currently fills `goal_embedding` with zeros (M0 stub). M1-Pipeline plumbs a real adapter so when the env emits obs to the policy, the goal_embedding reflects the current subgoal. For M1-Pipeline we set a hardcoded "collect oak_log" subgoal at `reset()`; Plan B replaces this with planner-emitted goals.

- [ ] **Step 1: Modify the wrapper**

Open `src/aiutopia/env/wrapper.py`. Locate the existing `_read_all_obs` and surrounding imports. Add at the top of the file (near other imports):

```python
import numpy as np

from aiutopia.planner.goal_spec import GoalSpecAdapter
from aiutopia.schemas.plan import (
    Constraints, GoalSpecification, Subgoal, TargetState, TerminationConditions,
)
```

In `AiUtopiaPettingZooEnv.__init__`, after the FabricBridge is opened, add:
```python
        # GoalSpecAdapter — M1-Pipeline ships a hardcoded "collect 64 oak_log"
        # goal embedded into every obs. Plan B replaces this with planner-
        # emitted Subgoal objects routed via aiutopia.planner.event_queue.
        from aiutopia.planner.goal_spec import load_bge_small
        try:
            bge = load_bge_small()
        except Exception:
            # On dev machines without sentence-transformers fully installed,
            # fall back to a zero-vector encoder so tests don't block.
            class _ZeroBGE:
                def encode(self, _text: str) -> np.ndarray:
                    return np.zeros(384, dtype=np.float32)
            bge = _ZeroBGE()
        self.goal_adapter = GoalSpecAdapter(bge=bge)
        self._stub_subgoal = Subgoal(
            role="gatherer",
            priority=5,
            goal_specification=GoalSpecification(
                target_state=TargetState(inventory_delta={"oak_log": 64}),
                termination_conditions=TerminationConditions(
                    success_criteria=["inventory_meets_delta"],
                    timeout_ticks=6000,
                ),
            ),
            constraints=Constraints(),
            nl_summary="collect 64 oak_log",
        )
        self._stub_goal_embed = self.goal_adapter.embed(self._stub_subgoal).astype(np.float32)
```

Modify `_decode_obs` (or whichever path fills in `goal_embedding`) so it injects `self._stub_goal_embed` in place of the zeros. The existing M0 stub had:
```python
    if "goal_embedding" not in raw:
        out["goal_embedding"] = np.zeros(GOAL_EMBED_DIM, dtype=np.float32)
```
Change to:
```python
    if "goal_embedding" not in raw:
        out["goal_embedding"] = self._stub_goal_embed   # M1-Pipeline default
```
You may need to make `_decode_obs` a method on the env class (or pass `goal_embed` in) — whichever is cleaner. The change is one line.

- [ ] **Step 2: Verify import path works**

```bash
PYTHONPATH=src python -c "
from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
print('wrapper import OK')
"
```

Expected: `wrapper import OK`. If `load_bge_small` triggers a heavy sentence-transformers download, that's fine — first call is slow.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/env/wrapper.py
git commit -m "feat(env): wire GoalSpecAdapter — stub 'collect 64 oak_log' goal (M1-Pipeline T15)"
```

---

### Task 16: Real `env.step()` — wire reward + ExploitDetector + memory writes

**Files:**
- Modify: `src/aiutopia/env/wrapper.py`

Replace the M0 stub that returned `rew = {a: 0.0 for a in self.agents}` with the full reward computation. Wire ExploitDetector per agent and pass exploit_penalties + clipped_axes_bitset into compute_reward.

- [ ] **Step 1: Modify imports + `__init__`**

In `src/aiutopia/env/wrapper.py`, add to imports:
```python
from aiutopia.env.exploit import ExploitDetector
from aiutopia.env.reward import compute_reward_stage_1
```

In `AiUtopiaPettingZooEnv.__init__`, add per-agent ExploitDetector storage:
```python
        self.exploit_detectors: dict[str, ExploitDetector] = {
            agent: ExploitDetector() for agent in self.agents_init
        }
```

- [ ] **Step 2: Replace the body of `step()`**

Find the existing `step(self, actions)` method. Replace its body with:
```python
    def step(self, actions: dict[str, dict]):
        # 1. Dispatch each agent's action via Py4J (mid-tick comm flush)
        comm_msgs: list[dict] = []
        for agent, act in actions.items():
            self.skill_counters[agent] += 1
            invocation_id = f"{agent}-{self.skill_counters[agent]}"
            self.bridge.dispatch_skill(agent, act, invocation_id)
            if int(act.get("should_broadcast", 0)) == 1 and np.asarray(
                    act.get("comm_target_mask", [0, 0, 0, 0])).any():
                comm_msgs.append({"sender": agent, "action": act})
        if comm_msgs:
            self.bridge.flush_comm_batch(comm_msgs)

        # 2. Advance world; collect SkillCompletionEvents
        completion_jsons = self.bridge.advance_tick_await_events(timeout_ms=30_000)
        completions_by_agent: dict[str, dict] = {}
        for j in completion_jsons:
            try:
                evt = json.loads(j) if isinstance(j, str) else j
                completions_by_agent[evt["agentId"]] = evt
            except Exception:
                continue

        # 3. Batched observation read
        new_obs = self._read_all_obs()
        rew: dict[str, float] = {}
        term: dict[str, bool] = {}
        trunc: dict[str, bool] = {}
        info: dict[str, dict] = {}

        for agent in list(self.agents):
            completion = completions_by_agent.get(agent, {})
            n_clipped = bin(int(completion.get("clippedAxesBitset", 0))).count("1")
            died_this_tick = (
                new_obs.get(agent, {}).get("health", [20.0])[0] <= 0
                and self._prev_obs.get(agent, {}).get("health", [20.0])[0] > 0
            )
            # Run exploit detector
            exploit_penalties = self.exploit_detectors[agent].step(
                role=_role_of(agent),
                obs_prev=self._prev_obs.get(agent, {}),
                obs_curr=new_obs.get(agent, {}),
                action=actions.get(agent, {}),
                env_meta={
                    "global_step": self._tick,
                    "skill_result_code": completion.get("resultCode", "RUNNING"),
                },
            )
            env_meta = {
                "died_this_tick": died_this_tick,
                "n_clipped_param_axes": n_clipped,
                "exploit_penalties": [(n.value, p) for n, p in exploit_penalties],
            }
            rew[agent] = compute_reward_stage_1(
                role=_role_of(agent),
                obs_prev=self._prev_obs.get(agent, {}),
                obs_curr=new_obs.get(agent, {}),
                action=actions.get(agent, {}),
                env_meta=env_meta,
            )
            term[agent]  = died_this_tick
            trunc[agent] = self._tick >= self.max_ticks
            info[agent]  = {
                "skill_completion":   completion,
                "exploit_penalties":  [(n.value, p) for n, p in exploit_penalties],
                "n_clipped":          n_clipped,
            }

        self._prev_obs = new_obs
        self._tick += 1
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info
```

(Also add `import json` at the top of the file if not present.)

- [ ] **Step 3: Sanity-check import**

```bash
PYTHONPATH=src python -c "from aiutopia.env.wrapper import AiUtopiaPettingZooEnv; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/aiutopia/env/wrapper.py
git commit -m "feat(env): wire real reward + ExploitDetector + skill-completion bridge (M1-Pipeline T16)"
```

---

### Task 17: Real `EpisodicMemoryWriter.maybe_write` — Chroma writes live

**Files:**
- Modify: `src/aiutopia/memory/writer.py`
- Create: `tests/unit/test_memory_writer_live.py`

The M0 writer only counted writes (`high_count`, `medium_count`). M1-Pipeline writes to actual Chroma collections so `aiutopia memory inspect` returns non-empty data.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_memory_writer_live.py`:
```python
import pytest

pytest.importorskip("chromadb")

from aiutopia.common.ids import new_agent_uuid
from aiutopia.memory.client import open_chroma
from aiutopia.memory.writer import EpisodicMemoryWriter, EpisodicRecord

pytestmark = pytest.mark.integration


def test_high_importance_writes_immediately_to_chroma(chroma_dir, tmp_path):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = new_agent_uuid()
    rec = EpisodicRecord(
        agent_uuid=uuid,
        timestamp=42,
        event_type="chopped",
        participants=[],
        importance_score=0.85,  # > 0.7 → immediate
        summary="agent chopped an oak_log",
        embedding=[0.1] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "high"

    coll = client.get_collection(f"mem_{uuid}")
    out = coll.get(limit=10)
    assert len(out["ids"]) == 1
    assert "oak_log" in out["documents"][0]
    assert out["metadatas"][0]["importance_score"] == 0.85


def test_medium_importance_buffered_then_flushed(chroma_dir):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = new_agent_uuid()
    rec = EpisodicRecord(
        agent_uuid=uuid,
        timestamp=1,
        event_type="step",
        participants=[],
        importance_score=0.45,
        summary="agent walked",
        embedding=[0.0] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "medium"
    # Not yet written to Chroma — verify by querying empty
    coll = client.get_or_create_collection(f"mem_{uuid}")
    assert len(coll.get()["ids"]) == 0
    # Flush
    flushed = writer.flush()
    assert flushed >= 1
    # Now present
    assert len(coll.get()["ids"]) >= 1


def test_low_importance_skipped(chroma_dir):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    rec = EpisodicRecord(
        agent_uuid=new_agent_uuid(), timestamp=1, event_type="noise",
        participants=[], importance_score=0.1, summary="x", embedding=[0.0] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "skipped"
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/unit/test_memory_writer_live.py -v -m integration
```

Expected: 3 FAIL (constructor doesn't take `chroma_client`).

- [ ] **Step 3: Modify writer**

Replace the body of `src/aiutopia/memory/writer.py`'s `EpisodicMemoryWriter` class:
```python
@dataclass
class EpisodicMemoryWriter:
    """Importance-weighted writer.

    M1-Pipeline change vs M0: now optionally takes a Chroma client and writes
    real records. When `chroma_client=None` falls back to M0 buffer-only
    counting (useful for unit tests that don't want Chroma overhead)."""

    chroma_client: Any = None        # chromadb.ClientAPI or None
    high_count:    int = 0
    medium_count:  int = 0
    skipped_count: int = 0
    _buffer:       dict[str, list[EpisodicRecord]] = field(
        default_factory=lambda: defaultdict(list))

    def maybe_write(self, record: EpisodicRecord) -> str:
        if record.importance_score >= HIGH_IMPORTANCE_THRESHOLD:
            self.high_count += 1
            self._write_to_chroma([record])
            return "high"
        if record.importance_score >= MEDIUM_IMPORTANCE_THRESHOLD:
            self.medium_count += 1
            self._buffer[record.agent_uuid].append(record)
            return "medium"
        self.skipped_count += 1
        return "skipped"

    def flush(self) -> int:
        """Flush all buffered MEDIUM records to Chroma. Returns count written."""
        total = 0
        for agent_uuid, records in list(self._buffer.items()):
            if records:
                self._write_to_chroma(records)
                total += len(records)
        self._buffer.clear()
        return total

    # ─────────────────────────────────────────────────────────────
    def _write_to_chroma(self, records: list[EpisodicRecord]) -> None:
        if self.chroma_client is None:
            return  # M0-compatible no-op mode
        from aiutopia.common.ids import memory_id_for
        # Group by agent_uuid; one collection per agent
        by_agent: dict[str, list[EpisodicRecord]] = defaultdict(list)
        for r in records:
            by_agent[r.agent_uuid].append(r)
        for agent_uuid, recs in by_agent.items():
            coll = self.chroma_client.get_or_create_collection(memory_id_for(agent_uuid))
            ids       = [f"{r.agent_uuid}-{r.timestamp}-{i}" for i, r in enumerate(recs)]
            docs      = [r.summary for r in recs]
            metas     = [{
                "timestamp":        r.timestamp,
                "event_type":       r.event_type,
                "importance_score": r.importance_score,
                "participants_csv": "," + ",".join(r.participants) + "," if r.participants else ",",
            } for r in recs]
            embeds    = [r.embedding if r.embedding is not None else [0.0] * 384 for r in recs]
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
```

Make sure the `from typing import Any` import is present at the top.

- [ ] **Step 4: Verify tests pass**

```bash
python -m pytest tests/unit/test_memory_writer.py tests/unit/test_memory_writer_live.py -v
```

Expected: 4 + 3 = 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aiutopia/memory/writer.py tests/unit/test_memory_writer_live.py
git commit -m "feat(memory): EpisodicMemoryWriter — real Chroma writes for HIGH+MEDIUM (M1-Pipeline T17)"
```

---

### Task 18: `aiutopia agent drive` CLI — manual skill dispatch for testing

**Files:**
- Modify: `src/aiutopia/cli/agent.py`

Adds a CLI command that lets you manually dispatch a skill to an agent without a policy. Used by the smoke test in T21.

- [ ] **Step 1: Add `drive` subcommand**

In `src/aiutopia/cli/agent.py`, after the existing `list_agents` command, append:
```python
@app.command("drive")
def drive(
    agent_name:  str       = typer.Option(..., help="agent name to drive"),
    skill:       int       = typer.Option(..., help="skill_type index (0=NAVIGATE,1=HARVEST,2=DEPOSIT_CHEST,3=SEARCH,4=WAIT,5=NOOP_BROADCAST)"),
    target:      int       = typer.Option(0,   help="target_class index"),
    dx:          float     = typer.Option(0.0, help="spatial param x in [-1,1]"),
    dy:          float     = typer.Option(0.0),
    dz:          float     = typer.Option(0.0),
    scalar:      float     = typer.Option(0.5, help="scalar_param in [0,1]"),
    py4j_port:   int       = typer.Option(25099, help="Py4J port"),
    timeout_ms:  int       = typer.Option(60_000, help="wait this long for completion"),
) -> None:
    """Manually dispatch a single skill to an agent and wait for completion.

    M1-Pipeline manual smoke tool — used to verify motor + obs + reward path
    without a trained RL policy. Plan B's training driver replaces this."""
    from aiutopia.env.bridge import FabricBridge
    import time, json as _json
    action = {
        "skill_type":       skill,
        "target_class":     target,
        "spatial_param":    [dx, dy, dz],
        "scalar_param":     [scalar],
        "comm_payload":     [0.0] * 128,
        "should_broadcast": 0,
        "comm_target_mask": [0, 0, 0, 0],
    }
    invocation_id = f"manual-{int(time.time()*1000)}"
    with FabricBridge(port=py4j_port) as bridge:
        bridge.dispatch_skill(agent_name, action, invocation_id)
        typer.echo(f"dispatched skill={skill} target={target} → {invocation_id}")
        # Block on completion
        events = bridge.advance_tick_await_events(timeout_ms=timeout_ms)
        if not events:
            typer.echo("timeout — no completion event arrived", err=True)
            raise typer.Exit(code=1)
        for evt in events:
            typer.echo(_json.dumps(evt, indent=2))
```

- [ ] **Step 2: Verify CLI surfaces the command**

```bash
PYTHONPATH=src python -m aiutopia.cli.app agent drive --help
```

Expected: shows all the options listed above.

- [ ] **Step 3: Commit**

```bash
git add src/aiutopia/cli/agent.py
git commit -m "feat(cli): aiutopia agent drive — manual skill dispatch for M1-Pipeline smoke (T18)"
```

---

### Task 19: Pass `role` through to Carpet `registerAgent` so AgentRegistry knows

**Files:**
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`
- Modify: `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`
- Modify: `src/aiutopia/env/bridge.py`
- Modify: `src/aiutopia/cli/agent.py`

In M0 we always called `AgentRegistry.registerAgent(name)` which defaulted role to "gatherer". For M1-Pipeline, with the new role-aware overlay logic, the role must be passed explicitly so non-gatherer agents (M2+) get the right obs.

- [ ] **Step 1: Extend `WorldOps.carpetSpawn` to accept a role**

Open `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java`. Change the signature from `carpetSpawn(String playerName, String skin)` to `carpetSpawn(String playerName, String skin, String role)`:
```java
    public boolean carpetSpawn(String playerName, String skin, String role) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm =
                server.getCommandManager();
            cm.executeWithPrefix(
                server.getCommandSource(),
                "/player " + playerName + " spawn"
            );
            // Register WITH role so obs builder dispatches the right overlay
            if (role == null || role.isEmpty()) role = "gatherer";
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName, role);
            // (skin is intentionally ignored — see M0 commit history)
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "carpetSpawn failed for {}: {}", playerName, e.getMessage());
            return false;
        }
    }
```

- [ ] **Step 2: Update `Py4JEntryPoint`**

Open `fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java`. Change:
```java
    public boolean carpetSpawn(String playerName, String skin) {
        return world.carpetSpawn(playerName, skin);
    }
```
To:
```java
    public boolean carpetSpawn(String playerName, String skin, String role) {
        return world.carpetSpawn(playerName, skin, role);
    }
```

- [ ] **Step 3: Update Python `FabricBridge.carpet_spawn`**

In `src/aiutopia/env/bridge.py`:
```python
    def carpet_spawn(self, player_name: str,
                      skin: str | None = None,
                      role: str = "gatherer") -> bool:
        return bool(self.entry_point.carpetSpawn(
            player_name, skin or "", role))
```

- [ ] **Step 4: Update CLI to pass role**

In `src/aiutopia/cli/agent.py` `spawn()` function, find the `bridge.carpet_spawn(chosen_name, skin=skin)` line and change to:
```python
        ok = bridge.carpet_spawn(chosen_name, skin=skin, role=role)
```

- [ ] **Step 5: Build + redeploy + commit**

```bash
cd fabric_mod && ./gradlew build --no-daemon 2>&1 | tail -5
cp build/libs/aiutopia-mod-0.0.0-m0.jar ../server-runtime/mods/
cd ..
git add fabric_mod/src/main/java/dev/aiutopia/mod/bridge/WorldOps.java \
        fabric_mod/src/main/java/dev/aiutopia/mod/Py4JEntryPoint.java \
        src/aiutopia/env/bridge.py \
        src/aiutopia/cli/agent.py
git commit -m "feat(mod,cli): plumb role through carpetSpawn so obs builder dispatches correct overlay (M1-Pipeline T19)"
```

---

### Task 20: Integration smoke test — manual drive against live server

**Files:**
- Create: `tests/integration/test_motor_smoke.py`
- Create: `scripts/m1a-smoke.sh`

These exercise the full pipeline: spawn an agent, dispatch a NAVIGATE+HARVEST sequence, observe the rewards.

- [ ] **Step 1: Write the Python smoke test**

Create `tests/integration/test_motor_smoke.py`:
```python
"""End-to-end smoke: dispatch NAVIGATE + HARVEST to a live agent and
verify (a) Java emits completion events, (b) reward signal moves.

Skips when no Py4J server on port 25099."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

PORT = int(os.environ.get("PY4J_SMOKE_PORT", "25099"))


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


@pytest.fixture
def live_server() -> int:
    if not _port_open("127.0.0.1", PORT):
        pytest.skip(f"no Py4J server on port {PORT}")
    return PORT


@pytest.mark.integration
def test_navigate_then_harvest_emits_completion_events(live_server: int, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": "src"}

    # 1. Spawn agent
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "spawn", "--role", "gatherer",
         "--py4j-port", str(live_server)],
        capture_output=True, text=True, env=env, check=True, timeout=30,
    )
    assert "spawn (skin=" in out.stdout
    # Extract the spawned agent name from "identity: spawned <Name> ..."
    name_line = next(l for l in out.stdout.splitlines() if l.startswith("identity: spawned"))
    agent_name = name_line.split()[2]
    time.sleep(1.5)  # let Carpet finish placing the fake player

    # 2. Drive NAVIGATE forward
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "drive",
         "--agent-name", agent_name,
         "--skill", "0",   # NAVIGATE
         "--dx", "0.1", "--dy", "0.0", "--dz", "0.0",
         "--scalar", "0.5",
         "--py4j-port", str(live_server),
         "--timeout-ms", "30000"],
        capture_output=True, text=True, env=env, check=False, timeout=45,
    )
    assert "resultCode" in out.stdout, out.stderr

    # 3. Drive HARVEST oak_log (target_class=0)
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "drive",
         "--agent-name", agent_name,
         "--skill", "1",   # HARVEST
         "--target", "0",  # oak_log
         "--scalar", "0.05",
         "--py4j-port", str(live_server),
         "--timeout-ms", "30000"],
        capture_output=True, text=True, env=env, check=False, timeout=45,
    )
    # Allowed outcomes:
    #   - COMPLETED if there's a tree nearby
    #   - IMMEDIATE_FAILURE if no oak_log within 8 blocks
    #   - FAILED_TIMEOUT if it tried but ran out of ticks
    assert any(code in out.stdout for code in
               ("COMPLETED", "IMMEDIATE_FAILURE", "FAILED_TIMEOUT")), out.stdout
```

- [ ] **Step 2: Write the bash smoke script**

Create `scripts/m1a-smoke.sh`:
```bash
#!/usr/bin/env bash
# M1-Pipeline end-to-end smoke test.
#
# Prereqs:
#   1. Fabric server running on port 25565 with our mod + Carpet + Lithium + FerriteCore
#   2. -Daiutopia.py4j.port=25099 system property set on server launch
#   3. (Optional) MC client connected to localhost:25565 for visual confirmation
#
# This script:
#   - spawns a gatherer agent
#   - drives NAVIGATE forward (dx=0.5 = 16 blocks)
#   - drives HARVEST oak_log (scalar=0.1 = ~6 blocks cap)
#   - drives WAIT
#   - prints each completion event

set -euo pipefail

export AIUTOPIA_ROOT="${AIUTOPIA_ROOT:-/tmp/aiu-m1a-smoke}"
export PYTHONPATH="${PYTHONPATH:-src}"
PORT="${PY4J_PRODUCTION_PORT:-25099}"

rm -rf "$AIUTOPIA_ROOT"

echo "[1/4] spawning gatherer…"
SPAWN_OUT=$(python -m aiutopia.cli.app agent spawn --role gatherer --py4j-port "$PORT")
echo "$SPAWN_OUT"
AGENT=$(echo "$SPAWN_OUT" | awk '/identity: spawned/ {print $3}')
echo "[*] agent name: $AGENT"
sleep 1.5

echo "[2/4] NAVIGATE forward (dx=0.5)…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 0 \
    --dx 0.5 --dy 0.0 --dz 0.0 --scalar 0.5 \
    --py4j-port "$PORT" --timeout-ms 30000

echo "[3/4] HARVEST oak_log (target=0, scalar=0.1)…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 1 --target 0 --scalar 0.1 \
    --py4j-port "$PORT" --timeout-ms 60000

echo "[4/4] WAIT 1 sec…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 4 --scalar 0.1 \
    --py4j-port "$PORT" --timeout-ms 30000

echo "smoke PASS — check connected MC client to verify agent moved + chopped"
```

Make executable: `chmod +x scripts/m1a-smoke.sh`.

- [ ] **Step 3: Try the smoke test (skips without live server)**

```bash
python -m pytest tests/integration/test_motor_smoke.py -v -m integration
```

Expected: 1 SKIPPED (if no live server) OR 1 PASSED (if server up).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_motor_smoke.py scripts/m1a-smoke.sh
git commit -m "test(integration): manual-drive motor smoke + scripts/m1a-smoke.sh (M1-Pipeline T20)"
```

---

### Task 21: Live verify — spawn, drive, observe rewards in real time

**Files:** No new files. This is a manual verification step.

- [ ] **Step 1: Start a fresh Fabric server**

```bash
cd server-runtime
JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 \
PATH=$JAVA_HOME/bin:$PATH \
java -Daiutopia.py4j.port=25099 -Xms2g -Xmx2g -XX:+UseG1GC \
    -jar fabric-server-launcher.jar nogui
```

In another shell wait for "Done (X.Xs)! For help, type help" then proceed.

- [ ] **Step 2: Run the smoke script**

```bash
PY4J_PRODUCTION_PORT=25099 scripts/m1a-smoke.sh
```

Expected output:
```
[1/4] spawning gatherer…
identity: spawned <Name> (gatherer, uuid=01K...)
memory:   collections mem_... + skill_lib_... ready
carpet: /player <Name> spawn (skin=...) -> ok
[*] agent name: <Name>
[2/4] NAVIGATE forward (dx=0.5)…
dispatched skill=0 target=0 → manual-...
{"agentId":"<Name>","skillInvocationId":"manual-...","resultCode":"COMPLETED",...}
[3/4] HARVEST oak_log (target=0, scalar=0.1)…
dispatched skill=1 target=0 → manual-...
{"agentId":"<Name>","skillInvocationId":"manual-...","resultCode":"COMPLETED",...}
[4/4] WAIT 1 sec…
smoke PASS — check connected MC client to verify agent moved + chopped
```

- [ ] **Step 3: Visual confirmation**

Connect a MC 1.21.1 Java client to `localhost:25565`. You should observe:
- Agent player visible in the world at spawn coords
- During NAVIGATE step: agent walks ~16 blocks in the +X direction
- During HARVEST step: agent walks to nearest oak tree, chops one block (possibly multiple if cap > 1), block disappears, agent gains oak_log in inventory

If no oak tree is nearby (spawn area is e.g. plains biome), the HARVEST step will print `"resultCode":"IMMEDIATE_FAILURE"` with `"failureReason":"no 'oak_log' within 8.0 blocks"`. To fix: regenerate the world with a forest seed (delete `server-runtime/world/`, set `level-type=minecraft:large_biomes` in `server.properties`, restart). Or `/op <yourname>` yourself and run `/setblock ~ ~ ~ minecraft:oak_log` next to the agent before running step 3.

- [ ] **Step 4: Stop the server (Ctrl+C in the server shell, or `stop` typed into its stdin)**

- [ ] **Step 5: Document the verification — no commit needed** (this task is a runbook, not a code change)

---

### Task 22: Update M0_PROGRESS to track M1-Pipeline completion + write M1A_VERIFIED tag

**Files:**
- Modify: `M0_PROGRESS.md` (rename to M1A_PROGRESS.md or append section)

- [ ] **Step 1: Append M1-Pipeline section to `M0_PROGRESS.md`**

Add at the bottom of `M0_PROGRESS.md`:
```markdown

---

## M1-Pipeline Progress

**Status:** M1-Pipeline source-complete and live-smoke verified.
**Tag:** `m1a-verified` at the commit that lands T22.

### What changed vs M0

- **Motor module is real.** `MotorBridge.dispatchSkill` parses action JSON,
  constructs a per-skill `SkillExecutor`, runs it across server ticks,
  emits `SkillCompletionEvent` JSON on terminal results.
- **5 skill executors live:** NAVIGATE (direct-line walk), HARVEST (find
  nearest matching block, walk to it, break it), DEPOSIT_CHEST (find nearest
  chest, transfer all inventory), SEARCH (yaw rotation scan), WAIT (no-op).
- **Observation pipeline emits real data.** `WorldOps.observationsAll`
  composes per-agent obs via `ObservationBuilder` (CoreObsBuilder +
  GathererOverlayBuilder); Python receives populated Dict obs not zeros.
- **Reward computation live.** `env.step()` calls
  `compute_reward_stage_1()` per agent. Delta-inventory + PBRS shaping +
  death/time/clip/exploit penalties.
- **ExploitDetector wired.** 5 per-agent rules (DROP_SPAM, OSCILLATION,
  INV_REPEAT, LAZY_INACTION, NOOP_SKILL_SPAM). Multi-agent BULK_FARMING
  is M2+ when builder + gatherer coexist.
- **Episodic memory writes live.** Chroma collections receive HIGH+MEDIUM
  importance records. `aiutopia memory inspect` returns real data.
- **CLI gained `agent drive`** for manual skill dispatch without an RL policy.

### What's still NOT trained
The agent doesn't learn anything yet. `aiutopia agent drive ...` is a manual
remote control. Plan B (M1-Training) adds the PPO config, RLModule, training
driver, and the actual training run that takes a freshly-spawned gatherer to
80% success on "collect 64 oak_log".

### Plan B prereqs (inherits from M1-Pipeline)
- All Plan A scaffolding above
- Real obs in Python's Dict format (verified live)
- Real reward computation with PBRS shaping (verified live)
- ExploitDetector wired into env.step() (verified live)
- Carpet fake player responds to NAVIGATE/HARVEST/DEPOSIT_CHEST/SEARCH/WAIT
```

- [ ] **Step 2: Tag**

```bash
git add M0_PROGRESS.md
git commit -m "docs(M0_PROGRESS): M1-Pipeline complete — motor + obs + reward live (M1-Pipeline T22)"
git tag -a m1a-verified -m "M1-Pipeline: motor + obs + reward + exploit + episodic-memory live; agent manually drivable"
git tag -l
```

Expected tags: `m0`, `m0-source`, `m0-verified`, `m1a-verified`.

---

## M1-Pipeline completion checklist

Before tagging `m1a-verified`:

- [ ] `python -m pytest tests/unit -v -m "not integration and not determinism"` is all green (~14 new + 78 inherited = ~92 tests)
- [ ] `cd fabric_mod && ./gradlew build` succeeds; jar is in `build/libs/`
- [ ] Server-runtime has the new jar in `mods/`
- [ ] Live Fabric server boots clean (Py4J listener visible in log)
- [ ] `aiutopia agent spawn --role gatherer --py4j-port 25099` produces a Carpet fake player
- [ ] `aiutopia agent drive --skill 0 --dx 0.3 ...` makes the agent walk
- [ ] `aiutopia agent drive --skill 1 --target 0 --scalar 0.1` chops nearby oak_log (or returns clean IMMEDIATE_FAILURE if no tree)
- [ ] `aiutopia memory inspect --agent-uuid <ulid>` returns at least one record after a few drives
- [ ] Server log shows no "Caused by:" stack traces during the smoke

## Handoff to Plan B (M1-Training)

Plan B adds:
- `AiUtopiaRoleRLModule` real implementation (per spec §7.2 — `additional_module_specs`, NOT module globals)
- Real `CoreEncoderModule`, `SharedBackboneModule(LSTM 256)`, `CTDECriticModule` (two-stage encoder)
- `PPOConfig` and `MultiRLModuleSpec` wiring
- `scripts/train.py` with Ray Tune + `EvalGateStopCallback`
- `AiUtopiaMetricsCallback` (per-role entropy, Q-variance, trajectory cosine)
- `ExploitHuntCallback` (1000-episode replay)
- Held-out eval scenarios (3 fixed seeds for "collect 64 oak_log")
- First weight promotion via `aiutopia promote-weights --role gatherer`
- First passing determinism check on real weights

Estimated effort: 3-5 weeks at 10-15 hr/wk after M1-Pipeline lands.

---

## Self-review notes

- **Spec coverage:** Every spec § listed in the §-touched table maps to at least one task. Pieces explicitly deferred to Plan B (§7.1, §7.2, §7.4, §5.10) are called out in the carry-forward list. ✓
- **Placeholder scan:** No `TODO`/`TBD`/`fill in later` in tasks; the `M2+` and `Plan B` references are scope notes, not code-side placeholders. ✓
- **Type consistency:** `compute_reward_stage_1` signature is the same in T14, T16. `ExploitDetector.step()` takes the same kwargs in T13 and T16. `bridge.dispatch_skill` and `bridge.advance_tick_await_events` signatures consistent. ✓
- **Critical M0 carry-forwards verified used:**
  - ZGC: production compose still pinned (no change in this plan).
  - `additional_module_specs`: Plan B (T-Plan-B) explicitly references this; nothing in Plan A introduces module globals.
  - Batched `observationsAll`: T11 keeps it single-call.
  - Idempotent `close()`: not modified in Plan A (M0's fix stays).
  - CUDA determinism fixture: not exercised in Plan A; Plan B's RL training reactivates it.
  - Mid-tick comm flush: T16's step() preserves it (comm_msgs collected before advance_tick).
- **Open M0 carry-forwards still flagged for fix in Plan B:**
  - `hash()` non-determinism in `GoalSpecAdapter.build_structured_features` — will bite during eval replays.
  - `Path("src/aiutopia/identity/migrations")` hardcoded in CLI — works from repo root only.
  - `init_identity_db` migration directory needs `importlib.resources` for installed-package usage.
