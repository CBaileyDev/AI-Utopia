package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.BlockState;
import net.minecraft.entity.MovementType;
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
 *  SEARCH_RADIUS = 16 to match the gatherer obs scan radius — the policy
 *  sees resources to 16 blocks; harvest should be able to reach the same.
 *
 *  Sequencing per tick:
 *    1. If no current target block, scan radius for a match. None → COMPLETED if
 *       we've broken at least one block this dispatch, else IMMEDIATE_FAILURE.
 *    2. If target found but >2 blocks away, move toward it via agent.move(SELF, ...).
 *    3. If within 2 blocks, break it (1 tick), increment broken-count,
 *       clear target. Loop to step 1.
 *    4. If broken-count >= cap, COMPLETED.
 *
 *  Note on item pickup: world.breakBlock drops items as entities; the
 *  Carpet fake player picks them up over the next 1-2 ticks, so inventory
 *  delta (and reward signal) may lag the COMPLETED event by one tick.
 *  Acceptable for M1A; for strict-deterministic Plan B replays consider
 *  using Block.dropStacks(...) + direct inventory insertion.
 */
public class HarvestSkill implements SkillExecutor {

    private static final double MAX_SEARCH_RADIUS = 16.0;
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

        // R15: accept scalar OR 1-element array for scalar_param. Default = 1 block.
        double scalar = NavigateSkill.readScalar(action, "scalar_param", 1.0 / MAX_QUANTITY);
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
            return SkillResult.FAILED_TIMEOUT;
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
            // Move toward it via vanilla collision-aware movement
            Vec3d dir = targetCenter.subtract(here).normalize();
            double step = Math.min(WALK_PER_TICK, dist - REACH_RADIUS);
            float yaw = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
            agent.setYaw(yaw);
            agent.move(MovementType.SELF, dir.multiply(step));
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
        // Two-pass search: prefer ground-reachable matches (dy ∈ [-2, +1])
        // to avoid stalling on canopy logs the agent can't jump to. If no
        // ground-level match is in range, fall back to the full vertical range
        // (caller treats that as a request to give up after `brokenCount > 0`).
        Optional<BlockPos> ground = scanShell(world, origin, substr, -2, 1);
        if (ground.isPresent()) return ground;
        int radius = (int) Math.ceil(MAX_SEARCH_RADIUS);
        return scanShell(world, origin, substr, -radius, radius);
    }

    private static Optional<BlockPos> scanShell(ServerWorld world, BlockPos origin,
                                                  String substr, int dyMin, int dyMax) {
        int radius = (int) Math.ceil(MAX_SEARCH_RADIUS);
        BlockPos best = null;
        double   bestDist = Double.MAX_VALUE;
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = dyMin; dy <= dyMax; dy++) {
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
