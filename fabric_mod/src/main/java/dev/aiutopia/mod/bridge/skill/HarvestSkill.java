package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.entity.MovementType;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.event.GameEvent;

import java.util.List;
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
 *  MAX_SEARCH_RADIUS = 48 (N21) is DELIBERATELY DECOUPLED from (and wider than)
 *  the gatherer obs scan radius of 16. The policy only PERCEIVES resources to 16
 *  blocks, but HARVEST chains internally (re-scan + walk to the next-nearest log
 *  as it runs), so it must be able to FIND logs the policy can't see — otherwise,
 *  once the agent clears the near cluster and the tail strands >16 blocks from its
 *  rest position, findNearest returns empty and chaining halts (the seed-3 55/64
 *  stall). 48 ≈ the 33×33 arena diagonal, so chaining can always reach the next
 *  log from any rest position. (Was 16, which incorrectly mirrored the obs scan.)
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

    private static final double MAX_SEARCH_RADIUS = 48.0;
    // N16b: with REACH=3.0, the collision attractor at dist≈3.0+ε still
    // pinned the agent (`step = min(WALK, dist-3.0)` shrunk to ~2e-4 b/tick
    // because the agent's bbox tangents the log column at horizontal 0.8
    // from log center, giving 3D dist = sqrt(0.64+2.25+remaining_dz²) that
    // floats just above 3.0 for the second-nearest log of the ring).
    // Bumped 3.0 -> 4.5 (vanilla creative reach is ~5 blocks — bedrock
    // creative players can break a log from 4.5 away). Combined with the
    // `step = WALK_PER_TICK` (no-shrink) fix below, this eliminates the
    // attractor entirely: agent always walks full speed and vanilla AABB
    // collision stops it at the right horizontal position; reach check
    // then succeeds because 4.5 generously covers any collision-pinned pos.
    private static final double REACH_RADIUS      = 4.5;
    private static final double REACH_RADIUS_SQ   = REACH_RADIUS * REACH_RADIUS;
    private static final double WALK_PER_TICK     = 4.3 / 20.0;
    private static final int    MAX_QUANTITY      = 64;

    // N16: no-progress watchdog. If the agent fails to make detectable
    // progress toward currentTarget for this many ticks while currentTarget
    // is set, abandon it and rescan. If findNearest returns the same block
    // we just abandoned, return FAILED_TIMEOUT with a clear reason.
    private static final int    STALL_TICK_BUDGET = 20;
    private static final double STALL_DIST_EPSILON_SQ = 1e-4;

    private String targetSubstr;
    private int    cap;
    private int    brokenCount = 0;
    private BlockPos currentTarget;
    private BlockPos lastAbandonedTarget;     // N16
    private Vec3d  lastPos;                   // N16
    private int    stuckTicks;                // N16
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
            BlockPos cand = found.get();
            // N16: if findNearest picks the same block we just abandoned for
            // being unreachable, don't loop on it forever — fail clearly.
            if (lastAbandonedTarget != null && cand.equals(lastAbandonedTarget)) {
                failureReason = "harvest stalled — only reachable target ("
                    + cand + ") cannot be approached within REACH_RADIUS="
                    + REACH_RADIUS;
                return SkillResult.FAILED_TIMEOUT;
            }
            currentTarget = cand;
            lastPos       = agent.getPos();
            stuckTicks    = 0;
        }
        Vec3d targetCenter = Vec3d.ofCenter(currentTarget);
        Vec3d here = agent.getPos();
        // N16: use squared distance with a small epsilon so float-precision
        // edges (dist == 2.0 + 1 ULP from the geometric attractor where the
        // agent's bbox tangents the log's column) don't trigger the
        // shrinking-step infinite stall.
        double distSq = here.squaredDistanceTo(targetCenter);
        if (distSq > REACH_RADIUS_SQ + 1e-3) {
            // N16: no-progress watchdog. If pos hasn't changed measurably
            // since last tick, count toward stall budget; on budget exhaust,
            // abandon this target and rescan (lastAbandonedTarget prevents
            // immediate re-selection on the next tick).
            if (lastPos != null && here.squaredDistanceTo(lastPos) < STALL_DIST_EPSILON_SQ) {
                stuckTicks++;
                if (stuckTicks >= STALL_TICK_BUDGET) {
                    lastAbandonedTarget = currentTarget;
                    currentTarget = null;
                    stuckTicks = 0;
                    return SkillResult.RUNNING;
                }
            } else {
                stuckTicks = 0;
            }
            lastPos = here;
            // N16b: ALWAYS use full WALK_PER_TICK. The old shrinking-step
            // formula `step = min(WALK, dist - REACH_RADIUS)` produces
            // sub-1e-3 b/tick steps whenever collision pins the agent at
            // dist ≈ REACH_RADIUS + ε — re-creating the precision attractor
            // bug at any radius value. With full walk speed, vanilla AABB
            // collision stops the agent at the natural geometric limit;
            // overshoot is impossible because the *next* tick's reach
            // check sees dist <= REACH_RADIUS and enters the break branch.
            Vec3d dir = targetCenter.subtract(here).normalize();
            float yaw = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
            agent.setYaw(yaw);
            agent.move(MovementType.SELF, dir.multiply(WALK_PER_TICK));
            return SkillResult.RUNNING;
        }
        // Within reach — break the block and DIRECTLY transfer drops into
        // the agent's inventory. The old `world.breakBlock(pos, true, agent)`
        // path relied on the Carpet fake player auto-pickup'ing the ItemEntity
        // that breakBlock spawns at the block's center — but with the bumped
        // REACH_RADIUS=4.5 (needed to escape the float-precision attractor),
        // the agent stops 3-4 blocks AWAY from the log when it breaks, well
        // outside vanilla's ~1.5b auto-pickup radius. Result: drops spawn,
        // never get picked up, despawn after 100s, and reward never fires
        // because `_inventory_from_obs` stays empty. The N16c fix bypasses
        // entity-form drops entirely: compute drops via loot table, insert
        // directly into the agent's inventory, then air-out the block.
        BlockState state = world.getBlockState(currentTarget);
        if (!Registries.BLOCK.getId(state.getBlock()).toString().contains(targetSubstr)) {
            // Target changed under us (already broken, race with another
            // agent, etc.). Clear and re-scan.
            currentTarget = null;
            return SkillResult.RUNNING;
        }
        // 1. Compute drops via the block's loot table (oak_log → 1 oak_log).
        //    breakingEntity=agent lets loot tables that key on entity (eg
        //    looting/silk-touch) work correctly when tools are added later.
        List<ItemStack> drops = Block.getDroppedStacks(
            state, world, currentTarget,
            state.hasBlockEntity() ? world.getBlockEntity(currentTarget) : null,
            agent, ItemStack.EMPTY);
        // 2. Insert each stack into the agent's inventory. offerOrDrop is
        //    the right method: it tries to merge into existing stacks first,
        //    fills empty slots, and only spawns an ItemEntity for excess
        //    that won't fit (i.e. only if the inventory is full).
        for (ItemStack drop : drops) {
            agent.getInventory().offerOrDrop(drop);
        }
        agent.getInventory().markDirty();
        // 3. Replace the block with air + emit the standard break event so
        //    listeners (particle/sound, sculk sensors, etc.) still fire.
        world.setBlockState(currentTarget, Blocks.AIR.getDefaultState(),
            Block.NOTIFY_ALL);
        world.syncWorldEvent(2001, currentTarget, Block.getRawIdFromState(state));
        world.emitGameEvent(agent, GameEvent.BLOCK_DESTROY, currentTarget);
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
