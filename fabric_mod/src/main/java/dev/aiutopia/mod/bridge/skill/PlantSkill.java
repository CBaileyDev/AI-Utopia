package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.entity.MovementType;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.event.GameEvent;

/**
 * M2 PLANT: plant a seed on tilled soil.
 *
 * Params: target_location (3-tuple [x, y, z]), seed_type (int, 0=wheat_seeds default).
 * Success: CROPS block placed at age 1.
 *
 * Sequencing per tick:
 *   1. Parse target location from action JSON.
 *   2. If target not in range (>4.5 blocks), walk toward it.
 *   3. If target is Farmland (age ≥ 0) and in reach, place CROPS (age 1).
 *   4. COMPLETED on success; IMMEDIATE_FAILURE if target is not farmland.
 */
public class PlantSkill implements SkillExecutor {

    private static final double REACH_RADIUS   = 4.5;
    private static final double REACH_RADIUS_SQ = REACH_RADIUS * REACH_RADIUS;
    private static final double WALK_PER_TICK  = 4.3 / 20.0;
    private static final int    STALL_TICK_BUDGET = 20;
    private static final double STALL_DIST_EPSILON_SQ = 1e-4;

    private BlockPos targetPos;
    private Vec3d lastPos;
    private int stuckTicks;
    private long ticksRemaining;
    private int clipped;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        // Parse target_location: expected as a 3-element array [x, y, z]
        if (!action.has("target_location")) {
            failureReason = "PLANT requires target_location array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        var arr = action.getAsJsonArray("target_location");
        if (arr.size() != 3) {
            failureReason = "target_location must be length-3 array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        int x = arr.get(0).getAsInt();
        int y = arr.get(1).getAsInt();
        int z = arr.get(2).getAsInt();
        this.targetPos = new BlockPos(x, y, z);
        this.lastPos = agent.getPos();
        this.stuckTicks = 0;
        this.ticksRemaining = action.has("timeout_ticks")
            ? action.get("timeout_ticks").getAsLong()
            : 600L;
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        if (--ticksRemaining <= 0) {
            failureReason = "plant timeout";
            return SkillResult.FAILED_TIMEOUT;
        }
        ServerWorld world = (ServerWorld) agent.getWorld();
        Vec3d targetCenter = Vec3d.ofCenter(targetPos);
        Vec3d here = agent.getPos();
        double distSq = here.squaredDistanceTo(targetCenter);

        // If target is not in reach, walk toward it
        if (distSq > REACH_RADIUS_SQ + 1e-3) {
            // Stall detection
            if (lastPos != null && here.squaredDistanceTo(lastPos) < STALL_DIST_EPSILON_SQ) {
                stuckTicks++;
                if (stuckTicks >= STALL_TICK_BUDGET) {
                    failureReason = "plant stalled — cannot reach target";
                    return SkillResult.FAILED_TIMEOUT;
                }
            } else {
                stuckTicks = 0;
            }
            lastPos = here;

            // Walk toward target
            Vec3d dir = targetCenter.subtract(here).normalize();
            float yaw = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
            agent.setYaw(yaw);
            agent.move(MovementType.SELF, dir.multiply(WALK_PER_TICK));
            return SkillResult.RUNNING;
        }

        // Within reach — check if block is Farmland
        BlockState state = world.getBlockState(targetPos);
        String blockId = Registries.BLOCK.getId(state.getBlock()).toString();
        if (!blockId.equals("farmland")) {
            failureReason = "target block is not farmland: " + blockId;
            return SkillResult.IMMEDIATE_FAILURE;
        }

        // Plant CROPS (wheat) at age 1 (newly planted)
        BlockState crop = Blocks.WHEAT.getDefaultState()
            .with(net.minecraft.block.CropBlock.AGE, 1);
        world.setBlockState(targetPos, crop, Block.NOTIFY_ALL);
        world.emitGameEvent(agent, GameEvent.BLOCK_PLACE, targetPos);

        return SkillResult.COMPLETED;
    }

    @Override public int clippedAxes()     { return clipped; }
    @Override public String failureReason() { return failureReason; }
}
