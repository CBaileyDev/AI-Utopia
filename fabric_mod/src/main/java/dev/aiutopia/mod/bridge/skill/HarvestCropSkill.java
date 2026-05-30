package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.block.CropBlock;
import net.minecraft.entity.MovementType;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.event.GameEvent;

import java.util.List;
import java.util.Optional;

/**
 * M2 HARVEST_CROP: break a crop at ripeness (age 7–8).
 *
 * Params: target_location (3-tuple [x, y, z]).
 * Success: crop removed at age 7+, seeds + food dropped and inserted into inventory.
 *
 * Sequencing per tick:
 *   1. Parse target location from action JSON.
 *   2. If target not in range (>4.5 blocks), walk toward it.
 *   3. If target is a CROPS block at age 7+, harvest (loot table drops).
 *   4. COMPLETED on harvest; IMMEDIATE_FAILURE if crop is too young (age < 7).
 *
 * Uses the same direct-drop-to-inventory pattern as HarvestSkill (N16c)
 * to avoid item-entity race conditions and ensure deterministic reward
 * signal firing when inventory updates.
 */
public class HarvestCropSkill implements SkillExecutor {

    private static final double REACH_RADIUS   = 4.5;
    private static final double REACH_RADIUS_SQ = REACH_RADIUS * REACH_RADIUS;
    private static final double WALK_PER_TICK  = 4.3 / 20.0;
    private static final int    STALL_TICK_BUDGET = 20;
    private static final double STALL_DIST_EPSILON_SQ = 1e-4;
    private static final int    MIN_CROP_AGE    = 7;  // ripeness threshold

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
            failureReason = "HARVEST_CROP requires target_location array";
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
            failureReason = "harvest_crop timeout";
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
                    failureReason = "harvest_crop stalled — cannot reach target";
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

        // Within reach — check if block is a ripe crop
        BlockState state = world.getBlockState(targetPos);
        String blockId = Registries.BLOCK.getId(state.getBlock()).toString();
        if (!blockId.contains("wheat") && !blockId.contains("carrot") && !blockId.contains("potato")) {
            failureReason = "target block is not a crop: " + blockId;
            return SkillResult.IMMEDIATE_FAILURE;
        }

        // Check crop age (use CropBlock.AGE property, default 0-7 for wheat)
        if (!state.getProperties().contains(CropBlock.AGE)) {
            failureReason = "target crop has no AGE property";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        int age = state.get(CropBlock.AGE);
        if (age < MIN_CROP_AGE) {
            failureReason = "crop too young (age " + age + ", min " + MIN_CROP_AGE + ")";
            return SkillResult.IMMEDIATE_FAILURE;
        }

        // Harvest: compute drops via loot table and insert into inventory
        var drops = Block.getDroppedStacks(
            state, world, targetPos,
            state.hasBlockEntity() ? world.getBlockEntity(targetPos) : null,
            agent, net.minecraft.item.ItemStack.EMPTY);

        for (var drop : drops) {
            agent.getInventory().offerOrDrop(drop);
        }
        agent.getInventory().markDirty();

        // Replace with air
        world.setBlockState(targetPos, Blocks.AIR.getDefaultState(), Block.NOTIFY_ALL);
        world.syncWorldEvent(2001, targetPos, Block.getRawIdFromState(state));
        world.emitGameEvent(agent, GameEvent.BLOCK_DESTROY, targetPos);

        return SkillResult.COMPLETED;
    }

    @Override public int clippedAxes()     { return clipped; }
    @Override public String failureReason() { return failureReason; }
}
