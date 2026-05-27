package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.block.entity.ChestBlockEntity;
import net.minecraft.entity.MovementType;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.inventory.Inventory;
import net.minecraft.item.ItemStack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.Optional;

/** §4.2 DEPOSIT_CHEST: locate the nearest chest within MAX_CHEST_RADIUS,
 *  walk to it, transfer items from the MAIN_SIZE=36 main inventory slots
 *  (NOT armor / off-hand) to the chest.
 *
 *  M1-Pipeline simplification: deposits everything in the main inventory,
 *  not just items matching target_class. Full filtering is Plan B (or M2
 *  when builder needs specific items kept).
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
            agent.setYaw((float) Math.toDegrees(Math.atan2(-dir.x, dir.z)));
            agent.move(MovementType.SELF, dir.multiply(step));
            return SkillResult.RUNNING;
        }
        ServerWorld world = (ServerWorld) agent.getWorld();
        var be = world.getBlockEntity(chestPos);
        if (!(be instanceof ChestBlockEntity chest)) {
            failureReason = "chest disappeared mid-deposit";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        depositMain(agent, chest);
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

    private static void depositMain(ServerPlayerEntity agent, Inventory chest) {
        // Only main inventory (36 slots) — leaves armor + offhand untouched (R17).
        var pi = agent.getInventory();
        for (int slot = 0; slot < PlayerInventory.MAIN_SIZE; slot++) {
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
