package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.BlockState;
import net.minecraft.block.CropBlock;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;

import static dev.aiutopia.mod.obs.ObsJsonWriter.*;

/**
 * M2 farmer overlay. Emits:
 *   - g_crop_grid: 32×32 single-y-layer, per-cell crop age 0–8 (or 0 if no crop)
 *   - g_ripeness: scalar 0–1, fraction of crops in grid at age 8
 *
 * Used by the farmer role to perceive field state and target harvest/plant actions.
 */
public final class FarmerOverlayBuilder {

    private static final int GRID_RADIUS = 16;  // 32×32 grid centered on agent

    public void populate(JsonObject obs, ServerPlayerEntity agent, MinecraftServer server) {
        ServerWorld world = (ServerWorld) agent.getWorld();
        BlockPos origin = agent.getBlockPos();

        // 1. Scan 32×32 grid at agent's y-level (± 0 for crops on the same level)
        int gridSize = 2 * GRID_RADIUS;
        float[][] grid = new float[gridSize][gridSize];
        int totalCrops = 0;
        int ripeCrops = 0;

        for (int dx = -GRID_RADIUS; dx < GRID_RADIUS; dx++) {
            for (int dz = -GRID_RADIUS; dz < GRID_RADIUS; dz++) {
                BlockPos p = origin.add(dx, 0, dz);
                BlockState s = world.getBlockState(p);
                String id = Registries.BLOCK.getId(s.getBlock()).toString();

                // Check if this is a crop block (wheat, carrot, potato, etc.)
                if (isCropBlock(id)) {
                    // Extract age (default CropBlock.AGE property, 0-7 for wheat)
                    int age = s.getProperties().contains(CropBlock.AGE)
                        ? s.get(CropBlock.AGE)
                        : 0;
                    grid[dx + GRID_RADIUS][dz + GRID_RADIUS] = (float) age;
                    totalCrops++;
                    if (age >= 7) {  // ripeness threshold matches HarvestCropSkill
                        ripeCrops++;
                    }
                } else {
                    // No crop or non-crop block: age 0
                    grid[dx + GRID_RADIUS][dz + GRID_RADIUS] = 0.0f;
                }
            }
        }

        // 2. Pack g_crop_grid as flat JSON (32 × 32 = 1024 floats)
        JsonArray gridArr = new JsonArray(gridSize * gridSize);
        for (int x = 0; x < gridSize; x++) {
            for (int z = 0; z < gridSize; z++) {
                gridArr.add(grid[x][z]);
            }
        }
        obs.add("g_crop_grid", gridArr);

        // 3. g_ripeness: fraction of crops at age 8 (max age for wheat/carrot/potato)
        double ripeness = totalCrops > 0 ? (double) ripeCrops / totalCrops : 0.0;
        obs.addProperty("g_ripeness", Math.min(1.0, ripeness));
    }

    private static boolean isCropBlock(String blockId) {
        return blockId.contains("wheat") ||
               blockId.contains("carrot") ||
               blockId.contains("potato") ||
               blockId.contains("beetroot") ||
               blockId.contains("torchflower");
    }
}
