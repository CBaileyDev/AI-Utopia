package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.BlockState;
import net.minecraft.block.entity.ChestBlockEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import static dev.aiutopia.mod.obs.ObsJsonWriter.*;

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
                // NATURAL-TERRAIN FIX: scan top-down for the TOPMOST LOG (channel 0)
                // within ±3 y of agent. Skip air, leaves, and any non-log block —
                // do NOT break on the first non-air. This makes a log under a leaf
                // canopy visible (the obs scan now matches HARVEST's log-specific
                // search). On the bare training arena the topmost block in-band IS
                // the log, so this is byte-identical to the old topmost-non-air scan
                // (golden trace / transfer gate unaffected). Mirrors the sim's
                // gatherer_nearest_columns (topmost log per column within ±3).
                for (int dy = 3; dy >= -3; dy--) {
                    BlockPos p = origin.add(dx, dy, dz);
                    BlockState s = world.getBlockState(p);
                    if (s.isAir()) continue;
                    String id = Registries.BLOCK.getId(s.getBlock()).toString();
                    Integer channel = matchChannel(id);
                    if (channel == null || channel != 0) {
                        continue;  // not a log (leaf / other block) — keep scanning down
                    }
                    grid[dx + GRID_RADIUS][dz + GRID_RADIUS][channel] = 1.0f;
                    if (Math.sqrt(dx*dx + dy*dy + dz*dz) <= SCAN_RADIUS) {
                        nearby.add(new NearbyResource(dx, dy, dz, id));
                        totalCount++;
                    }
                    break;  // only the topmost LOG per (dx, dz)
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

        // 6. R4: nearest-resource + nearest-chest distances. Python's
        // _normalize_raw turns these into target_resource_in_range +
        // target_chest_in_range booleans for action_mask.py.
        //
        // SENTINEL_NO_TARGET (999.0) is used when no target was found —
        // larger than any plausible REACH_RADIUS, so the derived boolean
        // is unambiguously False.
        double nearestResDist = nearby.isEmpty()
            ? SENTINEL_NO_TARGET
            : Math.sqrt(nearby.get(0).distSq());
        obs.add("nearest_resource_distance", vec1(nearestResDist));
        obs.add("nearest_chest_distance",
            vec1(findNearestChestDistance(world, origin)));
    }

    private static final double SENTINEL_NO_TARGET = 999.0;
    private static final int    CHEST_SCAN_RADIUS  = 8;

    private static double findNearestChestDistance(ServerWorld world, BlockPos origin) {
        double best = SENTINEL_NO_TARGET;
        for (int dx = -CHEST_SCAN_RADIUS; dx <= CHEST_SCAN_RADIUS; dx++) {
            for (int dy = -CHEST_SCAN_RADIUS; dy <= CHEST_SCAN_RADIUS; dy++) {
                for (int dz = -CHEST_SCAN_RADIUS; dz <= CHEST_SCAN_RADIUS; dz++) {
                    BlockPos p = origin.add(dx, dy, dz);
                    if (world.getBlockEntity(p) instanceof ChestBlockEntity) {
                        double d = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        if (d < best) best = d;
                    }
                }
            }
        }
        return best;
    }

    private static Integer matchChannel(String blockId) {
        for (var e : CHANNEL.entrySet()) {
            if (blockId.contains(e.getKey())) return e.getValue();
        }
        return null;
    }

    private static double typeIdForEntity(net.minecraft.entity.Entity e) {
        // R12: bitwise mask the sign bit instead of Math.abs.
        // Math.abs(Integer.MIN_VALUE) == Integer.MIN_VALUE (still negative)
        // would produce a negative type_id; Python expects [0, 1].
        String id = Registries.ENTITY_TYPE.getId(e.getType()).toString();
        return ((id.hashCode() & 0x7FFFFFFF) % 256) / 256.0;
    }

    private record NearbyResource(int dx, int dy, int dz, String id) {
        double distSq() { return dx*dx + dy*dy + dz*dz; }
    }
}
