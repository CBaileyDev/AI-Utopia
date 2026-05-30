package dev.aiutopia.mod.obs;

import com.google.gson.JsonObject;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.server.network.ServerPlayerEntity;

/**
 * Explorer role observations: biome, richness score, nearest resources.
 *
 * Phase 2b: Minimal stub. Returns exploration-specific obs.
 * Phase 2.5+: Add richness computation, resource detection.
 */
public class ExplorerOverlayBuilder {

    public void populate(JsonObject obs, ServerPlayerEntity player,
                         net.minecraft.server.MinecraftServer server) {
        // Phase 2b stub: minimal required fields for Explorer obs space
        // Phase 2.5+ will add richness computation, resource detection, hostiles

        // Richness score (placeholder: always 0)
        obs.addProperty("g_richness_score", 0.0f);

        // Nearest resources (placeholder: empty array)
        obs.add("g_nearest_resources", new com.google.gson.JsonArray());

        // Hostiles nearby (placeholder: 0 - Phase 2.5+ will scan)
        obs.addProperty("g_hostiles_nearby", 0);
    }
}
