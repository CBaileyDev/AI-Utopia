package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

import static dev.aiutopia.mod.obs.ObsJsonWriter.*;

/** §4.1 universal core obs.
 *
 *  Type contract with Python (R2 fix):
 *   - Scalar Box((1,)) gym fields (health, hunger, saturation, armor_value,
 *     tick_in_episode, goal_ticks_left, time_of_day, light_level) are emitted
 *     as 1-element JSON arrays so Python's np.asarray gives shape (1,).
 *   - Discrete / MultiDiscrete fields (biome_id, main_hand_item_id,
 *     off_hand_item_id, inv_slot_item_ids) emit raw integer registry IDs,
 *     NOT registry-name strings. spaces.py expects ints in [0, N_ITEMS=2048)
 *     and [0, N_BIOMES=64); ints come from ItemIdTable (contiguous remap, NOT
 *     `rawId & 0x3FF`), exposed to Python via Py4JEntryPoint#getItemIdNameTable;
 *     biome IDs are
 *     mapped via biomeRawId() (the dynamic registry is per-world).
 *   - String fields agent_uuid + agent_name + role_id are auxiliary —
 *     Python's _normalize_raw uses them to derive role_one_hot and
 *     agent_uuid_embed and then strips them.
 *
 *  goal_embedding comes from Python's GoalSpecAdapter — not emitted here. */
public final class CoreObsBuilder {

    /** Single source of truth for the obs-side item-id encoding.
     *
     *  N9: the M1A code used `Registries.ITEM.getRawId(item) & 0x3FF`, which
     *  is catastrophic — vanilla MC 1.21.1 has ~1300 items, so the mask
     *  silently overwrote oak_log / stone / cobblestone / every basic block
     *  with `base+1024` items (spawn eggs). Agents literally could not see
     *  oak_log in their inventory regardless of the reward path.
     *
     *  The fix: a contiguous remap via {@link ItemIdTable}, exported to
     *  Python via {@link dev.aiutopia.mod.Py4JEntryPoint#getItemIdNameTable()}
     *  so reward.py can translate obs ints back to LOG_VALUE keys. */
    public static int maskedItemId(Item item) {
        return ItemIdTable.get().idOf(item);
    }

    public void populate(JsonObject obs, ServerPlayerEntity agent, MinecraftServer server) {
        // Auxiliary string fields — used by Python's _normalize_raw to derive
        // role_one_hot + agent_uuid_embed, then dropped before gym validation.
        obs.addProperty("agent_uuid", agent.getUuidAsString());
        obs.addProperty("agent_name", agent.getGameProfile().getName());
        obs.addProperty("role_id",
            dev.aiutopia.mod.agent.AgentRegistry.roleOf(agent.getGameProfile().getName()));

        long worldTime = server.getOverworld().getTime();
        obs.add("tick_in_episode", vec1(worldTime % 24_000L));   // Box((1,))

        var pos = agent.getPos();
        obs.add("position", vec(pos.x, pos.y, pos.z));
        var vel = agent.getVelocity();
        obs.add("velocity", vec(vel.x, vel.y, vel.z));
        obs.add("yaw_pitch", vec2(agent.getYaw(), agent.getPitch()));

        // R2: wrap scalar stats in 1-element arrays for Box((1,)) compatibility.
        obs.add("health",      vec1(agent.getHealth()));
        obs.add("hunger",      vec1(agent.getHungerManager().getFoodLevel()));
        obs.add("saturation",  vec1(agent.getHungerManager().getSaturationLevel()));
        obs.add("armor_value", vec1(agent.getArmor()));

        // Inventory — 36 slots, raw int item id (NOT registry-name string).
        // Registries.ITEM.getRawId returns a stable per-MC-version int.
        var inv = agent.getInventory();
        JsonArray itemIds = new JsonArray();
        JsonArray counts  = new JsonArray();
        for (int i = 0; i < 36; i++) {
            ItemStack s = inv.getStack(i);
            // Clamp to N_ITEMS=1024 (spaces.py). Modded item IDs may exceed
            // 1024 in a heavily-modded server; for the M1A vanilla baseline
            // we modulo as a safety net so we never blow the Discrete bound.
            itemIds.add(maskedItemId(s.getItem()));
            counts.add(s.getCount());
        }
        obs.add("inv_slot_item_ids", itemIds);
        obs.add("inv_slot_counts",   counts);
        obs.addProperty("main_hand_item_id",
            maskedItemId(agent.getMainHandStack().getItem()));
        obs.addProperty("off_hand_item_id",
            maskedItemId(agent.getOffHandStack().getItem()));

        // goal_ticks_left default 0 — Python's _normalize_raw replaces this with
        // the real ticks-left from the current Subgoal stub.
        obs.add("goal_ticks_left", vec1(0));

        // World state
        obs.add("time_of_day",
            vec1(server.getOverworld().getTimeOfDay() % 24_000L));   // Box((1,))
        boolean raining   = server.getOverworld().isRaining();
        boolean thundering = server.getOverworld().isThundering();
        obs.addProperty("weather", thundering ? 2 : (raining ? 1 : 0));   // Discrete(3) — bare int OK
        obs.addProperty("biome_id", biomeRawId(server, agent));            // Discrete(64) — bare int OK
        obs.add("light_level",
            vec1(server.getOverworld().getLightLevel(agent.getBlockPos())));   // Box((1,))

        // comm_payloads + comm_metadata are placed by Python from the CommBus
        // ring buffer — we don't emit them here.
        // action_mask is computed in Python from the symbolic obs above.
    }

    /** R3 fix: biomes are a dynamic registry in MC 1.21.1. Resolve the
     *  RegistryEntry → RegistryKey → Identifier → stable int.
     *  Modulo into [0, 64) to fit the Discrete(N_BIOMES) gym space. Falls
     *  back to bucket 0 if the biome has no key (datapack edge case). */
    private static int biomeRawId(MinecraftServer server, ServerPlayerEntity agent) {
        var entry = server.getOverworld().getBiome(agent.getBlockPos());
        return entry.getKey()
            .map(k -> Math.abs(k.getValue().toString().hashCode()) & 0x3F)
            .orElse(0);
    }
}
