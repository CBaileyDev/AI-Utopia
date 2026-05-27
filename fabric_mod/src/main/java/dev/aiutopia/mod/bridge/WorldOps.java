package dev.aiutopia.mod.bridge;

import net.minecraft.server.MinecraftServer;

/** §7.3 world inspection + reset operations. M0 stubs return an empty JSON object. */
public class WorldOps {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

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

    /** Reset world to the given seed. M0 stub. */
    public void resetWorld(long seed) {
        // TODO M1: kill all entities, regenerate world from seed.
    }

    /** Spawn a Carpet fake player. Returns true on success.
     *  Requires Carpet to be installed on the running server.
     *
     *  If `skin` is non-null and non-empty, follows the spawn with
     *  `/player <name> loadProfile <skin>` so the agent appears with
     *  the chosen skin instead of the default. Failure to apply the
     *  skin is logged but does NOT fail the overall spawn — the
     *  agent still exists, just with the default skin. */
    public boolean carpetSpawn(String playerName, String skin, String role) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm =
                server.getCommandManager();
            // MC 1.21.1 `CommandManager.executeWithPrefix(ServerCommandSource, String)`
            // returns void. Brigadier exceptions propagate as RuntimeException; a
            // failed command (e.g. Carpet not installed) silently no-ops. We trust
            // success if no exception fires and register the agent name.
            cm.executeWithPrefix(
                server.getCommandSource(),
                "/player " + playerName + " spawn"
            );
            // Register WITH role so obs builder dispatches the right overlay
            if (role == null || role.isEmpty()) role = "gatherer";
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName, role);

            // NOTE: Carpet 1.4.147 (MC 1.21.1) does not expose a /player <name>
            // loadProfile <skin> subcommand. The fake player's skin is derived
            // from Mojang's account lookup for `playerName` itself when the
            // server is in online-mode. In offline-mode (our smoke setup) the
            // skin shows as default Steve/Alex. M5+ may revisit by switching
            // to a custom entity type with explicit GameProfile assignment.
            // The `skin` parameter is currently ignored on the Carpet path;
            // kept in the method signature for forward compatibility.
            if (skin != null && !skin.isEmpty()) {
                dev.aiutopia.mod.AiUtopiaMod.LOG.debug(
                    "skin '{}' ignored (Carpet 1.4.x has no loadProfile); "
                    + "use online-mode + matching MC account for skin",
                    skin);
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "carpetSpawn failed for {}: {}", playerName, e.getMessage());
            return false;
        }
    }

    private final java.util.Random epRand = new java.util.Random();

    /** Per-episode reset for training:
     *   - teleport agent to spawn (64, 66, -48) — 1 block above grass floor
     *   - clear agent inventory
     *   - air-fill the arena above the grass floor (Y=66..70), preserves grass at Y=60..65
     *   - place a ring of 8 oak_log blocks at Y=66 (on top of grass), radius 4-6, seeded
     *  Fast (~10ms). */
    public boolean resetEpisode(String playerName, long seed) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/tp " + playerName + " 64 66 -48");
            cm.executeWithPrefix(src, "/clear " + playerName);
            cm.executeWithPrefix(src,
                "/fill 48 66 -64 80 70 -32 air replace");

            // Seeded ring: 8 logs at radius 4-6 on top of grass (Y=66), angles jittered by seed
            epRand.setSeed(seed);
            for (int i = 0; i < 8; i++) {
                double theta = (2.0 * Math.PI * i / 8.0) + (epRand.nextDouble() - 0.5) * 0.4;
                int r = 4 + epRand.nextInt(3);              // 4, 5, or 6
                int x = 64 + (int) Math.round(r * Math.cos(theta));
                int z = -48 + (int) Math.round(r * Math.sin(theta));
                cm.executeWithPrefix(src, "/setblock " + x + " 66 " + z + " oak_log");
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "resetEpisode failed for {} seed={}: {}", playerName, seed, e.getMessage());
            return false;
        }
    }

    /** One-time setup at server boot when training mode is active. Idempotent.
     *  Bakes a deterministic flat-grass arena around (64, 65, -48) and force-loads
     *  the chunks so /setblock/setblock-style commands during resetEpisode always
     *  succeed regardless of the world's natural terrain. */
    public boolean setupTrainingScene() {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/difficulty peaceful");
            cm.executeWithPrefix(src, "/time set noon");
            cm.executeWithPrefix(src, "/gamerule doDaylightCycle false");
            cm.executeWithPrefix(src, "/gamerule doMobSpawning false");
            // Keep training arena chunks resident so block ops always land
            cm.executeWithPrefix(src, "/forceload add 32 -64 96 -16");
            // Bake a solid grass floor and clear arena air above it
            cm.executeWithPrefix(src, "/fill 48 60 -64 80 64 -32 grass_block replace");
            cm.executeWithPrefix(src, "/fill 48 65 -64 80 80 -32 air replace");
            // N12: dropped from 300 -> 60 TPS. At 300 TPS, two distinct
            // concurrent-modification crashes were reproducible: CME in
            // Lithium's forEachInBox during EntityPlayerMPFake tick, and
            // AIOOBE in fastutil LongOpenHashSet.rehash during chunk tick.
            // 60 TPS (3x vanilla speed) still gives a meaningful training
            // throughput boost without exposing those races.
            cm.executeWithPrefix(src, "/tick rate 60.0");

            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "setupTrainingScene failed: {}", e.getMessage());
            return false;
        }
    }
}
