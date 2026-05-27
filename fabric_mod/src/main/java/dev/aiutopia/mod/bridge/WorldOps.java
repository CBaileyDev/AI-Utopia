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
    public boolean carpetSpawn(String playerName, String skin) {
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
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName);

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
}
