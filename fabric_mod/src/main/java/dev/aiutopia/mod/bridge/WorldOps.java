package dev.aiutopia.mod.bridge;

import net.minecraft.server.MinecraftServer;

/** §7.3 world inspection + reset operations. M0 stubs return an empty JSON object. */
public class WorldOps {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

    /** Batched observation read — single JSON blob with every agent on
     *  this env. M0 stub: returns "{}" so Python can parse it. */
    public String observationsAll() {
        // TODO M1: assemble per-agent obs from world state + Carpet APIs.
        return "{}";
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
            int result = cm.executeWithPrefix(
                server.getCommandSource(),
                "/player " + playerName + " spawn"
            );
            if (result <= 0) return false;
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName);

            if (skin != null && !skin.isEmpty()) {
                int skinResult = cm.executeWithPrefix(
                    server.getCommandSource(),
                    "/player " + playerName + " loadProfile " + skin
                );
                if (skinResult <= 0) {
                    dev.aiutopia.mod.AiUtopiaMod.LOG.warn(
                        "loadProfile {} failed for {}; using default skin",
                        skin, playerName);
                }
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "carpetSpawn failed for {}: {}", playerName, e.getMessage());
            return false;
        }
    }
}
