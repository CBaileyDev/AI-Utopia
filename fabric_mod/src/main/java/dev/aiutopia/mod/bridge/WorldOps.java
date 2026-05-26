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
}
