package dev.aiutopia.mod.obs;

import com.google.gson.JsonObject;
import dev.aiutopia.mod.agent.AgentRegistry;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.1 obs Dict builder. Composes core + role-specific overlay into a
 *  single JsonObject keyed by agent_id; WorldOps.observationsAll wraps
 *  all agents into one outer JsonObject. */
public final class ObservationBuilder {

    private final CoreObsBuilder           core   = new CoreObsBuilder();
    private final GathererOverlayBuilder   gather = new GathererOverlayBuilder();
    private final FarmerOverlayBuilder     farmer = new FarmerOverlayBuilder();

    /** Build the full obs JsonObject for one agent. */
    public JsonObject buildForAgent(ServerPlayerEntity agent, MinecraftServer server) {
        JsonObject obs = new JsonObject();
        core.populate(obs, agent, server);
        // M1-Pipeline ships gatherer overlay only. Other roles are M2-M4.
        String name = agent.getGameProfile().getName();
        String role = AgentRegistry.roleOf(name);
        if (role.equals("gatherer")) {
            gather.populate(obs, agent, server);
        } else if (role.equals("farmer")) {
            farmer.populate(obs, agent, server);
        }
        return obs;
    }
}
