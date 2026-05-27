package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.2 SEARCH: rotate yaw 360° over scalar_param × MAX_DURATION ticks.
 *  No world side-effect; the observation builder picks up resources that
 *  enter the agent's scan radius regardless of yaw, so SEARCH is mostly
 *  a semantic distinction from WAIT (different reward shaping potential
 *  in Plan B). */
public class SearchSkill implements SkillExecutor {
    private static final int MAX_DURATION = 200;  // 10 s @ 20 TPS

    private long endTick;
    private long startTick;
    private float startYaw;
    private int clipped;
    private String failureReason = "";

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        double scalar = NavigateSkill.readScalar(action, "scalar_param", 0.5);
        if (scalar < 0.0 || scalar > 1.0) {
            clipped |= 0b1000;
            scalar = Math.max(0.0, Math.min(1.0, scalar));
        }
        long duration = Math.max(1L, (long) Math.round(scalar * MAX_DURATION));
        startTick = server.getOverworld().getTime();
        endTick   = startTick + duration;
        startYaw  = agent.getYaw();
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        long now = server.getOverworld().getTime();
        if (now >= endTick) return SkillResult.COMPLETED;
        long total = endTick - startTick;
        float progress = (float)(now - startTick) / total;
        agent.setYaw(startYaw + progress * 360.0f);
        return SkillResult.RUNNING;
    }

    @Override public int clippedAxes()     { return clipped; }
    @Override public String failureReason() { return failureReason; }
}
