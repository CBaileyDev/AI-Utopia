package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** §4.2 WAIT: pass scalar_param × MAX_WAIT ticks. Useful for letting the
 *  world catch up (sleep through night, wait for a crop to grow).
 *
 *  scalar_param is normalized [0, 1]; clip out-of-range values and flag bit 3
 *  of clippedAxes (the scalar axis). Default 0.05 ≈ half a second. */
public class WaitSkill implements SkillExecutor {
    private static final long MAX_WAIT = 200L;   // 10 s @ 20 TPS

    private long ticksRemaining;
    private int  clipped;

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        double scalar = NavigateSkill.readScalar(action, "scalar_param", 0.05);
        if (scalar < 0.0 || scalar > 1.0) {
            clipped |= 0b1000;
            scalar = Math.max(0.0, Math.min(1.0, scalar));
        }
        ticksRemaining = Math.max(1L, (long) Math.round(scalar * MAX_WAIT));
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        return --ticksRemaining <= 0 ? SkillResult.COMPLETED : SkillResult.RUNNING;
    }

    @Override public int    clippedAxes()   { return clipped; }
    @Override public String failureReason() { return "";      }
}
