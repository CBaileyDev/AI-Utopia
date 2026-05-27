package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.entity.MovementType;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.Vec3d;

/** §4.2 NAVIGATE: walk in a direct line toward (spatial_param × max_range)
 *  blocks from the agent's current position. RUNNING until within 1.0 block
 *  of the target, then COMPLETED.
 *
 *  spatial_param is normalized [-1, 1]^3; multiplied by MAX_NAV_RANGE=32
 *  blocks gives the displacement vector. Clipping happens at start():
 *  if requested |dx| > MAX_NAV_RANGE, clamp and set the clippedAxes bit.
 *
 *  Movement uses {@code agent.move(MovementType.SELF, ...)} so vanilla AABB
 *  collision applies (the agent bumps into walls + falls with gravity).
 *  The agent will NOT auto-step 1-block ledges — that's Carpet ActionPack
 *  territory, deferred to Plan B. For flat-plains M1 verification this is
 *  sufficient. */
public class NavigateSkill implements SkillExecutor {

    private static final double MAX_NAV_RANGE        = 32.0;  // blocks
    private static final double ARRIVAL_RADIUS       = 1.0;   // blocks
    private static final double WALK_SPEED_PER_TICK  = 4.3 / 20.0;  // ~0.215 b/tick

    private Vec3d targetPos;
    private long  ticksRemaining;
    private int   clipped;
    private String failureReason = "";

    /** Accept scalar_param as either a 1-element array OR a bare number.
     *  Gym sometimes sends one or the other depending on Box((1,)) vs
     *  scalar Discrete encoding (R15). Default returned if absent. */
    static double readScalar(JsonObject action, String key, double dflt) {
        if (!action.has(key)) return dflt;
        JsonElement el = action.get(key);
        if (el.isJsonArray() && el.getAsJsonArray().size() >= 1) {
            return el.getAsJsonArray().get(0).getAsDouble();
        }
        if (el.isJsonPrimitive() && el.getAsJsonPrimitive().isNumber()) {
            return el.getAsDouble();
        }
        return dflt;
    }

    @Override
    public SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server) {
        if (!action.has("spatial_param")) {
            failureReason = "NAVIGATE requires spatial_param array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        var arr = action.getAsJsonArray("spatial_param");
        if (arr.size() != 3) {
            failureReason = "spatial_param must be length-3 array";
            return SkillResult.IMMEDIATE_FAILURE;
        }
        double[] raw = { arr.get(0).getAsDouble(), arr.get(1).getAsDouble(), arr.get(2).getAsDouble() };
        // Clip each axis to [-1, 1] and track which were out of range
        for (int i = 0; i < 3; i++) {
            if (raw[i] < -1.0 || raw[i] > 1.0) {
                clipped |= (1 << i);
                raw[i] = Math.max(-1.0, Math.min(1.0, raw[i]));
            }
        }
        Vec3d origin = agent.getPos();
        this.targetPos = new Vec3d(
            origin.x + raw[0] * MAX_NAV_RANGE,
            origin.y + raw[1] * 8.0,            // vertical range is tighter
            origin.z + raw[2] * MAX_NAV_RANGE
        );
        // timeout_ticks default 6000 (5 min); honor JSON override if present
        this.ticksRemaining = action.has("timeout_ticks")
            ? action.get("timeout_ticks").getAsLong()
            : 6000L;
        return SkillResult.RUNNING;
    }

    @Override
    public SkillResult tick(ServerPlayerEntity agent, MinecraftServer server) {
        if (--ticksRemaining <= 0) {
            failureReason = "navigate timeout — never reached target";
            return SkillResult.FAILED_TIMEOUT;
        }
        Vec3d here  = agent.getPos();
        Vec3d delta = targetPos.subtract(here);
        double dist = delta.length();
        if (dist <= ARRIVAL_RADIUS) {
            return SkillResult.COMPLETED;
        }
        // Step toward target at walk speed — use agent.move() not setPosition()
        // so vanilla collision logic stops the agent at walls.
        Vec3d dir   = delta.normalize();
        double step = Math.min(WALK_SPEED_PER_TICK, dist);
        Vec3d stepVec = dir.multiply(step);
        // Face the direction of travel (yaw from dx, dz; pitch from dy)
        float yaw   = (float) Math.toDegrees(Math.atan2(-dir.x, dir.z));
        float pitch = (float) Math.toDegrees(-Math.asin(dir.y));
        agent.setYaw(yaw);
        agent.setPitch(pitch);
        agent.move(MovementType.SELF, stepVec);
        return SkillResult.RUNNING;
    }

    @Override public int clippedAxes()    { return clipped;  }
    @Override public String failureReason() { return failureReason; }
}
