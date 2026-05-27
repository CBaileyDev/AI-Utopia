package dev.aiutopia.mod.bridge.skill;

import com.google.gson.JsonObject;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

/** A skill that drives an agent (Carpet fake player) for one or more ticks.
 *
 *  Lifecycle:
 *   1. MotorBridge constructs the executor and calls {@link #start(ServerPlayerEntity, JsonObject, MinecraftServer)}
 *      once with the agent and the decoded action JSON.
 *   2. MotorBridge calls {@link #tick(ServerPlayerEntity, MinecraftServer)} on every server tick
 *      until it returns a terminal {@link SkillResult}.
 *   3. On terminal result, MotorBridge emits a SkillCompletionEvent and clears
 *      the agent's current executor slot.
 *
 *  Implementations MUST:
 *   - Be stateful per dispatch (not singletons; constructed fresh each dispatch).
 *   - Bound their tick budget so an infinite-loop bug can't lock up the server
 *     (timeoutTicks parameter is passed in JSON; check it).
 *   - Report IMMEDIATE_FAILURE if start() preconditions fail (e.g., HARVEST
 *     called but target_class refers to a block not in range).
 *   - Track which continuous params they had to clip and expose them via
 *     {@link #clippedAxes()} (consumed by Python for γ_clip).
 */
public interface SkillExecutor {

    /** Initialize the skill from action JSON. Returns IMMEDIATE_FAILURE if
     *  preconditions are violated and the skill cannot start; otherwise
     *  RUNNING (skill needs more ticks) or COMPLETED (skill done immediately
     *  — used for instant skills like WAIT(scalar=0) or NOOP_BROADCAST). */
    SkillResult start(ServerPlayerEntity agent, JsonObject action, MinecraftServer server);

    /** Advance the skill one tick. Returns RUNNING to continue or a terminal
     *  SkillResult. */
    SkillResult tick(ServerPlayerEntity agent, MinecraftServer server);

    /** Bitset of continuous-param axes that had to be clipped on start()
     *  (bit 0 = spatial.dx, 1 = spatial.dy, 2 = spatial.dz, 3 = scalar). */
    int clippedAxes();

    /** Optional human-readable reason for IMMEDIATE_FAILURE / FAILED_TIMEOUT;
     *  surfaced in the SkillCompletionEvent. Empty string when not set. */
    String failureReason();
}
