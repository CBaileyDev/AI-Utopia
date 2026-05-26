package dev.aiutopia.mod.bridge;

import java.util.Collections;
import java.util.List;
import net.minecraft.server.MinecraftServer;

/**
 * Motor module — dispatches parameterized skill primitives to the world.
 * M0 stubs all methods so the bridge surface is callable end-to-end;
 * real Baritone / Carpet integrations land in M1+.
 *
 * §6.3 invariants:
 *   - dispatchSkill(...) returns immediately; completion is signalled via
 *     advanceTickAwaitEvents().
 *   - Idempotency: skillInvocationId is unique per (agent, dispatch);
 *     duplicate IDs are silently de-duped.
 */
public class MotorBridge {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

    /** Dispatch a parameterized skill. M0 stub: enqueues a no-op on the
     *  server thread so we exercise the public scheduling API. M1 replaces
     *  the lambda body with the actual skill dispatch.
     *
     *  NOTE: `MinecraftServer.execute(Runnable)` is the public API for
     *  scheduling work on the main thread. `ServerTask` is package-private
     *  and cannot be instantiated from outside `net.minecraft.server`. */
    public void dispatchSkill(String agentId, String encodedAction,
                              String skillInvocationId) {
        if (server != null) {
            server.execute(() -> { /* no-op stub — M1 wires real motor */ });
        }
    }

    /** Advance one tick; return agent_ids that completed a skill this tick.
     *  M0 stub: returns empty list immediately. */
    public List<String> advanceTickAwaitEvents(long timeoutMs) {
        return Collections.emptyList();
    }
}
