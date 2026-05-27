package dev.aiutopia.mod.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import dev.aiutopia.mod.AiUtopiaMod;
import dev.aiutopia.mod.bridge.skill.*;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;

import java.util.*;
import java.util.concurrent.*;

/** §7.3 + §4.6 motor module. Owns:
 *   - the per-agent active skill slot (max 1 in flight per agent)
 *   - per-agent skill_invocation_id counter for idempotent dispatch dedupe
 *   - the per-tick driver that calls SkillExecutor.tick() and emits
 *     SkillCompletionEvent on terminal results
 *   - the synchronous gate that Py4J's advanceTickAwaitEvents() waits on
 *
 * Thread model: server tick events run on the server thread; Py4J calls
 * arrive on Py4J's gateway thread. We marshal Py4J→server via the
 * server.execute(Runnable) queue and signal Py4J→server via thread-safe
 * concurrent maps + a BlockingQueue of completion events. */
public class MotorBridge {

    /** Emitted to Python when a skill terminates. */
    public static final class CompletionEvent {
        public final String agentId;
        public final String skillInvocationId;
        public final String resultCode;        // SkillResult.name()
        public final String failureReason;
        public final int    clippedAxesBitset; // bits 0=dx,1=dy,2=dz,3=scalar
        public CompletionEvent(String a, String i, SkillResult r, String why, int clip) {
            this.agentId = a; this.skillInvocationId = i;
            this.resultCode = r.name(); this.failureReason = why; this.clippedAxesBitset = clip;
        }
    }

    private MinecraftServer server;
    private final Gson gson = new Gson();

    // active skill per agent (server-thread access only)
    private final Map<String, ActiveDispatch> active = new HashMap<>();

    // idempotency: dispatched invocation ids per agent (server-thread access only)
    private final Map<String, Set<String>> dispatched = new HashMap<>();

    // completion events, drained by advanceTickAwaitEvents (multi-thread)
    private final BlockingQueue<CompletionEvent> completedQueue = new LinkedBlockingQueue<>();

    private static final class ActiveDispatch {
        final SkillExecutor executor;
        final String skillInvocationId;
        ActiveDispatch(SkillExecutor e, String i) { this.executor = e; this.skillInvocationId = i; }
    }

    // R11: Fabric event system has no unregister; guard so reconnects don't
    // pile up duplicate callbacks (each tick would otherwise run N times).
    private static volatile boolean tickRegistered = false;

    public void attachServer(MinecraftServer server) {
        if (this.server != null) return;  // already attached — idempotent
        this.server = server;
        if (!tickRegistered) {
            // Register the per-tick driver. END_SERVER_TICK fires once per server
            // tick after world simulation; we tick each active executor.
            // NOTE: this callback fires for ALL MotorBridge instances ever
            // attached. The lambda below captures `this`, but the static
            // guard means we only ever register one — for the current
            // process's single Py4JEntryPoint singleton, that's correct.
            ServerTickEvents.END_SERVER_TICK.register(this::onServerTick);
            tickRegistered = true;
        }
    }

    public void detachServer() {
        this.server = null;
        active.clear();
        dispatched.clear();
        completedQueue.clear();
        // tickRegistered intentionally NOT reset — see attachServer note above.
    }

    /** Py4J entry point. agentId is the Carpet player name; encodedAction is the
     *  full action_dict JSON-serialized by Python. */
    public void dispatchSkill(String agentId, String encodedAction, String skillInvocationId) {
        if (server == null) return;
        // Marshal to server thread for safe state access
        server.execute(() -> dispatchOnServerThread(agentId, encodedAction, skillInvocationId));
    }

    private void dispatchOnServerThread(String agentId, String encodedAction, String skillInvocationId) {
        // Idempotency: drop duplicate invocation ids
        Set<String> seen = dispatched.computeIfAbsent(agentId, k -> new HashSet<>());
        if (!seen.add(skillInvocationId)) {
            AiUtopiaMod.LOG.debug("dropping duplicate dispatch {} for {}", skillInvocationId, agentId);
            return;
        }
        // Pre-empt any active skill for this agent (emit ABORTED)
        ActiveDispatch prev = active.remove(agentId);
        if (prev != null) {
            completedQueue.offer(new CompletionEvent(
                agentId, prev.skillInvocationId, SkillResult.ABORTED,
                "preempted by " + skillInvocationId, prev.executor.clippedAxes()
            ));
        }
        // Find the agent player
        ServerPlayerEntity agent = server.getPlayerManager().getPlayer(agentId);
        if (agent == null) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "agent player not found: " + agentId, 0
            ));
            return;
        }
        // Parse JSON
        JsonObject action;
        try {
            action = gson.fromJson(encodedAction, JsonObject.class);
        } catch (Exception e) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "bad action JSON: " + e.getMessage(), 0
            ));
            return;
        }
        // Construct executor for the skill_type
        int skillType = action.has("skill_type") ? action.get("skill_type").getAsInt() : -1;
        SkillExecutor exec = newExecutorForSkillType(skillType);
        if (exec == null) {
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, SkillResult.IMMEDIATE_FAILURE,
                "unknown skill_type: " + skillType, 0
            ));
            return;
        }
        // Start the executor
        SkillResult started = exec.start(agent, action, server);
        if (started != SkillResult.RUNNING) {
            // Instant terminal — emit completion right away
            completedQueue.offer(new CompletionEvent(
                agentId, skillInvocationId, started, exec.failureReason(), exec.clippedAxes()
            ));
            return;
        }
        active.put(agentId, new ActiveDispatch(exec, skillInvocationId));
    }

    /** GATHERER skill_type indices match Python's spaces.py:
     *  0 NAVIGATE  1 HARVEST  2 DEPOSIT_CHEST  3 SEARCH  4 WAIT  5 NOOP_BROADCAST */
    private SkillExecutor newExecutorForSkillType(int skillType) {
        return switch (skillType) {
            case 0 -> new NavigateSkill();
            case 1 -> new HarvestSkill();
            case 2 -> new DepositChestSkill();
            case 3 -> new SearchSkill();
            case 4 -> new WaitSkill();
            case 5 -> new WaitSkill();   // NOOP_BROADCAST = WaitSkill(0); comm payload is handled by CommBus
            default -> null;
        };
    }

    private void onServerTick(MinecraftServer server) {
        if (this.server == null || active.isEmpty()) return;
        // Snapshot to avoid ConcurrentModificationException on emit-and-remove
        var entries = new ArrayList<>(active.entrySet());
        for (var entry : entries) {
            String agentId  = entry.getKey();
            ActiveDispatch d = entry.getValue();
            ServerPlayerEntity agent = server.getPlayerManager().getPlayer(agentId);
            if (agent == null) {
                active.remove(agentId);
                pruneDispatched(agentId, d.skillInvocationId);  // R13
                completedQueue.offer(new CompletionEvent(
                    agentId, d.skillInvocationId, SkillResult.ABORTED,
                    "agent disappeared", d.executor.clippedAxes()
                ));
                continue;
            }
            SkillResult r;
            try {
                r = d.executor.tick(agent, server);
            } catch (Exception e) {
                AiUtopiaMod.LOG.error("skill {} crashed for {}: {}",
                                       d.skillInvocationId, agentId, e.getMessage(), e);
                r = SkillResult.IMMEDIATE_FAILURE;
            }
            if (r != SkillResult.RUNNING) {
                active.remove(agentId);
                pruneDispatched(agentId, d.skillInvocationId);  // R13
                completedQueue.offer(new CompletionEvent(
                    agentId, d.skillInvocationId, r,
                    d.executor.failureReason(), d.executor.clippedAxes()
                ));
            }
        }
    }

    /** R13: drop the invocation id from the per-agent dedupe set once the
     *  skill has terminated. The dedupe window is "in-flight", not "ever
     *  seen" — keeping terminated ids forever leaks memory across long
     *  runs and provides no extra correctness (Python won't reuse ids). */
    private void pruneDispatched(String agentId, String skillInvocationId) {
        Set<String> seen = dispatched.get(agentId);
        if (seen != null) {
            seen.remove(skillInvocationId);
            if (seen.isEmpty()) {
                dispatched.remove(agentId);
            }
        }
    }

    /** Py4J entry point: blocks up to timeoutMs for at least one CompletionEvent
     *  to arrive, then drains all pending completions. Returns the list of
     *  CompletionEvent JSON strings (one per terminated skill since last call).
     *
     *  R14: use a plain ArrayList for drainTo. The anonymous-AbstractCollection
     *  hack the earlier draft used worked on current JDKs by coincidence
     *  (LinkedBlockingQueue.drainTo happens to call c.add(e)) but the Collection
     *  contract permits other strategies. */
    public java.util.List<String> advanceTickAwaitEvents(long timeoutMs) {
        java.util.List<String> out = new ArrayList<>();
        try {
            // Block up to timeoutMs for the first event
            CompletionEvent first = completedQueue.poll(timeoutMs, TimeUnit.MILLISECONDS);
            if (first != null) {
                out.add(gson.toJson(first));
                // Drain any others without further blocking
                List<CompletionEvent> drained = new ArrayList<>();
                completedQueue.drainTo(drained);
                for (CompletionEvent e : drained) {
                    out.add(gson.toJson(e));
                }
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        return out;
    }
}
