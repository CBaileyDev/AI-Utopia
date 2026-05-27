package dev.aiutopia.mod;

import dev.aiutopia.mod.bridge.CommBus;
import dev.aiutopia.mod.bridge.MotorBridge;
import dev.aiutopia.mod.bridge.WorldOps;
import net.minecraft.server.MinecraftServer;

/**
 * Surface exposed to Python via Py4J. Each method here is callable from
 * Python with normal type marshalling. Keep signatures stable — Python
 * code depends on them.
 *
 * §7.3 invariants:
 *   - observationsAll() returns ONE JSON blob per env per tick (batched).
 *   - motorBridge() handles SkillCompletionEvent via ack-based callbacks.
 *   - commBus().flushBatch(...) is called mid-tick by the env wrapper.
 */
public class Py4JEntryPoint {
    private MinecraftServer server;
    private final MotorBridge motor   = new MotorBridge();
    private final CommBus     commBus = new CommBus();
    private final WorldOps    world   = new WorldOps();

    public void attachServer(MinecraftServer server) {
        this.server = server;
        this.motor.attachServer(server);
        this.world.attachServer(server);
    }

    public void detachServer() {
        this.server = null;
        this.motor.detachServer();
        this.world.detachServer();
    }

    // ───── methods called from Python ─────

    /** Stub heartbeat — returns "ok" if the server is attached. */
    public String health() {
        return server == null ? "stopped" : "ok";
    }

    /** Batched observation read — one JSON blob containing all agents
     *  on this env's server. */
    public String observationsAll() {
        return world.observationsAll();
    }

    public MotorBridge motorBridge() { return motor; }
    public CommBus     commBus()     { return commBus; }

    /** Reset the world to the given seed. M0 stub. */
    public void resetWorld(long seed) {
        world.resetWorld(seed);
    }

    /** Advance one tick and return the list of agent_ids whose current
     *  skill completed this tick. Times out after timeoutMs and returns
     *  an empty list. M0 stub: returns [] immediately. */
    public java.util.List<String> advanceTickAwaitEvents(long timeoutMs) {
        return motor.advanceTickAwaitEvents(timeoutMs);
    }

    /** Drain queued ChatEvents. Called from Python planner each tick. */
    public java.util.List<String> drainChatEvents() {
        return dev.aiutopia.mod.chat.ChatEventBuffer.drainAll();
    }

    /** Spawn a Carpet fake player with optional skin. Returns true on success. */
    public boolean carpetSpawn(String playerName, String skin, String role) {
        return world.carpetSpawn(playerName, skin, role);
    }

    /** DEV/SMOKE ONLY: run an arbitrary server command (e.g. /setblock).
     *  Plan B should restrict this behind an auth check or remove it entirely
     *  before exposing the server publicly. Returns true if the command
     *  completed without throwing; false on exception or detached server. */
    public boolean runCommand(String command) {
        if (server == null) return false;
        try {
            server.getCommandManager().executeWithPrefix(
                server.getCommandSource(), command);
            return true;
        } catch (Exception e) {
            AiUtopiaMod.LOG.warn("runCommand failed for {!r}: {}", command, e.getMessage());
            return false;
        }
    }

    /** Per-episode reset with a seed for deterministic log placement. */
    public boolean resetEpisode(String playerName, long seed) {
        return world.resetEpisode(playerName, seed);
    }

    /** One-time training-scene setup. */
    public boolean setupTrainingScene() {
        return world.setupTrainingScene();
    }

    /** N9: returns the full masked item-id → simple-name table that the obs
     *  builder uses for `inv_slot_item_ids` / `main_hand_item_id` / `off_hand_item_id`.
     *
     *  Python's reward.py (LOG_VALUE keys) is keyed by unprefixed item names
     *  (e.g. "oak_log"), but the obs space emits int IDs (`rawId & 0x3FF`).
     *  Without this table, _inventory_from_obs falls back to "item_{N}" which
     *  never matches LOG_VALUE → identically zero primary reward.
     *
     *  Collision policy: with ~1300 vanilla items and 1024 buckets, ~25% of
     *  buckets collide after `& 0x3FF`. We iterate `Registries.ITEM` in
     *  registration order (its natural iteration order) and last-write-wins.
     *  In practice the only items M1B needs (oak_log, dirt, stone, ...) have
     *  small raw IDs and don't collide. WARNs are emitted for visibility. */
    public java.util.Map<Integer, String> getItemIdNameTable() {
        return dev.aiutopia.mod.obs.ItemIdTable.get().exportIdToPath();
    }
}
