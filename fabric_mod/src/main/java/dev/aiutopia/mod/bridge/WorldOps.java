package dev.aiutopia.mod.bridge;

import net.minecraft.server.MinecraftServer;

/** §7.3 world inspection + reset operations. M0 stubs return an empty JSON object. */
public class WorldOps {
    private MinecraftServer server;

    public void attachServer(MinecraftServer server) { this.server = server; }
    public void detachServer()                       { this.server = null;   }

    /** §7.3 batched observation read — single JSON blob with every agent on
     *  this env. Iterates AgentRegistry.snapshot(), composes per-agent obs
     *  via ObservationBuilder. Skips agents that are not currently on the
     *  server (e.g. dead, pre-spawn). */
    public String observationsAll() {
        if (server == null) return "{}";
        com.google.gson.JsonObject root = new com.google.gson.JsonObject();
        var builder = new dev.aiutopia.mod.obs.ObservationBuilder();
        for (String name : dev.aiutopia.mod.agent.AgentRegistry.snapshot()) {
            var player = server.getPlayerManager().getPlayer(name);
            if (player == null) continue;   // not currently connected
            root.add(name, builder.buildForAgent(player, server));
        }
        return root.toString();
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
    public boolean carpetSpawn(String playerName, String skin, String role) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm =
                server.getCommandManager();
            // MC 1.21.1 `CommandManager.executeWithPrefix(ServerCommandSource, String)`
            // returns void. Brigadier exceptions propagate as RuntimeException; a
            // failed command (e.g. Carpet not installed) silently no-ops. We trust
            // success if no exception fires and register the agent name.
            cm.executeWithPrefix(
                server.getCommandSource(),
                "/player " + playerName + " spawn"
            );
            // Register WITH role so obs builder dispatches the right overlay
            if (role == null || role.isEmpty()) role = "gatherer";
            dev.aiutopia.mod.agent.AgentRegistry.registerAgent(playerName, role);

            // NOTE: Carpet 1.4.147 (MC 1.21.1) does not expose a /player <name>
            // loadProfile <skin> subcommand. The fake player's skin is derived
            // from Mojang's account lookup for `playerName` itself when the
            // server is in online-mode. In offline-mode (our smoke setup) the
            // skin shows as default Steve/Alex. M5+ may revisit by switching
            // to a custom entity type with explicit GameProfile assignment.
            // The `skin` parameter is currently ignored on the Carpet path;
            // kept in the method signature for forward compatibility.
            if (skin != null && !skin.isEmpty()) {
                dev.aiutopia.mod.AiUtopiaMod.LOG.debug(
                    "skin '{}' ignored (Carpet 1.4.x has no loadProfile); "
                    + "use online-mode + matching MC account for skin",
                    skin);
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "carpetSpawn failed for {}: {}", playerName, e.getMessage());
            return false;
        }
    }

    private final java.util.Random epRand = new java.util.Random();

    /** Per-episode reset for training:
     *   - teleport agent to spawn (64, 66, -48) — 1 block above grass floor
     *   - clear agent inventory
     *   - air-fill the arena above the grass floor (Y=66..70), preserves grass at Y=60..65
     *   - place a 64-oak_log FLAT grid at Y=66 (on top of grass), seeded jitter
     *
     *  P0 fix: the M1B eval gate is "collect 64 oak_log in one episode," but the
     *  old reset seeded only 8 logs with no refill, making the gate unmeetable by
     *  construction. We place exactly 64 oak_log per episode (N21 Inc2: as 16
     *  vertical 4-tall trunks, was a flat single-log grid).
     *
     *  REACHABILITY (see HarvestSkill.java): REACH_RADIUS=4.5; the skill has no
     *  climb/jump and findNearest PREFERS the ground band dy ∈ [-2,+1]. A 4-tall
     *  trunk (Y=65..68) is nonetheless FULLY clearable, EMPIRICALLY VERIFIED (N21,
     *  scripts/n21_breaktiming_determinism.py: all seeds reach 64/64 deterministic-
     *  ally): findNearest's pass-2 reaches the upper logs, and after the base logs
     *  are broken the agent walks INTO the cleared column (horizontal→0) so the top
     *  log (Y=69 center 69.5, vertical 4.5 from feet Y=65) is within REACH. Do NOT
     *  exceed height 4 without a climb model — tops above Y=68 are out of reach from
     *  the ground (the seed-3 reachability trap).
     *
     *  Layout: 16 trunks on a 4×4 grid spaced 7 — x = 52+7*col (52..73),
     *  z = -61+7*row (-61..-40), each a 4-log stack at Y=65..68, all inside
     *  [48,80]×[-64,-32]. The grid straddles the spawn tile (64,-48). Each trunk
     *  gets a seeded ±1 (x,z) jitter (so eval seeds 1/2/3 differ); jittered coords
     *  are clamped, unique, and off the spawn tile. MUST match sim/world.py
     *  reset() byte-for-byte (draw order). Fast (~10ms). */
    public boolean resetEpisode(String playerName, long seed) {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/tp " + playerName + " 64 66 -48");
            cm.executeWithPrefix(src, "/clear " + playerName);
            // N21: equip a stone axe in the main hand so HARVEST's survival
            // break-timing (BREAK_TICKS_PER_LOG) reflects a real tool, not a
            // bare hand. /item replace guarantees the main hand each episode.
            cm.executeWithPrefix(src, "/item replace entity " + playerName
                + " weapon.mainhand with minecraft:stone_axe");
            // Clear from Y=65 (trunk base, on the grass) up so last episode's
            // logs — including the Y=65 base — are removed before re-placing.
            cm.executeWithPrefix(src,
                "/fill 48 65 -64 80 70 -32 air replace");

            // N21 Inc2: 16 vertical BARE oak trunks (4 logs each, Y=65..68) on a
            // 4×4 spaced grid, seeded ±1 (x,z) jitter per trunk. MUST match
            // sim/world.py reset() BYTE-FOR-BYTE: identical base formulas, the
            // SAME nextInt(3) draw order (x THEN z), identical clamp + dedup-nudge,
            // and a FIXED height 4 (no height draw — keeps the RNG draw pattern the
            // same 2-per-trunk surface). 16×4 = 64 logs (gate unchanged); height 4
            // ≤ REACH 4.5 so every log is ground-reachable (no climbing); NO leaves
            // (collidable — would block the straight-line agent.move walk).
            epRand.setSeed(seed);
            final int SPAWN_X = 64, SPAWN_Z = -48;
            final int MIN_X = 48, MAX_X = 80, MIN_Z = -64, MAX_Z = -32;
            final int TREE_GRID = 4, TRUNK_H = 4;
            java.util.HashSet<Long> used = new java.util.HashSet<>();
            for (int row = 0; row < TREE_GRID; row++) {
                for (int col = 0; col < TREE_GRID; col++) {
                    int baseX = 52 + 7 * col;     // 52,59,66,73
                    int baseZ = -61 + 7 * row;    // -61,-54,-47,-40
                    // Seeded jitter in {-1,0,+1} on each axis (x THEN z).
                    int x = baseX + (epRand.nextInt(3) - 1);
                    int z = baseZ + (epRand.nextInt(3) - 1);
                    x = Math.max(MIN_X, Math.min(MAX_X, x));
                    z = Math.max(MIN_Z, Math.min(MAX_Z, z));
                    // Resolve collisions with the spawn tile or an already-placed
                    // trunk by nudging deterministically through neighbor offsets.
                    // 16 trunks spaced 7 apart in a 33×33 arena -> a free cell
                    // always exists.
                    while ((x == SPAWN_X && z == SPAWN_Z) || used.contains(key(x, z))) {
                        if (x < MAX_X)      x += 1;
                        else if (z < MAX_Z) z += 1;
                        else if (x > MIN_X) x -= 1;
                        else                z -= 1;
                    }
                    used.add(key(x, z));
                    // Stack the trunk: oak_log at Y=65 (base, on the grass) ..
                    // 65+TRUNK_H-1 (=Y=68 for height 4). Rooted, not floating.
                    for (int dy = 0; dy < TRUNK_H; dy++) {
                        cm.executeWithPrefix(src,
                            "/setblock " + x + " " + (65 + dy) + " " + z + " oak_log");
                    }
                }
            }
            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "resetEpisode failed for {} seed={}: {}", playerName, seed, e.getMessage());
            return false;
        }
    }

    /** Pack (x,z) into a single long for the dedup set used by resetEpisode. */
    private static long key(int x, int z) {
        return (((long) x) << 32) ^ (z & 0xffffffffL);
    }

    /** One-time setup at server boot when training mode is active. Idempotent.
     *  Bakes a deterministic flat-grass arena around (64, 65, -48) and force-loads
     *  the chunks so /setblock/setblock-style commands during resetEpisode always
     *  succeed regardless of the world's natural terrain. */
    public boolean setupTrainingScene() {
        if (server == null) return false;
        try {
            net.minecraft.server.command.CommandManager cm = server.getCommandManager();
            net.minecraft.server.command.ServerCommandSource src = server.getCommandSource();
            cm.executeWithPrefix(src, "/difficulty peaceful");
            cm.executeWithPrefix(src, "/time set noon");
            cm.executeWithPrefix(src, "/gamerule doDaylightCycle false");
            cm.executeWithPrefix(src, "/gamerule doMobSpawning false");
            // Keep training arena chunks resident so block ops always land
            cm.executeWithPrefix(src, "/forceload add 32 -64 96 -16");
            // Bake a solid grass floor and clear arena air above it
            cm.executeWithPrefix(src, "/fill 48 60 -64 80 64 -32 grass_block replace");
            cm.executeWithPrefix(src, "/fill 48 65 -64 80 80 -32 air replace");
            // N12: dropped from 300 -> 60 TPS. At 300 TPS, two distinct
            // concurrent-modification crashes were reproducible: CME in
            // Lithium's forEachInBox during EntityPlayerMPFake tick, and
            // AIOOBE in fastutil LongOpenHashSet.rehash during chunk tick.
            // 60 TPS (3x vanilla speed) still gives a meaningful training
            // throughput boost without exposing those races.
            cm.executeWithPrefix(src, "/tick rate 60.0");

            return true;
        } catch (Exception e) {
            dev.aiutopia.mod.AiUtopiaMod.LOG.error(
                "setupTrainingScene failed: {}", e.getMessage());
            return false;
        }
    }
}
