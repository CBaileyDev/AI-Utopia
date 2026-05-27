package dev.aiutopia.mod.agent;

import java.util.Set;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** In-process registry of AI agent player names + their roles.
 *  Populated by Py4J (CLI `aiutopia agent spawn`). Mixins + obs builder
 *  consult this to decide whether to filter / which overlay to emit.
 *  Thread-safe; reads are lock-free. */
public final class AgentRegistry {
    private static final Map<String, String> AGENT_ROLES = new ConcurrentHashMap<>();

    private AgentRegistry() {}

    /** M0 backward-compat — defaults role to "gatherer" when role unspecified. */
    public static void registerAgent(String playerName) {
        AGENT_ROLES.putIfAbsent(playerName, "gatherer");
    }
    public static void registerAgent(String playerName, String role) {
        AGENT_ROLES.put(playerName, role);
    }
    public static void unregisterAgent(String playerName) {
        AGENT_ROLES.remove(playerName);
    }
    public static boolean isAgent(String playerName) {
        return AGENT_ROLES.containsKey(playerName);
    }
    public static String roleOf(String playerName) {
        return AGENT_ROLES.getOrDefault(playerName, "");
    }
    public static Set<String> snapshot() {
        return Set.copyOf(AGENT_ROLES.keySet());
    }
}
