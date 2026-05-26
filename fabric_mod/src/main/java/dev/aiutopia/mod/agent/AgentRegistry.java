package dev.aiutopia.mod.agent;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** In-process registry of AI agent player names.
 *  Populated by Py4J (CLI `aiutopia agent spawn`). Mixins read this to
 *  decide whether to filter / block. Thread-safe; reads are lock-free. */
public final class AgentRegistry {
    private static final Set<String> AGENT_NAMES = ConcurrentHashMap.newKeySet();

    private AgentRegistry() {}

    public static void registerAgent(String playerName) {
        AGENT_NAMES.add(playerName);
    }
    public static void unregisterAgent(String playerName) {
        AGENT_NAMES.remove(playerName);
    }
    public static boolean isAgent(String playerName) {
        return AGENT_NAMES.contains(playerName);
    }
    public static Set<String> snapshot() {
        return Set.copyOf(AGENT_NAMES);
    }
}
