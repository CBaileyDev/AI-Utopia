package dev.aiutopia.mod.chat;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/** Bounded FIFO of ChatEvent JSON blobs awaiting Python pickup.
 *  Python pulls via Py4J at the start of each planner tick. */
public final class ChatEventBuffer {
    private static final int MAX_SIZE = 256;
    private static final Deque<String> QUEUE = new ArrayDeque<>();

    private ChatEventBuffer() {}

    public static synchronized void push(String chatEventJson) {
        if (QUEUE.size() >= MAX_SIZE) QUEUE.pollFirst();   // drop oldest
        QUEUE.addLast(chatEventJson);
    }

    public static synchronized List<String> drainAll() {
        List<String> out = new ArrayList<>(QUEUE);
        QUEUE.clear();
        return out;
    }

    public static synchronized int size() { return QUEUE.size(); }
}
