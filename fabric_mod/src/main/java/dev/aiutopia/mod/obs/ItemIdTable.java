package dev.aiutopia.mod.obs;

import dev.aiutopia.mod.AiUtopiaMod;
import net.minecraft.item.Item;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** N9: contiguous Item → gym-int-id mapping shared by the obs builder and
 *  the Py4J export.
 *
 *  WHY THIS EXISTS:
 *    The M1A obs builder used `Registries.ITEM.getRawId(item) & 0x3FF` to
 *    fit the int into the gym Discrete(1024) bound. Vanilla MC 1.21.1
 *    registers ~1300 items, so ~25% of low-id items (oak_log, stone,
 *    cobblestone, every basic block) get OVERWRITTEN in obs by later
 *    spawn-egg items with `rawId = base + 1024`. The mask is a silent
 *    information-destroyer: agents could never see oak_log in their
 *    inventory, regardless of the reward path.
 *
 *  HOW THIS WORKS:
 *    On first use we iterate `Registries.ITEM` in registration order and
 *    assign contiguous IDs starting at 0. AIR is the first vanilla item,
 *    so AIR → 0 — this matches reward.py's "iid == 0 → no item" sentinel.
 *
 *    The mapping is built once and cached. Reverse map (id → path name)
 *    is exposed via {@link #exportIdToPath()} for Py4J transport to
 *    Python's reward.py module.
 *
 *  N_ITEMS sync: spaces.py declares N_ITEMS = 2048 (gym Discrete bound).
 *    Items with assigned id >= N_ITEMS are clamped to 0 with a WARN at
 *    boot. Today that's zero items (vanilla 1.21.1 ~1300 < 2048). */
public final class ItemIdTable {

    /** Must match spaces.py N_ITEMS (gym Discrete upper bound, exclusive).
     *  Vanilla MC 1.21.1 registry size ≈ 1300; 2048 leaves room for a few
     *  modded items without breaking the gym contract. */
    public static final int N_ITEMS = 2048;

    private static final ItemIdTable INSTANCE = new ItemIdTable();

    private final Map<Item, Integer> itemToId;
    private final Map<Integer, String> idToPath;
    private final int overflowCount;

    private ItemIdTable() {
        Map<Item, Integer> i2id = new HashMap<>();
        Map<Integer, String> id2p = new LinkedHashMap<>();   // preserves insert order
        int next = 0;
        int overflow = 0;
        for (Item item : Registries.ITEM) {
            if (next >= N_ITEMS) {
                overflow++;
                continue;
            }
            Identifier id = Registries.ITEM.getId(item);
            i2id.put(item, next);
            id2p.put(next, id.getPath());
            next++;
        }
        this.itemToId = i2id;
        this.idToPath = id2p;
        this.overflowCount = overflow;
        AiUtopiaMod.LOG.info(
            "ItemIdTable: assigned {} contiguous IDs (N_ITEMS={}, overflow={})",
            next, N_ITEMS, overflow);
        if (overflow > 0) {
            AiUtopiaMod.LOG.warn(
                "ItemIdTable: {} items exceeded N_ITEMS={} and were clamped to id=0; "
              + "bump N_ITEMS in spaces.py + ItemIdTable.java if these matter",
                overflow, N_ITEMS);
        }
    }

    public static ItemIdTable get() { return INSTANCE; }

    /** Lookup the contiguous gym ID for an Item. Returns 0 (AIR sentinel)
     *  for any item that overflowed N_ITEMS during init. */
    public int idOf(Item item) {
        Integer v = itemToId.get(item);
        return v == null ? 0 : v;
    }

    /** Snapshot map id → unprefixed item name (e.g. "oak_log"). Used by
     *  {@link dev.aiutopia.mod.Py4JEntryPoint#getItemIdNameTable()} so
     *  Python's reward.py can map obs ints back to the names LOG_VALUE
     *  uses as keys. */
    public Map<Integer, String> exportIdToPath() {
        return new HashMap<>(idToPath);
    }

    public int size() { return idToPath.size(); }
}
