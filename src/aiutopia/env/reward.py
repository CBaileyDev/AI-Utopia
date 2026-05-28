"""§5 reward architecture — stage 1 only for M1-Pipeline.

Stage 2 (multi-objective + curriculum decay) and stage 3 (village
scarcity weights + LLM-driven targets) are deferred to M2-M5."""
from __future__ import annotations

from typing import Literal

RoleId = Literal["gatherer", "builder", "farmer", "defender"]


# §5.7 — VPT-normalized log-scaled potentials
LOG_VALUE: dict[str, float] = {
    # raw materials
    "oak_log":          1.000,
    "oak_planks":       0.050,
    "stick":            0.0625,
    "cobblestone":      1.0 / 11.0,
    "stone":            0.500,
    "coal":             0.400,
    "iron_ore":         4.000,
    "iron_ingot":       5.000,
    "gold_ingot":       3.000,
    "diamond":          8.000,
    # food
    "wheat":            0.1875,
    "bread":            0.375,
    "porkchop":         0.375,   "cooked_porkchop":  0.500,
    "beef":             0.375,   "cooked_beef":      0.500,
    "chicken":          0.375,   "cooked_chicken":   0.500,
    "carrot":           0.1875,
    "apple":            0.250,
    # crafted
    "crafting_table":   1.000,
    "furnace":          1.000,
    "wooden_pickaxe":   1.000,
    "stone_pickaxe":    1.500,
    "iron_pickaxe":     4.000,
    "wooden_sword":     0.500,
    "stone_sword":      1.000,
    "iron_sword":       4.000,
    "wooden_axe":       0.500,   "stone_axe":        1.000,    "iron_axe":         4.000,
    "wooden_hoe":       0.500,   "stone_hoe":        1.000,    "iron_hoe":         4.000,
    # armor
    "leather_helmet":   0.500,   "iron_helmet":      2.000,
    "leather_chestplate": 1.000, "iron_chestplate":  4.000,
    "leather_leggings": 0.875,   "iron_leggings":    3.500,
    "leather_boots":    0.500,   "iron_boots":       2.000,
    # placeables
    "torch":            0.125,
    "oak_door":         0.500,
    "glass_pane":       0.500,
    "ladder":           0.250,
    "fence":            0.125,
    "chest":            1.000,
}


# Per-role anti-hoarding caps. `_default` applies to any item not in this role's dict.
ROLE_INVENTORY_CAPS: dict[RoleId, dict[str, int]] = {
    "gatherer":  {
        "oak_log": 256, "cobblestone": 256, "stone": 128,
        "coal": 128, "iron_ore": 128, "iron_ingot": 64,
        "diamond": 16, "wheat": 64, "bread": 32,
        "stick": 64, "oak_planks": 128,
        "wooden_pickaxe": 4, "stone_pickaxe": 4, "iron_pickaxe": 2,
        "_default": 64,
    },
    "builder":   {
        "oak_log": 128, "oak_planks": 512, "cobblestone": 512,
        "stone": 256, "torch": 128, "oak_door": 16,
        "glass_pane": 64, "ladder": 32, "fence": 64,
        "chest": 8, "iron_ingot": 16,
        "_default": 32,
    },
    "farmer":    {
        "wheat": 256, "bread": 128, "carrot": 64,
        "porkchop": 32, "beef": 32, "chicken": 32,
        "cooked_porkchop": 32, "cooked_beef": 32, "cooked_chicken": 32,
        "_default": 16,
    },
    "defender":  {
        "iron_sword": 4, "iron_pickaxe": 2,
        "iron_helmet": 2, "iron_chestplate": 2,
        "iron_leggings": 2, "iron_boots": 2,
        "bread": 16, "cooked_beef": 16,
        "_default": 8,
    },
}


def tech_tree_potential(inventory: dict[str, int], role: str) -> float:
    """Φ(s) for PBRS shaping. Capped per-role (anti-hoarding) and weighted by
    LOG_VALUE (VPT-normalized). Used by `compute_reward()` as:
        r_pbrs = γ · Φ(s') − Φ(s),   γ = 0.99
    Absolute scale matters less than monotonicity for PBRS — the difference
    is what feeds the reward.
    """
    if role not in ROLE_INVENTORY_CAPS:
        raise KeyError(f"unknown role: {role!r}")
    caps = ROLE_INVENTORY_CAPS[role]
    default_cap = caps.get("_default", 32)
    total = 0.0
    for item, qty in inventory.items():
        if item not in LOG_VALUE:
            continue
        cap = caps.get(item, default_cap)
        total += min(qty, cap) * LOG_VALUE[item]
    return total


# ---------------------------------------------------------------------
# Stage-1 reward composition (§5.1 + §5.2 stage-1 branch only).
# Stages 2 + 3 (multi-objective + scarcity-weighted) are M2-M5 work.
# ---------------------------------------------------------------------

GAMMA          = 0.99    # PBRS discount
DEATH_PENALTY  = 10.0
TIME_PENALTY   = 0.001
GAMMA_CLIP     = 0.05    # per axis (§5.5)


def _delta_inventory(prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
    """Positive: item gained. Negative: item lost. Ignores zero deltas."""
    keys = set(prev) | set(curr)
    return {k: curr.get(k, 0) - prev.get(k, 0)
            for k in keys
            if curr.get(k, 0) - prev.get(k, 0) != 0}


def _inventory_from_obs(obs: dict) -> dict[str, int]:
    """Reconstruct {item_id: count} dict from the obs slot arrays.

    M1A defect surfaced during T21 v11: the obs space declares
    `inv_slot_item_ids` as MultiDiscrete (numeric IDs), but this
    function originally assumed strings like 'minecraft:oak_log'.
    Now handles both: strings are split as before; ints/numpy-ints
    map via _ITEM_ID_TO_NAME (Java-side ItemId table). Unknown int
    IDs become 'item_{N}' — they won't match LOG_VALUE keys, so they
    produce zero reward (silent), but the pipeline doesn't crash.

    Until the int→name table is wired through from the Java side
    (T-followup), real training will see zero positive reward for
    gathering. See NEXT_SESSION.md N3 caveat.
    """
    items = obs.get("inv_slot_item_ids", [])
    counts = obs.get("inv_slot_counts", [])
    out: dict[str, int] = {}
    for item, count in zip(items, counts):
        try:
            cnt = int(count)
        except (TypeError, ValueError):
            cnt = 0
        if cnt <= 0:
            continue
        if isinstance(item, str):
            if not item:
                continue
            key = item.split(":", 1)[-1]
        else:
            # int / numpy.int64 — name lookup via Java-aligned table
            iid = int(item)
            if iid == 0:                # canonical "no item" sentinel
                continue
            key = _ITEM_ID_TO_NAME.get(iid, f"item_{iid}")
        out[key] = out.get(key, 0) + cnt
    return out


# Java-side ItemId table — Python-side mirror of the contiguous mapping
# emitted by `dev.aiutopia.mod.obs.ItemIdTable` and shipped to Python by
# `Py4JEntryPoint.getItemIdNameTable()`. The wrapper updates this dict
# at env init from the live Java side (the authoritative source); the
# eager seed below is the **N14 defensive fallback** so reward shaping
# still works if:
#   (a) the wrapper's update path silently no-ops (e.g. log filtering
#       hides the failure during training, see N14),
#   (b) downstream code imports `reward` without ever touching the
#       FabricBridge (unit tests, offline reward replays),
#   (c) the Py4J connection is unavailable but a checkpoint's actions
#       are being scored against a replay buffer.
# Values were probed from a live MC 1.21.1 server with the current mod
# build (commit 0a5a6e5) — see scripts/probe_item_id_table.py. The
# in-process Java fetch overwrites/augments these entries at env init.
_ITEM_ID_TO_NAME: dict[int, str] = {
    # core blocks
    0: "air",
    1: "stone",
    27: "grass_block",
    28: "dirt",
    35: "cobblestone",
    36: "oak_planks",
    56: "bedrock",
    57: "sand",
    61: "gravel",
    64: "iron_ore",
    132: "oak_log",
    291: "torch",
    299: "chest",
    300: "crafting_table",
    302: "furnace",
    303: "ladder",
    357: "glass_pane",
    711: "oak_door",
    # food + raw materials
    800: "apple",
    803: "coal",
    805: "diamond",
    811: "iron_ingot",
    815: "gold_ingot",
    848: "stick",
    854: "wheat",
    855: "bread",
    881: "porkchop",
    882: "cooked_porkchop",
    909: "water_bucket",
    988: "beef",
    989: "cooked_beef",
    990: "chicken",
    991: "cooked_chicken",
    1097: "carrot",
    # tools (wooden / stone / iron)
    818: "wooden_sword",
    820: "wooden_pickaxe",
    821: "wooden_axe",
    822: "wooden_hoe",
    823: "stone_sword",
    825: "stone_pickaxe",
    826: "stone_axe",
    827: "stone_hoe",
    833: "iron_sword",
    835: "iron_pickaxe",
    836: "iron_axe",
    837: "iron_hoe",
    # armor
    856: "leather_helmet",
    857: "leather_chestplate",
    858: "leather_leggings",
    859: "leather_boots",
    864: "iron_helmet",
    865: "iron_chestplate",
    866: "iron_leggings",
    867: "iron_boots",
}


def _gatherer_primary_signal(prev_inv: dict[str, int],
                              curr_inv: dict[str, int]) -> float:
    """§5.4 — `Σ_r delta_inv[r] * potential[r]` (VPT-normalized).

    Uses LOG_VALUE directly (not the capped potential) so each block
    chopped gives the same reward, regardless of how much the agent
    has already hoarded. Capping is the PBRS potential's job (anti-
    hoarding pressure)."""
    delta = _delta_inventory(prev_inv, curr_inv)
    return sum(delta.get(item, 0) * value for item, value in LOG_VALUE.items())


def compute_reward_stage_1(*,
                            role: str,
                            obs_prev: dict,
                            obs_curr: dict,
                            action: dict,
                            env_meta: dict) -> float:
    """§5.1 stage-1 reward composition for solo per-role pretraining.

        r_stage_1 = r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip

    Args:
      role: must match a RoleId.
      obs_prev / obs_curr: dicts with inv_slot_item_ids + inv_slot_counts.
      action: contains skill_type at minimum.
      env_meta: dict with keys:
        - died_this_tick: bool
        - n_clipped_param_axes: int (0..4)
        - exploit_penalties: list[(name, positive_penalty_value)]

    Returns the scalar reward for this tick.
    """
    prev_inv = _inventory_from_obs(obs_prev)
    curr_inv = _inventory_from_obs(obs_curr)

    # M1-Pipeline only ships gatherer primary signal. Other roles get 0
    # until M2-M4 add their primary signals.
    if role == "gatherer":
        r_primary = _gatherer_primary_signal(prev_inv, curr_inv)
    else:
        r_primary = 0.0

    phi_prev = tech_tree_potential(prev_inv, role)
    phi_curr = tech_tree_potential(curr_inv, role)
    r_pbrs   = GAMMA * phi_curr - phi_prev

    r_death  = DEATH_PENALTY if env_meta.get("died_this_tick", False) else 0.0
    r_time   = TIME_PENALTY
    r_clip   = GAMMA_CLIP * int(env_meta.get("n_clipped_param_axes", 0))
    r_exploits = sum(p for _, p in env_meta.get("exploit_penalties", []))

    return r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip
