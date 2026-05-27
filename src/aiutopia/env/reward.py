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
