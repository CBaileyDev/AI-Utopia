"""§5 reward architecture — stage 1 only for M1-Pipeline.

Stage 2 (multi-objective + curriculum decay) and stage 3 (village
scarcity weights + LLM-driven targets) are deferred to M2-M5."""

from __future__ import annotations

from typing import Literal

import numpy as np

RoleId = Literal["gatherer", "builder", "farmer", "defender"]


# §5.7 — VPT-normalized log-scaled potentials
LOG_VALUE: dict[str, float] = {
    # raw materials
    "oak_log": 1.000,
    "oak_planks": 0.050,
    "stick": 0.0625,
    "cobblestone": 1.0 / 11.0,
    "stone": 0.500,
    "coal": 0.400,
    "iron_ore": 4.000,
    "iron_ingot": 5.000,
    "gold_ingot": 3.000,
    "diamond": 8.000,
    # food
    "wheat": 0.1875,
    "bread": 0.375,
    "porkchop": 0.375,
    "cooked_porkchop": 0.500,
    "beef": 0.375,
    "cooked_beef": 0.500,
    "chicken": 0.375,
    "cooked_chicken": 0.500,
    "carrot": 0.1875,
    "apple": 0.250,
    # crafted
    "crafting_table": 1.000,
    "furnace": 1.000,
    "wooden_pickaxe": 1.000,
    "stone_pickaxe": 1.500,
    "iron_pickaxe": 4.000,
    "wooden_sword": 0.500,
    "stone_sword": 1.000,
    "iron_sword": 4.000,
    "wooden_axe": 0.500,
    "stone_axe": 1.000,
    "iron_axe": 4.000,
    "wooden_hoe": 0.500,
    "stone_hoe": 1.000,
    "iron_hoe": 4.000,
    # armor
    "leather_helmet": 0.500,
    "iron_helmet": 2.000,
    "leather_chestplate": 1.000,
    "iron_chestplate": 4.000,
    "leather_leggings": 0.875,
    "iron_leggings": 3.500,
    "leather_boots": 0.500,
    "iron_boots": 2.000,
    # placeables
    "torch": 0.125,
    "oak_door": 0.500,
    "glass_pane": 0.500,
    "ladder": 0.250,
    "fence": 0.125,
    "chest": 1.000,
}


# Per-role TASK-ITEM allowlists (M1B single-attractor gate).
#
# WHY: The gatherer reward previously had a cobblestone *attractor* through
# TWO channels — (a) `_gatherer_primary_signal` summed over EVERY LOG_VALUE
# entry (so mining cobblestone paid 1/11 ≈ 0.0909 per block), and (b)
# `tech_tree_potential` (the PBRS Φ) ALSO weighted cobblestone. With two
# off-task channels both paying out, 3 of 4 PPO instances learned to mine
# cobblestone instead of oak_log, diluting the gradient on the actual M1B
# objective (the oak_log gate).
#
# FIX: For a role that has a task allowlist, BOTH the primary signal AND the
# PBRS potential are restricted to ONLY the items in its allowlist. For the
# gatherer that means a clean SINGLE attractor on oak_log: cobblestone and all
# other off-task items contribute 0 to r_primary AND to Φ. The PBRS channel
# must not reward off-task cobblestone — otherwise the shaping term re-creates
# the very attractor we removed from r_primary.
#
# Roles WITHOUT an entry here (builder/farmer/defender) are UNCHANGED: they
# fall back to the full LOG_VALUE-weighted potential over their whole capped
# inventory (those roles' r_primary is already 0 in stage 1, but their Φ feeds
# PBRS and must keep its existing multi-item shape for M2-M4).
ROLE_TASK_ITEMS: dict[RoleId, frozenset[str]] = {
    "gatherer": frozenset({"oak_log"}),
}


# Per-role anti-hoarding caps. `_default` applies to any item not in this role's dict.
ROLE_INVENTORY_CAPS: dict[RoleId, dict[str, int]] = {
    "gatherer": {
        "oak_log": 256,
        "cobblestone": 256,
        "stone": 128,
        "coal": 128,
        "iron_ore": 128,
        "iron_ingot": 64,
        "diamond": 16,
        "wheat": 64,
        "bread": 32,
        "stick": 64,
        "oak_planks": 128,
        "wooden_pickaxe": 4,
        "stone_pickaxe": 4,
        "iron_pickaxe": 2,
        "_default": 64,
    },
    "builder": {
        "oak_log": 128,
        "oak_planks": 512,
        "cobblestone": 512,
        "stone": 256,
        "torch": 128,
        "oak_door": 16,
        "glass_pane": 64,
        "ladder": 32,
        "fence": 64,
        "chest": 8,
        "iron_ingot": 16,
        "_default": 32,
    },
    "farmer": {
        "wheat": 256,
        "bread": 128,
        "carrot": 64,
        "porkchop": 32,
        "beef": 32,
        "chicken": 32,
        "cooked_porkchop": 32,
        "cooked_beef": 32,
        "cooked_chicken": 32,
        "_default": 16,
    },
    "defender": {
        "iron_sword": 4,
        "iron_pickaxe": 2,
        "iron_helmet": 2,
        "iron_chestplate": 2,
        "iron_leggings": 2,
        "iron_boots": 2,
        "bread": 16,
        "cooked_beef": 16,
        "_default": 8,
    },
}


def tech_tree_potential(inventory: dict[str, int], role: str) -> float:
    """Φ(s) for PBRS shaping. Capped per-role (anti-hoarding) and weighted by
    LOG_VALUE (VPT-normalized). Used by `compute_reward()` as:
        r_pbrs = γ · Φ(s') − Φ(s),   γ = 0.99
    Absolute scale matters less than monotonicity for PBRS — the difference
    is what feeds the reward.

    M1B single-attractor: if `role` has a ROLE_TASK_ITEMS allowlist, Φ counts
    ONLY those task items (gatherer ⇒ oak_log). This keeps the PBRS channel
    from re-introducing an off-task attractor (e.g. cobblestone) that we
    removed from the primary signal. Roles without an allowlist sum over their
    full capped inventory, exactly as before.
    """
    if role not in ROLE_INVENTORY_CAPS:
        raise KeyError(f"unknown role: {role!r}")
    caps = ROLE_INVENTORY_CAPS[role]
    default_cap = caps.get("_default", 32)
    task_items = ROLE_TASK_ITEMS.get(role)  # None ⇒ no allowlist (count all)
    total = 0.0
    for item, qty in inventory.items():
        if item not in LOG_VALUE:
            continue
        if task_items is not None and item not in task_items:
            continue  # off-task item: excluded from this role's Φ
        cap = caps.get(item, default_cap)
        total += min(qty, cap) * LOG_VALUE[item]
    return total


# ---------------------------------------------------------------------
# Stage-1 reward composition (§5.1 + §5.2 stage-1 branch only).
# Stages 2 + 3 (multi-objective + scarcity-weighted) are M2-M5 work.
# ---------------------------------------------------------------------

GAMMA = 0.99  # PBRS discount
DEATH_PENALTY = 10.0
TIME_PENALTY = 0.001
GAMMA_CLIP = 0.05  # per axis (§5.5)


def _delta_inventory(prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
    """Positive: item gained. Negative: item lost. Ignores zero deltas."""
    keys = set(prev) | set(curr)
    return {
        k: curr.get(k, 0) - prev.get(k, 0) for k in keys if curr.get(k, 0) - prev.get(k, 0) != 0
    }


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
    for item, count in zip(items, counts, strict=False):
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
            if iid == 0:  # canonical "no item" sentinel
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


def _gatherer_primary_signal(
    prev_inv: dict[str, int], curr_inv: dict[str, int], role: str = "gatherer"
) -> float:
    """§5.4 — `Σ_r delta_inv[r] * potential[r]` (VPT-normalized).

    Uses LOG_VALUE directly (not the capped potential) so each block
    chopped gives the same reward, regardless of how much the agent
    has already hoarded. Capping is the PBRS potential's job (anti-
    hoarding pressure).

    M1B single-attractor: restricted to the role's ROLE_TASK_ITEMS
    allowlist (gatherer ⇒ {oak_log}). Off-task items (cobblestone,
    stone, ...) contribute 0, so this channel is a clean single
    attractor on oak_log and does not compete with the PPO gradient.
    A role with no allowlist would fall back to the full LOG_VALUE sum
    (no such role uses this path in stage 1)."""
    delta = _delta_inventory(prev_inv, curr_inv)
    task_items = ROLE_TASK_ITEMS.get(role)  # None ⇒ no allowlist (count all)
    return sum(
        delta.get(item, 0) * value
        for item, value in LOG_VALUE.items()
        if task_items is None or item in task_items
    )


def compute_reward_stage_1(
    *, role: str, obs_prev: dict, obs_curr: dict, action: dict, env_meta: dict
) -> float:
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
    
    M2 extension: Dispatches to role-specific reward function (explorer, farmer).
    """
    if role == "gatherer":
        return _compute_reward_stage_1_gatherer(
            obs_prev=obs_prev, obs_curr=obs_curr, action=action, env_meta=env_meta
        )
    elif role == "explorer":
        return _compute_reward_stage_1_explorer(
            obs_prev=obs_prev, obs_curr=obs_curr, action=action, env_meta=env_meta
        )
    elif role == "farmer":
        return _compute_reward_stage_1_farmer(
            obs_prev=obs_prev, obs_curr=obs_curr, action=action, env_meta=env_meta
        )
    else:
        raise ValueError(f"unknown role: {role!r}")


def _compute_reward_stage_1_gatherer(
    *, obs_prev: dict, obs_curr: dict, action: dict, env_meta: dict
) -> float:
    """Gatherer stage-1 reward (original logic, renamed for clarity)."""
    prev_inv = _inventory_from_obs(obs_prev)
    curr_inv = _inventory_from_obs(obs_curr)

    r_primary = _gatherer_primary_signal(prev_inv, curr_inv, "gatherer")

    phi_prev = tech_tree_potential(prev_inv, "gatherer")
    phi_curr = tech_tree_potential(curr_inv, "gatherer")
    r_pbrs = GAMMA * phi_curr - phi_prev

    r_death = DEATH_PENALTY if env_meta.get("died_this_tick", False) else 0.0
    r_time = TIME_PENALTY
    r_clip = GAMMA_CLIP * int(env_meta.get("n_clipped_param_axes", 0))
    r_exploits = sum(p for _, p in env_meta.get("exploit_penalties", []))

    return r_primary + r_pbrs - r_death - r_time - r_exploits - r_clip


def explorer_potential(obs_curr: dict, decay_coeff: float = 1.0) -> float:
    """§D1a — Φ(s) for Explorer's discovery shaping.
    
    Encourages richness progress (approaching forest). Decays over episode.
    M2 spec: shaping optional; primary signal is sparse +1.0 discovery bonus.
    For now, returns 0 (decay to no shaping; primary signal only).
    """
    # richness_score: 0 (no resources nearby) to 1 (dense forest)
    # For Stage 1, primary signal is the discovery gate; shaping is optional
    # and can be added in M2.2 if needed.
    return 0.0


def farmer_potential(obs_curr: dict, decay_coeff: float = 1.0) -> float:
    """§M2 — Φ(s) for Farmer's temporal credit assignment shaping.
    
    Three terms: planting progress, ripeness progress, timeliness.
    Scaled to 0–100 range to match tech_tree_potential's magnitude (CTDE).
    """
    # Term 1: Planting progress (0–1, then scaled)
    planted_count = obs_curr.get("f_planted_count", (0,))[0] if isinstance(obs_curr.get("f_planted_count"), (list, np.ndarray)) else 0
    if isinstance(planted_count, np.ndarray):
        planted_count = planted_count.item()
    planted_progress = min(1.0, float(planted_count) / 64.0)

    # Term 2: Ripeness progress (already 0–1)
    ripeness = obs_curr.get("f_ripeness", (0,))[0] if isinstance(obs_curr.get("f_ripeness"), (list, np.ndarray)) else 0
    if isinstance(ripeness, np.ndarray):
        ripeness = ripeness.item()
    ripeness_progress = float(ripeness)

    # Term 3: Timeliness (inverse staleness)
    crop_grid = obs_curr.get("f_crop_grid")
    time_at_ripeness = obs_curr.get("f_time_at_ripeness")
    timeliness = 0.0
    if crop_grid is not None and time_at_ripeness is not None:
        crop_grid = np.asarray(crop_grid)
        time_at_ripeness = np.asarray(time_at_ripeness)
        ripe_cells = crop_grid == 8
        if ripe_cells.any():
            staleness = np.minimum(1.0, time_at_ripeness[ripe_cells] / 50.0)
            timeliness = float(np.mean(1.0 - staleness))

    # Composite with decay + magnitude scaling
    phi_unscaled = (
        0.15 * planted_progress +
        0.50 * ripeness_progress +
        0.35 * timeliness
    ) * decay_coeff

    return 100.0 * phi_unscaled


def _compute_reward_stage_1_explorer(
    *, obs_prev: dict, obs_curr: dict, action: dict, env_meta: dict
) -> float:
    """Explorer stage-1 reward (M2 design spec).
    
    r_explorer = r_discovery + r_pbrs - r_death - r_time - r_stuck
    
    Primary signal: +1.0 when richness_score crosses discovery threshold.
    Shaping: optional progress/coverage terms (deferred to M2.2).
    """
    richness_prev = obs_prev.get("g_richness_score", (0,))[0] if isinstance(obs_prev.get("g_richness_score"), (list, np.ndarray)) else 0
    richness_curr = obs_curr.get("g_richness_score", (0,))[0] if isinstance(obs_curr.get("g_richness_score"), (list, np.ndarray)) else 0

    if isinstance(richness_prev, np.ndarray):
        richness_prev = richness_prev.item()
    if isinstance(richness_curr, np.ndarray):
        richness_curr = richness_curr.item()

    # Sparse discovery: +1.0 when richness crosses threshold (8 logs in 16-block window)
    discovery_threshold = 0.125  # 8 / 64
    r_discovery = 1.0 if (richness_prev < discovery_threshold <= richness_curr) else 0.0

    # Shaping (optional): progress toward richness
    # For M2.1, defer this; primary signal only
    r_pbrs = 0.0

    r_death = DEATH_PENALTY if env_meta.get("died_this_tick", False) else 0.0
    r_time = TIME_PENALTY
    r_stuck = 0.1 if env_meta.get("no_movement_ticks", 0) > 10 else 0.0

    return r_discovery + r_pbrs - r_death - r_time - r_stuck


def _compute_reward_stage_1_farmer(
    *, obs_prev: dict, obs_curr: dict, action: dict, env_meta: dict
) -> float:
    """Farmer stage-1 reward (M2 design spec).
    
    r_farmer = r_principal + r_pbrs - r_death - r_time - r_exploits - r_clip
    
    Primary: +1.0 per unique harvested cell.
    Shaping: farmer_potential with temporal decay.
    Exploits: re-planting, unripe harvest, idle waiting.
    """
    # Sparse principal: incremented per unique harvested cell
    harvested_prev = obs_prev.get("f_harvested_count", (0,))[0] if isinstance(obs_prev.get("f_harvested_count"), (list, np.ndarray)) else 0
    harvested_curr = obs_curr.get("f_harvested_count", (0,))[0] if isinstance(obs_curr.get("f_harvested_count"), (list, np.ndarray)) else 0

    if isinstance(harvested_prev, np.ndarray):
        harvested_prev = harvested_prev.item()
    if isinstance(harvested_curr, np.ndarray):
        harvested_curr = harvested_curr.item()

    harvested_delta = max(0, int(harvested_curr) - int(harvested_prev))
    r_principal = float(harvested_delta)

    # PBRS with decay
    tick_curr = obs_curr.get("tick_in_episode", (0,))[0]
    if isinstance(tick_curr, np.ndarray):
        tick_curr = tick_curr.item()
    max_ticks = env_meta.get("max_episode_ticks", 1000)
    decay_curr = max(0.0, 1.0 - float(tick_curr) / float(max_ticks))
    decay_next = max(0.0, 1.0 - (float(tick_curr) + 1.0) / float(max_ticks))

    phi_prev = farmer_potential(obs_prev, decay_coeff=decay_curr)
    phi_curr = farmer_potential(obs_curr, decay_coeff=decay_next)
    r_pbrs = GAMMA * phi_curr - phi_prev

    # Universal penalties
    r_death = DEATH_PENALTY if env_meta.get("died_this_tick", False) else 0.0
    r_time = TIME_PENALTY
    r_clip = GAMMA_CLIP * int(env_meta.get("n_clipped_param_axes", 0))

    # Farmer-specific exploit penalties
    exploit_penalties = env_meta.get("exploit_penalties", [])
    r_exploits = sum(p for _, p in exploit_penalties)

    return r_principal + r_pbrs - r_death - r_time - r_exploits - r_clip
