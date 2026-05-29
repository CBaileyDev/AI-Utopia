"""Byte-faithful gatherer obs adapter: ``SimWorld`` -> obs dict matching the
REAL env obs (captured golden trace), NOT merely the declared space.

Ground-truth note (from tests/fixtures/gatherer_obs_trace_seed1.json, captured
off the live env): the real obs DEVIATES from build_role_observation_space in
shape — ``g_resource_grid`` arrives FLAT ``(6144,)`` (Java emits a flat JsonArray
and wrapper._decode_obs does ``np.asarray`` without reshape), and
``g_richness_score`` / ``biome_id`` / ``weather`` / ``main_hand_item_id`` /
``off_hand_item_id`` are 0-d scalars. The policy consumes this (flattened)
format, so the sim MUST match it for transfer — replicating the nominal space
would actually break parity. The 27 keys here are exactly the keys the real obs
carries (== space keys; the auxiliary ``nearest_*_distance`` are consumed by the
action-mask and dropped, not emitted).

Spatial fields replicate GathererOverlayBuilder.java:
  - origin = floor(agent_pos) -> blockpos (bx,by,bz). Agent settles at obs-y=65,
    logs at y=66, so a log's dy = +1 (confirmed: g_nearest_resources[*][1] = 0.125).
  - g_resource_grid[dx+16][dz+16][0] = 1 (channel 0=log) for alive logs in the
    32x32 window; flattened x-z-c (C-order) to (6144,).
  - g_nearest_resources: alive logs within Euclidean SCAN_RADIUS=16 (with dy),
    top-8 by distance, row [dx/16, dy/8, dz/16, 0, 1, 1]; zero-padded.
  - g_richness_score = min(1, count_within_radius / 64).

Constant fields are set to the captured-arena values where they matter
(health/hunger=20, weather=1, biome=33, light=12, time=6000); the pure-noise
fields the policy should ignore (velocity, yaw_pitch, tick_in_episode,
goal_ticks_left) are left at simple defaults — NONE of these are golden-asserted
(see DYNAMIC_FIELDS in the parity test), so they don't gate Phase-A fidelity.

IMPORT-LIGHT: numpy + env.spaces + env.action_mask + env._embeds + env.reward
(the pure ``_ITEM_ID_TO_NAME`` table). Never the wrapper / chroma / py4j / torch.
"""

from __future__ import annotations

import math

import numpy as np

from aiutopia.env._embeds import _agent_uuid_embed, gatherer_goal_embedding_stub
from aiutopia.env.action_mask import compute_gatherer_action_mask
from aiutopia.env.reward import _ITEM_ID_TO_NAME

# name -> id (reverse of reward's id->name table) for inventory slot packing.
_NAME_TO_ID: dict[str, int] = {name: i for i, name in _ITEM_ID_TO_NAME.items()}

GRID_RADIUS = 16
SCAN_RADIUS = 16.0
REACH_RADIUS_BLOCKS = 4.5
SENTINEL_NO_TARGET = 999.0
INV_SLOTS = 36
_AGENT_ID = "gatherer_0"
# N21 Inc1: the agent is equipped with a stone axe each episode (WorldOps.reset-
# Episode /item replace). The real obs reports item id 826 in inventory slot 0 and
# in main_hand; the sim must mirror it so the obs the policy sees matches real.
STONE_AXE_ID = 826

# Captured-arena constants (tests/fixtures/gatherer_obs_trace_seed1.json). These
# are the real env's M1B-arena values; matching them keeps the constant inputs
# the policy sees identical between sim and real. Not golden-asserted.
_WEATHER = 1
_BIOME_ID = 33
_LIGHT_LEVEL = 12
_TIME_OF_DAY = 6000

# Reuse the SAME constant embeds the real env produces (single source of truth).
_UUID_EMBED = _agent_uuid_embed(_AGENT_ID).astype(np.float32)
_GOAL_EMBED = gatherer_goal_embedding_stub().astype(np.float32)
_ROLE_ONE_HOT = np.array([1, 0, 0, 0], dtype=np.int32)  # gatherer


def _inv_slots(inventory: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pack {item_name: count} into (item_ids[36], counts[36]).

    Mirrors how the real fake player's inventory presents after /clear + the
    stone-axe equip + pickups: slot 0 holds the stone axe (N21 Inc1), harvested
    items fill from slot 1, each slot stacking to 64 (M1B's 64-oak_log cap fits one
    slot). Empty slots are id 0, count 0.
    """
    ids = np.zeros(INV_SLOTS, dtype=np.int32)
    counts = np.zeros(INV_SLOTS, dtype=np.int32)
    # N21 Inc1: slot 0 is the equipped stone axe (real obs slot 0 == 826).
    ids[0] = STONE_AXE_ID
    counts[0] = 1
    slot = 1
    for name, qty in inventory.items():
        remaining = int(qty)
        while remaining > 0 and slot < INV_SLOTS:
            take = min(remaining, 64)
            ids[slot] = _NAME_TO_ID.get(name, 0)
            counts[slot] = take
            remaining -= take
            slot += 1
    return ids, counts


def gatherer_nearest_columns(
    world,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int, int, int, int, int]]]:
    """Per-(x,z)-column TOPMOST-log-within-±3 nearest list — the SINGLE source of
    truth for both g_nearest_resources AND the decision-core instance pointer.

    Returns ``(col_top, nearby)``:
      - ``col_top``: dict (dx,dz)->topmost log dy in [-3,+3] for every in-grid
        column (drives g_resource_grid).
      - ``nearby``: the SCAN_RADIUS subset sorted by (distSq, dx, dz), each entry
        ``(x, z, dx, dy, dz, dist_sq)`` with absolute world (x,z). g_nearest_resources
        row k == nearby[k]; the action ``target_class`` (reinterpreted as an instance
        pointer in decision-core mode) indexes ``nearby`` -> the trunk at world (x,z).

    Mirrors GathererOverlayBuilder: scan each column top-down dy=+3..-3, take the
    topmost non-air (==topmost log for our bare-trunk arena); logs at dy+4 are above
    the window and never seen. Iteration order dx outer [-16,16), dz inner, so the
    stable distSq sort reproduces the equidistant tie-break the golden trace caught.
    """
    bx = int(math.floor(float(world.agent_pos[0])))
    by = int(math.floor(float(world.agent_pos[1])))
    bz = int(math.floor(float(world.agent_pos[2])))
    col_top: dict[tuple[int, int], int] = {}
    for i in range(world.logs.shape[0]):
        if not bool(world.log_alive[i]):
            continue
        lx, ly, lz = (int(world.logs[i][0]), int(world.logs[i][1]), int(world.logs[i][2]))
        dx, dy, dz = lx - bx, ly - by, lz - bz
        if not (-GRID_RADIUS <= dx < GRID_RADIUS and -GRID_RADIUS <= dz < GRID_RADIUS):
            continue
        if not (-3 <= dy <= 3):
            continue
        cur = col_top.get((dx, dz))
        if cur is None or dy > cur:
            col_top[(dx, dz)] = dy
    nearby: list[tuple[int, int, int, int, int, int]] = []
    for dx in range(-GRID_RADIUS, GRID_RADIUS):
        for dz in range(-GRID_RADIUS, GRID_RADIUS):
            dy = col_top.get((dx, dz))
            if dy is None:
                continue
            dist_sq = dx * dx + dy * dy + dz * dz
            if math.sqrt(dist_sq) <= SCAN_RADIUS:
                nearby.append((bx + dx, bz + dz, dx, dy, dz, dist_sq))
    nearby.sort(key=lambda r: (r[5], r[2], r[4]))  # (distSq, dx, dz)
    return col_top, nearby


def build_gatherer_obs(world, harvest_mask_on_perception: bool = False) -> dict:
    """Build the gatherer obs dict from SimWorld state, matching the real obs.

    ``harvest_mask_on_perception`` (decision-core): when True, HARVEST/MINE is
    masked-VALID whenever a resource is in PERCEPTION (g_nearest non-empty), not
    only within REACH — because the decision-core MINE walks to the pointed
    instance. This makes greedy decode MINE a visible trunk and NAVIGATE only when
    blind (the correct explore behavior). Default False = the in-reach mask the
    real obs / golden trace use (unchanged)."""
    # --- spatial (GathererOverlayBuilder parity; see gatherer_nearest_columns) ---
    col_top, nearby = gatherer_nearest_columns(world)
    grid = np.zeros((2 * GRID_RADIUS, 2 * GRID_RADIUS, 6), dtype=np.float32)
    for (dx, dz), _dy in col_top.items():
        grid[dx + GRID_RADIUS][dz + GRID_RADIUS][0] = 1.0  # channel 0 = log

    g_resource_grid = grid.reshape(-1)  # FLAT (6144,) x-z-c, matches real
    g_nearest = np.zeros((8, 6), dtype=np.float32)
    for k in range(min(8, len(nearby))):
        _x, _z, dx, dy, dz, _ = nearby[k]
        g_nearest[k] = [dx / SCAN_RADIUS, dy / 8.0, dz / SCAN_RADIUS, 0.0, 1.0, 1.0]

    g_richness = np.float32(min(1.0, len(nearby) / 64.0))
    nearest_res_dist = math.sqrt(nearby[0][5]) if nearby else SENTINEL_NO_TARGET
    nearest_chest_dist = SENTINEL_NO_TARGET  # no chest in the M1B arena

    inv_ids, inv_counts = _inv_slots(world.inventory)

    # --- action mask via the SAME builder the wrapper uses ---
    mask = compute_gatherer_action_mask(
        {
            "inv_slot_counts": inv_counts.tolist(),
            "target_resource_in_range": (
                bool(nearby) if harvest_mask_on_perception
                else nearest_res_dist <= REACH_RADIUS_BLOCKS
            ),
            "target_chest_in_range": nearest_chest_dist <= REACH_RADIUS_BLOCKS,
            "health": 20.0,
        }
    )

    return {
        # constant embeds (identical to real via shared _embeds)
        "agent_uuid_embed": _UUID_EMBED,
        "role_one_hot": _ROLE_ONE_HOT,
        "goal_embedding": _GOAL_EMBED,
        # dynamic — derived from world state
        "position": world.agent_pos.astype(np.float32),
        "tick_in_episode": np.array([int(world.tick)], dtype=np.int32),
        "inv_slot_item_ids": inv_ids,
        "inv_slot_counts": inv_counts,
        "g_resource_grid": g_resource_grid,
        "g_nearest_resources": g_nearest,
        "g_richness_score": g_richness,
        "g_hostiles_nearby": np.zeros((4, 4), dtype=np.float32),
        "action_mask": mask,
        # near-constant / noise fields (not golden-asserted; defaults)
        "velocity": np.zeros(3, dtype=np.float32),
        "yaw_pitch": np.zeros(2, dtype=np.float32),
        "health": np.array([20.0], dtype=np.float32),
        "hunger": np.array([20.0], dtype=np.float32),
        "saturation": np.array([20.0], dtype=np.float32),
        "armor_value": np.array([0.0], dtype=np.float32),
        "main_hand_item_id": np.int32(STONE_AXE_ID),
        "off_hand_item_id": np.int32(0),
        "goal_ticks_left": np.array([0], dtype=np.int32),
        "time_of_day": np.array([_TIME_OF_DAY], dtype=np.int32),
        "weather": np.int32(_WEATHER),
        "biome_id": np.int32(_BIOME_ID),
        "light_level": np.array([_LIGHT_LEVEL], dtype=np.int32),
        "comm_payloads": np.zeros((32, 128), dtype=np.float32),
        "comm_metadata": np.zeros((32, 8), dtype=np.float32),
    }
