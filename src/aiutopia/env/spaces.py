"""§4.1, §4.2 — Per-role observation + action Dict spaces.

M0 implements gatherer only. M2 adds explorer + farmer.
"""
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from gymnasium.spaces import Dict as DictSpace

# Fixed constants — must agree with Java side (motor_module.encode_action).
N_ITEMS         = 2048     # N9: contiguous remap via Java ItemIdTable (no
                           # longer a sparse mod-1024 mask). Vanilla MC 1.21.1
                           # ~1300 items; 2048 gives headroom + must match
                           # ItemIdTable.N_ITEMS on the Java side.
N_BIOMES        = 64
INV_SLOTS       = 36
GOAL_EMBED_DIM  = 512
COMM_PAYLOAD_DIM= 128
COMM_BUFFER_SLOTS= 32      # §3 carry-forward (1.6 s history at 20 TPS)

N_GATHERER_SKILLS         = 6   # navigate, harvest, deposit_chest, search, wait, noop_broadcast
N_EXPLORER_SKILLS         = 1   # bearing only (Discrete 8, reinterpreted from target_class)
N_FARMER_SKILLS           = 7   # plow, plant, harvest, navigate, wait, etc.
N_TARGET_CLASSES_PER_ROLE = 64  # block_pos / resource_id / chest_id / direction_bias index

CORE_KEYS = (
    "agent_uuid_embed", "role_one_hot", "tick_in_episode",
    "position", "velocity", "yaw_pitch", "health", "hunger",
    "saturation", "armor_value",
    "inv_slot_item_ids", "inv_slot_counts",
    "main_hand_item_id", "off_hand_item_id",
    "goal_embedding", "goal_ticks_left",
    "time_of_day", "weather", "biome_id", "light_level",
    "comm_payloads", "comm_metadata",
    "action_mask",
)
GATHERER_KEYS = (
    "g_resource_grid", "g_nearest_resources",
    "g_richness_score", "g_hostiles_nearby",
)
EXPLORER_KEYS = (
    "g_resource_grid", "g_nearest_resources",
    "g_richness_score",  # richness is the key signal for forest discovery
)
FARMER_KEYS = (
    "f_crop_grid", "f_ripeness", "f_planted_count",
    "f_harvested_count", "f_harvested_mask", "f_time_at_ripeness",
)


def _action_mask_space(n_skills: int, n_targets: int) -> DictSpace:
    return DictSpace({
        "skill_type":       MultiBinary(n_skills),
        "target_per_skill": MultiBinary((n_skills, n_targets)),
        "comm_payload":     MultiBinary(1),
        "should_broadcast": MultiBinary(2),
    })


def _core_space() -> dict:
    return {
        "agent_uuid_embed":  Box(-1, 1,  (384,), np.float32),
        "role_one_hot":      MultiBinary(4),
        "tick_in_episode":   Box(0, 24_000, (1,), np.int32),
        "position":          Box(-3e7, 3e7, (3,), np.float32),
        "velocity":          Box(-10, 10,  (3,), np.float32),
        "yaw_pitch":         Box(-180, 180, (2,), np.float32),
        "health":            Box(0, 20, (1,), np.float32),
        "hunger":            Box(0, 20, (1,), np.float32),
        "saturation":        Box(0, 20, (1,), np.float32),
        "armor_value":       Box(0, 20, (1,), np.float32),
        "inv_slot_item_ids": MultiDiscrete([N_ITEMS] * INV_SLOTS),
        "inv_slot_counts":   Box(0, 64, (INV_SLOTS,), np.int32),
        "main_hand_item_id": Discrete(N_ITEMS),
        "off_hand_item_id":  Discrete(N_ITEMS),
        "goal_embedding":    Box(-3, 3, (GOAL_EMBED_DIM,), np.float32),
        "goal_ticks_left":   Box(0, 12_000, (1,), np.int32),
        "time_of_day":       Box(0, 24_000, (1,), np.int32),
        "weather":           Discrete(3),
        "biome_id":          Discrete(N_BIOMES),
        "light_level":       Box(0, 15, (1,), np.int32),
        "comm_payloads":     Box(-1, 1, (COMM_BUFFER_SLOTS, COMM_PAYLOAD_DIM),
                                 np.float32),
        "comm_metadata":     Box(0, 1, (COMM_BUFFER_SLOTS, 8), np.float32),
    }


def _gatherer_overlay() -> dict:
    return {
        "g_resource_grid":     Box(0, 1, (32, 32, 6), np.float32),
        "g_nearest_resources": Box(-1, 1, (8, 6), np.float32),
        "g_richness_score":    Box(0, 1, (1,), np.float32),
        "g_hostiles_nearby":   Box(0, 1, (4, 4), np.float32),
    }


def _explorer_overlay() -> dict:
    """Explorer obs: richness + nearest resources (no grid conv, minimal obs)."""
    return {
        "g_resource_grid":     Box(0, 1, (32, 32, 6), np.float32),  # provided but may not be used
        "g_nearest_resources": Box(-1, 1, (8, 6), np.float32),
        "g_richness_score":    Box(0, 1, (1,), np.float32),
    }


def _farmer_overlay() -> dict:
    """Farmer obs: crop grid + ripeness + planted/harvested counts."""
    return {
        "f_crop_grid":          Box(0, 8, (32, 32), np.uint8),     # crop age per cell
        "f_ripeness":           Box(0, 1, (1,), np.float32),       # fraction of cells at stage 8
        "f_planted_count":      Box(0, 1024, (1,), np.int32),      # unique cells planted this episode
        "f_harvested_count":    Box(0, 1024, (1,), np.int32),      # cells harvested this episode
        "f_harvested_mask":     MultiBinary((32, 32)),             # has each cell been harvested?
        "f_time_at_ripeness":   Box(0, 100, (32, 32), np.int32),   # ticks since ripeness
    }


def build_role_observation_space(role: str, stage: int) -> DictSpace:
    if role == "gatherer":
        spaces = _core_space()
        spaces.update(_gatherer_overlay())
        spaces["action_mask"] = _action_mask_space(
            N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE
        )
        return DictSpace(spaces)
    elif role == "explorer":
        spaces = _core_space()
        spaces.update(_explorer_overlay())
        spaces["action_mask"] = _action_mask_space(
            8, 8  # explorer action_mask for discrete 8-bearing (simplified)
        )
        return DictSpace(spaces)
    elif role == "farmer":
        spaces = _core_space()
        spaces.update(_farmer_overlay())
        spaces["action_mask"] = _action_mask_space(
            N_FARMER_SKILLS, N_TARGET_CLASSES_PER_ROLE
        )
        return DictSpace(spaces)
    else:
        raise NotImplementedError(
            f"role {role!r} obs space not implemented"
        )


def build_role_action_space(role: str) -> DictSpace:
    if role == "gatherer":
        return DictSpace({
            "skill_type":       Discrete(N_GATHERER_SKILLS),
            "target_class":     Discrete(N_TARGET_CLASSES_PER_ROLE),
            "spatial_param":    Box(-1, 1, (3,), np.float32),
            "scalar_param":     Box(0, 1, (1,), np.float32),
            "comm_payload":     Box(-1, 1, (COMM_PAYLOAD_DIM,), np.float32),
            "should_broadcast": Discrete(2),
            "comm_target_mask": MultiBinary(4),
        })
    elif role == "explorer":
        # Explorer outputs a discrete 8-bearing via target_class reinterpretation
        return DictSpace({
            "target_class":     Discrete(8),  # bearing: N/NE/E/SE/S/SW/W/NW
        })
    elif role == "farmer":
        return DictSpace({
            "skill_type":       Discrete(N_FARMER_SKILLS),
            "target_class":     Discrete(N_TARGET_CLASSES_PER_ROLE),
            "spatial_param":    Box(-1, 1, (3,), np.float32),
            "scalar_param":     Box(0, 1, (1,), np.float32),
            "comm_payload":     Box(-1, 1, (COMM_PAYLOAD_DIM,), np.float32),
            "should_broadcast": Discrete(2),
            "comm_target_mask": MultiBinary(4),
        })
    else:
        raise NotImplementedError(
            f"role {role!r} action space not implemented"
        )
