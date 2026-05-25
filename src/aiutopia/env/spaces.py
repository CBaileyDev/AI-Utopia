"""§4.1, §4.2 — Per-role observation + action Dict spaces.

M0 implements gatherer only. Other roles raise NotImplementedError until
their milestones (builder M2, farmer M3, defender M4)."""
from __future__ import annotations

from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiBinary, MultiDiscrete
import numpy as np

# Fixed constants — must agree with Java side (motor_module.encode_action).
N_ITEMS         = 1024     # MC 1.21 item-id space (sparse OK; this is the cap)
N_BIOMES        = 64
INV_SLOTS       = 36
GOAL_EMBED_DIM  = 512
COMM_PAYLOAD_DIM= 128
COMM_BUFFER_SLOTS= 32      # §3 carry-forward (1.6 s history at 20 TPS)

N_GATHERER_SKILLS         = 6   # navigate, harvest, deposit_chest, search, wait, noop_broadcast
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


def build_role_observation_space(role: str, stage: int) -> DictSpace:
    if role != "gatherer":
        raise NotImplementedError(
            f"role {role!r} obs space not implemented in M0 (see milestone map)"
        )
    spaces = _core_space()
    spaces.update(_gatherer_overlay())
    spaces["action_mask"] = _action_mask_space(
        N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE
    )
    return DictSpace(spaces)


def build_role_action_space(role: str) -> DictSpace:
    if role != "gatherer":
        raise NotImplementedError(
            f"role {role!r} action space not implemented in M0"
        )
    return DictSpace({
        "skill_type":       Discrete(N_GATHERER_SKILLS),
        "target_class":     Discrete(N_TARGET_CLASSES_PER_ROLE),
        "spatial_param":    Box(-1, 1, (3,), np.float32),
        "scalar_param":     Box(0, 1, (1,), np.float32),
        "comm_payload":     Box(-1, 1, (COMM_PAYLOAD_DIM,), np.float32),
        "should_broadcast": Discrete(2),
        "comm_target_mask": MultiBinary(4),
    })
