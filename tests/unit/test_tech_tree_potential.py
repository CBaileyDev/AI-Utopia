import pytest

from aiutopia.env.reward import (
    LOG_VALUE,
    ROLE_INVENTORY_CAPS,
    ROLE_TASK_ITEMS,
    tech_tree_potential,
)


def test_log_value_table_has_oak_log_at_1_point_0() -> None:
    assert LOG_VALUE["oak_log"] == 1.0


def test_potential_zero_on_empty_inventory() -> None:
    assert tech_tree_potential({}, "gatherer") == 0.0


def test_potential_grows_with_inventory() -> None:
    p1 = tech_tree_potential({"oak_log": 1}, "gatherer")
    p10 = tech_tree_potential({"oak_log": 10}, "gatherer")
    p100 = tech_tree_potential({"oak_log": 100}, "gatherer")
    assert p1 < p10
    # Per §5.7, gatherer cap on oak_log is 256, so 100 < cap and the growth is linear
    assert p10 < p100


def test_potential_capped_per_role() -> None:
    # Gatherer cap on oak_log is 256; potential clamped above
    p256 = tech_tree_potential({"oak_log": 256}, "gatherer")
    p1000 = tech_tree_potential({"oak_log": 1000}, "gatherer")
    assert p256 == p1000  # cap applied


def test_unrecognized_item_ignored() -> None:
    p = tech_tree_potential({"unobtainium": 999}, "gatherer")
    assert p == 0.0


def test_per_role_caps_differ() -> None:
    # Builder has higher oak_planks cap than gatherer
    builder_cap = ROLE_INVENTORY_CAPS["builder"]["oak_planks"]
    gatherer_cap = ROLE_INVENTORY_CAPS["gatherer"]["oak_planks"]
    assert builder_cap > gatherer_cap


def test_rejects_unknown_role() -> None:
    with pytest.raises(KeyError):
        tech_tree_potential({"oak_log": 1}, "wizard")


# --- M1B single-attractor: gatherer Φ excludes off-task items ---------------


def test_gatherer_task_items_is_oak_log_only() -> None:
    assert ROLE_TASK_ITEMS["gatherer"] == frozenset({"oak_log"})


def test_gatherer_potential_ignores_cobblestone() -> None:
    # Off-task cobblestone must contribute 0 to the gatherer's PBRS potential,
    # otherwise the shaping channel re-creates the cobblestone attractor.
    assert tech_tree_potential({"cobblestone": 50}, "gatherer") == 0.0


def test_gatherer_potential_cobblestone_does_not_change_oak_log_phi() -> None:
    phi_logs_only = tech_tree_potential({"oak_log": 10}, "gatherer")
    phi_logs_plus_cobble = tech_tree_potential({"oak_log": 10, "cobblestone": 99}, "gatherer")
    # Φ counts only oak_log: adding cobblestone does not change it.
    assert phi_logs_plus_cobble == phi_logs_only
    assert phi_logs_only == 10 * LOG_VALUE["oak_log"]


def test_gatherer_potential_oak_log_scales_to_cap() -> None:
    cap = ROLE_INVENTORY_CAPS["gatherer"]["oak_log"]
    assert tech_tree_potential({"oak_log": 5}, "gatherer") == 5 * 1.0
    # capped at 256 * 1.0
    assert tech_tree_potential({"oak_log": cap + 100}, "gatherer") == cap * 1.0


def test_builder_potential_still_counts_cobblestone() -> None:
    # Regression guard: roles WITHOUT a task allowlist are UNCHANGED and still
    # weight cobblestone (and all other LOG_VALUE items) in their PBRS Φ.
    assert "builder" not in ROLE_TASK_ITEMS
    builder_cobble_cap = ROLE_INVENTORY_CAPS["builder"]["cobblestone"]
    n = 10  # below the builder cobblestone cap
    assert n < builder_cobble_cap
    expected = n * LOG_VALUE["cobblestone"]
    assert tech_tree_potential({"cobblestone": n}, "builder") == expected


def test_other_roles_have_no_task_allowlist() -> None:
    for role in ("builder", "farmer", "defender"):
        assert role not in ROLE_TASK_ITEMS
