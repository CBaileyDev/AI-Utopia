import pytest

from aiutopia.env.reward import (
    LOG_VALUE,
    ROLE_INVENTORY_CAPS,
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
