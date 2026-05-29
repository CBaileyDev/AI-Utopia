"""Phase-0 fix #3 unit tests for the M1 gate predicate.

Targets `_gatherer_collected_64_oak_log` directly with hand-built `final_obs`
dicts so the test is fully isolated from the heavyweight env/wrapper chain
(`AiUtopiaPettingZooEnv` is imported lazily inside `run_scenario`; only the
leaf module reward.py is imported at module scope). No live Fabric server is
touched.

The predicate was DOUBLY broken before this fix:
  (a) FALSE-POSITIVE: a fallback summed ALL slot counts when no oak_log was
      found, so 64 cobblestone falsely passed the gate.
  (b) latent FALSE-NEGATIVE: the obs space declares `inv_slot_item_ids` as
      MultiDiscrete (INTEGER ids), but the old check did `"oak_log" in str(i)`
      which never matches an int id — the fallback was the only thing that
      ever made the predicate pass in production. Deleting (a) without fixing
      (b) would have made the gate never pass.

The fix resolves the slot arrays via reward._inventory_from_obs (the same
function/table the reward uses), so the gate handles BOTH int and string ids
and matches the exact canonical name "oak_log".
"""

from __future__ import annotations

import pytest

from aiutopia.env.reward import _ITEM_ID_TO_NAME
from aiutopia.train.scenario_runner import (
    M1_GATE_ENV_STEP_BUDGET,
    M1_OAK_LOG_TARGET,
    M1_SCENARIOS,
    _gatherer_collected_64_oak_log,
)


def _final_obs(item_ids: list, counts: list[int]) -> dict:
    """Build a final_obs dict in the shape run_scenario produces:
    {"gatherer_0": {"inv_slot_item_ids": [...], "inv_slot_counts": [...]}}."""
    return {
        "gatherer_0": {
            "inv_slot_item_ids": item_ids,
            "inv_slot_counts": counts,
        }
    }


# Integer IDs derived from the eager-seed table (NOT hardcoded guesses) so the
# int-ID tests stay valid if the table is renumbered. Unit tests run without
# env init, so only the static seed is present (Java augmentation never fires).
def _id_for(name: str) -> int:
    try:
        return next(i for i, n in _ITEM_ID_TO_NAME.items() if n == name)
    except StopIteration:  # pragma: no cover - guards a table regression
        pytest.skip(
            f"{name!r} not in eager-seed _ITEM_ID_TO_NAME; "
            "int-ID test cannot run without env init"
        )


OAK_LOG_ID = _id_for("oak_log")
COBBLESTONE_ID = _id_for("cobblestone")


# ───── string-id cases (task spec; production also emits namespaced strings
# only if Java ever switches — _inventory_from_obs handles them either way) ─────


def test_exactly_64_oak_log_string_passes() -> None:
    obs = _final_obs(["minecraft:oak_log"], [64])
    assert _gatherer_collected_64_oak_log(obs) is True


def test_63_oak_log_string_fails() -> None:
    obs = _final_obs(["minecraft:oak_log"], [63])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_64_cobblestone_string_no_oak_log_fails() -> None:
    """Regression guard for the DELETED false-pass fallback: a non-oak_log
    inventory must NOT satisfy the gate even when total counts >= 64."""
    obs = _final_obs(["minecraft:cobblestone"], [64])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_60_oak_log_plus_10_cobblestone_string_fails() -> None:
    obs = _final_obs(
        ["minecraft:oak_log", "minecraft:cobblestone"],
        [60, 10],
    )
    assert _gatherer_collected_64_oak_log(obs) is False


def test_64_oak_log_plus_10_cobblestone_string_passes() -> None:
    obs = _final_obs(
        ["minecraft:oak_log", "minecraft:cobblestone"],
        [64, 10],
    )
    assert _gatherer_collected_64_oak_log(obs) is True


def test_bare_oak_log_id_matches() -> None:
    obs = _final_obs(["oak_log"], [64])
    assert _gatherer_collected_64_oak_log(obs) is True


# ───── integer-id cases (production reality: MultiDiscrete obs) ─────
# These are the cases the OLD predicate got wrong: int ids never matched the
# string check, so only the false-pass fallback ever made it return True.


def test_exactly_64_oak_log_int_passes() -> None:
    obs = _final_obs([OAK_LOG_ID], [64])
    assert _gatherer_collected_64_oak_log(obs) is True


def test_63_oak_log_int_fails() -> None:
    obs = _final_obs([OAK_LOG_ID], [63])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_64_cobblestone_int_no_oak_log_fails() -> None:
    """The critical regression: 64 cobblestone (int id) must NOT pass.
    Under the old fallback this FALSELY passed; under the old string-only
    check the SAME inventory would also wrongly pass via the fallback."""
    obs = _final_obs([COBBLESTONE_ID], [64])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_60_oak_log_plus_10_cobblestone_int_fails() -> None:
    obs = _final_obs([OAK_LOG_ID, COBBLESTONE_ID], [60, 10])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_64_oak_log_plus_10_cobblestone_int_passes() -> None:
    obs = _final_obs([OAK_LOG_ID, COBBLESTONE_ID], [64, 10])
    assert _gatherer_collected_64_oak_log(obs) is True


def test_oak_log_split_across_slots_int_sums() -> None:
    """oak_log spread over multiple stacks (int ids) is summed across slots."""
    obs = _final_obs([OAK_LOG_ID, OAK_LOG_ID, COBBLESTONE_ID], [32, 32, 64])
    assert _gatherer_collected_64_oak_log(obs) is True


# ───── edge cases ─────


def test_empty_inventory_fails() -> None:
    obs = _final_obs([], [])
    assert _gatherer_collected_64_oak_log(obs) is False


def test_missing_gatherer_agent_fails_gracefully() -> None:
    """A final_obs with no gatherer_0 (e.g. pruned agent) must not pass."""
    assert _gatherer_collected_64_oak_log({}) is False


# ───── budget reconciliation (fix #3(b)) ─────


def test_scenario_budget_is_full_gate_horizon() -> None:
    """All M1 scenarios must allow the full 1000-env-step gate horizon, not
    the previous premature 300-step cap. Both the runner loop bound and the
    env truncation counter are in ENV STEPS, so 1000 here == 1000 env steps."""
    assert M1_GATE_ENV_STEP_BUDGET == 1000
    assert len(M1_SCENARIOS) == 3
    for sc in M1_SCENARIOS:
        assert sc.max_ticks == 1000
        assert sc.success is _gatherer_collected_64_oak_log


def test_oak_log_target_constant() -> None:
    assert M1_OAK_LOG_TARGET == 64
