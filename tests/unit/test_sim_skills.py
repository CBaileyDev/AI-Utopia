# tests/unit/test_sim_skills.py
import numpy as np

from aiutopia.sim.skills import apply_skill
from aiutopia.sim.world import SimWorld


def _harvest(target_class=0, scalar=1 / 64):
    return {
        "skill_type": 1,
        "target_class": target_class,
        "spatial_param": np.zeros(3, np.float32),
        "scalar_param": np.array([scalar], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


def test_harvest_collects_one_log_and_moves_into_reach():
    w = SimWorld()
    w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(scalar=1 / 64))  # cap=1
    assert w.inventory.get("oak_log", 0) == 1
    assert comp["resultCode"] == "COMPLETED"
    assert int(w.log_alive.sum()) == 63  # exactly one removed


def test_harvest_cap_collects_up_to_cap():
    w = SimWorld()
    w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(scalar=1.0))  # cap=64
    assert w.inventory.get("oak_log", 0) == 64
    assert int(w.log_alive.sum()) == 0


def test_harvest_fails_when_no_logs_left():
    w = SimWorld()
    w.reset(seed=1)
    w.log_alive[:] = False
    w, comp = apply_skill(w, _harvest())
    assert comp["resultCode"] in ("FAILED_TIMEOUT", "IMMEDIATE_FAILURE")
    assert w.inventory.get("oak_log", 0) == 0


def test_harvest_wrong_target_class_collects_nothing():
    # Phase-C transfer fix: target_class must match oak_log (0). Any other class
    # matches no resource in the M1B arena -> IMMEDIATE_FAILURE, 0 collected,
    # mirroring real MC. (Without this gate the sim ignored target_class, the
    # policy learned an arbitrary 54, and real MC collected 0.)
    w = SimWorld()
    w.reset(seed=1)
    w, comp = apply_skill(w, _harvest(target_class=54))  # the observed bad class
    assert comp["resultCode"] == "IMMEDIATE_FAILURE"
    assert w.inventory.get("oak_log", 0) == 0
    assert int(w.log_alive.sum()) == 64  # nothing harvested
