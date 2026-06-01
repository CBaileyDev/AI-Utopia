"""Reward-config externalization loader tests.

With NO config file present every value must be byte-identical to the
historical hardcoded literal — that keeps the parity + reward suites green.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aiutopia.env import reward as rwd
from aiutopia.env.reward import (
    build_default_reward_config,
    deep_merge_reward_config,
    load_reward_config,
    reward_config_to_serializable,
)


def test_default_constants_unchanged() -> None:
    """Module constants are byte-identical to the historical literals."""
    assert rwd.LOG_VALUE["oak_log"] == 1.000
    assert rwd.LOG_VALUE["cobblestone"] == 1.0 / 11.0
    assert rwd.LOG_VALUE["diamond"] == 8.000
    assert len(rwd.LOG_VALUE) == 48
    assert rwd.GAMMA == 0.99
    assert rwd.TIME_PENALTY == 0.001
    assert rwd.DEATH_PENALTY == 10.0
    assert rwd.ROLE_TASK_ITEMS["gatherer"] == frozenset({"oak_log"})
    assert rwd.ROLE_INVENTORY_CAPS["gatherer"]["oak_log"] == 256


def test_build_default_matches_module_constants() -> None:
    """The serialized default config reproduces the constants exactly."""
    cfg = build_default_reward_config()
    assert cfg["log_value"]["cobblestone"] == 1.0 / 11.0
    assert cfg["pbrs"]["gamma"] == 0.99
    assert cfg["pbrs"]["time_penalty"] == 0.001
    assert cfg["pbrs"]["death_penalty"] == 10.0
    assert cfg["role_task_items"]["gatherer"] == ["oak_log"]
    assert cfg["role_caps"]["gatherer"]["oak_log"] == 256


def test_deep_merge_partial_keeps_other_items() -> None:
    """A partial overlay updates one key and preserves the rest."""
    base = build_default_reward_config()
    merged = deep_merge_reward_config(base, {"log_value": {"oak_log": 2.0}})
    assert merged["log_value"]["oak_log"] == 2.0
    assert merged["log_value"]["cobblestone"] == 1.0 / 11.0
    assert merged["pbrs"]["gamma"] == 0.99
    merged2 = deep_merge_reward_config(base, {"pbrs": {"death_penalty": 5.0}})
    assert merged2["pbrs"]["death_penalty"] == 5.0
    assert merged2["pbrs"]["gamma"] == 0.99


def test_serializable_is_json_safe() -> None:
    """Reward_config_to_serializable produces a JSON-encodable structure."""
    cfg = build_default_reward_config()
    serial = reward_config_to_serializable(cfg)
    back = json.loads(json.dumps(serial))
    assert back["role_task_items"]["gatherer"] == ["oak_log"]


def test_load_without_file_equals_defaults(tmp_path, monkeypatch) -> None:
    """No overlay file → loaded config equals the defaults exactly."""
    monkeypatch.setenv("AIUTOPIA_REWARD_CONFIG", str(tmp_path / "absent.json"))
    assert load_reward_config() == build_default_reward_config()


def test_load_with_overlay_merges(tmp_path, monkeypatch) -> None:
    """An on-disk overlay is deep-merged over the defaults."""
    cfg_file = tmp_path / "rewards.json"
    cfg_file.write_text(json.dumps({"pbrs": {"death_penalty": 3.0}}), encoding="utf-8")
    monkeypatch.setenv("AIUTOPIA_REWARD_CONFIG", str(cfg_file))
    loaded = load_reward_config()
    assert loaded["pbrs"]["death_penalty"] == 3.0
    assert loaded["pbrs"]["gamma"] == 0.99
    assert loaded["log_value"]["oak_log"] == 1.000


def test_overlay_rebinds_constants_and_reward_math(tmp_path) -> None:
    """Overlay overrides the module constants AND flows into the reward math."""
    overlay = tmp_path / "rewards.json"
    overlay.write_text(
        json.dumps({"log_value": {"oak_log": 2.0}, "pbrs": {"death_penalty": 4.0}}),
        encoding="utf-8",
    )
    # primary signal for 1 harvested oak_log = 1 * 2.0 = 2.0 (plus PBRS/time);
    # with the default oak_log=1.0 it would be ~1.0, so > 1.9 proves the override.
    script = (
        "from aiutopia.env import reward as r\n"
        "assert r.LOG_VALUE['oak_log'] == 2.0, r.LOG_VALUE['oak_log']\n"
        "assert r.DEATH_PENALTY == 4.0, r.DEATH_PENALTY\n"
        "assert r.ROLE_TASK_ITEMS['gatherer'] == frozenset({'oak_log'})\n"
        "prev = {'inv_slot_item_ids': ['oak_log'], 'inv_slot_counts': [0]}\n"
        "curr = {'inv_slot_item_ids': ['oak_log'], 'inv_slot_counts': [1]}\n"
        "meta = {'died_this_tick': False, 'n_clipped_param_axes': 0, "
        "'exploit_penalties': []}\n"
        "rw = r.compute_reward_stage_1(role='gatherer', obs_prev=prev, "
        "obs_curr=curr, action={'skill_type': 1}, env_meta=meta)\n"
        "assert rw > 1.9, rw\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["AIUTOPIA_REWARD_CONFIG"] = str(overlay)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
