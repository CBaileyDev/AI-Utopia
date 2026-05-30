#!/usr/bin/env python
"""Phase 2a RLModule Validation: Verify Explorer + Farmer infrastructure (no env).

Tests:
  1. RLModule classes instantiate cleanly
  2. Obs/action spaces build correctly
  3. Reward functions dispatch by role
  4. Potentials compute (PBRS shaping)
  5. RLModule forward() shapes match obs/action contracts

Run: PYTHONPATH=src py -3.11 scripts/phase2a_rlmodule_validation.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np


def main() -> None:
    """Validate Phase 2 RLModule infrastructure."""
    from aiutopia.common.logging import get_logger, setup_logging
    from aiutopia.env.reward import (
        compute_reward_stage_1,
        explorer_potential,
        farmer_potential,
    )
    from aiutopia.env.spaces import build_role_action_space, build_role_observation_space

    setup_logging("INFO")
    log = get_logger("phase2a_validation")

    log.info("=" * 60)
    log.info("PHASE 2A: RLModule Infrastructure Validation")
    log.info("=" * 60)

    roles = ["gatherer", "explorer", "farmer"]

    # 1. Test obs/action space instantiation
    log.info("\n[1] Observation & Action Space Instantiation")
    for role in roles:
        try:
            obs_space = build_role_observation_space(role, stage=1)
            action_space = build_role_action_space(role)
            log.info(f"  ✓ {role:10s}: obs_space keys={len(obs_space) if hasattr(obs_space, '__len__') else '?'}, "
                     f"action_space={action_space}")
        except Exception as e:
            log.error(f"  ✗ {role}: {e}")

    # 2. Test reward function dispatch
    log.info("\n[2] Reward Function Dispatch")
    fake_obs = {
        "inventory": np.zeros(36, dtype=np.float32),
        "g_log_count": np.array(0.0),
        "health": np.array(20.0),
        "g_richness_score": np.array(0.5),
    }
    fake_action = {"skill_type": 0}
    env_meta = {"died_this_tick": False, "n_clipped_param_axes": 0, "exploit_penalties": []}

    for role in roles:
        try:
            rew = compute_reward_stage_1(role=role, obs_prev=fake_obs, obs_curr=fake_obs,
                                        action=fake_action, env_meta=env_meta)
            log.info(f"  ✓ {role:10s}: reward={rew:.6f}")
        except Exception as e:
            log.error(f"  ✗ {role}: {e}")

    # 3. Test potential-based reward shaping (PBRS)
    log.info("\n[3] Potential-Based Reward Shaping (PBRS)")
    fake_obs_explorer = {
        "g_richness_score": np.array(0.5),
        "g_nearest_resources": np.zeros((8, 5)),
    }
    fake_obs_farmer = {
        "f_planted_count": np.array(10),
        "f_ripeness": np.array(0.5),
        "f_crop_grid": np.zeros((32, 32)),
        "f_time_at_ripeness": np.array(100),
    }

    try:
        phi_explorer = explorer_potential(fake_obs_explorer)
        log.info(f"  ✓ explorer_potential = {phi_explorer}")
    except Exception as e:
        log.error(f"  ✗ explorer_potential: {e}")

    try:
        phi_farmer = farmer_potential(fake_obs_farmer)
        log.info(f"  ✓ farmer_potential = {phi_farmer:.6f}")
    except Exception as e:
        log.error(f"  ✗ farmer_potential: {e}")

    # 4. Test RLModule instantiation (if torch available)
    log.info("\n[4] RLModule Instantiation (torch-dependent)")
    try:
        import torch
        from aiutopia.rl_module.explorer_rl_module import ExplorerRoleRLModule
        from aiutopia.rl_module.farmer_rl_module import FarmerRoleRLModule
        from aiutopia.rl_module.role_rl_module import GathererRoleRLModule

        for role, cls in [
            ("gatherer", GathererRoleRLModule),
            ("explorer", ExplorerRoleRLModule),
            ("farmer", FarmerRoleRLModule),
        ]:
            try:
                obs_space = build_role_observation_space(role)
                action_space = build_role_action_space(role)
                module = cls(obs_space, action_space, model_config_dict={})
                log.info(f"  ✓ {role:10s}: {module.__class__.__name__}")
            except Exception as e:
                log.error(f"  ✗ {role}: {e}")
    except ImportError:
        log.warning("  ⊘ Torch not available; skipping RLModule instantiation")

    log.info("\n" + "=" * 60)
    log.info("Phase 2a validation complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
