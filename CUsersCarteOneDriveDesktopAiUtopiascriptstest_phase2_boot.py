"""Boot test for Phase 2 RLMODULE wiring (Explorer + Farmer).

This script validates that:
1. All new RLModule classes can be imported
2. Observation/action spaces build correctly
3. Configuration factories don't error on role setup
4. Policy mapping resolves all roles
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all new modules import without error."""
    print("Testing imports...")
    try:
        from aiutopia.rl_module.explorer_rl_module import ExplorerRoleRLModule
        from aiutopia.rl_module.farmer_rl_module import FarmerRoleRLModule
        print("  ✓ Explorer and Farmer RLModules imported")
        
        from aiutopia.rl_module.actor_head import (
            explorer_action_dist_config,
            farmer_action_dist_config,
            ExplorerActorHead,
            FarmerActorHead,
        )
        print("  ✓ Actor heads imported")
        
        from aiutopia.rl_module.role_encoder import (
            ExplorerRoleEncoder,
            FarmerRoleEncoder,
        )
        print("  ✓ Role encoders imported")
        
        from aiutopia.env.spaces import build_role_observation_space, build_role_action_space
        print("  ✓ Space builders imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spaces():
    """Test that obs/action spaces build for all roles."""
    print("\nTesting observation and action spaces...")
    try:
        from aiutopia.env.spaces import build_role_observation_space, build_role_action_space
        
        for role in ["gatherer", "explorer", "farmer"]:
            obs_space = build_role_observation_space(role, stage=1)
            act_space = build_role_action_space(role)
            print(f"  ✓ {role}: obs {len(obs_space.spaces)} keys, action {len(act_space.spaces)} keys")
        
        return True
    except Exception as e:
        print(f"  ✗ Space building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test that config factories resolve roles."""
    print("\nTesting config factories...")
    try:
        from aiutopia.train.config import _get_role_rl_module_class
        
        for role in ["gatherer", "explorer", "farmer"]:
            cls = _get_role_rl_module_class(role)
            print(f"  ✓ {role} -> {cls.__name__}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward():
    """Test that reward functions dispatch correctly."""
    print("\nTesting reward functions...")
    try:
        from aiutopia.env.reward import compute_reward_stage_1, farmer_potential, explorer_potential
        
        # Mock obs/action
        obs_dummy = {
            "inv_slot_item_ids": [0] * 36,
            "inv_slot_counts": [0] * 36,
            "g_richness_score": [0.1],
            "f_planted_count": [10],
            "f_harvested_count": [5],
            "f_crop_grid": [[0] * 32 for _ in range(32)],
            "f_ripeness": [0.3],
            "f_time_at_ripeness": [[0] * 32 for _ in range(32)],
            "tick_in_episode": [100],
        }
        env_meta = {"died_this_tick": False, "n_clipped_param_axes": 0, "exploit_penalties": []}
        
        # Test gatherer
        r_g = compute_reward_stage_1(
            role="gatherer", obs_prev=obs_dummy, obs_curr=obs_dummy,
            action={}, env_meta=env_meta
        )
        print(f"  ✓ gatherer reward: {r_g:.4f}")
        
        # Test explorer
        r_e = compute_reward_stage_1(
            role="explorer", obs_prev=obs_dummy, obs_curr=obs_dummy,
            action={}, env_meta=env_meta
        )
        print(f"  ✓ explorer reward: {r_e:.4f}")
        
        # Test farmer
        r_f = compute_reward_stage_1(
            role="farmer", obs_prev=obs_dummy, obs_curr=obs_dummy,
            action={}, env_meta=env_meta
        )
        print(f"  ✓ farmer reward: {r_f:.4f}")
        
        # Test potentials
        phi_e = explorer_potential(obs_dummy)
        phi_f = farmer_potential(obs_dummy)
        print(f"  ✓ explorer_potential: {phi_e:.4f}, farmer_potential: {phi_f:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Reward testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_imports()
    all_pass &= test_spaces()
    all_pass &= test_config()
    all_pass &= test_reward()
    
    print("\n" + "="*60)
    if all_pass:
        print("PHASE 2 BOOT TEST: PASS")
        sys.exit(0)
    else:
        print("PHASE 2 BOOT TEST: FAIL")
        sys.exit(1)
