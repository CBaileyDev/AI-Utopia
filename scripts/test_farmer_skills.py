#!/usr/bin/env python3
"""
Quick test of farmer skills: PLOW, PLANT, HARVEST_CROP.
Requires: instance-1 running on Py4J port 25001 with farmer agent spawned.
"""
import json
import time
import sys

# Add src to path
sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

def test_farmer_skills():
    config = Py4JConfig.from_env()
    bridge = FabricBridge(config.training_ports[0])  # Instance-1, port 25001
    bridge.open()
    
    try:
        # Spawn a farmer agent if not already spawned
        print("[setup] spawning farmer agent...")
        bridge.entry_point.carpetSpawn("farmer_test", "Steve", "farmer")
        time.sleep(2)
        
        # Read initial obs
        print("[obs] reading farmer observations...")
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        print(f"  Agents in obs: {list(obs.keys())}")
        
        if "farmer_test" in obs:
            farmer_obs = obs["farmer_test"]
            if "g_crop_grid" in farmer_obs:
                print(f"  ✓ g_crop_grid present (len={len(farmer_obs['g_crop_grid'])})")
            if "g_ripeness" in farmer_obs:
                print(f"  ✓ g_ripeness present (value={farmer_obs['g_ripeness']})")
        
        # Find a dirt block to test PLOW
        print("\n[plow] testing PLOW skill...")
        agent_pos = obs["farmer_test"]["position"]
        print(f"  Agent position: {agent_pos}")
        
        # Test PLOW: target a block 2 blocks away
        target_x = int(agent_pos[0]) + 3
        target_y = int(agent_pos[1])
        target_z = int(agent_pos[2])
        
        plow_action = {
            "skill_type": 6,  # PLOW
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 300
        }
        
        skill_id = "test_plow_1"
        bridge.entry_point.motorBridge().dispatchSkill(
            "farmer_test", json.dumps(plow_action), skill_id
        )
        print(f"  Dispatched PLOW to [{target_x}, {target_y}, {target_z}]")
        
        # Wait for skill to complete
        print("  Waiting for PLOW completion...")
        start_time = time.time()
        while time.time() - start_time < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(500)
            for event_str in events:
                event = json.loads(event_str)
                print(f"  ✓ PLOW result: {event['resultCode']} ({event['failureReason']})")
                if event['skillInvocationId'] == skill_id:
                    break
            if events:
                break
            time.sleep(0.1)
        
        # Test PLANT: plant at the same location
        print("\n[plant] testing PLANT skill...")
        plant_action = {
            "skill_type": 7,  # PLANT
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 300
        }
        
        skill_id = "test_plant_1"
        bridge.entry_point.motorBridge().dispatchSkill(
            "farmer_test", json.dumps(plant_action), skill_id
        )
        print(f"  Dispatched PLANT to [{target_x}, {target_y}, {target_z}]")
        
        # Wait for skill to complete
        print("  Waiting for PLANT completion...")
        start_time = time.time()
        while time.time() - start_time < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(500)
            for event_str in events:
                event = json.loads(event_str)
                print(f"  ✓ PLANT result: {event['resultCode']} ({event['failureReason']})")
                if event['skillInvocationId'] == skill_id:
                    break
            if events:
                break
            time.sleep(0.1)
        
        # Test HARVEST_CROP: try to harvest the newly planted crop (should fail - too young)
        print("\n[harvest_crop] testing HARVEST_CROP skill (should fail - crop too young)...")
        harvest_action = {
            "skill_type": 8,  # HARVEST_CROP
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 300
        }
        
        skill_id = "test_harvest_1"
        bridge.entry_point.motorBridge().dispatchSkill(
            "farmer_test", json.dumps(harvest_action), skill_id
        )
        print(f"  Dispatched HARVEST_CROP to [{target_x}, {target_y}, {target_z}]")
        
        # Wait for skill to complete
        print("  Waiting for HARVEST_CROP result...")
        start_time = time.time()
        while time.time() - start_time < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(500)
            for event_str in events:
                event = json.loads(event_str)
                print(f"  ✓ HARVEST_CROP result: {event['resultCode']} ({event['failureReason']})")
                if event['skillInvocationId'] == skill_id:
                    break
            if events:
                break
            time.sleep(0.1)
        
        # Read final obs
        print("\n[final_obs] reading farmer observations...")
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        if "farmer_test" in obs:
            farmer_obs = obs["farmer_test"]
            if "g_crop_grid" in farmer_obs:
                print(f"  ✓ g_crop_grid present (len={len(farmer_obs['g_crop_grid'])})")
            if "g_ripeness" in farmer_obs:
                print(f"  ✓ g_ripeness present (value={farmer_obs['g_ripeness']})")
        
        print("\n[done] farmer skills test complete.")
        
    finally:
        bridge.close()

if __name__ == "__main__":
    test_farmer_skills()
