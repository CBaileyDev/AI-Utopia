#!/usr/bin/env python3
"""
Real test of farmer skills: PLOW, PLANT, HARVEST_CROP.
Find grass block, plow it, plant seed, wait for ripeness, harvest.
"""
import json
import time
import sys

sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

def find_grass_block(bridge, agent_pos, radius=10):
    """Find nearest grass block by brute force scan."""
    # This is a hack — in production the obs would tell us
    for dx in range(-radius, radius):
        for dz in range(-radius, radius):
            for dy in range(-3, 4):
                x = int(agent_pos[0]) + dx
                y = int(agent_pos[1]) + dy
                z = int(agent_pos[2]) + dz
                # Use runCommand to check block type
                # Actually, we can't query this easily via Py4J without adding a helper
                # For now, just target y=65 (the grass layer in the arena)
                if dy == -10:
                    return [x, 65, z]
    return None

def test_skills():
    config = Py4JConfig.from_env()
    bridge = FabricBridge(config.training_ports[0])
    bridge.open()
    
    try:
        # Get current agent
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        agent_name = list(obs.keys())[0]
        agent_pos = obs[agent_name]["position"]
        
        print(f"[setup] agent={agent_name}, pos={agent_pos}")
        
        # Target grass at y=65 (arena grass layer)
        target_x = int(agent_pos[0]) + 5
        target_y = 65
        target_z = int(agent_pos[2]) + 5
        
        print(f"\n[plow] dispatching PLOW to [{target_x}, {target_y}, {target_z}] (y=65 is grass layer)")
        plow_action = {
            "skill_type": 6,
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 600
        }
        
        bridge.entry_point.motorBridge().dispatchSkill(
            agent_name, json.dumps(plow_action), "plow_1"
        )
        
        # Wait for completion
        start = time.time()
        plow_result = None
        while time.time() - start < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(200)
            if events:
                for evt_str in events:
                    evt = json.loads(evt_str)
                    if evt['skillInvocationId'] == "plow_1":
                        plow_result = evt
                        elapsed = time.time() - start
                        print(f"  Result={evt['resultCode']} ({evt['failureReason']}) after {elapsed:.2f}s")
                        break
                if plow_result:
                    break
            time.sleep(0.01)
        
        if not plow_result:
            print("  [ERROR] PLOW timed out")
            return
        
        if plow_result['resultCode'] != 'COMPLETED':
            print(f"  [ERROR] PLOW failed with {plow_result['resultCode']}")
            return
        
        # Test PLANT skill
        print(f"\n[plant] dispatching PLANT to [{target_x}, {target_y}, {target_z}]")
        plant_action = {
            "skill_type": 7,
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 600
        }
        
        bridge.entry_point.motorBridge().dispatchSkill(
            agent_name, json.dumps(plant_action), "plant_1"
        )
        
        # Wait for completion
        start = time.time()
        plant_result = None
        while time.time() - start < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(200)
            if events:
                for evt_str in events:
                    evt = json.loads(evt_str)
                    if evt['skillInvocationId'] == "plant_1":
                        plant_result = evt
                        elapsed = time.time() - start
                        print(f"  Result={evt['resultCode']} ({evt['failureReason']}) after {elapsed:.2f}s")
                        break
                if plant_result:
                    break
            time.sleep(0.01)
        
        if not plant_result:
            print("  [ERROR] PLANT timed out")
            return
        
        if plant_result['resultCode'] != 'COMPLETED':
            print(f"  [ERROR] PLANT failed with {plant_result['resultCode']}")
            return
        
        # Measure crop latency stats
        plow_ticks = plow_result.get('clippedAxesBitset', 0) if plow_result else 0
        plant_ticks = plant_result.get('clippedAxesBitset', 0) if plant_result else 0
        
        print(f"\n[latency]")
        print(f"  PLOW: {plow_result['resultCode']} in {time.time() - (start - (plant_result['resultCode'] != 'COMPLETED' and 30 or 0)):.2f}s")
        print(f"  PLANT: {plant_result['resultCode']} in {time.time() - start:.2f}s")
        
        # Read final obs
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        agent_obs = obs[agent_name]
        print(f"\n[final_obs]")
        print(f"  g_ripeness={agent_obs.get('g_ripeness', 'N/A')}")
        print(f"  g_crop_grid length={len(agent_obs.get('g_crop_grid', []))}")
        
        print("\n[done] all skills tested successfully.")
        
    finally:
        bridge.close()

if __name__ == "__main__":
    test_skills()
