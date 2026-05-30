#!/usr/bin/env python3
"""
Simple test of farmer skills: PLOW, PLANT.
"""
import json
import time
import sys

sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

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
        
        # Test PLOW skill
        target_x = int(agent_pos[0]) + 2
        target_y = int(agent_pos[1])
        target_z = int(agent_pos[2]) + 2
        
        print(f"\n[plow] dispatching PLOW to [{target_x}, {target_y}, {target_z}]")
        plow_action = {
            "skill_type": 6,
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 300
        }
        
        bridge.entry_point.motorBridge().dispatchSkill(
            agent_name, json.dumps(plow_action), "plow_test_1"
        )
        
        # Wait for completion
        start = time.time()
        while time.time() - start < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(500)
            if events:
                for evt_str in events:
                    evt = json.loads(evt_str)
                    if evt['skillInvocationId'] == "plow_test_1":
                        elapsed = time.time() - start
                        print(f"  Result={evt['resultCode']} ({evt['failureReason']}) after {elapsed:.1f}s")
                        break
                break
            time.sleep(0.05)
        
        # Test PLANT skill
        print(f"\n[plant] dispatching PLANT to [{target_x}, {target_y}, {target_z}]")
        plant_action = {
            "skill_type": 7,
            "target_location": [target_x, target_y, target_z],
            "timeout_ticks": 300
        }
        
        bridge.entry_point.motorBridge().dispatchSkill(
            agent_name, json.dumps(plant_action), "plant_test_1"
        )
        
        # Wait for completion
        start = time.time()
        while time.time() - start < 30:
            events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(500)
            if events:
                for evt_str in events:
                    evt = json.loads(evt_str)
                    if evt['skillInvocationId'] == "plant_test_1":
                        elapsed = time.time() - start
                        print(f"  Result={evt['resultCode']} ({evt['failureReason']}) after {elapsed:.1f}s")
                        break
                break
            time.sleep(0.05)
        
        # Read final obs to check crop grid
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        agent_obs = obs[agent_name]
        print(f"\n[final_obs] g_ripeness={agent_obs.get('g_ripeness', 'N/A')}")
        
        print("\n[done] skill test complete.")
        
    finally:
        bridge.close()

if __name__ == "__main__":
    test_skills()
