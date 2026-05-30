#!/usr/bin/env python3
"""
Quick test of farmer observation builder.
Requires: instance-1 running on Py4J port 25001 with a farmer agent.
"""
import json
import time
import sys

sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

def test_farmer_obs():
    config = Py4JConfig.from_env()
    bridge = FabricBridge(config.training_ports[0])
    bridge.open()
    
    try:
        # First, read what agents are currently in observations
        print("[obs] reading current observations...")
        obs_json = bridge.entry_point.observationsAll()
        obs = json.loads(obs_json)
        print(f"Current agents in obs: {list(obs.keys())}")
        
        if len(obs) == 0:
            print("No agents found. Attempting to spawn gatherer_test...")
            bridge.entry_point.carpetSpawn("gatherer_test", "Steve", "gatherer")
            time.sleep(1)
            obs_json = bridge.entry_point.observationsAll()
            obs = json.loads(obs_json)
            print(f"After spawn: {list(obs.keys())}")
        
        # Check first agent in obs
        if obs:
            agent_name = list(obs.keys())[0]
            agent_obs = obs[agent_name]
            print(f"\nObservations for {agent_name}:")
            print(f"  Keys: {list(agent_obs.keys())}")
            
            # Check for gatherer overlay
            if "g_resource_grid" in agent_obs:
                print(f"  [GATHERER] g_resource_grid present (len={len(agent_obs['g_resource_grid'])})")
            
            # Check for farmer overlay
            if "g_crop_grid" in agent_obs:
                print(f"  [FARMER] g_crop_grid present (len={len(agent_obs['g_crop_grid'])})")
            if "g_ripeness" in agent_obs:
                print(f"  [FARMER] g_ripeness present (value={agent_obs['g_ripeness']})")
        
        print("\n[done] observation test complete.")
        
    finally:
        bridge.close()

if __name__ == "__main__":
    test_farmer_obs()
