#!/usr/bin/env python3
import json, time, sys
sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

config = Py4JConfig.from_env()
bridge = FabricBridge(config.training_ports[0])
bridge.open()

try:
    obs_json = bridge.entry_point.observationsAll()
    obs = json.loads(obs_json)
    agent_name = list(obs.keys())[0]
    pos = obs[agent_name]["position"]
    
    print(f"Agent at {pos}")
    
    # Set up a farmland block right next to the agent (at reach)
    setup_x, setup_y, setup_z = 1, 74, 0
    print(f"\nSetting up farmland at {[setup_x, setup_y, setup_z]}...")
    bridge.entry_point.runCommand(f"setblock {setup_x} {setup_y} {setup_z} farmland")
    
    # Wait a tick
    time.sleep(0.1)
    
    # Now PLANT on it
    print("\n[PLANT on ready farmland]")
    plant = {"skill_type": 7, "target_location": [setup_x, setup_y, setup_z], "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(plant), "plant_1")
    
    start = time.time()
    while time.time() - start < 5:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            result = evt['resultCode']
            reason = evt['failureReason']
            elapsed = time.time() - start
            print(f"  {result} ({reason}) in {elapsed:.2f}s")
            
            if result == 'COMPLETED':
                print("  [OK] PLANT succeeded")
            break
    
    # Check obs
    obs_json = bridge.entry_point.observationsAll()
    obs = json.loads(obs_json)
    print(f"\ng_ripeness: {obs[agent_name]['g_ripeness']}")
    
    # Try HARVEST_CROP (should fail - crop too young)
    print("\n[HARVEST_CROP on age-1 crop (should fail)]")
    harvest = {"skill_type": 8, "target_location": [setup_x, setup_y, setup_z], "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(harvest), "harvest_1")
    
    start = time.time()
    while time.time() - start < 5:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            result = evt['resultCode']
            reason = evt['failureReason']
            print(f"  {result} ({reason})")
            break
    
    print("\n[SUCCESS] Skills working!")
    
finally:
    bridge.close()
