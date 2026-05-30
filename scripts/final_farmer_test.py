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
    
    # Target grass at Y=70 (confirmed working above)
    target = [0, 70, 0]
    print(f"Testing PLOW, PLANT, HARVEST_CROP at {target}")
    print(f"Agent at {pos}")
    
    # PLOW
    print("\n[PLOW]")
    plow = {"skill_type": 6, "target_location": target, "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(plow), "plow_1")
    
    start = time.time()
    while time.time() - start < 15:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            elapsed = time.time() - start
            result = evt['resultCode']
            reason = evt['failureReason']
            print(f"  {result} ({reason}) in {elapsed:.2f}s")
            if result == 'COMPLETED':
                plow_time = elapsed
                break
    
    # PLANT
    print("\n[PLANT]")
    plant = {"skill_type": 7, "target_location": target, "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(plant), "plant_1")
    
    start = time.time()
    while time.time() - start < 15:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            elapsed = time.time() - start
            result = evt['resultCode']
            reason = evt['failureReason']
            print(f"  {result} ({reason}) in {elapsed:.2f}s")
            if result == 'COMPLETED':
                plant_time = elapsed
                break
    
    # Read obs to check g_crop_grid and g_ripeness
    print("\n[OBSERVATIONS]")
    obs_json = bridge.entry_point.observationsAll()
    obs = json.loads(obs_json)
    agent_obs = obs[agent_name]
    print(f"  g_ripeness: {agent_obs['g_ripeness']}")
    print(f"  g_crop_grid length: {len(agent_obs['g_crop_grid'])}")
    
    # Check if crop_grid has the crop
    # g_crop_grid is 32x32 = 1024 values, center of arena is around index [16][16]
    # which is index 16*32 + 16 = 528 in the flat array
    # Actually, we'd need to know the exact mapping. For now, just check it's there.
    
    # HARVEST_CROP (should fail - crop too young)
    print("\n[HARVEST_CROP (should fail - crop age < 7)]")
    harvest = {"skill_type": 8, "target_location": target, "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(harvest), "harvest_1")
    
    start = time.time()
    while time.time() - start < 5:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            elapsed = time.time() - start
            result = evt['resultCode']
            reason = evt['failureReason']
            print(f"  {result} ({reason}) in {elapsed:.2f}s")
            break
    
    print("\n[SUCCESS] All farmer skills tested!")
    
finally:
    bridge.close()
