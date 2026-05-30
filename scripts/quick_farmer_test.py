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
    print(f"Agent: {agent_name} at {pos}")
    
    # Plow a block right at the agent's feet (should be on grass at y=65)
    target = [int(pos[0]), int(pos[1])-1, int(pos[2])]
    print(f"Plow at {target}")
    
    plow = {"skill_type": 6, "target_location": target, "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(plow), "p1")
    
    start = time.time()
    while time.time() - start < 10:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            print(f"PLOW: {evt['resultCode']} ({evt['failureReason']}) in {time.time()-start:.2f}s")
            break
    
    # Plant
    plant = {"skill_type": 7, "target_location": target, "timeout_ticks": 600}
    bridge.entry_point.motorBridge().dispatchSkill(agent_name, json.dumps(plant), "p2")
    
    start = time.time()
    while time.time() - start < 10:
        events = bridge.entry_point.motorBridge().advanceTickAwaitEvents(100)
        if events:
            evt = json.loads(events[0])
            print(f"PLANT: {evt['resultCode']} ({evt['failureReason']}) in {time.time()-start:.2f}s")
            break
    
    # Final obs
    obs_json = bridge.entry_point.observationsAll()
    obs = json.loads(obs_json)
    print(f"g_ripeness: {obs[agent_name]['g_ripeness']}")
    
finally:
    bridge.close()
