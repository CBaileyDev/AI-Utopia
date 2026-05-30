#!/usr/bin/env python3
import json, sys
sys.path.insert(0, 'src')

from aiutopia.common.config import Py4JConfig
from aiutopia.env.bridge import FabricBridge

config = Py4JConfig.from_env()
bridge = FabricBridge(config.training_ports[0])
bridge.open()

try:
    # Check what's around the agent
    obs_json = bridge.entry_point.observationsAll()
    obs = json.loads(obs_json)
    agent = list(obs.keys())[0]
    pos = obs[agent]["position"]
    print(f"Agent {agent} at {pos}")
    print(f"  Health: {obs[agent]['health']}")
    print(f"  Hunger: {obs[agent]['hunger']}")
    
    # Use runCommand to look around
    # Actually, let's just try plowing at various Y levels
    for dy in range(-5, 5):
        target = [0, int(pos[1]) + dy, 0]
        print(f"Testing Y={target[1]}...")
        result = bridge.entry_point.runCommand(f"setblock {target[0]} {target[1]} {target[2]} farmland")
        if result:
            print(f"  [OK] set to farmland")
            # Check what was there
            result = bridge.entry_point.runCommand(f"setblock {target[0]} {target[1]} {target[2]} grass_block")
            print(f"  [OK] reset to grass_block")
            break
        
finally:
    bridge.close()
