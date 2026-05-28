"""Confirm which logs remain in the arena after n16's two HARVEST attempts."""
from py4j.java_gateway import GatewayParameters, JavaGateway

gw = JavaGateway(gateway_parameters=GatewayParameters(port=25001, auto_convert=True, auto_field=False))
ep = gw.entry_point
LOGS = [(69,-48),(68,-45),(64,-43),(60,-45),(60,-49),(61,-51),(65,-54),(69,-51)]
for x, z in LOGS:
    ep.runCommand(f"/say PROBE_LOG_{x}_{z}")
    ep.runCommand(f"/execute if block {x} 66 {z} oak_log run say PROBE_FOUND_{x}_{z}")
    ep.runCommand(f"/execute unless block {x} 66 {z} oak_log run say PROBE_MISSING_{x}_{z}")
