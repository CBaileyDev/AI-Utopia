"""§7.3 — Py4J bridge wrapper.

Owns the JavaGateway lifecycle and the BATCHED observationsAll() call.
NOT per-agent observation(agent) — that pattern is forbidden per spec §4.6
(4× Py4J roundtrips/tick would cap throughput at ~300 agent-steps/sec).

`close()` is mandatory (PettingZoo lifecycle); without it, Ray worker
shutdown leaks Java processes that hold Py4J ports."""
from __future__ import annotations

import json
from typing import Any

from py4j.java_gateway import GatewayParameters, JavaGateway


class FabricBridge:
    """Single connection to one Fabric-side Py4J gateway."""

    def __init__(self, port: int):
        self.port = port
        self.gw: JavaGateway | None = None
        self.entry_point: Any = None

    def open(self) -> None:
        self.gw = JavaGateway(GatewayParameters(port=self.port, auto_field=True))
        self.entry_point = self.gw.entry_point

    def close(self) -> None:
        """Mandatory — see module docstring."""
        if self.gw is not None:
            try:
                self.gw.shutdown()
            finally:
                self.gw = None
                self.entry_point = None

    def __enter__(self) -> "FabricBridge":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ───── operations ─────
    def health(self) -> str:
        return str(self.entry_point.health())

    def observations_all(self) -> dict[str, dict]:
        """Single BATCHED call — returns dict mapping agent_id → obs_raw dict."""
        raw = str(self.entry_point.observationsAll())
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError(f"observationsAll must return a JSON object, got {type(parsed)}")
        return parsed

    def reset_world(self, seed: int) -> None:
        self.entry_point.resetWorld(int(seed))

    def advance_tick_await_events(self, timeout_ms: int = 30_000) -> list[str]:
        result = self.entry_point.advanceTickAwaitEvents(int(timeout_ms))
        return [str(x) for x in result]

    def dispatch_skill(self, agent_id: str, action_dict: dict,
                       skill_invocation_id: str) -> None:
        encoded = json.dumps(action_dict)
        self.entry_point.motorBridge().dispatchSkill(agent_id, encoded,
                                                      skill_invocation_id)

    def flush_comm_batch(self, messages: list[dict]) -> None:
        encoded = [json.dumps(m) for m in messages]
        # Py4J auto-converts Python list to java.util.List
        self.entry_point.commBus().flushBatch(encoded)

    def drain_chat_events(self) -> list[dict]:
        raw = self.entry_point.drainChatEvents()
        return [json.loads(str(x)) for x in raw]
