"""§4.7 — LLM planner scaffold. NOT runnable in M0.

The real planner (M5):
  - reads EventQueue (priority 0 ChatEvent > 1 FailureReport > 2 World > 3 Phase)
  - composes prompt with memory retrieval (§5.6)
  - calls Claude Haiku (5 s timeout, exp backoff)
  - 3-tier degradation: Haiku → Qwen 14B → halt + alert
  - writes to planner_state with planner_001_initial schema
"""
from __future__ import annotations


class StubPlanner:
    """M0 placeholder. Replaced by HaikuPlanner in M5."""

    def emit_paired_subgoals(self, role_pair: tuple[str, str]) -> dict:
        """Return a 2-subgoal LlmPlanOutput dict for M2 cooperative training."""
        raise NotImplementedError("M2 wires this with hand-coded blueprints")
