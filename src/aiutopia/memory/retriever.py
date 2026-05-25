"""§5.6 — Memory retrieval with tiered recency-decay per query intent."""
from __future__ import annotations

import math

# §5.6 tiered recency lambdas (decay per in-game day; 24_000 ticks/day)
RECENCY_LAMBDA_LONG_TERM      = 0.02     # decay at ~50 days
RECENCY_LAMBDA_GENERAL        = 0.04     # default
RECENCY_LAMBDA_TIME_SENSITIVE = 0.05     # decay at ~20 days

TICKS_PER_GAME_DAY = 24_000


def recency_score(now_tick: int, mem_tick: int, recency_lambda: float) -> float:
    """exp(-λ × age_days). Clamped to [0, 1]."""
    age_days = max(0.0, (now_tick - mem_tick) / TICKS_PER_GAME_DAY)
    return math.exp(-recency_lambda * age_days)


# Static query templates per intent (§5.6 — LLM-composed deferred to Phase 5+).
QUERY_TEMPLATES = {
    "general":          "general context for planning: goal={goal}, role={role}",
    "player_history":   "interactions with player {player_name}",
    "combat":           "combat danger threat involving {threat_types}",
    "funeral":          "death funeral memorial of {predecessor_agent_name}",
}


def render_query(intent: str, **kwargs: str) -> str:
    tpl = QUERY_TEMPLATES[intent]
    return tpl.format(**kwargs)
