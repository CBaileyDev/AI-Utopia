"""§3.3 — ChatBridge reply-type heuristic + drain loop scaffold.

In M0 this drains queued ChatEvents from the Java side and prints them.
M5 replaces the print with the real LLM round-trip + /tellraw response.
The classifier is committed in M0 so the heuristic is testable now."""
from __future__ import annotations

from aiutopia.schemas.enums import ExpectedReplyType


_IMPERATIVE_VERBS = frozenset({
    "come", "bring", "stop", "attack", "defend", "gather", "build", "move",
    "follow", "wait", "go", "drop", "pick", "kill", "heal", "trade",
})


def classify_reply_type(text: str) -> ExpectedReplyType:
    """§3.3 — heuristic for `expected_reply_type`.

    Upgrade to LLM classifier in Phase 5+ if heuristic miscategorizes >10%
    of messages."""
    if "?" in text:
        return "text"
    first_words = text.strip().lower().split()[:3]
    if any(w in _IMPERATIVE_VERBS for w in first_words):
        return "action_ack"
    return "none"
