from aiutopia.planner.chat_drain import classify_reply_type


def test_question_mark_classifies_as_text() -> None:
    assert classify_reply_type("where is the iron?") == "text"


def test_imperative_verb_classifies_as_action_ack() -> None:
    for cmd in ("come help me", "bring wood here", "attack the zombie",
                "defend the wall", "stop digging", "move to spawn",
                "build me a tower", "gather more stone"):
        assert classify_reply_type(cmd) == "action_ack", cmd


def test_statement_classifies_as_none() -> None:
    assert classify_reply_type("thanks") == "none"
    assert classify_reply_type("nice work") == "none"
