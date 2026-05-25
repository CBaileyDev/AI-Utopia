import numpy as np

from aiutopia.memory.retriever import recency_score, RECENCY_LAMBDA_GENERAL


def test_recency_score_at_zero_age_is_one() -> None:
    assert recency_score(now_tick=100, mem_tick=100,
                          recency_lambda=RECENCY_LAMBDA_GENERAL) == 1.0


def test_recency_score_decays_with_age() -> None:
    s_recent = recency_score(now_tick=200, mem_tick=100,
                               recency_lambda=RECENCY_LAMBDA_GENERAL)
    s_old    = recency_score(now_tick=200_000, mem_tick=100,
                               recency_lambda=RECENCY_LAMBDA_GENERAL)
    assert s_recent > s_old > 0.0


def test_recency_score_never_negative() -> None:
    # Even with negative age (memory from the future, shouldn't happen but…)
    s = recency_score(now_tick=0, mem_tick=100, recency_lambda=0.05)
    assert s >= 0.0


def test_recency_score_lambda_per_intent_constants_exist() -> None:
    from aiutopia.memory.retriever import (
        RECENCY_LAMBDA_LONG_TERM,
        RECENCY_LAMBDA_GENERAL,
        RECENCY_LAMBDA_TIME_SENSITIVE,
    )
    assert RECENCY_LAMBDA_LONG_TERM    < RECENCY_LAMBDA_GENERAL
    assert RECENCY_LAMBDA_GENERAL      < RECENCY_LAMBDA_TIME_SENSITIVE
