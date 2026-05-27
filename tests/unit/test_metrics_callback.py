import pytest

pytest.importorskip("ray")

from aiutopia.train.callbacks import AiUtopiaMetricsCallback


def test_callback_instantiates() -> None:
    cb = AiUtopiaMetricsCallback()
    result = {"info": {"learner": {"gatherer_policy": {"entropy": 1.42}}}}
    cb.on_train_result(algorithm=None, result=result)
    assert result["custom_metrics"]["gatherer_policy/entropy"] == 1.42


def test_callback_handles_missing_learner_info() -> None:
    cb = AiUtopiaMetricsCallback()
    result = {}
    cb.on_train_result(algorithm=None, result=result)
    assert result.get("custom_metrics", {}) == {}


def test_callback_accepts_metrics_logger_kwarg() -> None:
    """RLlibCallback v2.40+ passes metrics_logger as kwarg."""
    cb = AiUtopiaMetricsCallback()
    result = {"info": {"learner": {}}}
    cb.on_train_result(algorithm=None, metrics_logger=None, result=result)
