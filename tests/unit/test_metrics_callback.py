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


def test_exploit_hunt_callback_throttled() -> None:
    from aiutopia.train.callbacks import ExploitHuntCallback
    cb = ExploitHuntCallback(every_n_iters=3)
    result = {"env_runners": {"episode_extra_stats": {"exploit_drop_spam": 0.5}}}
    for i in range(1, 9):
        result["custom_metrics"] = {}
        cb.on_train_result(algorithm=None, result=result)
        if i % 3 == 0:
            assert "exploit_hunt/exploit_drop_spam" in result["custom_metrics"]
        else:
            assert "exploit_hunt/exploit_drop_spam" not in result["custom_metrics"]
