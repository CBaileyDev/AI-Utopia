from aiutopia.train.callbacks import EvalGateStopCallback


def _emit(cb, rate: float) -> dict:
    result = {"env_runners": {"episode_extra_stats": {"eval_m1_oak_log_success_rate": rate}}}
    cb.on_train_result(algorithm=None, result=result)
    return result


def test_gate_not_passed_until_three_consecutive() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    _emit(cb, 0.9)
    assert not cb.gate_passed
    _emit(cb, 0.85)
    assert not cb.gate_passed
    r = _emit(cb, 0.95)
    assert cb.gate_passed
    assert r["custom_metrics"]["M1/gate_passed"] == 1.0


def test_gate_resets_on_low_evaluation() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    _emit(cb, 0.9)
    _emit(cb, 0.85)
    _emit(cb, 0.5)
    assert not cb.gate_passed
    _emit(cb, 0.9)
    _emit(cb, 0.9)
    assert not cb.gate_passed
    _emit(cb, 0.9)
    assert cb.gate_passed


def test_gate_writes_zero_when_not_passed() -> None:
    cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    r = _emit(cb, 0.5)
    assert r["custom_metrics"]["M1/gate_passed"] == 0.0


def test_gate_ignores_results_without_evaluation_metric() -> None:
    cb = EvalGateStopCallback()
    cb.on_train_result(algorithm=None, result={})
    assert not cb.gate_passed


def test_m1_evaluation_scenario_callback_throttled() -> None:
    from aiutopia.train.callbacks import M1EvalScenarioCallback
    cb = M1EvalScenarioCallback(eval_interval=5)
    result = {}
    for _ in range(4):
        cb.on_train_result(algorithm=None, result=result)
    assert "eval_m1_oak_log_success_rate" not in (
        result.get("env_runners", {}).get("episode_extra_stats", {}))
