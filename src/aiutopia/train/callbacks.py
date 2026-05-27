"""Section 7.4 Custom Ray RLlib callbacks for AI Utopia training.

Uses the new RLlibCallback base class (Ray 2.40+); DefaultCallbacks is a
deprecated shim. All on_train_result signatures include metrics_logger.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from ray.rllib.callbacks.callbacks import RLlibCallback
except ImportError:                                       # 2.40 fallback path
    from ray.rllib.algorithms.callbacks import DefaultCallbacks as RLlibCallback


class AiUtopiaMetricsCallback(RLlibCallback):
    """Per-iteration metrics: per-policy entropy, vf_loss, kl."""

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        result.setdefault("custom_metrics", {})
        try:
            learner_info = result["info"]["learner"]
        except KeyError:
            return
        for policy_id, info in learner_info.items():
            for src, dst in (("entropy",       "entropy"),
                              ("policy_entropy","entropy"),
                              ("vf_loss",       "vf_loss"),
                              ("kl",            "kl")):
                val = info.get(src)
                if val is not None:
                    result["custom_metrics"][f"{policy_id}/{dst}"] = float(val)


class ExploitHuntCallback(RLlibCallback):
    """Section 4 exploit hunt — every N iterations, surface exploit-penalty
    aggregates per exploit type."""

    def __init__(self, *, every_n_iters: int = 200) -> None:
        super().__init__()
        self.every_n_iters = every_n_iters
        self._iter = 0

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        self._iter += 1
        if self._iter % self.every_n_iters != 0:
            return
        result.setdefault("custom_metrics", {})
        sampler = result.get("env_runners", result.get("sampler_results", {}))
        episode_stats = sampler.get("episode_extra_stats", {})
        for key, value in episode_stats.items():
            if key.startswith("exploit_"):
                result["custom_metrics"][f"exploit_hunt/{key}"] = float(value)


class EvalGateStopCallback(RLlibCallback):
    """Section 5.8 M1 evaluation gate: 80% success on collect-64-oak_log
    over 3 consecutive evaluations.

    Each evaluation arrives roughly once every `eval_interval` train iters
    (M1EvalScenarioCallback emits one rate per N iters), so 3 consecutive
    evaluations >= ~3*N train iters of sustained success.

    On pass: writes `custom_metrics["M1/gate_passed"] = 1.0`. Tune's
    stop dict watches that key and terminates the trial gracefully.
    """

    def __init__(self, *, milestone: str = "M1",
                  success_metric: str = "eval_m1_oak_log_success_rate",
                  threshold: float = 0.80,
                  consecutive_required: int = 3) -> None:
        super().__init__()
        self.milestone           = milestone
        self.success_metric      = success_metric
        self.threshold           = threshold
        self.consecutive_required = consecutive_required
        self._recent: deque[float] = deque(maxlen=consecutive_required)
        self.gate_passed = False

    def on_train_result(self, *, algorithm, metrics_logger=None,
                          result, **kwargs):
        sampler = result.get("env_runners", result.get("sampler_results", {}))
        rate = sampler.get("episode_extra_stats", {}).get(self.success_metric)
        if rate is None:
            return
        self._recent.append(float(rate))
        result.setdefault("custom_metrics", {})
        result["custom_metrics"][f"{self.milestone}/gate_history"] = list(self._recent)
        if (len(self._recent) == self.consecutive_required
            and all(r >= self.threshold for r in self._recent)):
            self.gate_passed = True
            result["custom_metrics"][f"{self.milestone}/gate_passed"] = 1.0
            # Tune stop dict watches custom_metrics/{milestone}/gate_passed
        else:
            result["custom_metrics"][f"{self.milestone}/gate_passed"] = 0.0
