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
