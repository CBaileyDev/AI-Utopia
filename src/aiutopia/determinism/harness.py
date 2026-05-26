"""§7.8 — Determinism utilities. Two metrics per §5.10:
  - action_argmax_divergence < 0.05
  - continuous_param_L2     < 0.10
over a 1000-tick replay window."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


EPS_ARGMAX = 0.05
EPS_L2     = 0.10


def configure_cuda_determinism() -> None:
    """Pin every cuDNN / cuBLAS knob that controls non-determinism.
    Call once at process start before importing torch CUDA ops.

    Without this, cuDNN autotuner randomizes per-process and the
    determinism gate becomes a flaky test."""
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class ReplayDivergence:
    action_argmax_divergence: float
    continuous_param_l2:      float

    @property
    def passes(self) -> bool:
        return (self.action_argmax_divergence < EPS_ARGMAX
                and self.continuous_param_l2 < EPS_L2)


def compute_divergence(trace_a: list[dict],
                       trace_b: list[dict]) -> ReplayDivergence:
    """Each trace entry must have keys `action_argmax: int` and
    `continuous_params: np.ndarray`."""
    if len(trace_a) != len(trace_b):
        raise ValueError(f"trace lengths differ: {len(trace_a)} vs {len(trace_b)}")
    if not trace_a:
        return ReplayDivergence(0.0, 0.0)

    argmax_div = float(np.mean([
        ta["action_argmax"] != tb["action_argmax"]
        for ta, tb in zip(trace_a, trace_b)
    ]))
    l2_div = float(np.mean([
        np.linalg.norm(np.asarray(ta["continuous_params"]) -
                       np.asarray(tb["continuous_params"]))
        for ta, tb in zip(trace_a, trace_b)
    ]))
    return ReplayDivergence(argmax_div, l2_div)
