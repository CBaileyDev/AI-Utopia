"""§7.2 — RLModule scaffold. NOT runnable in M0.

This file documents the intended shape so M1 implementation has a target
to fill in. Real implementation requires:
  - inheriting from ray.rllib.core.rl_module.torch.TorchRLModule
  - declaring shared submodules via MultiRLModuleSpec.additional_module_specs
    (NEVER module-level Python globals — Ray workers fork processes,
    Python globals don't share, and gradients silently never propagate
    to rollouts)
"""
from __future__ import annotations

# Intentionally NOT importing ray/torch here so cold imports stay light.
# M1 will replace this module with the real implementation.


class CoreEncoderModule:
    """§4.3 — universal core obs → 256-d feature. SHARED across roles via
    `additional_module_specs` on MultiRLModuleSpec."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1: implement against ray.rllib TorchRLModule")


class SharedBackboneModule:
    """§4.3 — Linear(448→384) + LSTM(384, hidden=256). SHARED across roles."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")


class CTDECriticModule:
    """§4.3 — two-stage encoder (per-agent → 128, then MLP). SHARED."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")


class AiUtopiaRoleRLModule:
    """§7.2 — per-role module: role encoder + (optional) pixel encoder + actor."""

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("M1")
