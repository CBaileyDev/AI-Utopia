"""Freeze-invariant test for the value-head warm-up in scripts/fast_train.py.

The warm-up phase must train ONLY the CTDE critic and leave the actor (core
encoder + shared backbone + role encoder + actor head) byte-stable, so a
BC-cloned policy is not destroyed by the initially-random value head before the
critic has learned to predict returns under it.

We test the extracted seams directly (no training subprocess):
  - split_actor_critic_params / build_optimizer / warmup_step.

`scripts/` is not on pythonpath (src-only), so load by file location like
tests/unit/test_sim_obs_parity.py does.
"""

import importlib.util
import pathlib

import numpy as np
import torch
from ray.rllib.core import Columns

_FAST_TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fast_train.py"
_spec = importlib.util.spec_from_file_location("_fast_train", _FAST_TRAIN_PY)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)


def _fixed_batch(mod, device, B=4, T=8):
    """A (B, T, ...) training batch from the gatherer obs space + a fixed state."""
    obs_space = _ft.build_role_observation_space("gatherer", stage=1)
    obs_space.seed(0)

    def _sample_leaf(space):
        return torch.as_tensor(
            np.stack([np.stack([space.sample() for _ in range(T)]) for _ in range(B)]),
            device=device,
        )

    def _build(space):
        if hasattr(space, "spaces"):  # Dict
            return {k: _build(sub) for k, sub in space.spaces.items()}
        return _sample_leaf(space)

    obs = _build(obs_space)
    hdim = mod.shared_backbone.lstm_hidden
    s_h = torch.zeros(B, hdim, device=device)
    s_c = torch.zeros(B, hdim, device=device)
    returns = torch.zeros(B, T, device=device)  # arbitrary value targets
    return obs, s_h, s_c, returns


def test_value_warmup_freezes_actor():
    """One value-only warm-up step: actor byte-stable, critic moved, v_loss finite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = _ft.build_module(device)
    actor_params, critic_params = _ft.split_actor_critic_params(mod)
    opt = _ft.build_optimizer(actor_params, critic_params, actor_lr=0.0, value_lr=3e-4)

    obs, s_h, s_c, returns = _fixed_batch(mod, device)

    def _logits():
        with torch.no_grad():
            out = mod._forward_train({Columns.OBS: obs, Columns.STATE_IN: {"h": s_h, "c": s_c}})
        return out[Columns.ACTION_DIST_INPUTS].detach().clone()

    before_logits = _logits()
    actor_before = [p.detach().clone() for p in actor_params]
    critic_before = [p.detach().clone() for p in critic_params]

    v_loss = _ft.warmup_step(mod, opt, obs, s_h, s_c, returns, vf_coeff=0.5, grad_clip=1.0)

    after_logits = _logits()

    for b, p in zip(actor_before, actor_params, strict=True):
        assert torch.equal(b, p.detach()), "actor param changed during value warm-up"
    assert torch.allclose(before_logits, after_logits, atol=1e-5), "actor logits drifted"
    moved = any(
        not torch.equal(b, p.detach()) for b, p in zip(critic_before, critic_params, strict=True)
    )
    assert moved, "critic did not update during value warm-up"
    assert np.isfinite(float(v_loss))


def test_actor_lr_schedule_phases():
    """Per-iter actor LR: 0 during warm-up, linear ramp, then full LR; phase tag."""
    full = 5e-5
    warmup, ramp = 25, 15

    # Warm-up phase: actor LR is exactly 0, phase tag "warmup".
    for it in (0, 10, 24):
        lr, phase = _ft.actor_lr_for_iter(it, warmup=warmup, ramp=ramp, full_lr=full)
        assert lr == 0.0
        assert phase == "warmup"

    # Ramp phase: first ramp iter is full/ramp, last ramp iter reaches full.
    lr0, phase0 = _ft.actor_lr_for_iter(warmup, warmup=warmup, ramp=ramp, full_lr=full)
    assert phase0 == "ramp"
    assert abs(lr0 - full * (1.0 / ramp)) < 1e-12
    lr_last, _ = _ft.actor_lr_for_iter(warmup + ramp - 1, warmup=warmup, ramp=ramp, full_lr=full)
    assert abs(lr_last - full) < 1e-12

    # Finetune phase: full LR, tag "finetune".
    lr_ft, phase_ft = _ft.actor_lr_for_iter(warmup + ramp, warmup=warmup, ramp=ramp, full_lr=full)
    assert abs(lr_ft - full) < 1e-12
    assert phase_ft == "finetune"

    # No warm-up, no ramp: always full LR, always "finetune".
    lr_n, phase_n = _ft.actor_lr_for_iter(0, warmup=0, ramp=0, full_lr=full)
    assert lr_n == full
    assert phase_n == "finetune"
