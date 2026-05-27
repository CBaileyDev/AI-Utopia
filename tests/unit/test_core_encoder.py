import numpy as np
import pytest
import torch
from gymnasium.spaces import Dict as DictSpace

from aiutopia.env.spaces import build_role_observation_space
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule, core_obs_flat_dim, flatten_core_obs_batched,
)


def _batched_sample(batch: int = 2) -> dict:
    space: DictSpace = build_role_observation_space("gatherer", stage=1)
    # Deviation from plan: skip nested Dict subspaces (e.g. action_mask), whose
    # samples are dicts and break np.stack -> torch.as_tensor. action_mask is
    # not in _CORE_KEYS_FOR_FLATTEN so this changes nothing under test.
    out: dict[str, list] = {}
    for _ in range(batch):
        s = space.sample()
        for k, v in s.items():
            if isinstance(v, dict):
                continue
            out.setdefault(k, []).append(np.asarray(v))
    return {k: torch.as_tensor(np.stack(vs)) for k, vs in out.items()}


def test_core_obs_flat_dim_is_consistent() -> None:
    dim = core_obs_flat_dim()
    # comm_payloads alone is 32*128=4096; floor at least that big
    assert dim > 4000
    assert dim < 20_000


def test_flatten_core_obs_batched_returns_BxD() -> None:
    obs = _batched_sample(batch=4)
    flat = flatten_core_obs_batched(obs)
    assert flat.ndim == 2
    assert flat.shape == (4, core_obs_flat_dim())
    assert flat.dtype == torch.float32


def test_flatten_core_obs_ignores_role_overlay() -> None:
    obs = _batched_sample(batch=2)
    obs2 = {**obs, "g_richness_score": torch.full_like(
        obs.get("g_richness_score", torch.zeros(2, 1)), 0.42)}
    a = flatten_core_obs_batched(obs)
    b = flatten_core_obs_batched(obs2)
    assert torch.allclose(a, b)


def test_core_encoder_module_output_shape_256() -> None:
    module = CoreEncoderModule(config={"core_hidden": [512, 256]})
    flat = torch.randn(4, core_obs_flat_dim())
    out = module(flat)
    assert out.shape == (4, 256)


def test_core_encoder_module_param_count_reasonable() -> None:
    module = CoreEncoderModule(config={"core_hidden": [512, 256]})
    n_params = sum(p.numel() for p in module.parameters())
    assert 1_000_000 < n_params < 20_000_000


def test_unflatten_role_obs_4d_passthrough() -> None:
    from aiutopia.rl_module.core_encoder import unflatten_role_obs
    t = torch.rand(4, 32, 32, 6)
    assert unflatten_role_obs(t, (32, 32, 6)).shape == (4, 32, 32, 6)


def test_unflatten_role_obs_flattened_reshape() -> None:
    from aiutopia.rl_module.core_encoder import unflatten_role_obs
    t = torch.rand(4, 32 * 32 * 6)
    out = unflatten_role_obs(t, (32, 32, 6))
    assert out.shape == (4, 32, 32, 6)


def test_unflatten_role_obs_wrong_shape_raises() -> None:
    from aiutopia.rl_module.core_encoder import unflatten_role_obs
    t = torch.rand(4, 999)
    with pytest.raises(ValueError, match="cannot reconcile"):
        unflatten_role_obs(t, (32, 32, 6))
