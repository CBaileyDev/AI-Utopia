import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from aiutopia.env.spaces import (
    build_role_action_space, build_role_observation_space,
)
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule


def _build_multi_module():
    obs_space    = build_role_observation_space("gatherer", stage=1)
    action_space = build_role_action_space("gatherer")
    spec = MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=obs_space,
                action_space=action_space,
                model_config={"role": "gatherer"},
            ),
        },
    )
    return spec.build()


def _sample_batched(obs_space, batch: int = 2):
    # Deviation from plan: skip nested Dict subspaces (e.g. action_mask), whose
    # samples are dicts and break np.stack -> torch.as_tensor. action_mask is
    # not consumed by the core/role encoders or actor head in M1 (it lives in
    # the action-masking connector). Mirrors the precedent in
    # tests/unit/test_core_encoder.py.
    import numpy as np
    out: dict[str, list] = {}
    for _ in range(batch):
        s = obs_space.sample()
        for k, v in s.items():
            if isinstance(v, dict):
                continue
            out.setdefault(k, []).append(np.asarray(v))
    return {k: torch.as_tensor(np.stack(vs)) for k, vs in out.items()}


def test_multi_rl_module_assembles() -> None:
    mrm = _build_multi_module()
    assert "gatherer_policy" in mrm


def test_forward_inference_produces_action_dist() -> None:
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    out = mrm["gatherer_policy"]._forward_inference({"obs": batched})
    assert "action_dist_inputs" in out
    assert out["action_dist_inputs"].shape == (2, 340)


def test_forward_train_emits_vf_preds() -> None:
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    out = mrm["gatherer_policy"]._forward_train({"obs": batched})
    assert "vf_preds" in out
    assert out["vf_preds"].shape == (2,)


def test_forward_train_with_time_dim() -> None:
    """RLlib passes (B, T, ...) when LSTM unroll is active."""
    mrm = _build_multi_module()
    obs_space = build_role_observation_space("gatherer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    # Add a time dim of 4
    batched_t = {k: v.unsqueeze(1).expand(2, 4, *v.shape[1:]).contiguous()
                 for k, v in batched.items()}
    out = mrm["gatherer_policy"]._forward_train({"obs": batched_t})
    assert out["action_dist_inputs"].shape == (2, 4, 340)
    assert out["vf_preds"].shape == (2, 4)


def test_initial_state_is_unbatched_rank_1() -> None:
    mrm = _build_multi_module()
    state = mrm["gatherer_policy"].get_initial_state()
    assert state["h"].shape == (256,)
    assert state["c"].shape == (256,)
