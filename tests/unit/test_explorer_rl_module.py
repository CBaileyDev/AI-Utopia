"""Explorer RLModule forward-contract tests.

Locks the SharedBackbone fusion contract for the explorer role: the explorer has
no grid_conv branch, so its RoleEncoder alone must supply the full 128-d role
feature (core 256 + role 128 + pixel 64 = 448 = FUSED_INPUT_DIM). A 64-d role
output fuses to 384 and raises a mat1/mat2 RuntimeError on the first forward.
"""

import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.rl_module.explorer_rl_module import ExplorerRoleRLModule


def _build_explorer():
    spec = MultiRLModuleSpec(
        rl_module_specs={
            "explorer_policy": RLModuleSpec(
                module_class=ExplorerRoleRLModule,
                observation_space=build_role_observation_space("explorer", stage=1),
                action_space=build_role_action_space("explorer"),
                model_config={"role": "explorer"},
            ),
        },
    )
    return spec.build()


def _sample_batched(obs_space, batch: int = 2):
    import numpy as np

    out: dict[str, list] = {}
    for _ in range(batch):
        s = obs_space.sample()
        for k, v in s.items():
            if isinstance(v, dict):
                continue
            out.setdefault(k, []).append(np.asarray(v))
    return {k: torch.as_tensor(np.stack(vs)) for k, vs in out.items()}


def test_explorer_forward_inference_runs() -> None:
    mrm = _build_explorer()
    obs_space = build_role_observation_space("explorer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    out = mrm["explorer_policy"]._forward_inference({"obs": batched})
    assert "action_dist_inputs" in out
    assert out["action_dist_inputs"].shape[0] == 2


def test_explorer_forward_train_with_time_dim() -> None:
    """RLlib passes (B, T, ...) under LSTM unroll; locks the time-folded path."""
    mrm = _build_explorer()
    obs_space = build_role_observation_space("explorer", stage=1)
    batched = _sample_batched(obs_space, batch=2)
    batched_t = {
        k: v.unsqueeze(1).expand(2, 4, *v.shape[1:]).contiguous()
        for k, v in batched.items()
    }
    out = mrm["explorer_policy"]._forward_train({"obs": batched_t})
    assert out["vf_preds"].shape == (2, 4)
