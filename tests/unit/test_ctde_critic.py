import torch

from aiutopia.rl_module.core_encoder import core_obs_flat_dim
from aiutopia.rl_module.ctde_critic import (
    CTDECriticModule, VILLAGE_INV_DIM, PER_AGENT_COMPRESSED_DIM,
)


def test_per_agent_compressed_dim_is_128() -> None:
    assert PER_AGENT_COMPRESSED_DIM == 128


def test_critic_output_shape() -> None:
    critic = CTDECriticModule(config={})
    batch = 4
    obs_dim = core_obs_flat_dim()
    all_agents = torch.randn(batch, 4, obs_dim)
    village_inv = torch.randn(batch, VILLAGE_INV_DIM)
    v = critic(all_agents, village_inv)
    assert v.shape == (batch,)


def test_critic_param_count_drops_vs_naive_mlp() -> None:
    critic = CTDECriticModule(config={})
    n = sum(p.numel() for p in critic.parameters())
    assert n < 10_000_000


def test_critic_handles_single_agent_padding() -> None:
    critic = CTDECriticModule(config={})
    obs_dim = core_obs_flat_dim()
    all_agents = torch.zeros(2, 4, obs_dim)
    all_agents[:, 0, :] = torch.randn(2, obs_dim)
    village_inv = torch.zeros(2, VILLAGE_INV_DIM)
    v = critic(all_agents, village_inv)
    assert not torch.isnan(v).any()
