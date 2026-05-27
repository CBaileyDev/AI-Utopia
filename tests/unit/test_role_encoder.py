import pytest
import torch

from aiutopia.rl_module.role_encoder import (
    GathererRoleEncoder, build_role_encoder,
)


def _gatherer_obs(batch: int = 2) -> dict:
    return {
        "g_resource_grid":     torch.rand(batch, 32, 32, 6),
        "g_nearest_resources": torch.rand(batch, 8, 6),
        "g_richness_score":    torch.rand(batch, 1),
        "g_hostiles_nearby":   torch.rand(batch, 4, 4),
    }


def test_gatherer_encoder_outputs_128_d() -> None:
    enc = GathererRoleEncoder(config={})
    feat = enc(_gatherer_obs(batch=2))
    assert feat.shape == (2, 128)


def test_build_role_encoder_gatherer_only_in_m1() -> None:
    assert isinstance(build_role_encoder("gatherer", {}), GathererRoleEncoder)
    for r in ("builder", "farmer", "defender"):
        with pytest.raises(NotImplementedError):
            build_role_encoder(r, {})
