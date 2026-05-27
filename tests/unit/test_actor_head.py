import pytest
import torch

from aiutopia.rl_module.actor_head import GathererActorHead, build_actor_head


def test_actor_head_output_dim_is_344() -> None:
    # T7.5 fix: MultiBinary(4) -> TorchMultiCategorical needs 2*4=8 logits,
    # not 4. Total: 256+8+2+2+6+6+64 = 344.
    head = GathererActorHead(config={})
    assert head.OUTPUT_DIM == 344


def test_actor_head_forward_shape() -> None:
    head = GathererActorHead(config={})
    batch = 4
    hidden = torch.randn(batch, 256)
    goal   = torch.randn(batch, 512)
    out = head(hidden, goal)
    assert out.shape == (batch, 344)


def test_build_actor_head_gatherer_only_in_m1() -> None:
    assert isinstance(build_actor_head("gatherer", {}), GathererActorHead)
    for r in ("builder", "farmer", "defender"):
        with pytest.raises(NotImplementedError):
            build_actor_head(r, {})
