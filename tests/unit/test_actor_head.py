import math

import pytest
import torch

from aiutopia.rl_module.actor_head import (
    LOG_STD_FOR_ZERO_ENTROPY,
    GathererActorHead,
    _comm_mask_slices,
    build_actor_head,
    gatherer_action_dist_config,
)


def test_actor_head_output_dim_is_344() -> None:
    # T7.5 fix: MultiBinary(4) -> TorchMultiCategorical needs 2*4=8 logits,
    # not 4. Total: 256+8+2+2+6+6+64 = 344.
    head = GathererActorHead(config={})
    assert head.OUTPUT_DIM == 344


def test_actor_head_forward_shape() -> None:
    head = GathererActorHead(config={})
    batch = 4
    hidden = torch.randn(batch, 256)
    goal = torch.randn(batch, 512)
    out = head(hidden, goal)
    assert out.shape == (batch, 344)


def test_build_actor_head_gatherer_only_in_m1() -> None:
    assert isinstance(build_actor_head("gatherer", {}), GathererActorHead)
    for r in ("builder", "farmer", "defender"):
        with pytest.raises(NotImplementedError):
            build_actor_head(r, {})


# ─────────────────────────────────────────────────────────────────────
# Fix #4 — dead comm-head masking (mask_comm). The comm_payload Gaussian
# (256 of 344 dims) and should_broadcast categorical are DEAD weight in
# single-agent M1B: the comm obs is all-zeros and comm has zero env effect,
# yet PPO maximizes their entropy (the 128-d Gaussian's unclamped log_std is
# the primary NaN-KL source). With mask_comm ON these sub-distributions must
# contribute ~0 entropy, ~0 KL, and receive ~0 gradient, while the emitted
# action dict shape stays byte-identical (the env contract is preserved).
# ─────────────────────────────────────────────────────────────────────

# Slice offsets are derived programmatically in actor_head; mirror here only
# for assertions. Alphabetical layout of the 344-d flat logits:
#   comm_payload      [0:256]   (128 mean | 128 log_std)
#   comm_target_mask  [256:264]
#   scalar_param      [264:266]
#   should_broadcast  [266:268]
#   skill_type        [268:274]
#   spatial_param     [274:280]
#   target_class      [280:344]
_COMM_PAYLOAD_MEAN = slice(0, 128)
_COMM_PAYLOAD_LOGSTD = slice(128, 256)
_SHOULD_BROADCAST = slice(266, 268)


def _build_dist(logits):
    from ray.rllib.core.distribution.torch.torch_distribution import (
        TorchMultiDistribution,
    )

    child_struct, input_lens = gatherer_action_dist_config()
    dist_cls = TorchMultiDistribution.get_partial_dist_cls(
        child_distribution_cls_struct=child_struct,
        input_lens=input_lens,
    )
    return dist_cls.from_logits(logits)


def test_comm_mask_slices_match_expected_layout() -> None:
    """Programmatic slice derivation lands on the documented comm offsets."""
    sl = _comm_mask_slices()
    assert sl["comm_payload_mean"] == (0, 128)
    assert sl["comm_payload_logstd"] == (128, 256)
    assert sl["should_broadcast"] == (266, 268)


def test_mask_off_is_byte_identical_to_raw_net() -> None:
    """mask_comm absent/OFF -> forward output equals the raw net (legacy)."""
    torch.manual_seed(0)
    head = GathererActorHead(config={})  # no mask_comm key -> OFF
    assert head.mask_comm is False
    hidden = torch.randn(4, 256)
    goal = torch.randn(4, 512)
    raw = head.net(torch.cat([hidden, goal], dim=-1))
    out = head(hidden, goal)
    assert torch.equal(out, raw)


def test_mask_on_zeros_comm_payload_mean_and_fixes_logstd() -> None:
    torch.manual_seed(0)
    head = GathererActorHead(config={"mask_comm": True})
    assert head.mask_comm is True
    out = head(torch.randn(4, 256), torch.randn(4, 512))
    mean = out[:, _COMM_PAYLOAD_MEAN]
    logstd = out[:, _COMM_PAYLOAD_LOGSTD]
    # Greedy/deterministic action uses the Gaussian mean -> exactly 0.
    assert torch.all(mean == 0.0)
    # log_std fixed to the zero-entropy constant across all dims/rows.
    assert torch.allclose(logstd, torch.full_like(logstd, LOG_STD_FOR_ZERO_ENTROPY))


def test_mask_on_should_broadcast_is_near_deterministic() -> None:
    head = GathererActorHead(config={"mask_comm": True})
    out = head(torch.randn(4, 256), torch.randn(4, 512))
    sb = out[:, _SHOULD_BROADCAST]
    # logits [+B, -B] -> always class 0, low entropy, identical across rows.
    assert torch.all(sb[:, 0] > sb[:, 1])
    assert torch.allclose(sb, sb[0:1].expand_as(sb))


def test_mask_on_comm_entropy_is_zero() -> None:
    """comm_payload Gaussian entropy == 0 and should_broadcast entropy ~ 0."""
    head = GathererActorHead(config={"mask_comm": True})
    logits = head(torch.randn(8, 256), torch.randn(8, 512))
    dist = _build_dist(logits)
    children = dist._flat_child_distributions  # alphabetical flatten order
    comm_payload_ent = children[0].entropy()  # index 0 = comm_payload
    should_broadcast_ent = children[3].entropy()  # index 3 = should_broadcast
    assert torch.allclose(comm_payload_ent, torch.zeros_like(comm_payload_ent), atol=1e-5)
    assert torch.all(should_broadcast_ent < 1e-3)


def test_mask_off_comm_entropy_is_nonzero_and_trainable() -> None:
    torch.manual_seed(1)
    head = GathererActorHead(config={})  # OFF
    logits = head(torch.randn(8, 256), torch.randn(8, 512))
    dist = _build_dist(logits)
    comm_payload_ent = dist._flat_child_distributions[0].entropy()
    # Legacy path: the dead Gaussian carries real (nonzero) entropy.
    assert torch.any(comm_payload_ent.abs() > 1e-3)


def test_mask_on_zeros_gradient_to_comm_head_rows() -> None:
    """Entropy loss flows ~0 gradient into the comm logit rows of the
    final Linear; the non-comm rows still receive gradient."""
    head = GathererActorHead(config={"mask_comm": True})
    logits = head(torch.randn(8, 256), torch.randn(8, 512))
    dist = _build_dist(logits)
    dist.entropy().sum().backward()
    grad = head.net[-1].weight.grad  # (344, hidden)
    comm_rows = torch.cat([grad[0:256], grad[266:268]], dim=0)
    other_rows = grad[280:344]  # target_class logits
    assert torch.allclose(comm_rows, torch.zeros_like(comm_rows), atol=1e-6)
    assert torch.any(other_rows.abs() > 1e-8)


def test_mask_off_passes_gradient_to_comm_head_rows() -> None:
    head = GathererActorHead(config={})  # OFF
    logits = head(torch.randn(8, 256), torch.randn(8, 512))
    dist = _build_dist(logits)
    dist.entropy().sum().backward()
    grad = head.net[-1].weight.grad
    comm_rows = grad[0:256]
    assert torch.any(comm_rows.abs() > 1e-8)


def test_mask_on_full_dist_samples_and_decodes() -> None:
    """Sanity: the full 344-d dist still samples; greedy decode runs and the
    emitted dict keeps comm_payload shape (128,) + should_broadcast."""
    from aiutopia.train.scenario_runner import _greedy_decode

    head = GathererActorHead(config={"mask_comm": True})
    logits = head(torch.randn(2, 256), torch.randn(2, 512))
    dist = _build_dist(logits)
    sample = dist.sample()
    assert sample["comm_payload"].shape == (2, 128)
    assert "should_broadcast" in sample

    decoded = _greedy_decode(logits[0])
    assert decoded["comm_payload"].shape == (128,)
    assert "should_broadcast" in decoded
    # Greedy comm_payload is the masked (zero) mean.
    assert math.isclose(float(abs(decoded["comm_payload"]).max()), 0.0, abs_tol=1e-6)
