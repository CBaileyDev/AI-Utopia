import torch

from aiutopia.rl_module.shared_backbone import SharedBackboneModule


def test_backbone_output_shape_with_state() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    batch, seq_len = 4, 1
    fused = torch.randn(batch, seq_len, 448)
    h0, c0 = b.initial_state(batch_size=batch, device=fused.device)
    out, (h1, c1) = b(fused, (h0, c0))
    assert out.shape == (batch, seq_len, 256)
    assert h1.shape == (1, batch, 256)
    assert c1.shape == (1, batch, 256)


def test_backbone_state_changes_with_input() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    fused = torch.randn(2, 1, 448)
    h0, c0 = b.initial_state(batch_size=2, device=fused.device)
    _, (h1, _) = b(fused, (h0, c0))
    assert not torch.allclose(h1, h0)


def test_backbone_initial_state_device() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    cpu = torch.device("cpu")
    h, c = b.initial_state(batch_size=3, device=cpu)
    assert h.device == cpu
    assert h.shape == (1, 3, 256)


def test_backbone_handles_multi_step_sequence() -> None:
    b = SharedBackboneModule(config={"lstm_hidden": 256})
    fused = torch.randn(1, 10, 448)
    h0, c0 = b.initial_state(batch_size=1, device=fused.device)
    out, _ = b(fused, (h0, c0))
    assert out.shape == (1, 10, 256)
