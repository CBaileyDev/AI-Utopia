"""Guard _greedy_decode's MASK-AWARE skill_type path (the N22 fix).

scenario_runner._greedy_decode applies the obs action_mask to the skill_type
slice so greedy eval can't argmax a known-invalid skill (e.g. HARVEST when no
resource is in reach → forced NAVIGATE). A regression here silently corrupts
every greedy eval (it was the source of the N22 greedy-vs-sampled confusion), so
the masked path needs explicit coverage. Slice layout (alphabetical): comm_payload
(256) + comm_target_mask(8) + scalar(2) + should_broadcast(2) = 268, then
skill_type(6) at [268:274]."""

import torch

from aiutopia.train.scenario_runner import _greedy_decode

_SKILL_OFFSET = 268
_N_SKILLS = 6
_FLAT_DIM = 344  # what _greedy_decode consumes: 256+8+2+2+6+6+64


def _flat_with_skill_logits(skill_logits: list[float]) -> torch.Tensor:
    flat = torch.zeros(_FLAT_DIM, dtype=torch.float32)
    flat[_SKILL_OFFSET : _SKILL_OFFSET + _N_SKILLS] = torch.tensor(
        skill_logits, dtype=torch.float32
    )
    return flat


# HARVEST=skill 1 is the unmasked argmax; NAVIGATE=skill 0 is the runner-up.
_HARVEST_TOP = [5.0, 10.0, 0.0, 0.0, 0.0, 0.0]


def test_no_mask_returns_unmasked_argmax():
    assert _greedy_decode(_flat_with_skill_logits(_HARVEST_TOP))["skill_type"] == 1


def test_skill_mask_blocks_the_argmax_skill():
    # HARVEST (index 1) masked OFF -> greedy must fall back to the next unmasked
    # skill (NAVIGATE, index 0), NOT the masked HARVEST.
    out = _greedy_decode(
        _flat_with_skill_logits(_HARVEST_TOP),
        action_mask={"skill_type": [1, 0, 1, 1, 1, 1]},
    )
    assert out["skill_type"] == 0


def test_all_zero_mask_does_not_blank_everything():
    # An all-zero mask must be IGNORED (not force -inf on every skill -> NaN/garbage);
    # the unmasked argmax stands.
    out = _greedy_decode(
        _flat_with_skill_logits(_HARVEST_TOP),
        action_mask={"skill_type": [0, 0, 0, 0, 0, 0]},
    )
    assert out["skill_type"] == 1


def test_mask_only_constrains_skill_type_not_other_heads():
    # Masking skill_type must not shift the other slices (target_class etc. still
    # decode from their own logits). target_class slice is all-zeros here -> argmax 0.
    out = _greedy_decode(
        _flat_with_skill_logits(_HARVEST_TOP),
        action_mask={"skill_type": [1, 0, 1, 1, 1, 1]},
    )
    assert out["skill_type"] == 0
    assert int(out["target_class"]) == 0
