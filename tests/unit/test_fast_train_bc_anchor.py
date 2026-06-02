"""Behavior-anchor invariants for the BC-anchor in scripts/fast_train.py.

RUN D2 showed plain PPO consolidation erodes a BC-cloned navigate (seed 2 -> 2/3)
because the decisive HARVEST-masked spawn states -- where NAVIGATE is mandatory --
are rare in randomized rollouts and decay untouched by the PPO gradient. The
BC-anchor re-applies the proven NAVIGATE-then-HARVEST supervision every finetune
iter on a FRESH force-masked spawn batch, with the scripted demonstrator as the
ground-truth reference.

These tests pin the two properties that make the anchor a correct consolidation
lever, exercised through the real force-masked sim path (no training subprocess):
  1. it pushes the policy TOWARD navigate on masked spawns (skill CE drops sharply
     under a few anchor gradient steps), and
  2. it touches ONLY the actor -- the CTDE critic is byte-stable -- so it composes
     with the value head / PPO value loss without corrupting the critic.

`scripts/` is not on pythonpath (src-only), so load by file location like
tests/unit/test_fast_train_value_warmup.py does.
"""

import importlib.util
import pathlib

import numpy as np
import torch

from aiutopia.sim.bc_demonstrator import demonstrate
from aiutopia.sim.skills import SKILL_NAVIGATE
from aiutopia.sim.vec_sim import VecGathererSim

_FAST_TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fast_train.py"
_spec = importlib.util.spec_from_file_location("_fast_train", _FAST_TRAIN_PY)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)


def _masked_anchor_batch(device, B=48, seed=1):
    """A force-masked spawn batch + the demonstrator's NAVIGATE-then-HARVEST targets."""
    sim = VecGathererSim(
        num_envs=B, max_episode_ticks=300, force_masked_spawn=True, randomize_layout=True
    )
    obs = sim.reset(np.arange(B, dtype=np.int64) + seed)
    obs_t = _ft.obs_to_tensors(obs, device)
    mask = obs_t["action_mask"]["skill_type"]
    demo = demonstrate(obs)
    exp_skill = torch.as_tensor(np.asarray(demo["skill_type"]), device=device).long()
    exp_spatial = torch.as_tensor(
        np.asarray(demo["spatial_param"]), device=device, dtype=torch.float32
    )
    return obs_t, mask, exp_skill, exp_spatial


def test_anchor_drives_policy_toward_navigate_on_masked_spawns():
    """A few anchor gradient steps sharply reduce the navigate skill-CE."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    mod = _ft.build_module(device)
    obs_t, mask, exp_skill, exp_spatial = _masked_anchor_batch(device)

    # force_masked_spawn => the decisive HARVEST-masked state, so the demonstrator
    # targets NAVIGATE on (essentially) every row: the batch genuinely exercises
    # the erosion-prone state, not an incidental harvest mix.
    nav_frac = float((exp_skill == SKILL_NAVIGATE).float().mean())
    assert nav_frac > 0.8, f"force-masked batch should be ~all-NAVIGATE, got {nav_frac:.2f}"

    hdim = mod.shared_backbone.lstm_hidden
    B = exp_skill.shape[0]
    zh = torch.zeros(B, hdim, device=device)
    zc = torch.zeros(B, hdim, device=device)

    with torch.no_grad():
        loss0, ce0, _mse0 = _ft.bc_anchor_loss(mod, obs_t, mask, exp_skill, exp_spatial, zh, zc)
    assert np.isfinite(float(loss0))

    opt = torch.optim.Adam(mod.parameters(), lr=3e-3)
    for _ in range(40):
        loss, _ce, _mse = _ft.bc_anchor_loss(mod, obs_t, mask, exp_skill, exp_spatial, zh, zc)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        _loss1, ce1, _mse1 = _ft.bc_anchor_loss(mod, obs_t, mask, exp_skill, exp_spatial, zh, zc)
    assert float(ce1) < 0.5 * float(ce0), (
        f"anchor did not pull the policy toward navigate: ce {float(ce0):.4f} -> {float(ce1):.4f}"
    )


def test_anchor_loss_leaves_critic_byte_stable():
    """The anchor loss is a policy-head loss; the CTDE critic must not move."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    mod = _ft.build_module(device)
    obs_t, mask, exp_skill, exp_spatial = _masked_anchor_batch(device)

    _actor, critic = _ft.split_actor_critic_params(mod)
    critic_before = [p.detach().clone() for p in critic]

    hdim = mod.shared_backbone.lstm_hidden
    B = exp_skill.shape[0]
    zh = torch.zeros(B, hdim, device=device)
    zc = torch.zeros(B, hdim, device=device)

    # Optimize over ALL params (as the trainer does): the anchor loss never reaches
    # the critic graph, so its grads are None and Adam skips them -> byte-stable.
    opt = torch.optim.Adam(mod.parameters(), lr=3e-3)
    for _ in range(5):
        loss, _ce, _mse = _ft.bc_anchor_loss(mod, obs_t, mask, exp_skill, exp_spatial, zh, zc)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for b, p in zip(critic_before, critic, strict=True):
        assert torch.equal(b, p.detach()), "CTDE critic moved under the BC-anchor loss"
