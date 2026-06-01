"""LSTM-faithful-replay invariant for the PPO UPDATE forward in fast_train.py.

The PPO update recomputes the policy logp of the collected actions to form the
new/old ratio. For a recurrent (LSTM) policy this is only correct if the UPDATE
forward reproduces the SAME per-timestep hidden states the COLLECTION forward
used. Collection threads the hidden state one tick at a time and RE-ZEROS it on
every episode ``done`` (auto-reset). A naive update forward that runs the LSTM
over the whole T-chunk from a single chunk-start state cannot re-zero mid-chunk,
so across an episode boundary the update logp diverges from the collection logp
and the PPO ratio is biased (catastrophic for a sharp BC clone).

This test builds a tiny synthetic rollout WITH a forced ``done`` in the middle
of the T-chunk, reproduces collection's step-by-step (re-zeroing) forward to get
the ground-truth per-timestep logp, then asserts the extracted ``update_forward``
helper reproduces it under UNCHANGED params with ``faithful=True`` (the fix) to
~1e-4 ACROSS the boundary.

`scripts/` is not on pythonpath (src-only), so load by file location like
tests/unit/test_fast_train_value_warmup.py does.
"""

import importlib.util
import pathlib

import numpy as np
import torch
from ray.rllib.core import Columns

_FAST_TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fast_train.py"
_spec = importlib.util.spec_from_file_location("_fast_train", _FAST_TRAIN_PY)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)


def _synthetic_rollout(mod, device, *, B=3, T=6, done_t=2, seed=0):
    # Step-by-step rollout (re-zeroing on done) over random obs; returns the
    # update buffers plus the ground-truth collection logp. A done is forced at
    # timestep done_t for env 0 only so the chunk crosses an episode boundary
    # for env 0 and the LSTM must re-zero there.
    obs_space = _ft.build_role_observation_space("gatherer", stage=1)
    obs_space.seed(seed)

    def _sample_obs():
        def _build(space):
            if hasattr(space, "spaces"):
                return {k: _build(sub) for k, sub in space.spaces.items()}
            return torch.as_tensor(np.stack([space.sample() for _ in range(B)]), device=device)

        return _build(obs_space)

    hdim = mod.shared_backbone.lstm_hidden
    zero_h = torch.zeros(hdim, device=device)
    zero_c = torch.zeros(hdim, device=device)
    state_h = torch.zeros(B, hdim, device=device)
    state_c = torch.zeros(B, hdim, device=device)

    buf_obs = []
    buf_act = []
    buf_logp = torch.zeros(T, B, device=device)
    buf_h0 = torch.zeros(T, B, hdim, device=device)
    buf_c0 = torch.zeros(T, B, hdim, device=device)
    done_flags = torch.zeros(T, B, device=device)

    for t in range(T):
        obs_t = _sample_obs()
        buf_obs.append(obs_t)
        buf_h0[t] = state_h
        buf_c0[t] = state_c
        with torch.no_grad():
            batch_in = {Columns.OBS: obs_t, Columns.STATE_IN: {"h": state_h, "c": state_c}}
            out = mod._forward_train(batch_in)
            raw_adi = out[Columns.ACTION_DIST_INPUTS]
            adi = _ft.apply_skill_mask(raw_adi, obs_t["action_mask"]["skill_type"])
            dist = mod.action_dist_cls.from_logits(adi)
            act = dist.sample()
            logp = dist.logp(act)
            st_out = out[Columns.STATE_OUT]
        buf_act.append({k: v.detach() for k, v in act.items()})
        buf_logp[t] = logp
        state_h = st_out["h"].detach()
        state_c = st_out["c"].detach()
        # Force a done for env 0 at done_t -> collection re-zeros that env's state.
        if t == done_t:
            done_flags[t, 0] = 1.0
        done = done_flags[t] > 0
        if bool(done.any()):
            state_h[done] = zero_h
            state_c[done] = zero_c

    # Stack into (mb=B, T, ...) update layout (env-major, t-minor).
    mb_obs = {}
    for k, v0 in buf_obs[0].items():
        if isinstance(v0, dict):
            mb_obs[k] = {sk: torch.stack([buf_obs[t][k][sk] for t in range(T)], dim=1) for sk in v0}
        else:
            mb_obs[k] = torch.stack([buf_obs[t][k] for t in range(T)], dim=1)
    skill_mask_seq = torch.stack([buf_obs[t]["action_mask"]["skill_type"] for t in range(T)], dim=1)
    act_seq = {k: torch.stack([buf_act[t][k] for t in range(T)], dim=1) for k in buf_act[0]}
    buf_h_seq = buf_h0.permute(1, 0, 2).contiguous()  # (B, T, H)
    buf_c_seq = buf_c0.permute(1, 0, 2).contiguous()
    coll_logp = buf_logp.T.contiguous()  # (B, T)
    return (
        mb_obs,
        skill_mask_seq,
        act_seq,
        buf_h0[0],
        buf_c0[0],
        buf_h_seq,
        buf_c_seq,
        coll_logp,
        T,
    )


def test_faithful_update_forward_matches_collection_across_boundary():
    """Faithful=True reproduces collection logp per timestep, across a done."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    mod = _ft.build_module(device)
    mod.eval()
    (
        mb_obs,
        skill_mask_seq,
        act_seq,
        s_h0,
        s_c0,
        buf_h_seq,
        buf_c_seq,
        coll_logp,
        T,
    ) = _synthetic_rollout(mod, device)

    with torch.no_grad():
        new_logp, ent, vf, _ = _ft.update_forward(
            mod,
            mb_obs,
            skill_mask_seq,
            act_seq,
            s_h0,
            s_c0,
            buf_h_seq,
            buf_c_seq,
            T=T,
            faithful=True,
        )

    # The shipped invariant: faithful replay == collection logp at every timestep,
    # INCLUDING after the forced episode boundary.
    #
    # Tolerance: collection runs T sequential length-1 LSTM forwards while the
    # faithful update folds them into one (B*T, 1) batched forward. The cell math
    # is identical, but float32 LSTM kernels accumulate in a different order when
    # batched vs sequential, so the per-element logp delta is hardware/torch-version
    # dependent (observed ~1e-4 on GPU, ~3e-4 on CPU/torch-2.12). 1e-3 is the robust
    # bound: it still sits a full 10x below the legacy start-state-only divergence
    # (>1e-2, asserted by the control test below), so the discriminating power that
    # makes this a real regression guard is preserved.
    assert new_logp.shape == coll_logp.shape
    max_err = float((new_logp - coll_logp).abs().max())
    assert max_err < 1e-3, f"faithful replay diverged from collection: max_err={max_err}"
    assert torch.isfinite(ent).all()
    assert torch.isfinite(vf).all()


def test_legacy_start_state_only_diverges_across_boundary():
    # The bug, made explicit (discriminating control): start-state-only replay
    # (faithful=False) keeps threading the LSTM across the forced done, so
    # post-boundary timesteps for the reset env diverge from collection's
    # re-zeroed logp. Guards against a regression that silently reverts the
    # update forward to the start-state-only path (which biases the PPO ratio).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    mod = _ft.build_module(device)
    mod.eval()
    done_t = 2
    (
        mb_obs,
        skill_mask_seq,
        act_seq,
        s_h0,
        s_c0,
        buf_h_seq,
        buf_c_seq,
        coll_logp,
        T,
    ) = _synthetic_rollout(mod, device, done_t=done_t)

    with torch.no_grad():
        legacy_logp, _, _, _ = _ft.update_forward(
            mod,
            mb_obs,
            skill_mask_seq,
            act_seq,
            s_h0,
            s_c0,
            buf_h_seq,
            buf_c_seq,
            T=T,
            faithful=False,
        )

    # env 0 was reset at done_t; timesteps AFTER the boundary (t > done_t) for
    # env 0 must differ from the collection logp (which re-zeroed there).
    post_err = float((legacy_logp[0, done_t + 1 :] - coll_logp[0, done_t + 1 :]).abs().max())
    assert post_err > 1e-2, (
        "legacy start-state-only replay unexpectedly matched collection across the "
        f"boundary (post_err={post_err}); the discriminating bug is not reproduced"
    )
