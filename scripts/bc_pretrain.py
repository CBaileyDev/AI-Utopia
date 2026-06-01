"""Behavior-cloning warm-start for the AiUtopia gatherer: NAVIGATE-then-HARVEST.

Seeds the navigate behavior PPO cannot bootstrap (Research/SEED1_HOLE_DIAGNOSIS.md).
Supervises the gatherer RLModule on the scripted NAVIGATE-then-HARVEST oracle
(sim.bc_demonstrator); collection is predominantly force_masked so EVERY episode
step0 is the decisive (zero-LSTM-state, HARVEST-masked) NAVIGATE example. LSTM state
is threaded and re-zeroed on done (eval-faithful); each timestep is a length-1 LSTM
segment fed its own stored start state. Per-head loss: CE on skill_type (mask-aware)
every step; MSE on spatial MEAN only on NAVIGATE rows; CE on target_class + MSE on
scalar MEAN only on HARVEST rows. Saves a state_dict loadable by fast_train
(--load-weights). See SUPERVISED_SESSION_LOG.md / the BC report for rationale.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from fast_train import (
    apply_skill_mask,
    build_module,
    obs_to_tensors,
    seed1_navigate_probe,
)
from ray.rllib.core import Columns
from torch import nn

from aiutopia.rl_module.actor_head import _slice_offsets
from aiutopia.sim.bc_demonstrator import demonstrate
from aiutopia.sim.skills import SKILL_HARVEST, SKILL_NAVIGATE
from aiutopia.sim.vec_sim import VecGathererSim

_OFF = _slice_offsets()
SCALAR_MEAN = _OFF["scalar_param"][0]
SKILL_S, SKILL_E = _OFF["skill_type"]
SPATIAL_MEAN_S = _OFF["spatial_param"][0]
TARGET_S, TARGET_E = _OFF["target_class"]


def expert_action_tensors(act_np, device):
    """Demonstrator numpy action to supervised targets on device."""
    return {
        "skill_type": torch.as_tensor(np.asarray(act_np["skill_type"]), device=device).long(),
        "target_class": torch.as_tensor(np.asarray(act_np["target_class"]), device=device).long(),
        "spatial_param": torch.as_tensor(
            np.asarray(act_np["spatial_param"]), device=device, dtype=torch.float32
        ),
        "scalar_param": torch.as_tensor(
            np.asarray(act_np["scalar_param"]).reshape(-1), device=device, dtype=torch.float32
        ),
    }


def _stack_obs(buf_obs, T):
    """Concatenate the T per-step obs dicts along the (already-flat) batch dim,
    recursing into nested sub-dicts (e.g. action_mask)."""
    out = {}
    for k, v0 in buf_obs[0].items():
        if isinstance(v0, dict):
            out[k] = {sk: torch.cat([buf_obs[t][k][sk] for t in range(T)], dim=0) for sk in v0}
        else:
            out[k] = torch.cat([buf_obs[t][k] for t in range(T)], dim=0)
    return out


def collect(sim, mod, device, T, B, hdim, zero_h, zero_c):
    """Roll the demonstrator T steps, threading + re-zeroing LSTM state."""
    obs = sim.last_obs
    state_h = zero_h.unsqueeze(0).expand(B, hdim).contiguous().clone()
    state_c = zero_c.unsqueeze(0).expand(B, hdim).contiguous().clone()

    buf_obs, buf_exp, buf_mask = [], [], []
    buf_h0 = torch.zeros(T, B, hdim, device=device)
    buf_c0 = torch.zeros(T, B, hdim, device=device)
    n_nav = n_harv = 0

    for t in range(T):
        obs_t = obs_to_tensors(obs, device)
        buf_obs.append(obs_t)
        buf_mask.append(obs_t["action_mask"]["skill_type"].clone())
        buf_h0[t] = state_h
        buf_c0[t] = state_c

        act_np = demonstrate(obs)
        buf_exp.append(expert_action_tensors(act_np, device))
        sk = np.asarray(act_np["skill_type"])
        n_nav += int((sk == SKILL_NAVIGATE).sum())
        n_harv += int((sk == SKILL_HARVEST).sum())

        with torch.no_grad():
            out = mod._forward_inference(
                {Columns.OBS: obs_t, Columns.STATE_IN: {"h": state_h, "c": state_c}}
            )
        state_h = out[Columns.STATE_OUT]["h"].detach()
        state_c = out[Columns.STATE_OUT]["c"].detach()

        obs, _rew, term, trunc = sim.step(act_np)
        done = torch.as_tensor(
            np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool), device=device
        )
        if bool(done.any()):
            state_h[done] = zero_h
            state_c[done] = zero_c

    sim.last_obs = obs
    return buf_obs, buf_exp, buf_mask, buf_h0, buf_c0, n_nav, n_harv


def bc_loss_for_chunk(mod, mb_obs, s_h, s_c, exp, skill_mask):
    """Single-step BC loss; each row carries its own start state."""
    out = mod._forward_train({Columns.OBS: mb_obs, Columns.STATE_IN: {"h": s_h, "c": s_c}})
    adi = out[Columns.ACTION_DIST_INPUTS]
    adi_masked = apply_skill_mask(adi, skill_mask)

    skill_logits = adi_masked[:, SKILL_S:SKILL_E]
    ce_skill = nn.functional.cross_entropy(skill_logits, exp["skill_type"])

    is_nav = exp["skill_type"] == SKILL_NAVIGATE
    is_harv = exp["skill_type"] == SKILL_HARVEST

    spatial_mean = adi[:, SPATIAL_MEAN_S : SPATIAL_MEAN_S + 3]
    if bool(is_nav.any()):
        mse_spatial = nn.functional.mse_loss(spatial_mean[is_nav], exp["spatial_param"][is_nav])
    else:
        mse_spatial = adi.sum() * 0.0

    if bool(is_harv.any()):
        target_logits = adi[:, TARGET_S:TARGET_E]
        ce_target = nn.functional.cross_entropy(
            target_logits[is_harv], exp["target_class"][is_harv]
        )
        scalar_mean = adi[:, SCALAR_MEAN : SCALAR_MEAN + 1].reshape(-1)
        mse_scalar = nn.functional.mse_loss(scalar_mean[is_harv], exp["scalar_param"][is_harv])
    else:
        ce_target = adi.sum() * 0.0
        mse_scalar = adi.sum() * 0.0

    loss = ce_skill + mse_spatial + 0.5 * ce_target + 0.1 * mse_scalar
    return loss, ce_skill.detach(), mse_spatial.detach(), ce_target.detach()


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-ep-ticks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-weights", type=str, default="weights/bc_gatherer.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    B, T = args.num_envs, args.horizon
    mod = build_module(device)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)

    sim = VecGathererSim(
        num_envs=B,
        max_episode_ticks=args.max_ep_ticks,
        force_masked_spawn=True,
        randomize_layout=True,
    )
    seeds = np.arange(B, dtype=np.int64) + 1 + args.seed * B
    sim.last_obs = sim.reset(seeds)

    init_state = mod.get_initial_state()
    hdim = init_state["h"].shape[0]
    zero_h = init_state["h"].to(device)
    zero_c = init_state["c"].to(device)

    print("BC before:", seed1_navigate_probe(mod, device), flush=True)

    for it in range(args.iters):
        t0 = time.perf_counter()
        buf_obs, buf_exp, buf_mask, buf_h0, buf_c0, n_nav, n_harv = collect(
            sim, mod, device, T, B, hdim, zero_h, zero_c
        )

        flat_obs = _stack_obs(buf_obs, T)
        flat_mask = torch.cat([buf_mask[t] for t in range(T)], dim=0)
        flat_h0 = buf_h0.reshape(T * B, hdim)
        flat_c0 = buf_c0.reshape(T * B, hdim)
        flat_exp = {
            "skill_type": torch.cat([buf_exp[t]["skill_type"] for t in range(T)], dim=0),
            "target_class": torch.cat([buf_exp[t]["target_class"] for t in range(T)], dim=0),
            "spatial_param": torch.cat([buf_exp[t]["spatial_param"] for t in range(T)], dim=0),
            "scalar_param": torch.cat([buf_exp[t]["scalar_param"] for t in range(T)], dim=0),
        }
        N = T * B

        losses, ce_sks, mse_sps, ce_tgs = [], [], [], []
        for _ep in range(args.epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, args.mb_size):
                idx = perm[start : start + args.mb_size]
                mb_obs = {}
                for k, v in flat_obs.items():
                    if isinstance(v, dict):
                        mb_obs[k] = {sk: sv[idx] for sk, sv in v.items()}
                    else:
                        mb_obs[k] = v[idx]
                exp = {k: v[idx] for k, v in flat_exp.items()}
                loss, ce_sk, mse_sp, ce_tg = bc_loss_for_chunk(
                    mod, mb_obs, flat_h0[idx], flat_c0[idx], exp, flat_mask[idx]
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mod.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss))
                ce_sks.append(float(ce_sk))
                mse_sps.append(float(mse_sp))
                ce_tgs.append(float(ce_tg))

        dt = time.perf_counter() - t0
        print(
            f"bc_iter {it:3d} | loss {np.mean(losses):.4f} "
            f"| ce_skill {np.mean(ce_sks):.4f} mse_spatial {np.mean(mse_sps):.5f} "
            f"ce_target {np.mean(ce_tgs):.4f} "
            f"| demo n_nav={n_nav} n_harv={n_harv} | {dt:.2f}s",
            flush=True,
        )

    print("BC after :", seed1_navigate_probe(mod, device), flush=True)

    if args.save_weights:
        os.makedirs(os.path.dirname(args.save_weights) or ".", exist_ok=True)
        torch.save(mod.state_dict(), args.save_weights)
        print(f"saved BC weights -> {args.save_weights}", flush=True)


if __name__ == "__main__":
    main()
