"""Lean, no-Ray PPO training loop for the AiUtopia gatherer over a vectorized sim.

The Ray RLlib trainer measures ~130 env-steps/s, but a step-time decomposition
showed ~65% of that is framework tax (ConnectorV2 pipelines, LSTM time-dim
zero-padding, per-minibatch machinery on the Windows in-driver learner) -- NOT
real compute (raw fwd+bwd on the learner shape is only ~0.84s of a 3.96s learner
step). A vectorized batched env (VecGathererSim, ~17,600 env-steps/s in
isolation) now exists. This script drives the SAME RLModule policy
(AiUtopiaRoleRLModule, gatherer) with a hand-written torch PPO loop over B
parallel envs to measure the real end-to-end ceiling once the framework tax is
removed, and to sanity-check that the loop learns (episode return should rise
toward ~127: the gatherer one-shots 64 oak_log via HARVEST, primary reward
64*1.0 = 64 plus PBRS telescoping, minus a small per-step time penalty).

Correctness notes (see report / commit message for the full rationale):
 - REUSE the module bound action dist (mod.action_dist_cls): flat-344 logits
   split into the 7 sub-dists in the exact alphabetical contract order; no
   hand-rolled slice decode. sample()->dict; logp/entropy->(B,) summed.
 - SKILL MASK: the RLModule does NOT apply the skill_type mask (it lived in the
   RLlib action-masking connector this loop bypasses). We apply the obs
   action_mask skill_type to slice [268:274] before from_logits, identically in
   the rollout AND update forwards so the PPO ratio is consistent and matches the
   mask-aware eval gate. Built with mask_comm=True (M1B gate path).
 - GAE term-vs-trunc: vec_sim overwrites the terminal obs on auto-reset, so the
   true final obs is lost. Bootstrap = 0 on term, = pre-step V(obs_t)=buf_val[t]
   on trunc, = next_val otherwise; lastgae reset to 0 on term|trunc.
 - LSTM update-forward approximation: the module takes one start-state per
   sequence and cannot re-zero mid-sequence, so a T-chunk crossing an episode
   boundary runs the LSTM across it during the update (collection is exact and
   DOES re-zero). Episodes here are short (one-shot harvest), so essentially
   EVERY T=32 chunk crosses a boundary (seq_cross ~1.0) -- this is structural,
   not a footnote. We measure its impact directly via the first-minibatch probe:
   on the first minibatch of the first epoch the params are unchanged from
   collection, so a faithful replay gives approx_kl/clipfrac ~0. Measured:
   first_mb KL ~0.001, clipfrac ~0.01 -- i.e. the cross-boundary LSTM replay
   does NOT materially corrupt the PPO ratio in practice, despite seq_cross~1.
   The exact (module-safe) fix would be to re-segment each rollout at episode
   boundaries and replay each segment from a zero start-state; the probe shows
   it is not needed for the gatherer. We log both seq_cross and the first_mb
   probe every iter so the approximation stays visible and falsifiable.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from torch import nn

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.rl_module.actor_head import _slice_offsets
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule
from aiutopia.sim.vec_sim import VecGathererSim

SKILL_SLICE = _slice_offsets()["skill_type"]
GAMMA = 0.99
LAMBDA = 0.95


def build_module(device):
    spec = MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=build_role_observation_space("gatherer", stage=1),
                action_space=build_role_action_space("gatherer"),
                model_config={
                    "role": "gatherer",
                    "max_seq_len": 32,
                    "actor_hidden": [256],
                    "mask_comm": True,
                    "core_encoder": {"core_hidden": [512, 256]},
                    "shared_backbone": {"lstm_hidden": 256},
                    "ctde_critic": {"critic_hidden": 256},
                },
            ),
        },
    ).build()
    return spec["gatherer_policy"].to(device)


def obs_to_tensors(obs, device):
    out = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            out[k] = obs_to_tensors(v, device)
        else:
            out[k] = torch.as_tensor(np.asarray(v)).to(device)
    return out


def apply_skill_mask(adi, skill_mask):
    s, e = SKILL_SLICE
    adi = adi.clone()
    mask = skill_mask.to(torch.bool)
    any_legal = mask.any(dim=-1, keepdim=True)
    keep = mask | (~any_legal)
    neg = torch.finfo(adi.dtype).min / 2
    adi[:, s:e] = torch.where(keep, adi[:, s:e], torch.full_like(adi[:, s:e], neg))
    return adi


def actions_to_numpy_for_env(act):
    """Convert the sampled action dict to numpy for vec_sim.step, clipping the
    bounded Box children to their declared ranges -- matching the eval gate
    (scenario_runner._greedy_decode). We clip a COPY for the env only; the RAW
    sampled action is stored in the buffer so the PPO logp is recomputed on the
    same raw value at update time (standard practice; keeps the ratio exact).
    Unbounded raw Gaussian spatial_param otherwise walks the agent out of the
    arena box in a handful of steps (OOB truncation).
    """
    out = {}
    for k, v in act.items():
        arr = v.detach().cpu().numpy()
        if k == "scalar_param":
            arr = np.clip(arr, 0.0, 1.0)
        elif k in ("spatial_param", "comm_payload"):
            arr = np.clip(arr, -1.0, 1.0)
        out[k] = arr
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb-envs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coeff", type=float, default=0.01)
    ap.add_argument("--vf-coeff", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-ep-ticks", type=int, default=300)
    ap.add_argument("--gate-check", action="store_true",
                    help="run 3 M1 scenarios (mask-aware greedy) post-train")
    ap.add_argument("--save-weights", type=str, default="",
                    help="path to torch.save the trained module state_dict")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    B, T = args.num_envs, args.horizon
    mod = build_module(device)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)

    sim = VecGathererSim(num_envs=B, max_episode_ticks=args.max_ep_ticks)
    seeds = np.arange(B, dtype=np.int64) + 1 + args.seed * B
    obs = sim.reset(seeds)

    init_state = mod.get_initial_state()
    hdim = init_state["h"].shape[0]
    state_h = init_state["h"].unsqueeze(0).expand(B, hdim).contiguous().to(device)
    state_c = init_state["c"].unsqueeze(0).expand(B, hdim).contiguous().to(device)
    zero_h = init_state["h"].to(device)
    zero_c = init_state["c"].to(device)

    sps = float("nan")
    collect_time = update_time = 0.0
    sps_hist: list[float] = []
    collect_hist: list[float] = []
    update_hist: list[float] = []
    for it in range(args.iters):
        t_iter0 = time.perf_counter()
        buf_obs = []
        buf_act = []
        buf_logp = torch.zeros(T, B, device=device)
        buf_val = torch.zeros(T, B, device=device)
        buf_rew = torch.zeros(T, B, device=device)
        buf_term = torch.zeros(T, B, device=device)
        buf_trunc = torch.zeros(T, B, device=device)
        buf_h0 = torch.zeros(T, B, hdim, device=device)
        buf_c0 = torch.zeros(T, B, hdim, device=device)
        # Full-episode returns (correct values from sim.last_episode_return,
        # which accumulates the WHOLE episode -- including the portion that
        # began in a previous rollout window). Split by term vs trunc.
        term_ep_returns: list[float] = []
        trunc_ep_returns: list[float] = []

        t_collect0 = time.perf_counter()
        for t in range(T):
            obs_t = obs_to_tensors(obs, device)
            buf_obs.append(obs_t)
            buf_h0[t] = state_h
            buf_c0[t] = state_c
            with torch.no_grad():
                out = mod._forward_train(
                    {Columns.OBS: obs_t, Columns.STATE_IN: {"h": state_h, "c": state_c}}
                )
                adi = apply_skill_mask(
                    out[Columns.ACTION_DIST_INPUTS], obs_t["action_mask"]["skill_type"]
                )
                dist = mod.action_dist_cls.from_logits(adi)
                act = dist.sample()
                logp = dist.logp(act)
                val = out[Columns.VF_PREDS]
                st_out = out[Columns.STATE_OUT]

            buf_act.append({k: v.detach() for k, v in act.items()})
            buf_logp[t] = logp
            buf_val[t] = val

            obs, rew, term, trunc = sim.step(actions_to_numpy_for_env(act))
            buf_rew[t] = torch.as_tensor(rew, device=device, dtype=torch.float32)
            term_t = torch.as_tensor(term, device=device, dtype=torch.float32)
            trunc_t = torch.as_tensor(trunc, device=device, dtype=torch.float32)
            buf_term[t] = term_t
            buf_trunc[t] = trunc_t
            # Snapshot the TRUE full-episode return for envs that just finished
            # (the sim wrote it into last_episode_return at this step; NaN
            # otherwise). Count a goal-termination as term even if the tick/OOB
            # truncation also fired on the same step.
            ler = sim.last_episode_return
            t_np = np.asarray(term, dtype=bool)
            tr_np = np.asarray(trunc, dtype=bool) & (~t_np)
            term_ep_returns.extend(ler[t_np & np.isfinite(ler)].tolist())
            trunc_ep_returns.extend(ler[tr_np & np.isfinite(ler)].tolist())

            state_h = st_out["h"].detach()
            state_c = st_out["c"].detach()
            done = (term_t + trunc_t) > 0
            if bool(done.any()):
                state_h[done] = zero_h
                state_c[done] = zero_c

        collect_time = time.perf_counter() - t_collect0

        with torch.no_grad():
            obs_t = obs_to_tensors(obs, device)
            out = mod._forward_train(
                {Columns.OBS: obs_t, Columns.STATE_IN: {"h": state_h, "c": state_c}}
            )
            last_val = out[Columns.VF_PREDS].detach()

        adv = torch.zeros(T, B, device=device)
        lastgae = torch.zeros(B, device=device)
        next_val = last_val
        for t in reversed(range(T)):
            term_t = buf_term[t]
            trunc_t = buf_trunc[t]
            done = (term_t + trunc_t).clamp(max=1.0)
            boot = torch.where(term_t > 0, torch.zeros_like(next_val), next_val)
            boot = torch.where(trunc_t > 0, buf_val[t], boot)
            delta = buf_rew[t] + GAMMA * boot - buf_val[t]
            lastgae = delta + GAMMA * LAMBDA * (1.0 - done) * lastgae
            adv[t] = lastgae
            next_val = buf_val[t]
        returns = adv + buf_val

        done_mask = (buf_term + buf_trunc).clamp(max=1.0)
        crosses = done_mask[:-1].sum(dim=0) > 0
        cross_frac = float(crosses.float().mean().item())

        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_update0 = time.perf_counter()
        env_idx = np.arange(B)
        kls, clipfracs, ents = [], [], []
        # First-minibatch probe: on the very first minibatch of the first epoch
        # the policy params are UNCHANGED from collection, so for a correct,
        # self-consistent replay approx_kl and clipfrac must be ~0. A materially
        # nonzero value here is direct evidence the start-state-only LSTM replay
        # (which cannot re-zero mid-sequence) is biasing the PPO ratio.
        first_mb_kl = float("nan")
        first_mb_clipfrac = float("nan")
        _probed = False
        for _ep in range(args.epochs):
            np.random.shuffle(env_idx)
            for start in range(0, B, args.mb_envs):
                mb = env_idx[start : start + args.mb_envs]
                mb_t = torch.as_tensor(mb, device=device, dtype=torch.long)
                mbb = len(mb)

                mb_obs = {}
                for k, v0 in buf_obs[0].items():
                    if isinstance(v0, dict):
                        mb_obs[k] = {
                            sk: torch.stack([buf_obs[t][k][sk][mb_t] for t in range(T)], dim=1)
                            for sk in v0
                        }
                    else:
                        mb_obs[k] = torch.stack([buf_obs[t][k][mb_t] for t in range(T)], dim=1)

                s_h = buf_h0[0][mb_t]
                s_c = buf_c0[0][mb_t]
                out = mod._forward_train(
                    {Columns.OBS: mb_obs, Columns.STATE_IN: {"h": s_h, "c": s_c}}
                )
                adi = out[Columns.ACTION_DIST_INPUTS]
                vf = out[Columns.VF_PREDS]

                skill_mask_seq = torch.stack(
                    [buf_obs[t]["action_mask"]["skill_type"][mb_t] for t in range(T)], dim=1
                )
                adi_flat = apply_skill_mask(
                    adi.reshape(-1, adi.shape[-1]),
                    skill_mask_seq.reshape(-1, skill_mask_seq.shape[-1]),
                )
                dist = mod.action_dist_cls.from_logits(adi_flat)

                act_flat = {}
                for k in buf_act[0]:
                    seq = torch.stack([buf_act[t][k][mb_t] for t in range(T)], dim=1)
                    if seq.ndim > 2:
                        act_flat[k] = seq.reshape(-1, *seq.shape[2:])
                    else:
                        act_flat[k] = seq.reshape(-1)
                new_logp = dist.logp(act_flat).reshape(mbb, T)
                ent = dist.entropy().reshape(mbb, T)

                old_logp = buf_logp[:, mb].T
                mb_adv = adv_norm[:, mb].T
                mb_ret = returns[:, mb].T

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (vf - mb_ret).pow(2).mean()
                ent_loss = ent.mean()
                loss = pg_loss + args.vf_coeff * v_loss - args.ent_coeff * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mod.parameters(), args.grad_clip)
                opt.step()

                with torch.no_grad():
                    logratio = new_logp - old_logp
                    approx_kl = ((torch.exp(logratio) - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip).float().mean()
                if not _probed:
                    first_mb_kl = float(approx_kl)
                    first_mb_clipfrac = float(clipfrac)
                    _probed = True
                kls.append(float(approx_kl))
                clipfracs.append(float(clipfrac))
                ents.append(float(ent_loss))

        update_time = time.perf_counter() - t_update0
        iter_time = time.perf_counter() - t_iter0

        # Full-episode return metrics (correct, from sim.last_episode_return).
        # Split by completion kind. term episodes reached the 64-oak goal and
        # should sit near ~127 (64 primary + PBRS telescoping); trunc episodes
        # are OOB / tick-budget and dominate early. The blended mean is reported
        # too for continuity, but term_mean + term_rate is the real M1 signal.
        all_returns = term_ep_returns + trunc_ep_returns
        ep_ret_mean = float(np.mean(all_returns)) if all_returns else float("nan")
        n_ep = len(all_returns)
        term_mean = float(np.mean(term_ep_returns)) if term_ep_returns else float("nan")
        n_term = len(term_ep_returns)
        term_rate = (n_term / n_ep) if n_ep > 0 else float("nan")

        sps = (B * T) / iter_time
        sps_hist.append(sps)
        collect_hist.append(collect_time)
        update_hist.append(update_time)
        ent_m = float(np.mean(ents))
        kl_m = float(np.mean(kls))
        cf_m = float(np.mean(clipfracs))
        print(
            f"iter {it:3d} | sps {sps:7.0f} | "
            f"collect {collect_time:5.2f}s update {update_time:5.2f}s | "
            f"ep_ret {ep_ret_mean:7.2f} (n={n_ep}) | "
            f"term_mean {term_mean:7.2f} (n_term={n_term}) term_rate {term_rate:.3f} | "
            f"ent {ent_m:.3f} KL {kl_m:.4f} clipfrac {cf_m:.3f} | "
            f"first_mb[KL {first_mb_kl:.5f} clipfrac {first_mb_clipfrac:.4f}] | "
            f"seq_cross {cross_frac:.2f}",
            flush=True,
        )

    warm = sps_hist[1:] if len(sps_hist) > 1 else sps_hist  # drop iter 0 (CUDA init)
    med_sps = float(np.median(warm)) if warm else float('nan')
    lo_sps = float(np.min(warm)) if warm else float('nan')
    hi_sps = float(np.max(warm)) if warm else float('nan')
    med_collect = float(np.median(collect_hist[1:] or collect_hist))
    med_update = float(np.median(update_hist[1:] or update_hist))
    print(
        f"\nHEADLINE: B={B} T={T} -> median {med_sps:.0f} env-steps/s end-to-end "
        f"over iters 1..{len(sps_hist) - 1} (range {lo_sps:.0f}-{hi_sps:.0f}); "
        f"per-iter median collect {med_collect:.2f}s / update {med_update:.2f}s. "
        f"RLlib baseline 130/s -> speedup {med_sps / 130:.1f}x (median).",
        flush=True,
    )

    if args.save_weights:
        torch.save(mod.state_dict(), args.save_weights)
        print(f'saved weights -> {args.save_weights}', flush=True)

    if args.gate_check:
        from aiutopia.train.scenario_runner import (  # noqa: PLC0415
            M1_SCENARIOS,
            aggregate_success_rate,
            run_scenario,
        )

        mod.eval()
        gate_results = []
        for sc in M1_SCENARIOS:
            res = run_scenario(
                sc,
                env_config={'active_roles': ['gatherer'], 'backend': 'sim'},
                rl_module=mod,
                device=device,
            )
            gate_results.append(res)
            print(f"  gate {res['name']}: success={res['success']}", flush=True)
        sr = aggregate_success_rate(gate_results)
        print(f'M1 GATE success_rate (mask-aware greedy eval): {sr:.3f}', flush=True)


if __name__ == "__main__":
    main()
