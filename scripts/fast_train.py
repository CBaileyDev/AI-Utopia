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
 - LSTM update-forward FIDELITY (update_forward, faithful=True): the module
   takes one start-state per sequence and cannot re-zero mid-sequence. Episodes
   here are short (one-shot harvest), so essentially EVERY T=32 chunk crosses an
   episode boundary (seq_cross ~1.0). The OLD start-state-only update ran the
   LSTM across the boundary while COLLECTION re-zeroed on done -> the update
   hidden-state trajectory diverged and the PPO ratio was biased. Benign for a
   high-entropy policy but CATASTROPHIC for a sharp BC clone (first_mb KL 0.71 at
   T=32), eroding a cloned navigate-then-harvest policy. FIX: truncated-BPTT(1)
   -- fold (mb, T) -> (mb*T, 1) and feed the STORED per-timestep collection
   states (buf_h0/buf_c0[t]) as STATE_IN, so the update state-trajectory ==
   collection's INCLUDING the post-done re-zero. With params unchanged the
   update logp == collection logp per timestep: the first-minibatch probe
   measures first_mb KL = -0.00000 at T=32 (was 0.71), held across all 100 iters
   of RUN D. We log seq_cross and the first_mb probe every iter so the fidelity
   stays visible and falsifiable.
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


def split_actor_critic_params(mod):
    """Partition named params: (actor, critic). Critic = ``ctde_critic.*`` only (raw-obs value path, freeze is clean)."""  # noqa: E501
    actor, critic = [], []
    for name, p in mod.named_parameters():
        (critic if name.startswith("ctde_critic") else actor).append(p)
    return actor, critic


def build_optimizer(actor_params, critic_params, *, actor_lr, value_lr):
    """Adam with two param-groups: [0]=actor (ramped lr), [1]=critic (value_lr)."""
    return torch.optim.Adam(
        [
            {"params": actor_params, "lr": actor_lr},
            {"params": critic_params, "lr": value_lr},
        ]
    )


def actor_lr_for_iter(it, *, warmup, ramp, full_lr):
    """Return (actor_lr, phase) for iter ``it``. Phase in warmup|ramp|finetune."""
    if it < warmup:
        return 0.0, "warmup"
    if ramp > 0 and it < warmup + ramp:
        step = (it - warmup) + 1  # 1..ramp -> lr full_lr*1/ramp .. full_lr
        return full_lr * (step / ramp), "ramp"
    return full_lr, "finetune"


def warmup_step(mod, opt, obs, s_h, s_c, returns, *, vf_coeff, grad_clip):
    """One VALUE-ONLY update; returns the value loss. No pg/ent/kl term."""
    out = mod._forward_train({Columns.OBS: obs, Columns.STATE_IN: {"h": s_h, "c": s_c}})
    vf = out[Columns.VF_PREDS]
    v_loss = 0.5 * (vf - returns).pow(2).mean()
    loss = vf_coeff * v_loss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(mod.parameters(), grad_clip)
    opt.step()
    return float(v_loss.detach())


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


# LSTM-faithfulness fix (the crux). COLLECTION threads the per-env hidden state
# one tick at a time and RE-ZEROS it on every episode `done` (auto-reset). The
# naive UPDATE forward runs the module's nn.LSTM over the whole T-chunk from a
# SINGLE chunk-start state, so it cannot re-zero mid-chunk: when a chunk crosses
# an episode boundary (almost always, for short gatherer episodes -> seq_cross
# ~1.0) the update hidden-state trajectory diverges from collection's, biasing
# the new/old PPO ratio. Benign for a high-entropy policy, CATASTROPHIC for a
# sharp BC clone (first_mb KL 0.5-0.7), which corrupts the update and erodes the
# cloned navigate-then-harvest policy.
#
# faithful=True reproduces the collection trajectory EXACTLY via truncated-BPTT
# (1 step): collapse (mb, T) -> (mb*T, 1) and feed the STORED per-timestep
# collection states (buf_h_seq/buf_c_seq, already detached => stop-grad on the
# carried state) as STATE_IN. Each timestep's forward then starts from precisely
# the hidden state collection used at that tick, INCLUDING the post-`done`
# re-zero, so with UNCHANGED params the update logp == collection logp per
# timestep (first_mb KL ~0) across boundaries. Gradient does not flow across
# timesteps through the recurrence (the accepted TBPTT-1 price; fine for
# consolidating an already-trained clone). faithful=False is the legacy
# start-state-only path (uses s_h0/s_c0 only), kept so the replay-invariant test
# can demonstrate the bug.
def update_forward(
    mod,
    mb_obs,
    skill_mask_seq,
    act_seq,
    s_h0,
    s_c0,
    buf_h_seq,
    buf_c_seq,
    *,
    T,
    faithful,
):
    """Recompute (new_logp, entropy, vf, adi_flat) for a PPO update minibatch."""
    mbb = skill_mask_seq.shape[0]
    if faithful:
        # TBPTT-1: fold each timestep into the batch as a length-1 sequence so it
        # starts from its STORED collection state. Row index == env*T + t
        # (env-major, t-minor), matching how mb_obs / skill / act / states stack.
        def _fold(v):
            if isinstance(v, dict):
                return {k: _fold(sub) for k, sub in v.items()}
            return v.reshape(mbb * T, 1, *v.shape[2:])  # (mb,T,...) -> (mb*T,1,...)

        obs_in = _fold(mb_obs)
        h_in = buf_h_seq.reshape(mbb * T, -1)
        c_in = buf_c_seq.reshape(mbb * T, -1)
        out = mod._forward_train({Columns.OBS: obs_in, Columns.STATE_IN: {"h": h_in, "c": c_in}})
        adi_seq = out[Columns.ACTION_DIST_INPUTS].reshape(mbb * T, -1)
        vf = out[Columns.VF_PREDS].reshape(mbb, T)
    else:
        out = mod._forward_train({Columns.OBS: mb_obs, Columns.STATE_IN: {"h": s_h0, "c": s_c0}})
        adi = out[Columns.ACTION_DIST_INPUTS]
        adi_seq = adi.reshape(-1, adi.shape[-1])
        vf = out[Columns.VF_PREDS]

    adi_flat = apply_skill_mask(adi_seq, skill_mask_seq.reshape(-1, skill_mask_seq.shape[-1]))
    dist = mod.action_dist_cls.from_logits(adi_flat)

    act_flat = {}
    for k, seq in act_seq.items():
        if seq.ndim > 2:
            act_flat[k] = seq.reshape(-1, *seq.shape[2:])
        else:
            act_flat[k] = seq.reshape(-1)
    new_logp = dist.logp(act_flat).reshape(mbb, T)
    ent = dist.entropy().reshape(mbb, T)
    return new_logp, ent, vf, adi_flat


def actions_to_numpy_for_env(act):
    """Convert the sampled action dict to numpy for vec_sim.step.

    Clips the bounded Box children (scalar/spatial/comm) to their declared ranges
    for the env ONLY (matching scenario_runner._greedy_decode); the RAW sampled
    action stays in the buffer so the PPO logp is recomputed on the same value
    (keeps the ratio exact). Unbounded raw spatial_param otherwise walks OOB fast.
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


def seed1_navigate_probe(mod, device) -> str:
    """seed_1 masked-spawn skill-type logits (the supervised leading indicator).

    Clean seed_1 scenario reset (no curriculum); reads RAW skill_type logits and
    reports the NAVIGATE(0) logit + rank vs SEARCH(3)/HARVEST(1). Ranks on RAW
    logits (masking HARVEST would make the rank trivial).
    """
    import torch  # noqa: PLC0415
    from ray.rllib.core import Columns  # noqa: PLC0415

    from aiutopia.sim.sim_env import AiUtopiaSimEnv  # noqa: PLC0415
    from aiutopia.train.scenario_runner import M1_SCENARIOS  # noqa: PLC0415

    sc = next(s for s in M1_SCENARIOS if s.seed == 1)
    env = AiUtopiaSimEnv(
        {"active_roles": ["gatherer"], "backend": "sim", "max_episode_ticks": sc.max_ticks}
    )
    try:
        obs, _ = env.reset(seed=sc.seed)
        agent_obs = obs["gatherer_0"]

        def _batch(v):
            if isinstance(v, dict):
                return {k: _batch(vv) for k, vv in v.items()}
            return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)

        state = mod.get_initial_state()
        state_in = {k: v.to(device).unsqueeze(0) for k, v in state.items()}
        with torch.no_grad():
            out = mod._forward_inference(
                {
                    Columns.OBS: {k: _batch(v) for k, v in agent_obs.items()},
                    Columns.STATE_IN: state_in,
                }
            )
        s, e = SKILL_SLICE
        logits = out[Columns.ACTION_DIST_INPUTS][0, s:e].detach().cpu().numpy()
        names = ["NAVIGATE", "HARVEST", "DEPOSIT", "SEARCH", "WAIT", "NOOP"]
        order = np.argsort(-logits)  # descending
        nav_rank = int(np.where(order == 0)[0][0]) + 1  # 1 == top
        masked = int(np.asarray(agent_obs["action_mask"]["skill_type"]).reshape(-1)[1]) == 0
        per = " ".join(f"{names[i]}={logits[i]:+.2f}" for i in range(len(logits)))
        return (
            f"seed1 NAVIGATE-logit: NAV={logits[0]:+.3f} (rank {nav_rank}/6) "
            f"vs HARVEST={logits[1]:+.3f} SEARCH={logits[3]:+.3f} "
            f"| harvest_masked={masked} | all[{per}]"
        )
    finally:
        env.close()


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
    # --- PPO stability (clip-only diverges past peak; KL holds it) ---
    ap.add_argument(
        "--kl-coeff",
        type=float,
        default=0.0,
        help="fixed KL-penalty coeff added to the loss (0 = off, current behavior)",
    )
    ap.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help="if >0, ADAPTIVE KL: scale kl_coeff *1.5 if iter-mean KL>1.5x target, "
        "/1.5 if <0.5x (RLlib-style). Needs --kl-coeff>0 as the starting value",
    )
    ap.add_argument(
        "--anneal-ent",
        action="store_true",
        help="linearly decay ent-coeff from --ent-coeff to ~0 over --iters",
    )
    ap.add_argument(
        "--lr-anneal",
        choices=["off", "linear", "cosine"],
        default="off",
        help="decay lr from --lr to ~0 over --iters (off=constant, current behavior)",
    )
    # --- TRAINING-only curriculum (parity-gated; eval stays clean) ---
    ap.add_argument(
        "--spawn-jitter",
        type=float,
        default=0.0,
        help="VecGathererSim spawn jitter (± blocks); >0 auto-enables randomize_layout",
    )
    ap.add_argument(
        "--approach-shaping",
        action="store_true",
        help="VecGathererSim PBRS toward nearest log while HARVEST masked",
    )
    ap.add_argument(
        "--force-masked-spawn",
        action="store_true",
        help="VecGathererSim: push agent out until HARVEST masked every episode",
    )
    ap.add_argument(
        "--gate-check",
        action="store_true",
        help="run 3 M1 scenarios (mask-aware greedy) post-train",
    )
    ap.add_argument(
        "--save-weights",
        type=str,
        default="",
        help="path to torch.save the trained module state_dict",
    )
    ap.add_argument(
        "--load-weights",
        type=str,
        default="",
        help="path to a state_dict to load BEFORE training (e.g. a BC warm-start). "
        "Combine with --iters 0 --gate-check to evaluate a checkpoint with no training.",
    )
    # --- BC consolidation: value warm-up + actor-freeze, then gentle LR ramp ---
    ap.add_argument(
        "--value-warmup-iters",
        type=int,
        default=0,
        help="first N iters train ONLY the value/critic; the actor is FROZEN "
        "(value-only loss + actor param-group lr=0) so a BC-cloned policy is byte-"
        "stable while the random critic learns to predict returns under it. Rollouts "
        "+ GAE are still real under the frozen policy. 0 = off.",
    )
    ap.add_argument(
        "--actor-lr",
        type=float,
        default=-1.0,
        help="actor (policy) LR for the finetune phase. <0 (default) = use --lr.",
    )
    ap.add_argument(
        "--value-lr",
        type=float,
        default=-1.0,
        help="critic (value) LR, CONSTANT through warm-up AND finetune. <0 = use --lr. "
        "Set higher than --actor-lr to converge the critic during warm-up.",
    )
    ap.add_argument(
        "--actor-lr-ramp",
        type=int,
        default=0,
        help="after warm-up, ramp the actor LR linearly from 0 to actor-lr over R "
        "iters so consolidation does not jolt the cloned policy. 0 = no ramp.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    B, T = args.num_envs, args.horizon
    mod = build_module(device)
    if args.load_weights:
        sd = torch.load(args.load_weights, map_location=device)
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        print(
            f"loaded weights <- {args.load_weights} "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )
    # Two param-groups (actor / critic) so the actor can be frozen (lr=0) during
    # the value warm-up and ramped afterward while the critic trains throughout at
    # --value-lr. group[0]=actor, group[1]=critic (see build_optimizer).
    actor_params, critic_params = split_actor_critic_params(mod)
    full_actor_lr = args.actor_lr if args.actor_lr >= 0.0 else args.lr
    value_lr = args.value_lr if args.value_lr >= 0.0 else args.lr
    warmup_iters = max(0, args.value_warmup_iters)
    ramp_iters = max(0, args.actor_lr_ramp)
    # During warm-up the actor group starts at lr=0 (belt-and-suspenders alongside
    # the value-only loss). actor_lr_for_iter is the single source of truth.
    init_actor_lr, _ = actor_lr_for_iter(
        0, warmup=warmup_iters, ramp=ramp_iters, full_lr=full_actor_lr
    )
    opt = build_optimizer(actor_params, critic_params, actor_lr=init_actor_lr, value_lr=value_lr)
    print(
        f"optimizer: actor params={len(actor_params)} critic params={len(critic_params)} "
        f"| value_warmup_iters={warmup_iters} actor_lr_ramp={ramp_iters} "
        f"full_actor_lr={full_actor_lr:g} value_lr={value_lr:g}",
        flush=True,
    )

    # TRAINING-only curriculum (parity-gated in VecGathererSim; default off). Any
    # curriculum flag implies randomize_layout (the scalar env gates all three on
    # it); without that the knobs would be silent no-ops.
    curriculum = args.spawn_jitter > 0.0 or args.approach_shaping or args.force_masked_spawn
    sim = VecGathererSim(
        num_envs=B,
        max_episode_ticks=args.max_ep_ticks,
        spawn_jitter=args.spawn_jitter,
        approach_shaping=args.approach_shaping,
        force_masked_spawn=args.force_masked_spawn,
        randomize_layout=curriculum,
    )
    seeds = np.arange(B, dtype=np.int64) + 1 + args.seed * B
    obs = sim.reset(seeds)

    # KL-penalty controller state (fixed or adaptive). kl_coeff is mutable across
    # iters when --target-kl>0 (RLlib-style adaptive). Defaults (0/0) reproduce the
    # prior clip-only loss bit-for-bit (the kl term is multiplied by 0).
    kl_coeff = float(args.kl_coeff)

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
        # Per-iter schedules (all OFF by default -> constant, prior behavior).
        frac = it / max(1, args.iters - 1)  # 0..1 across the run
        ent_coeff = args.ent_coeff * (1.0 - frac) if args.anneal_ent else args.ent_coeff

        # --- BC-consolidation phase + per-group LR (single source of truth) ---
        # actor LR: 0 during warm-up, ramped over --actor-lr-ramp, then full.
        actor_lr, phase = actor_lr_for_iter(
            it, warmup=warmup_iters, ramp=ramp_iters, full_lr=full_actor_lr
        )
        # --lr-anneal (if on) decays the FULL actor LR, applied only once the actor
        # is unfrozen (warm-up/ramp untouched so the freeze + ramp stay clean).
        if phase == "finetune" and args.lr_anneal != "off":
            denom = max(1, args.iters - 1 - (warmup_iters + ramp_iters))
            ft_frac = min(1.0, max(0.0, (it - (warmup_iters + ramp_iters)) / denom))
            if args.lr_anneal == "linear":
                actor_lr = full_actor_lr * (1.0 - ft_frac)
            else:  # cosine
                actor_lr = full_actor_lr * 0.5 * (1.0 + np.cos(np.pi * ft_frac))
        opt.param_groups[0]["lr"] = actor_lr  # actor group
        opt.param_groups[1]["lr"] = value_lr  # critic group (constant)
        is_warmup = phase == "warmup"
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
        kls, clipfracs, ents, v_losses = [], [], [], []
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

                if is_warmup:
                    # VALUE-ONLY warm-up: train ONLY the critic on this minibatch.
                    # mb_ret here uses GAE returns computed under the FROZEN policy
                    # (real targets). The pg/ent/kl terms are dropped entirely, so
                    # the actor sees zero gradient (hence zero Adam momentum) and
                    # the cloned policy is byte-stable. warmup_step also calls
                    # _forward_train, so the value path / LSTM threading matches the
                    # finetune path exactly.
                    mb_ret = returns[:, mb].T
                    vl = warmup_step(
                        mod,
                        opt,
                        mb_obs,
                        s_h,
                        s_c,
                        mb_ret,
                        vf_coeff=args.vf_coeff,
                        grad_clip=args.grad_clip,
                    )
                    v_losses.append(vl)
                    continue

                # LSTM-faithful PPO forward: per-timestep collection states fed as
                # STATE_IN (TBPTT-1) so the update state-trajectory == collection's
                # across auto-reset boundaries -> first_mb KL ~0 even for a sharp
                # BC clone. buf_h0/buf_c0[t] are the pre-step collection states.
                skill_mask_seq = torch.stack(
                    [buf_obs[t]["action_mask"]["skill_type"][mb_t] for t in range(T)], dim=1
                )
                act_seq = {
                    k: torch.stack([buf_act[t][k][mb_t] for t in range(T)], dim=1)
                    for k in buf_act[0]
                }
                buf_h_seq = torch.stack([buf_h0[t][mb_t] for t in range(T)], dim=1)
                buf_c_seq = torch.stack([buf_c0[t][mb_t] for t in range(T)], dim=1)
                new_logp, ent, vf, _adi = update_forward(
                    mod,
                    mb_obs,
                    skill_mask_seq,
                    act_seq,
                    s_h,
                    s_c,
                    buf_h_seq,
                    buf_c_seq,
                    T=T,
                    faithful=True,
                )

                old_logp = buf_logp[:, mb].T
                mb_adv = adv_norm[:, mb].T
                mb_ret = returns[:, mb].T

                logratio = new_logp - old_logp
                ratio = torch.exp(logratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (vf - mb_ret).pow(2).mean()
                ent_loss = ent.mean()
                # KL penalty (Schulman k3 estimator, GRAD-carrying so it actually
                # pulls the new policy back toward the OLD collection policy). With
                # kl_coeff=0 (default) this term is exactly 0 -> clip-only behavior.
                kl_mb = ((ratio - 1.0) - logratio).mean()
                loss = pg_loss + args.vf_coeff * v_loss - ent_coeff * ent_loss + kl_coeff * kl_mb

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
                v_losses.append(float(v_loss.detach()))

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
        # During warm-up the pg/ent/kl lists are empty (value-only step); guard the
        # means so a warm-up iter logs nan for those instead of a RuntimeWarning.
        ent_m = float(np.mean(ents)) if ents else float("nan")
        kl_m = float(np.mean(kls)) if kls else float("nan")
        cf_m = float(np.mean(clipfracs)) if clipfracs else float("nan")
        vloss_m = float(np.mean(v_losses)) if v_losses else float("nan")
        kl_coeff_logged = kl_coeff
        # ADAPTIVE KL (RLlib-style): nudge kl_coeff toward holding the iter-mean KL
        # near --target-kl. Only active when target_kl>0; finite-KL guard so a NaN
        # iter can't poison the controller. Logs the coeff USED this iter.
        if args.target_kl > 0.0 and np.isfinite(kl_m):
            if kl_m > 1.5 * args.target_kl:
                kl_coeff *= 1.5
            elif kl_m < 0.5 * args.target_kl:
                kl_coeff /= 1.5
        print(
            f"iter {it:3d} | {phase:8s} alr {actor_lr:.2e} | sps {sps:7.0f} | "
            f"collect {collect_time:5.2f}s update {update_time:5.2f}s | "
            f"ep_ret {ep_ret_mean:7.2f} (n={n_ep}) | "
            f"term_mean {term_mean:7.2f} (n_term={n_term}) term_rate {term_rate:.3f} | "
            f"v_loss {vloss_m:8.3f} | "
            f"ent {ent_m:.3f} KL {kl_m:.4f} clipfrac {cf_m:.3f} kl_coeff {kl_coeff_logged:.4f} | "
            f"first_mb[KL {first_mb_kl:.5f} clipfrac {first_mb_clipfrac:.4f}] | "
            f"seq_cross {cross_frac:.2f}",
            flush=True,
        )

    warm = sps_hist[1:] if len(sps_hist) > 1 else sps_hist  # drop iter 0 (CUDA init)
    med_sps = float(np.median(warm)) if warm else float("nan")
    lo_sps = float(np.min(warm)) if warm else float("nan")
    hi_sps = float(np.max(warm)) if warm else float("nan")
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
        print(f"saved weights -> {args.save_weights}", flush=True)

    if args.gate_check:
        from aiutopia.env.reward import _inventory_from_obs  # noqa: PLC0415
        from aiutopia.train.scenario_runner import (  # noqa: PLC0415
            M1_SCENARIOS,
            aggregate_success_rate,
            run_scenario,
        )

        mod.eval()
        # Supervised leading indicator: print the seed_1 NAVIGATE-logit line every
        # run (the basin-escape signal precedes the gate flipping).
        print(seed1_navigate_probe(mod, device), flush=True)
        gate_results = []
        for sc in M1_SCENARIOS:
            res = run_scenario(
                sc,
                # EVAL stays CLEAN: this env_config carries NO curriculum keys, so the
                # eval env defaults spawn_jitter/approach_shaping/force_masked_spawn
                # OFF and uses the FIXED scenario seed. Training-time curriculum cannot
                # leak into the gate measurement.
                env_config={"active_roles": ["gatherer"], "backend": "sim"},
                rl_module=mod,
                device=device,
            )
            gate_results.append(res)
            # Per-seed oak count (not just success): the SAME _inventory_from_obs the
            # gate predicate uses, over the final obs run_scenario returns.
            final = res.get("final_inventory", {}).get("gatherer_0", {})
            oak = _inventory_from_obs(final).get("oak_log", 0) if final else 0
            print(
                f"  gate {res['name']}: success={res['success']} oak={oak}",
                flush=True,
            )
        sr = aggregate_success_rate(gate_results)
        print(f"M1 GATE success_rate (mask-aware greedy eval): {sr:.3f}", flush=True)


if __name__ == "__main__":
    main()
