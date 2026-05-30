"""Section 7.1 PPO config builders."""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule

ENV_NAME = "aiutopia_minecraft"
SIM_ENV_NAME = "aiutopia_sim"


def register_aiutopia_env() -> None:
    """Idempotent registration of the real-MC env factory with Ray Tune.

    NOTE: imports ``env_factory`` (and thus ``AiUtopiaPettingZooEnv``) lazily so
    the heavy chromadb / py4j / sentence-transformers deps load only when the
    REAL backend is selected — the sim backend must never trigger this.
    """
    from aiutopia.train.env_factory import make_aiutopia_env_wrapped

    register_env(ENV_NAME, make_aiutopia_env_wrapped)


def register_aiutopia_sim_env() -> None:
    """Idempotent registration of the FAST-SIM env factory with Ray Tune.

    Mirrors ``register_aiutopia_env`` but imports the import-light
    ``sim_env_factory`` (only ``AiUtopiaSimEnv``), so selecting the sim backend
    pulls in NO chroma / py4j / torch-via-env / sentence-transformers.
    """
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env_wrapped

    register_env(SIM_ENV_NAME, make_aiutopia_sim_env_wrapped)


def _policy_mapping_fn(agent_id, episode=None, **kwargs):
    """New-API-stack signature: (agent_id, episode, **kwargs)."""
    return "gatherer_policy"


def m1_gatherer_config(
    *,
    # "real" -> live Minecraft via FabricBridge/Py4J (default; unchanged).
    # "sim"  -> the headless fast-sim (AiUtopiaSimEnv): points env= at the sim
    #           env name, registers ONLY the import-light sim factory, and drops
    #           the real-MC-only env_config keys (py4j_ports, tick_warp, ...).
    #           The SAME RLModule/spaces build against the sim env (its declared
    #           obs/action spaces are byte-identical to the real env's).
    backend: str = "real",
    # None -> derive one distinct port per runner (25001..25000+num_env_runners).
    # The wrapper maps worker_index % len(py4j_ports) -> port, so len(ports) MUST
    # equal num_env_runners or runners collide onto shared Fabric servers.
    py4j_ports: tuple[int, ...] | None = None,
    num_env_runners: int = 4,
    num_envs_per_env_runner: int = 2,
    # N17 finding: with `max_episode_ticks=2000` and the
    # 8-log seeded ring placed at episode start, the agent
    # depletes the ring in the first ~50-150 env steps and
    # then spends the remaining 1850+ steps with zero
    # oak_log signal — v18 inventory probes confirmed only
    # 1 of 4 instances accumulated oak_log past 22 (the
    # other 3 wandered off and mined cobblestone/deepslate).
    # Lowered to 300 so the wrapper's reset_episode keeps
    # re-placing the ring frequently enough that the on-
    # policy buffer is dominated by oak_log-relevant
    # trajectories. Eval gate is unchanged (scenarios cap
    # at 1000 ticks per Scenario.max_ticks).
    max_episode_ticks: int = 300,
    seed: int = 1,
    # N13 finding (v16 thread dump): at tick rate 60 (post-N12
    # crash fix) per-env-step latency is ~5-7s (vs ~1.4s at
    # tick 300). For batch=2048/4 workers = 512 steps/worker
    # × 7s = ~60 min/iter, far exceeding sample_timeout_s.
    # Shrunk batch 2048→256 + fragment 128→32 so each iter
    # cycle is 256/(4*32)=2 fragments × ~7s = ~14s sampling,
    # batch fill ~1 min, iter ~3 min. Worse sample efficiency
    # but trains end-to-end at tick 60 stability.
    rollout_fragment_length: int | str = 32,
    sample_timeout_s: float = 1800.0,
    # P0: was 256 (sized for 4 runners at tick-60, with minibatch=32 that was a
    # NaN-KL risk). With #4 masking the dead comm Gaussian the NaN source is
    # gone, so we use a larger batch: 768 = 12 runners x 32 fragment x 2 rounds,
    # minibatch=96. Scales the now-single-attractor signal into cleaner PPO
    # updates. (At 4 runners this is 6 rounds/iter — still fine under
    # sample_timeout_s.)
    train_batch_size: int = 768,
    # M2 decision-core experiment (sim backend only): demote HARVEST to mine the
    # policy-pointed instance (target_class indexes g_nearest_resources), run the
    # 2-cluster blind-explore arena, and enable PBRS distance shaping so the
    # policy can LEARN to explore-when-blind + select. Off => the proven
    # HARVEST-spam survival-forest path.
    decision_core: bool = False,
    # Natural-world mode (real backend only): peaceful=True (no hostile mobs),
    # arena_bounds_check=False (lift truncation box for far forests),
    # tick_warp=True (keep tick-warp for speed; drowning now survivable via
    # water_breathing, so no unobservable-death blocker).
    natural_world: bool = False,
) -> PPOConfig:
    """Section 7.1 M1 single-agent gatherer PPO config (new API stack).

    ``backend`` selects the env:
      - \"real\" (default): live Minecraft via Py4J — unchanged behavior.
      - \"sim\": the headless ``AiUtopiaSimEnv`` — registers only the import-light
        sim factory and uses a minimal env_config (no py4j_ports/tick_warp/etc.).

    ``natural_world`` (real backend only): sets peaceful=True (no hostile mobs),
      arena_bounds_check=False (lift truncation), and tick_warp=True (speed).
    """
    if backend not in ("real", "sim"):
        raise ValueError(f"backend must be 'real' or 'sim', got {backend!r}")

    if backend == "sim":
        register_aiutopia_sim_env()
        env_name = SIM_ENV_NAME
        # Minimal env_config: only the keys AiUtopiaSimEnv consumes. active_roles
        # is required (its __init__ KeyErrors without it). The real-MC-only keys
        # (py4j_ports, tick_warp, seed_strategy, per_worker_seed_offset,
        # enable_memory_writes) are intentionally dropped — the sim ignores
        # absent keys cleanly.
        env_config = {
            "stage": 1,
            "active_roles": ["gatherer"],
            "max_episode_ticks": max_episode_ticks,
            # Vary the arena layout per training episode so the policy sees the
            # full layout distribution (not a single-layout overfit); eval/transfer
            # pass fixed seeds and never set this.
            "randomize_layout": True,
            **(
                {
                    # M2: the POLICY drives instance-selection + blind-explore.
                    "decision_core": True,
                    # "clusters" (RANDOMIZED per seed): every episode has a blind hop,
                    # so the explore signal stays dominant. clusters-only + the bearing
                    # cue = held-out 5/5 (v5). DO NOT use "mixed" — trees episodes dilute
                    # the explore signal (the policy learns "never explore"); mixed
                    # REGRESSED to held-out 0/5 even WITH the cue (v6). Trees is a separate
                    # easier (no-blind-hop) layout; the decision-core's value is the
                    # clusters blind-explore.
                    "arena_mode": "clusters",
                    "arena_half": 34.0,         # roam far enough to reach cluster B
                    "distance_shaping": True,   # blind-only PBRS guides the explore hop
                    # Explorer-report sim: directional cue to the nearest resource
                    # (even beyond perception) -> the policy explores the RIGHT way,
                    # closing the blind-explore-direction gap (2/5 held-out failures).
                    "resource_bearing_cue": True,
                }
                if decision_core
                else {}
            ),
            # N21 Path A: the distance_shaping / failure_penalty / completion_bonus
            # flags were removed — they targeted a "teach the policy to NAVIGATE"
            # fix that is impossible given the obs is blind beyond 16 b (both sim
            # and real). Real MC clears the field via HARVEST's internal chaining
            # (restored in skills.py), so the policy is correctly HARVEST-only and
            # needs no nav incentive. See NEXT_SESSION.md "N21 FINAL".
        }
    else:
        register_aiutopia_env()
        env_name = ENV_NAME
        # Derive one Py4J port per runner unless explicitly overridden.
        if py4j_ports is None:
            py4j_ports = tuple(25001 + i for i in range(num_env_runners))
        env_config = {
            "stage": 1,
            "active_roles": ["gatherer"],
            "seed_strategy": "fixed_easy",
            "py4j_ports": list(py4j_ports),
            "tick_warp": True,
            "max_episode_ticks": max_episode_ticks,
            "per_worker_seed_offset": True,
            "enable_memory_writes": True,
            **(
                {
                    "peaceful": True,
                    "arena_bounds_check": False,
                }
                if natural_world
                else {}
            ),
        }

    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .environment(
            env=env_name,
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            rollout_fragment_length=rollout_fragment_length,
            sample_timeout_s=sample_timeout_s,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=0.5,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=min(256, train_batch_size // 8),
            num_epochs=5,
            gamma=0.99,
            lr=3.0e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            grad_clip=1.0,
            # NO legacy model={"use_lstm": True} block — the custom RLModule
            # owns the LSTM. Ray ignores this block under new API stack
            # but its presence is misleading.
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "gatherer_policy": RLModuleSpec(
                        module_class=AiUtopiaRoleRLModule,
                        observation_space=build_role_observation_space("gatherer", stage=1),
                        action_space=build_role_action_space("gatherer"),
                        model_config={
                            "role": "gatherer",
                            "max_seq_len": 32,
                            "actor_hidden": [256],
                            # Fix #4: mask the DEAD comm heads in single-agent
                            # M1B. comm_payload (a 128-d diag Gaussian = 256 of
                            # the 344 action-dist dims) and should_broadcast are
                            # unused here — the comm OBSERVATION is all-zeros and
                            # comm has zero env effect — yet PPO maximizes their
                            # entropy (the unclamped log_std drives the learner
                            # entropy 198->202->NaN seen in progress.csv). With
                            # this ON those sub-dists are forced constant:
                            # ~0 entropy, ~0 KL, ~0 gradient. The action SPACE is
                            # unchanged (policy still emits a zeroed comm_payload
                            # + should_broadcast), so the env contract holds and
                            # M2+ MARL flips this OFF to restore live comm.
                            "mask_comm": True,
                            "core_encoder": {"core_hidden": [512, 256]},
                            "shared_backbone": {"lstm_hidden": 256},
                            "ctde_critic": {"critic_hidden": 256},
                        },
                    ),
                },
            )
        )
        .multi_agent(
            policies={"gatherer_policy"},
            policy_mapping_fn=_policy_mapping_fn,
            policies_to_train=["gatherer_policy"],
        )
        .resources(num_cpus_for_main_process=2)
        .reporting(
            # N19-followup: was 200. With max_episode_ticks=300 and
            # 64 env_steps/env/iter (4 envs × 32 fragment), we'd need
            # ~940 training iters before the 200-episode smoothing
            # window fills — far longer than M1B targets (~100-200 iters
            # to converge). Lowered to 20 so episode_return_mean
            # populates within the first few iters that produce
            # episode terminations.
            metrics_num_episodes_for_smoothing=20,
            keep_per_episode_custom_metrics=True,  # T10 reads exploit_* stats
        )
        .checkpointing(
            export_native_model_files=True,
            checkpoint_trainable_policies_only=True,
        )
        .debugging(seed=seed)
    )
    return cfg
