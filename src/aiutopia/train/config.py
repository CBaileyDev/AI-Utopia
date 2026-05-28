"""Section 7.1 PPO config builders."""
from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule


ENV_NAME = "aiutopia_minecraft"


def register_aiutopia_env() -> None:
    """Idempotent registration of the env factory with Ray Tune."""
    from aiutopia.train.env_factory import make_aiutopia_env_wrapped
    register_env(ENV_NAME, make_aiutopia_env_wrapped)


def _policy_mapping_fn(agent_id, episode=None, **kwargs):
    """New-API-stack signature: (agent_id, episode, **kwargs)."""
    return "gatherer_policy"


def m1_gatherer_config(*,
                        py4j_ports:        tuple[int, ...] = (25001, 25002, 25003, 25004),
                        num_env_runners:   int = 4,
                        num_envs_per_env_runner: int = 2,
                        # N16-followup: pre-N16, env.step took ~5-7s and 12_000
                        # ticks would have meant 16-23h per episode (no episode
                        # ever ended → episode_return_mean stayed undefined).
                        # Post-N16, env.step is ~0.15s, so 12_000 = ~30min/ep
                        # still wasteful. Eval gate measures "64 oak_log within
                        # 1000 env steps", so 2000 gives the policy 2x slack
                        # for learning without dragging the on-policy buffer.
                        max_episode_ticks: int = 2_000,
                        seed:              int = 1,
                        # N13 finding (v16 thread dump): at tick rate 60 (post-N12
                        # crash fix) per-env-step latency is ~5-7s (vs ~1.4s at
                        # tick 300). For batch=2048/4 workers = 512 steps/worker
                        # × 7s = ~60 min/iter, far exceeding sample_timeout_s.
                        # Shrunk batch 2048→256 + fragment 128→32 so each iter
                        # cycle is 256/(4*32)=2 fragments × ~7s = ~14s sampling,
                        # batch fill ~1 min, iter ~3 min. Worse sample efficiency
                        # but trains end-to-end at tick 60 stability.
                        rollout_fragment_length: int | str = 32,
                        sample_timeout_s:        float     = 1800.0,
                        train_batch_size:        int       = 256,
                        ) -> PPOConfig:
    """Section 7.1 M1 single-agent gatherer PPO config (new API stack)."""
    register_aiutopia_env()

    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .environment(
            env=ENV_NAME,
            env_config={
                "stage":                 1,
                "active_roles":          ["gatherer"],
                "seed_strategy":         "fixed_easy",
                "py4j_ports":            list(py4j_ports),
                "tick_warp":             True,
                "max_episode_ticks":     max_episode_ticks,
                "per_worker_seed_offset": True,
                "enable_memory_writes":  True,
            },
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
                            "core_encoder":    {"core_hidden": [512, 256]},
                            "shared_backbone": {"lstm_hidden": 256},
                            "ctde_critic":     {"critic_hidden": 256},
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
            metrics_num_episodes_for_smoothing=200,
            keep_per_episode_custom_metrics=True,   # T10 reads exploit_* stats
        )
        .checkpointing(
            export_native_model_files=True,
            checkpoint_trainable_policies_only=True,
        )
        .debugging(seed=seed)
    )
    return cfg
