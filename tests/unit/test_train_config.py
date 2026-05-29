import pytest

pytest.importorskip("ray")

from aiutopia.train.config import m1_gatherer_config, ENV_NAME


def test_m1_gatherer_config_builds() -> None:
    cfg = m1_gatherer_config()
    d = cfg.to_dict()
    # P0: batch raised 256->768 (12 runners x 32 x 2 rounds; minibatch 96) now
    # that #4's comm-head mask removed the NaN-KL that the old small minibatch=32
    # risked. minibatch = min(256, 768//8) = 96.
    assert d["train_batch_size"] == 768
    assert d["minibatch_size"] == 96
    assert d["num_epochs"] == 5
    assert d["gamma"] == 0.99
    assert d["lr"] == 3.0e-4
    assert d["num_env_runners"] == 4
    assert d["num_envs_per_env_runner"] == 2
    assert d["num_learners"] == 1
    assert d["num_gpus_per_learner"] == 0.5
    assert d["env_config"]["stage"] == 1
    assert d["env_config"]["active_roles"] == ["gatherer"]
    assert d["env_config"]["tick_warp"] is True
    # New API stack must be on
    assert d.get("enable_rl_module_and_learner") is True or \
           d.get("_enable_new_api_stack") is True or \
           cfg.enable_rl_module_and_learner is True


def test_m1_gatherer_config_env_registered() -> None:
    from ray.tune.registry import _global_registry, ENV_CREATOR
    m1_gatherer_config()
    assert _global_registry.contains(ENV_CREATOR, ENV_NAME)


def test_policy_mapping_fn_new_api_signature() -> None:
    from aiutopia.train.config import _policy_mapping_fn
    assert _policy_mapping_fn("gatherer_0", episode=None) == "gatherer_policy"
    assert _policy_mapping_fn("gatherer_0") == "gatherer_policy"
    # Extra kwargs ignored
    assert _policy_mapping_fn("gatherer_0", episode=None, worker=None) == "gatherer_policy"
