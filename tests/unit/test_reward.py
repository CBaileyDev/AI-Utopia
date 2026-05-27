from aiutopia.env.reward import (
    DEATH_PENALTY,
    GAMMA,
    GAMMA_CLIP,
    TIME_PENALTY,
    compute_reward_stage_1,
)


def _obs(inv: dict[str, int]) -> dict:
    return {
        "inv_slot_item_ids": list(inv.keys()),
        "inv_slot_counts":   list(inv.values()),
    }


def _env_meta(died: bool = False, n_clipped: int = 0,
              exploit_penalties: list = None) -> dict:
    return {
        "died_this_tick": died,
        "n_clipped_param_axes": n_clipped,
        "exploit_penalties": exploit_penalties or [],
    }


def test_zero_reward_on_no_change() -> None:
    obs = _obs({"oak_log": 5})
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=obs,
                                obs_curr=obs,
                                action={"skill_type": 4},
                                env_meta=_env_meta())
    # Expected: r_pbrs = 0.99 * phi(5) - phi(5) = -0.01 * phi(5)
    # Time penalty = -0.001
    # No primary signal change
    assert r < 0   # negative time penalty + small negative PBRS shift
    assert r > -0.1  # but tiny


def test_positive_reward_on_oak_log_gain() -> None:
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=_obs({"oak_log": 0}),
                                obs_curr=_obs({"oak_log": 1}),
                                action={"skill_type": 1},
                                env_meta=_env_meta())
    # delta-inventory: +1 oak_log x LOG_VALUE[oak_log]=1.0 = +1.0 primary
    # PBRS: 0.99 * 1.0 - 0.0 = +0.99
    # Total ~ +2 - 0.001 (time)
    assert r > 1.5


def test_death_penalty() -> None:
    obs = _obs({})
    r = compute_reward_stage_1(role="gatherer",
                                obs_prev=obs,
                                obs_curr=obs,
                                action={"skill_type": 4},
                                env_meta=_env_meta(died=True))
    assert r < -DEATH_PENALTY + 1.0  # roughly -10


def test_clip_penalty() -> None:
    obs = _obs({})
    r1 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(n_clipped=0))
    r2 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(n_clipped=3))
    assert r2 == r1 - 3 * GAMMA_CLIP


def test_exploit_penalty_subtracted() -> None:
    obs = _obs({})
    r1 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta())
    r2 = compute_reward_stage_1(role="gatherer",
                                 obs_prev=obs, obs_curr=obs,
                                 action={"skill_type": 4},
                                 env_meta=_env_meta(exploit_penalties=[
                                     ("drop_spam", 0.5), ("oscillation", 0.3)
                                 ]))
    assert r2 == r1 - (0.5 + 0.3)


def test_gamma_is_0_99() -> None:
    assert GAMMA == 0.99


def test_time_penalty_is_0_001() -> None:
    assert TIME_PENALTY == 0.001
