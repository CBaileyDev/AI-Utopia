from aiutopia.memory.writer import importance_score, IMPORTANCE_WEIGHTS


def test_importance_weights_sum_to_one() -> None:
    assert abs(sum(IMPORTANCE_WEIGHTS.values()) - 1.0) < 1e-9


def test_importance_score_all_zero_inputs_is_zero() -> None:
    s = importance_score(
        abs_reward_norm=0.0, novel_state=0.0, comm_norm=0.0,
        player_proximity=0.0, threat_level=0.0, planner_event=0.0,
    )
    assert s == 0.0


def test_importance_score_all_one_inputs_is_one() -> None:
    s = importance_score(
        abs_reward_norm=1.0, novel_state=1.0, comm_norm=1.0,
        player_proximity=1.0, threat_level=1.0, planner_event=1.0,
    )
    assert abs(s - 1.0) < 1e-9


def test_importance_score_weighted_combination_matches_spec() -> None:
    # Only abs_reward_norm = 1, others 0 → should equal 0.30 weight
    s = importance_score(
        abs_reward_norm=1.0, novel_state=0.0, comm_norm=0.0,
        player_proximity=0.0, threat_level=0.0, planner_event=0.0,
    )
    assert abs(s - IMPORTANCE_WEIGHTS["abs_reward"]) < 1e-9
