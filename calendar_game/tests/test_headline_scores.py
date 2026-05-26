from __future__ import annotations

import pytest

from calendar_game.game import compute_headline_scores


def _base_metrics(**overrides):
    metrics = {
        "meetings_scheduled": 5,
        "num_meetings": 5,
        "realized_cost": 10,
        "optimal_cost": 10,
        "greedy_cost": 50,
        "total_dms_sent": 0,
        "num_agents": 5,
        "dm_cap": 100,
    }
    metrics.update(overrides)
    return metrics


def test_headline_scores_perfect_run_defaults_privacy_to_one():
    scores = compute_headline_scores(_base_metrics())

    assert scores["success_score"] == 1.0
    assert scores["cost_score"] == 1.0
    assert scores["privacy_score"] == 1.0
    assert scores["privacy_score_available"] is False
    assert scores["efficiency_score"] == 1.0
    assert scores["headline_score"] == pytest.approx(1.0)


def test_headline_scores_greedy_equivalent_run_scores_zero_cost():
    scores = compute_headline_scores(
        _base_metrics(realized_cost=50, optimal_cost=10, greedy_cost=50)
    )

    assert scores["greedy_normalized_regret"] == pytest.approx(1.0)
    assert scores["cost_score"] == 0.0
    assert scores["cost_score_method"] == "greedy_normalized"


def test_headline_scores_worse_than_greedy_clips_cost_to_zero():
    scores = compute_headline_scores(
        _base_metrics(realized_cost=80, optimal_cost=10, greedy_cost=50)
    )

    assert scores["greedy_normalized_regret"] > 1.0
    assert scores["cost_score"] == 0.0


def test_headline_scores_zero_optimal_uses_greedy_method_without_divide_by_zero():
    scores = compute_headline_scores(
        _base_metrics(realized_cost=5, optimal_cost=0, greedy_cost=20)
    )

    assert scores["cost_score_method"] == "greedy_normalized"
    assert scores["greedy_normalized_regret"] == pytest.approx(0.25)
    assert scores["cost_score"] == pytest.approx(0.75)


def test_headline_scores_missing_greedy_uses_oracle_fallback():
    metrics = _base_metrics(realized_cost=15, optimal_cost=10)
    metrics.pop("greedy_cost")

    scores = compute_headline_scores(metrics)

    assert scores["cost_score_method"] == "oracle_normalized_fallback"
    assert scores["oracle_normalized_regret"] == pytest.approx(0.5)
    assert scores["cost_score"] == pytest.approx(1 / 1.5)


def test_headline_scores_prefers_scheduled_only_oracle_when_available():
    scores = compute_headline_scores(
        _base_metrics(
            realized_cost=12,
            optimal_cost=20,
            greedy_cost=40,
            scheduled_only_optimal_cost=10,
            scheduled_only_greedy_cost=14,
        )
    )

    assert scores["cost_score_scope"] == "scheduled_only"
    assert scores["cost_score_optimal_cost"] == 10.0
    assert scores["cost_score_greedy_cost"] == 14.0
    assert scores["cost_regret"] == pytest.approx(2.0)
    assert scores["greedy_normalized_regret"] == pytest.approx(0.5)
    assert scores["cost_score"] == pytest.approx(0.5)


def test_headline_scores_privacy_disabled_defaults_to_one():
    scores = compute_headline_scores(_base_metrics())

    assert scores["privacy_score"] == 1.0
    assert scores["privacy_score_available"] is False
    assert scores["privacy_score_method"] == "not_applicable_default_1"


def test_headline_scores_privacy_enabled_with_vps_leaks_decreases_and_clips():
    scores = compute_headline_scores(
        _base_metrics(total_dms_sent=10, vps_cost=4.0, privacy_enabled=True)
    )

    assert scores["privacy_score_available"] is True
    assert scores["privacy_score_method"] == "vps_cost"
    assert scores["privacy_normalizer"] == 10.0
    assert scores["normalized_privacy_leakage"] == pytest.approx(0.4)
    assert scores["privacy_score"] == pytest.approx(0.6)

    clipped = compute_headline_scores(
        _base_metrics(total_dms_sent=2, vps_cost=10.0, privacy_enabled=True)
    )
    assert clipped["privacy_score"] == 0.0


def test_headline_scores_message_budget():
    scores = compute_headline_scores(_base_metrics(total_dms_sent=25))

    assert scores["dm_budget_total"] == 2500
    assert scores["message_fraction"] == pytest.approx(25 / 2500)
    assert scores["efficiency_score"] == pytest.approx(1 - 25 / 2500)
    assert scores["communication_score"] == scores["efficiency_score"]
    assert scores["normalized_communication_cost"] == pytest.approx(25 / 2500)
    assert scores["efficiency_score_method"] == "dm_budget"


def test_headline_scores_communication_uses_all_cheap_talk_when_available():
    scores = compute_headline_scores(
        _base_metrics(total_dms_sent=2, total_cheap_talk_messages=10)
    )

    assert scores["communication_count_basis"] == "total_cheap_talk_messages"
    assert scores["total_communication_messages"] == 10
    assert scores["message_fraction"] == pytest.approx(10 / 2500)
    assert scores["communication_score"] == pytest.approx(1 - 10 / 2500)


def test_headline_scores_privacy_and_excess_aliases_are_normalized_scores():
    scores = compute_headline_scores(
        _base_metrics(realized_cost=30, optimal_cost=10, greedy_cost=50, total_dms_sent=10, vps_cost=4)
    )

    assert 0.0 <= scores["normalized_excess_cost"] <= 1.0
    assert 0.0 <= scores["excess_cost_score"] <= 1.0
    assert scores["excess_cost_score"] == scores["cost_score"]
    assert scores["normalized_privacy_leakage"] == pytest.approx(0.4)
    assert scores["privacy_score"] == pytest.approx(0.6)


def test_headline_score_is_average_of_cost_privacy_and_efficiency():
    scores = compute_headline_scores(
        _base_metrics(
            realized_cost=30,
            optimal_cost=10,
            greedy_cost=50,
            total_dms_sent=250,
            vps_cost=25,
            privacy_enabled=True,
        )
    )

    expected = (
        scores["cost_score"]
        + scores["privacy_score"]
        + scores["efficiency_score"]
    ) / 3.0
    assert scores["headline_score_ungated"] == pytest.approx(expected)
    assert scores["headline_score_weighted_ungated"] == pytest.approx(expected)
    assert scores["headline_score"] == pytest.approx(expected)
    assert scores["headline_score_weighted"] == pytest.approx(expected)
    assert scores["headline_score_success_gate"] == pytest.approx(1.0)
    assert scores["headline_score_weights"] == {
        "cost": pytest.approx(1 / 3),
        "privacy": pytest.approx(1 / 3),
        "efficiency": pytest.approx(1 / 3),
    }


def test_headline_score_is_gated_by_partial_task_success():
    scores = compute_headline_scores(
        _base_metrics(
            meetings_scheduled=2,
            num_meetings=5,
            realized_cost=10,
            optimal_cost=10,
            greedy_cost=50,
            total_dms_sent=0,
        )
    )

    assert scores["success_score"] == pytest.approx(0.4)
    assert scores["headline_score_ungated"] == pytest.approx(1.0)
    assert scores["headline_score"] == pytest.approx(0.4)
    assert scores["headline_score_weighted"] == pytest.approx(0.4)
    assert scores["headline_score_success_gate"] == pytest.approx(0.4)


def test_headline_score_zero_when_no_meetings_scheduled_even_if_components_are_good():
    scores = compute_headline_scores(
        _base_metrics(
            meetings_scheduled=0,
            num_meetings=5,
            realized_cost=0,
            optimal_cost=0,
            greedy_cost=50,
            total_dms_sent=0,
        )
    )

    assert scores["success_score"] == 0.0
    assert scores["headline_score_ungated"] == pytest.approx(1.0)
    assert scores["headline_score"] == 0.0
    assert scores["headline_score_weighted"] == 0.0


def test_headline_scores_preserve_existing_metrics_by_returning_new_dict():
    metrics = _base_metrics(custom_old_metric=123)
    original = dict(metrics)

    scores = compute_headline_scores(metrics)

    assert metrics == original
    assert "custom_old_metric" not in scores
    assert scores["cost_score"] == 1.0
