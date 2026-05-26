from __future__ import annotations

import pytest

from calendar_game.dataset import CalendarGameDataset
from calendar_game.evaluate import summarize_game_df


def _trace(game_id: str, contributions: list[dict], *, scheduled_excess: float, full_excess: float) -> dict:
    num_agents = len(contributions)
    per_agent_scheduled = [row["scheduled_only_excess_burden"] for row in contributions]
    per_agent_fairness = [row["fairness_cost"] for row in contributions]
    per_agent_cost = [max(0.0, value) for value in per_agent_scheduled]
    return {
        "game_id": game_id,
        "config": {
            "experiment_name": "exp",
            "experiment_run_id": "run",
            "seed": 1,
            "num_agents": num_agents,
            "num_participants": 2,
            "density": 0.2,
            "task_path": "tasks/calbench_90_uniform.jsonl",
            "agents": [{"model": "model-a"} for _ in range(num_agents)],
        },
        "events": [
            {
                "type": "game_start",
                "data": {
                    "num_slots": 4,
                    "optimal_cost": 10.0,
                    "greedy_cost": 12.0,
                },
            }
        ],
        "metrics": {
            "num_agents": num_agents,
            "num_meetings": 2,
            "meetings_scheduled": 1,
            "coordination_rate": 0.5,
            "realized_cost": 10.0 + full_excess,
            "optimal_cost": 10.0,
            "scheduled_only_optimal_cost": 4.0,
            "scheduled_only_excess_cost": scheduled_excess,
            "total_scheduled_only_excess_burden": scheduled_excess,
            "legacy_total_full_task_excess_burden": full_excess,
            "per_agent_scheduled_only_excess_burden": per_agent_scheduled,
            "per_agent_fairness_cost": per_agent_fairness,
            "fairness_cost": sum(per_agent_fairness) / num_agents,
            "fairness": 0.5,
            "total_cheap_talk_messages": 12,
            "total_dms_sent": 5,
            "total_groupchat_messages_sent": 7,
            "total_participant_groupchat_messages_sent": 3,
            "total_all_groupchat_messages_sent": 4,
            "efficiency": 12,
        },
        "final_state": {
            "per_agent_cost": per_agent_cost,
            "per_agent_fallback_cost": [0.0 for _ in range(num_agents)],
            "oracle_per_agent_cost": [0.0 for _ in range(num_agents)],
            "legacy_per_agent_full_task_excess_burden": [full_excess / num_agents for _ in range(num_agents)],
            "per_agent_excess_burden": per_agent_scheduled,
            "contribution_scores": contributions,
            "round_outcomes": [],
        },
    }


def test_dataset_exports_paper_metric_fields_and_legacy_aliases():
    ds = CalendarGameDataset.from_traces(
        [
            _trace(
                "g1",
                [
                    {
                        "agent_id": 0,
                        "task_success": 0.5,
                        "coordinated_participant_rounds": 1,
                        "participant_rounds": 2,
                        "messages_sent": 4,
                        "messages_per_scheduled_meeting": 4.0,
                        "scheduled_only_excess_burden": 3.0,
                        "fairness_cost": 2.0,
                    },
                    {
                        "agent_id": 1,
                        "task_success": 1.0,
                        "coordinated_participant_rounds": 1,
                        "participant_rounds": 1,
                        "messages_sent": 0,
                        "messages_per_scheduled_meeting": 0.0,
                        "scheduled_only_excess_burden": -1.0,
                        "fairness_cost": 2.0,
                    },
                ],
                scheduled_excess=2.0,
                full_excess=20.0,
            )
        ]
    )

    game_row = ds.to_game_df().iloc[0]
    assert game_row["excess_cost"] == 2.0
    assert game_row["scheduled_only_excess_cost"] == 2.0
    assert game_row["legacy_full_task_excess_cost"] == 20.0
    assert game_row["fairness_cost"] == 2.0
    assert game_row["legacy_fairness_ratio"] == pytest.approx(0.5)

    agent_rows = ds.to_agent_df().sort_values("agent_id").to_dict("records")
    assert agent_rows[0]["coordination_cost"] == 3.0
    assert agent_rows[0]["excess_cost"] == 3.0
    assert agent_rows[0]["communication_efficiency"] == 4.0
    assert agent_rows[1]["coordination_cost"] == -1.0
    assert agent_rows[1]["communication_efficiency"] == 0.0


def test_evaluate_summary_averages_per_agent_game_values_not_pooled_ratios():
    ds = CalendarGameDataset.from_traces(
        [
            _trace(
                "g1",
                [
                    {
                        "agent_id": 0,
                        "task_success": 0.5,
                        "coordinated_participant_rounds": 1,
                        "participant_rounds": 2,
                        "messages_sent": 4,
                        "messages_per_scheduled_meeting": 4.0,
                        "scheduled_only_excess_burden": 3.0,
                        "fairness_cost": 2.0,
                    },
                    {
                        "agent_id": 1,
                        "task_success": 1.0,
                        "coordinated_participant_rounds": 1,
                        "participant_rounds": 1,
                        "messages_sent": 0,
                        "messages_per_scheduled_meeting": 0.0,
                        "scheduled_only_excess_burden": -1.0,
                        "fairness_cost": 2.0,
                    },
                ],
                scheduled_excess=2.0,
                full_excess=20.0,
            ),
            _trace(
                "g2",
                [
                    {
                        "agent_id": 0,
                        "task_success": 0.0,
                        "coordinated_participant_rounds": 0,
                        "participant_rounds": 10,
                        "messages_sent": 2,
                        "messages_per_scheduled_meeting": 2.0,
                        "scheduled_only_excess_burden": 8.0,
                        "fairness_cost": 4.0,
                    },
                    {
                        "agent_id": 1,
                        "task_success": 1.0,
                        "coordinated_participant_rounds": 1,
                        "participant_rounds": 1,
                        "messages_sent": 6,
                        "messages_per_scheduled_meeting": 6.0,
                        "scheduled_only_excess_burden": 0.0,
                        "fairness_cost": 4.0,
                    },
                ],
                scheduled_excess=8.0,
                full_excess=100.0,
            ),
        ]
    )

    summary = summarize_game_df(ds.to_game_df(), ds.to_agent_df()).iloc[0]

    assert summary["task_success_mean"] == pytest.approx((0.5 + 1.0 + 0.0 + 1.0) / 4)
    assert summary["coordination_rate_mean"] == pytest.approx((0.5 + 1.0 + 0.0 + 1.0) / 4)
    assert summary["coordination_cost_mean"] == pytest.approx((3.0 - 1.0 + 8.0 + 0.0) / 4)
    assert summary["excess_cost_mean"] == pytest.approx((3.0 - 1.0 + 8.0 + 0.0) / 4)
    assert summary["communication_efficiency_mean"] == pytest.approx((4.0 + 0.0 + 2.0 + 6.0) / 4)
    assert summary["fairness_cost_mean"] == pytest.approx((2.0 + 2.0 + 4.0 + 4.0) / 4)
    assert summary["legacy_full_task_excess_cost_mean"] == pytest.approx((20.0 + 100.0) / 2)
