from calendar_game.ratings import (
    SCORE_MARGIN_RATING_VARIANT,
    extract_calendar_rating_event,
    load_target_vps_from_game_target_csv,
    load_target_vps_from_pair_csv,
)


def test_score_margin_variant_counts_varied_errand_units_against_oracle():
    trace = {
        "game_id": "g1",
        "config": {
            "game_name": "calendar",
            "num_agents": 2,
            "task_path": "tasks/calbench_90_varied.jsonl",
            "task_id": "synthetic_varied",
            "agents": [
                {"model": "model-a"},
                {"model": "model-b"},
            ],
        },
        "metrics": {
            "per_agent_scheduled_only_excess_burden": [7, 0],
            "realized_cost": 117,
            "optimal_cost": 17,
            "scheduled_only_oracle_assignments": {"1": 0},
        },
        "final_state": {
            "contribution_scores": [
                {"agent_id": 0, "coordination_rate": 1.0},
                {"agent_id": 1, "coordination_rate": 1.0},
            ],
            "calendars": [
                [None, {"errand_id": 1, "cost": 100}, {"errand_id": 2, "cost": 10}],
                [None, None, None],
            ],
        },
        "events": [
            {
                "type": "batch_applied",
                "data": {
                    "agent_id": 0,
                    "actions": [
                        {"type": "reschedule", "item_id": 1, "from_slot": 0, "to_slot": 1},
                        {"type": "reschedule", "item_id": 2, "from_slot": 2, "to_slot": 1},
                    ],
                },
            }
        ],
    }
    task_scenario = {
        "calendars": [
            [{"errand_id": 1, "cost": 100}, None, {"errand_id": 2, "cost": 10}],
            [None, None, None],
        ],
        "meetings": [{"id": 1, "participants": [0, 1]}],
    }

    default_event = extract_calendar_rating_event(trace, task_scenario=task_scenario)
    margin_event = extract_calendar_rating_event(
        trace,
        rating_variant=SCORE_MARGIN_RATING_VARIANT,
        task_scenario=task_scenario,
    )

    assert default_event.metric_scores["excess_cost"] == {"agent_0": 7.0, "agent_1": 0.0}
    assert margin_event.metric_scores["excess_cost"] == {"agent_0": 10.0, "agent_1": 0.0}
    assert margin_event.metadata["raw_metric_scores"]["excess_cost"] == {"agent_0": 7.0, "agent_1": 0.0}
    assert margin_event.metadata["raw_game_metric_scores"]["excess_cost"] == 7.0
    assert margin_event.metadata["raw_game_metric_scores"]["legacy_full_task_excess_cost"] == 100.0
    assert margin_event.metadata["metric_score_margins"]["coordination_ratio"] == 0.1
    assert margin_event.metadata["metric_score_margins"]["excess_cost"] == 10.0


def test_score_margin_variant_uses_raw_uniform_excess_cost():
    trace = {
        "game_id": "g2",
        "config": {
            "game_name": "calendar",
            "num_agents": 2,
            "task_path": "tasks/calbench_90_uniform.jsonl",
            "task_id": "synthetic_uniform",
            "agents": [
                {"model": "model-a"},
                {"model": "model-b"},
            ],
        },
        "metrics": {
            "per_agent_scheduled_only_excess_burden": [9, 0],
            "scheduled_only_oracle_assignments": {},
        },
        "final_state": {
            "contribution_scores": [
                {"agent_id": 0, "coordination_rate": 1.0},
                {"agent_id": 1, "coordination_rate": 1.0},
            ],
            "calendars": [[None], [None]],
        },
        "events": [
            {
                "type": "batch_applied",
                "data": {
                    "agent_id": 0,
                    "actions": [
                        {"type": "reschedule", "item_id": item_id, "from_slot": 0, "to_slot": 1}
                        for item_id in range(1, 8)
                    ],
                },
            }
        ],
    }
    task_scenario = {
        "calendars": [
            [{"errand_id": item_id, "cost": 1} for item_id in range(1, 8)],
            [None for _ in range(7)],
        ],
        "meetings": [],
    }

    event = extract_calendar_rating_event(
        trace,
        rating_variant=SCORE_MARGIN_RATING_VARIANT,
        task_scenario=task_scenario,
    )

    assert event.metric_scores["excess_cost"]["agent_0"] == 7.0
    assert event.metadata["metric_score_margins"]["excess_cost"] == 10.0


def test_rating_event_uses_explicit_rating_player_id():
    trace = {
        "game_id": "game-1",
        "config": {
            "game_name": "calendar",
            "num_agents": 2,
            "agents": [
                {
                    "type": "dspy",
                    "model": "publishers/google/models/gemini-3-flash-preview",
                    "rating_player_id": "gemini-3-flash-preview-redteam-c006",
                },
                {"type": "llm", "model": "publishers/google/models/gemini-3-flash-preview"},
            ],
        },
        "metrics": {
            "coordination_rate": 1.0,
            "per_agent_scheduled_only_excess_burden": [0.0, 1.0],
        },
        "final_state": {
            "contribution_scores": [
                {"agent_id": 0, "coordination_rate": 1.0},
                {"agent_id": 1, "coordination_rate": 0.5},
            ],
        },
    }

    event = extract_calendar_rating_event(trace)

    assert event is not None
    assert event.participants[0].player_id == "gemini-3-flash-preview-redteam-c006"
    assert event.participants[1].player_id == "publishers/google/models/gemini-3-flash-preview"


def test_rating_event_uses_task_success_and_scheduled_only_excess_without_clamp():
    trace = {
        "game_id": "g-paper",
        "config": {
            "game_name": "calendar",
            "num_agents": 2,
            "agents": [{"model": "model-a"}, {"model": "model-b"}],
        },
        "metrics": {
            "coordination_rate": 1.0,
            "realized_cost": 10.0,
            "optimal_cost": 1.0,
            "total_scheduled_only_excess_burden": 1.0,
            "per_agent_scheduled_only_excess_burden": [2.0, -1.0],
            "legacy_per_agent_full_task_excess_burden": [20.0, 10.0],
        },
        "final_state": {
            "contribution_scores": [
                {"agent_id": 0, "task_success": 0.5, "coordination_rate": 1.0},
                {"agent_id": 1, "task_success": 1.0, "coordination_rate": 1.0},
            ],
        },
    }

    event = extract_calendar_rating_event(trace)

    assert event is not None
    assert event.metric_scores["coordination_ratio"] == {"agent_0": 0.5, "agent_1": 1.0}
    assert event.metric_scores["excess_cost"] == {"agent_0": 2.0, "agent_1": -1.0}
    assert event.metadata["raw_metric_scores"]["excess_cost"] == {"agent_0": 2.0, "agent_1": -1.0}
    assert event.metadata["raw_game_metric_scores"]["excess_cost"] == 1.0
    assert event.metadata["raw_game_metric_scores"]["legacy_full_task_excess_cost"] == 9.0


def test_rating_vps_loaders_default_to_uniform_excess_vps(tmp_path):
    pair_csv = tmp_path / "pair_round_vps.csv"
    pair_csv.write_text(
        "\n".join(
            [
                "game_id,target_agent,target_is_participant,observer_is_participant,weight_mode,vps_loss",
                "g1,0,true,true,uniform,2.5",
                "g1,0,true,true,cost,99.0",
                "g1,1,true,true,uniform,1.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target_csv = tmp_path / "game_target_summary.csv"
    target_csv.write_text(
        "\n".join(
            [
                "game_id,target_agent,weight_mode,excess_calibrated_vps_loss_total",
                "g1,0,uniform,3.0",
                "g1,0,cost,88.0",
                "g1,1,uniform,4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_target_vps_from_pair_csv(pair_csv) == {"g1": {0: 2.5, 1: 1.5}}
    assert load_target_vps_from_game_target_csv(target_csv) == {"g1": {0: 3.0, 1: 4.0}}
