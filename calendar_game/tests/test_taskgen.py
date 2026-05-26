from calendar_game.taskgen import (
    build_task,
    build_tasks_from_config,
    count_solvable_assignments,
    derive_blocked_errand_task,
    select_balanced_bucket_tasks,
)


def test_count_solvable_assignments_uses_feasible_fraction():
    task = {
        "task_id": "manual_fraction",
        "params": {
            "num_slots": 3,
        },
        "calendars": [
            [None, {"errand_id": 1, "cost": 1}, {"errand_id": 2, "cost": 1, "blocked": True}],
            [None, None, None],
        ],
        "meetings": [
            {"id": 1, "participants": [0, 1], "duration": 1, "cost": 1},
        ],
    }

    solvable, total, fraction = count_solvable_assignments(task)

    assert solvable == 2
    assert total == 3
    assert fraction == 2 / 3


def test_count_solvable_assignments_uses_ordered_distinct_denominator():
    task = {
        "task_id": "manual_permutation_denominator",
        "params": {
            "num_slots": 3,
        },
        "calendars": [
            [None, None, {"errand_id": 1, "cost": 1, "blocked": True}],
            [None, None, None],
        ],
        "meetings": [
            {"id": 1, "participants": [0], "duration": 1, "cost": 1},
            {"id": 2, "participants": [1], "duration": 1, "cost": 1},
        ],
    }

    solvable, total, fraction = count_solvable_assignments(task)

    assert solvable == 4
    assert total == 6
    assert fraction == 4 / 6


def test_solvable_fraction_scorer_buckets_lower_fraction_as_harder():
    config = {
        "setting_name": "fraction_bucket_smoke",
        "seed_base": 710_000,
        "candidates_per_config": 6,
        "selected_per_bucket": 1,
        "difficulty_scorer": "cp_sat_solvable_assignment_fraction",
        "configs": [
            {
                "total_agents": 3,
                "subset_size": 2,
                "num_slots": 8,
                "num_meetings": 2,
                "density": 0.6,
                "pref_level": 1,
                "meeting_cost_level": 1,
                "errand_cost_level": 1,
            }
        ],
    }

    tasks, _summary = build_tasks_from_config(config)
    by_bucket = {task["difficulty"]: task for task in tasks}

    assert by_bucket["easy"]["difficulty_score"] >= by_bucket["medium"]["difficulty_score"]
    assert by_bucket["medium"]["difficulty_score"] >= by_bucket["hard"]["difficulty_score"]
    assert all("solvable_assignment_fraction" in task for task in tasks)
    assert all(task["params"].get("blocked_errands_per_agent") in {None, 2, 4, 6} for task in tasks)


def test_derive_blocked_errand_task_remains_feasible():
    source = build_task({
        "task_id": "source_5a3p5m",
        "seed": 323032,
        "total_agents": 5,
        "subset_size": 3,
        "density": 0.8,
        "pref_level": 1,
        "num_meetings": 5,
    })

    blocked = derive_blocked_errand_task(source, blocked_errands_per_agent=1)

    assert blocked["feasible"]
    assert blocked["optimal"]["cost"] is not None
    assert blocked["greedy"]["cost"] is not None
    assert blocked["params"]["blocked_errands_per_agent"] == 1
    assert blocked["blocked_errand_count"] == 5
    for calendar in blocked["calendars"]:
        assert sum(
            1
            for slot in calendar
            if isinstance(slot, dict) and slot.get("blocked")
        ) == 1


def test_build_task_samples_blocked_errands_before_remaining_density():
    task = build_task({
        "task_id": "initial_blocked_density",
        "seed": 323033,
        "total_agents": 5,
        "subset_size": 3,
        "density": 0.8,
        "pref_level": 1,
        "num_slots": 16,
        "num_meetings": 5,
        "blocked_errands_per_agent": 4,
    })

    assert task["feasible"]
    assert task["params"]["blocked_errands_per_agent"] == 4
    assert task["blocked_errand_count"] == 20
    for calendar in task["calendars"]:
        errands = [slot for slot in calendar if isinstance(slot, dict) and "errand_id" in slot]
        blocked = [slot for slot in errands if slot.get("blocked")]
        assert len(errands) == 11
        assert len(blocked) == 4


def test_bucket_selection_balances_blocked_counts_when_available():
    bucket_tasks = [
        {
            "task_id": f"candidate_{index}",
            "params": {
                "blocked_errands_per_agent": blocked_count,
                "density": 0.8,
                "num_slots": 16,
                "total_agents": 5,
            },
        }
        for index, blocked_count in enumerate([2, 2, 4, 4, 6, 6])
    ]

    selected, balance = select_balanced_bucket_tasks(
        bucket_tasks,
        3,
        config={
            "blocked_errands_per_agent_values": [2, 4, 6],
            "density": 0.8,
            "num_slots": 16,
            "total_agents": 5,
        },
        config_idx=0,
        bucket="easy",
    )

    assert [task["params"]["blocked_errands_per_agent"] for task in selected] == [2, 4, 6]
    assert balance["fallback_count"] == 0
    assert balance["selected_counts"] == {"2": 1, "4": 1, "6": 1}


def test_default_config_samples_blocked_counts_then_buckets_by_cp_sat_fraction():
    config = {
        "setting_name": "blocked_sampled_fraction_bucket_smoke",
        "seed_base": 720_000,
        "candidates_per_config": 9,
        "selected_per_bucket": 1,
        "blocked_errands_per_agent_values": [2, 4, 6],
        "configs": [
            {
                "total_agents": 5,
                "subset_size": 3,
                "num_slots": 16,
                "num_meetings": 5,
                "density": 0.8,
                "pref_level": 1,
                "meeting_cost_level": 1,
                "errand_cost_level": 1,
            }
        ],
    }

    tasks, summary = build_tasks_from_config(config)
    by_bucket = {task["difficulty"]: task for task in tasks}

    assert summary["difficulty_bucket_scope"] == "config_relative_tertile"
    assert summary["bucket_counts"] == {"easy": 1, "medium": 1, "hard": 1}
    assert "selected_blocked_errands_per_agent_balance" in summary["configs"][0]
    assert by_bucket["easy"]["difficulty_score"] >= by_bucket["medium"]["difficulty_score"]
    assert by_bucket["medium"]["difficulty_score"] >= by_bucket["hard"]["difficulty_score"]
    assert all(task["params"].get("blocked_errands_per_agent") in {None, 2, 4, 6} for task in tasks)
    assert all("solvable_assignment_fraction" in task for task in tasks)


def test_config_can_vary_agent_calendar_and_blocked_loads_per_task():
    config = {
        "setting_name": "varied_agent_load_smoke",
        "seed_base": 730_000,
        "candidates_per_config": 9,
        "selected_per_bucket": 1,
        "blocked_errands_per_agent_values": [2, 4, 6],
        "configs": [
            {
                "total_agents": 5,
                "subset_size": 3,
                "num_slots": 16,
                "num_meetings": 5,
                "density": 0.8,
                "vary_agent_densities": True,
                "agent_density_values": [0.6, 0.8, 1.0],
                "vary_blocked_errands_per_agent": True,
                "pref_level": 1,
                "meeting_cost_level": 1,
                "errand_cost_level": 1,
            }
        ],
    }

    tasks, summary = build_tasks_from_config(config)

    for task in tasks:
        filled_counts = [sum(slot is not None for slot in calendar) for calendar in task["calendars"]]
        blocked_counts = [
            sum(isinstance(slot, dict) and bool(slot.get("blocked")) for slot in calendar)
            for calendar in task["calendars"]
        ]
        assert len(set(filled_counts)) > 1
        assert len(set(blocked_counts)) > 1
        assert len(set(task["params"]["agent_densities"])) > 1
        assert len(set(task["params"]["blocked_errands_per_agent"])) > 1

    for config_summary in summary["configs"]:
        for bucket_summary in config_summary["selected_blocked_errands_per_agent_balance"].values():
            assert bucket_summary["selection_strategy"] == "per_agent_pair_balance"
            assert "agent_pair_counts" in bucket_summary
            assert bucket_summary["agent_pair_max_count_range"] <= 1


def test_build_tasks_from_config_generates_at_least_ten_configurations():
    config = {
        "setting_name": "engine_smoke_10_configs",
        "seed_base": 700_000,
        "candidates_per_config": 3,
        "selected_per_bucket": 1,
        "configs": [
            {
                "total_agents": [2, 3],
                "subset_size": 2,
                "num_slots": 8,
                "num_meetings": 1,
                "densities": [0.25, 0.35, 0.45, 0.55, 0.65],
                "pref_level": 1,
                "meeting_cost_level": 1,
                "errand_cost_level": 1,
            }
        ],
    }

    tasks, summary = build_tasks_from_config(config)

    assert summary["total_configs"] == 10
    assert summary["total_tasks"] == 30
    assert len(tasks) == 30
    assert summary["bucket_counts"] == {"easy": 10, "medium": 10, "hard": 10}
    assert len({
        (
            task["params"]["total_agents"],
            task["params"]["subset_size"],
            task["params"]["density"],
            task["params"]["pref_level"],
            task["params"]["num_meetings"],
            task["params"]["num_slots"],
        )
        for task in tasks
    }) == 10
    assert all(task["feasible"] for task in tasks)
    assert all(task["optimal"]["cost"] is not None for task in tasks)
    assert all(task["params"].get("blocked_errands_per_agent") in {None, 2, 4, 6} for task in tasks)
