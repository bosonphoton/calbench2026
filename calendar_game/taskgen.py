"""Generate durable calendar task fixtures."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import combinations, product
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import yaml

from calendar_game.scenario import generate_scenario
from calendar_game.solver import (
    apply_schedule,
    count_feasible_assignments_cp_sat,
    solve_greedy,
    solve_optimal,
)

TASK_SCHEMA_VERSION = 1
NUM_SLOTS = 16
NUM_MEETINGS = 1
DURATION = 1

TOTAL_AGENT_VALUES = [2, 3, 4, 5]
DENSITY_VALUES = [0.6, 0.8]
PREF_LEVEL_VALUES = [1, 2, 3]
BALANCED_PREF_LEVEL_VALUES = [1, 3]
BALANCED_NUM_MEETINGS = 3
BALANCED_CANDIDATES_PER_CONFIG = 30
BALANCED_SELECTED_PER_BUCKET = 1
FIVE_MEETING_NUM_MEETINGS = 5
FIVE_MEETING_TOTAL_AGENT_VALUES = [2, 5]
FIVE_MEETING_DENSITY_VALUES = [0.6, 0.8, 1.0]
FIVE_MEETING_SUBSET_VALUES_BY_AGENT_COUNT = {
    2: [2],
    5: [3, 5],
}
PREFERENCE_ERRAND_COST_VALUES = [1, 10, 100]
FULL_DATASET_CANDIDATES_PER_CONFIG = 100
FULL_DATASET_SELECTED_PER_BUCKET = 4
DEFAULT_BLOCKED_ERRANDS_PER_AGENT = 6
MINIMAL_RATIO_NUM_MEETINGS = 3
MINIMAL_RATIO_DENSITY = 0.8
MINIMAL_RATIO_COST_MULTIPLIERS = [2, 3, 4]
COST_RATIO_MULTIPLIERS = [2, 3, 4]
INITIAL_PRIOR_MEETING_MULTIPLIERS = [5, 10, 20, 100]
BALANCED_UNIFORM_SOURCE = "tasks/balanced_uniform_cost_v1.jsonl"
DIFFICULTY_BUCKETS = ["easy", "medium", "hard"]
DEFAULT_DIFFICULTY_SCORER = "cp_sat_solvable_assignment_fraction"
DEFAULT_DIFFICULTY_BUCKET_MODE = "cp_sat_tertile"
DEFAULT_BLOCKED_ERRANDS_PER_AGENT_VALUES = [2, 4, 6]
DIFFICULTY_SCORER_DESCRIPTIONS = {
    "optimal_cost_per_participant_slot": (
        "optimal total move cost divided by total participant-slots "
        "across all meetings"
    ),
    "optimal_evictions_per_participant_slot": (
        "optimal number of displaced errands divided by total participant-slots "
        "across all meetings"
    ),
    "optimal_cost_per_total_agent": "optimal total move cost divided by total agents",
    "cp_sat_solvable_assignment_fraction": (
        "fraction of ordered distinct meeting-slot assignments that satisfy "
        "the CP-SAT feasibility constraints; lower fractions are harder"
    ),
}

DifficultyScorer = Callable[[dict[str, Any]], float]
DIFFICULTY_SCORER_LOWER_IS_HARDER = {"cp_sat_solvable_assignment_fraction"}

INITIAL_SMALL_TASKS: list[dict[str, Any]] = [
    {"task_id": "t001_easy_2a_2p_sparse_flat", "seed": 1001, "total_agents": 2, "subset_size": 2, "density": 0.3, "pref_level": 1},
    {"task_id": "t002_2a_2p_medium_flat", "seed": 1002, "total_agents": 2, "subset_size": 2, "density": 0.5, "pref_level": 1},
    {"task_id": "t003_2a_2p_dense_flat", "seed": 1003, "total_agents": 2, "subset_size": 2, "density": 0.8, "pref_level": 1},
    {"task_id": "t004_2a_2p_dense_cost3", "seed": 1004, "total_agents": 2, "subset_size": 2, "density": 0.8, "pref_level": 3},
    {"task_id": "t005_3a_2p_medium_flat", "seed": 1005, "total_agents": 3, "subset_size": 2, "density": 0.5, "pref_level": 1},
    {"task_id": "t006_3a_3p_medium_flat", "seed": 1006, "total_agents": 3, "subset_size": 3, "density": 0.5, "pref_level": 1},
    {"task_id": "t007_3a_2p_dense_cost2", "seed": 1007, "total_agents": 3, "subset_size": 2, "density": 0.8, "pref_level": 2},
    {"task_id": "t008_4a_2p_medium_cost2", "seed": 1008, "total_agents": 4, "subset_size": 2, "density": 0.5, "pref_level": 2},
    {"task_id": "t009_4a_4p_dense_cost2", "seed": 1009, "total_agents": 4, "subset_size": 4, "density": 0.8, "pref_level": 2},
    {"task_id": "t010_5a_2p_medium_cost3", "seed": 1010, "total_agents": 5, "subset_size": 2, "density": 0.5, "pref_level": 3},
    {"task_id": "t011_5a_3p_dense_cost3", "seed": 1011, "total_agents": 5, "subset_size": 3, "density": 0.8, "pref_level": 3},
    {"task_id": "t012_hard_5a_5p_dense_cost3", "seed": 1012, "total_agents": 5, "subset_size": 5, "density": 0.8, "pref_level": 3},
]


def select_participants(seed: int, total_agents: int, subset_size: int) -> list[int]:
    if subset_size > total_agents:
        raise ValueError("subset_size cannot exceed total_agents")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_agents), subset_size))


def select_participant_lists(seed: int, total_agents: int, subset_size: int, num_meetings: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        sorted(rng.sample(range(total_agents), subset_size))
        for _ in range(num_meetings)
    ]


def select_balanced_participant_lists(
    seed: int,
    total_agents: int,
    subset_size: int,
    num_meetings: int,
) -> list[list[int]]:
    """Select participant subsets while balancing appearances when possible."""
    if subset_size == total_agents:
        return [list(range(total_agents)) for _ in range(num_meetings)]

    total_appearances = subset_size * num_meetings
    if total_appearances % total_agents != 0:
        return select_participant_lists(seed, total_agents, subset_size, num_meetings)

    target = total_appearances // total_agents
    rng = random.Random(seed)
    combos: list[tuple[int, ...]] = []

    def build(start: int, current: list[int]) -> None:
        if len(current) == subset_size:
            combos.append(tuple(current))
            return
        for agent_id in range(start, total_agents):
            current.append(agent_id)
            build(agent_id + 1, current)
            current.pop()

    build(0, [])
    rng.shuffle(combos)
    counts = {agent_id: 0 for agent_id in range(total_agents)}
    chosen: list[tuple[int, ...]] = []

    def search() -> bool:
        if len(chosen) == num_meetings:
            return all(count == target for count in counts.values())
        viable = [
            combo for combo in combos
            if all(counts[agent_id] < target for agent_id in combo)
        ]
        viable.sort(key=lambda combo: (sum(counts[agent_id] for agent_id in combo), rng.random()))
        for combo in viable:
            chosen.append(combo)
            for agent_id in combo:
                counts[agent_id] += 1
            if search():
                return True
            for agent_id in combo:
                counts[agent_id] -= 1
            chosen.pop()
        return False

    if not search():
        return select_participant_lists(seed, total_agents, subset_size, num_meetings)
    return [list(combo) for combo in chosen]


def assign_balanced_speaker_orders(
    seed: int,
    participant_lists: list[list[int]],
    total_agents: int,
) -> list[list[int]]:
    """Order each participant list while balancing first-speaker exposure.

    The first speaker for each meeting is chosen from that meeting's participants.
    When the meeting/agent counts allow exact balance, this gives each agent the
    same number of first-speaker turns. Otherwise it keeps the spread as small
    as the participant lists permit.
    """
    rng = random.Random(seed)
    first_counts = {agent_id: 0 for agent_id in range(total_agents)}
    normalized_lists = [sorted(set(participants)) for participants in participant_lists]

    for normalized in normalized_lists:
        if not normalized:
            raise ValueError("meeting participants cannot be empty")

    target = len(participant_lists) // total_agents
    if len(participant_lists) % total_agents == 0 and target > 0:
        appearance_counts = {agent_id: 0 for agent_id in range(total_agents)}
        for participants in normalized_lists:
            for agent_id in participants:
                appearance_counts[agent_id] += 1
        if all(count >= target for count in appearance_counts.values()):
            choices = deepcopy(normalized_lists)
            for choice in choices:
                rng.shuffle(choice)
            exact_firsts: list[int | None] = [None] * len(choices)

            def search(index: int) -> bool:
                if index == len(choices):
                    return all(count == target for count in first_counts.values())
                candidates = sorted(
                    choices[index],
                    key=lambda agent_id: (first_counts[agent_id], rng.random()),
                )
                for agent_id in candidates:
                    if first_counts[agent_id] >= target:
                        continue
                    first_counts[agent_id] += 1
                    exact_firsts[index] = agent_id
                    if search(index + 1):
                        return True
                    exact_firsts[index] = None
                    first_counts[agent_id] -= 1
                return False

            if search(0):
                return [
                    [
                        int(exact_firsts[index]),
                        *[
                            agent_id for agent_id in normalized_lists[index]
                            if agent_id != exact_firsts[index]
                        ],
                    ]
                    for index in range(len(participant_lists))
                ]

    ordered: list[list[int]] = []
    for normalized in normalized_lists:
        shuffled = list(normalized)
        rng.shuffle(shuffled)
        first = min(shuffled, key=lambda agent_id: (first_counts[agent_id], rng.random()))
        first_counts[first] += 1
        rest = [agent_id for agent_id in normalized if agent_id != first]
        ordered.append([first, *rest])

    return ordered


def ensure_task_speaker_orders(task: dict[str, Any]) -> None:
    if all("speaker_order" in meeting for meeting in task.get("meetings", [])):
        return
    params = task["params"]
    participant_lists = [list(meeting["participants"]) for meeting in task["meetings"]]
    speaker_orders = assign_balanced_speaker_orders(
        int(task["seed"]),
        participant_lists,
        int(params["total_agents"]),
    )
    for meeting, speaker_order in zip(task["meetings"], speaker_orders, strict=True):
        meeting["speaker_order"] = speaker_order


def participant_slots(task: dict[str, Any]) -> int:
    return sum(
        int(meeting["duration"]) * len(meeting["participants"])
        for meeting in task["meetings"]
    )


def optimal_cost_per_participant_slot(task: dict[str, Any]) -> float:
    slots = participant_slots(task)
    if slots <= 0:
        raise ValueError(f"{task['task_id']}: cannot score task with no participant slots")
    return float(task["optimal"]["cost"]) / slots


def optimal_evictions_per_participant_slot(task: dict[str, Any]) -> float:
    slots = participant_slots(task)
    if slots <= 0:
        raise ValueError(f"{task['task_id']}: cannot score task with no participant slots")
    return float(task["optimal_evictions"]) / slots


def optimal_cost_per_total_agent(task: dict[str, Any]) -> float:
    return float(task["optimal"]["cost"]) / int(task["params"]["total_agents"])


def count_solvable_assignments(task: dict[str, Any]) -> tuple[int, int, float]:
    """Count CP-SAT-feasible ordered distinct meeting-slot assignments.

    The denominator is ``num_slots! / (num_slots - num_meetings)!``. The
    numerator counts assignments in that same no-slot-reuse space by asking
    the CP-SAT feasibility model to enumerate all valid solutions.
    """
    calendars = task["calendars"]
    meetings = task["meetings"]
    num_slots = int(task["params"]["num_slots"])
    num_meetings = len(meetings)
    if num_meetings > num_slots:
        return 0, 0, 0.0
    total_assignments = math.factorial(num_slots) // math.factorial(num_slots - num_meetings)
    if total_assignments <= 0:
        return 0, 0, 0.0

    solvable_assignments = count_feasible_assignments_cp_sat(
        calendars,
        meetings,
        num_slots,
        require_distinct_slots=True,
    )
    return solvable_assignments, total_assignments, solvable_assignments / total_assignments


def cp_sat_solvable_assignment_fraction(task: dict[str, Any]) -> float:
    solvable, total, fraction = count_solvable_assignments(task)
    task["solvable_assignment_count"] = solvable
    task["total_assignment_count"] = total
    task["solvable_assignment_fraction"] = fraction
    return fraction


DIFFICULTY_SCORERS: dict[str, DifficultyScorer] = {
    "cp_sat_solvable_assignment_fraction": cp_sat_solvable_assignment_fraction,
    "optimal_cost_per_total_agent": optimal_cost_per_total_agent,
    "optimal_cost_per_participant_slot": optimal_cost_per_participant_slot,
    "optimal_evictions_per_participant_slot": optimal_evictions_per_participant_slot,
}


def difficulty_score_sort_value(task: dict[str, Any], scorer_name: str | None = None) -> float:
    scorer = scorer_name or str(task.get("difficulty_scorer", DEFAULT_DIFFICULTY_SCORER))
    score = float(task["difficulty_score"])
    return -score if scorer in DIFFICULTY_SCORER_LOWER_IS_HARDER else score


def normalized_difficulty_value(score: float, min_score: float, max_score: float, scorer_name: str) -> float:
    if max_score == min_score:
        return 0.0
    if scorer_name in DIFFICULTY_SCORER_LOWER_IS_HARDER:
        return (max_score - score) / (max_score - min_score)
    return (score - min_score) / (max_score - min_score)


def score_task_difficulty(task: dict[str, Any], scorer_name: str = DEFAULT_DIFFICULTY_SCORER) -> dict[str, Any]:
    try:
        scorer = DIFFICULTY_SCORERS[scorer_name]
    except KeyError as exc:
        choices = ", ".join(sorted(DIFFICULTY_SCORERS))
        raise ValueError(f"unknown difficulty scorer {scorer_name!r}; choose one of: {choices}") from exc

    slots = participant_slots(task)
    total_agents = int(task["params"]["total_agents"])
    task["participant_slots"] = slots
    task["difficulty_scorer"] = scorer_name
    task["difficulty_score"] = scorer(task)
    task["cost_per_total_agent"] = float(task["optimal"]["cost"]) / total_agents
    return task


def build_task(spec: dict[str, Any], *, difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER, skip_optimal: bool = False) -> dict[str, Any]:
    num_meetings = spec.get("num_meetings", NUM_MEETINGS)
    num_slots = spec.get("num_slots", NUM_SLOTS)
    errand_cost_multiplier = spec.get("errand_cost_multiplier", 1)
    errand_cost_values = spec.get("errand_cost_values")
    errand_cost_level = spec.get("errand_cost_level")
    meeting_cost_level = spec.get("meeting_cost_level", 1)
    agent_densities = spec.get("agent_densities")
    blocked_errands_per_agent = spec.get("blocked_errands_per_agent")
    participant_lists = spec.get("participant_lists")
    if participant_lists is None:
        if num_meetings == 1:
            participant_lists = [select_participants(spec["seed"], spec["total_agents"], spec["subset_size"])]
        else:
            participant_lists = select_balanced_participant_lists(
                spec["seed"],
                spec["total_agents"],
                spec["subset_size"],
                num_meetings,
            )
    speaker_orders = spec.get("speaker_orders")
    if speaker_orders is None:
        speaker_orders = assign_balanced_speaker_orders(
            spec["seed"],
            participant_lists,
            spec["total_agents"],
        )
    scenario_skip_optimal = skip_optimal or bool(errand_cost_values)
    scenario = generate_scenario(
        spec["seed"],
        spec["total_agents"],
        num_slots,
        spec["density"],
        spec["pref_level"],
        num_meetings,
        participant_lists=participant_lists,
        speaker_orders=speaker_orders,
        force_witness_errand=True,
        skip_optimal=scenario_skip_optimal,
        errand_cost_multiplier=errand_cost_multiplier,
        errand_cost_level=errand_cost_level,
        meeting_cost_level=meeting_cost_level,
        agent_densities=agent_densities,
        blocked_errands_per_agent=blocked_errands_per_agent,
    )
    if errand_cost_values:
        assign_balanced_errand_costs(
            scenario["calendars"],
            cost_values=list(errand_cost_values),
            seed=spec["seed"],
        )
        recompute_scenario_solutions(scenario, skip_optimal=skip_optimal)
    optimal_evictions = (
        None
        if skip_optimal
        else count_evictions_for_assignments(
            scenario["calendars"],
            scenario["meetings"],
            scenario["optimal"]["assignments"],
        )
    )

    task = {
        "task_id": spec["task_id"],
        "seed": spec["seed"],
        "version": TASK_SCHEMA_VERSION,
        "params": {
            "total_agents": spec["total_agents"],
            "subset_size": spec["subset_size"],
            "num_slots": num_slots,
            "num_meetings": num_meetings,
            "duration": DURATION,
            "density": spec["density"],
            "agent_densities": scenario.get("agent_densities"),
            "pref_level": spec["pref_level"],
            "errand_cost_multiplier": errand_cost_multiplier,
            "meeting_cost_range": [1, meeting_cost_level],
            "errand_cost_range": (
                [min(errand_cost_values), max(errand_cost_values)]
                if errand_cost_values
                else [
                    errand_cost_multiplier,
                    (errand_cost_level if errand_cost_level is not None else spec["pref_level"]) * errand_cost_multiplier,
                ]
            ),
        },
        "calendars": scenario["calendars"],
        "meetings": scenario["meetings"],
        "witness_solution": scenario["witness_solution"],
        "optimal": scenario["optimal"],
        "greedy": scenario["greedy"],
        "feasible": scenario["feasible"],
        "difficulty": spec.get("difficulty"),
        "config_normalized_difficulty": spec.get("config_normalized_difficulty"),
        "optimal_evictions": optimal_evictions,
    }
    if task["difficulty"] is None:
        task.pop("difficulty")
    if task["config_normalized_difficulty"] is None:
        task.pop("config_normalized_difficulty")
    if task["optimal_evictions"] is None:
        task.pop("optimal_evictions")
    if errand_cost_values:
        task["params"]["errand_cost_values"] = list(errand_cost_values)
    if blocked_errands_per_agent is not None:
        task["params"]["blocked_errands_per_agent"] = scenario.get("blocked_errands_per_agent")
        task["blocked_slots_by_agent"] = {
            str(agent_id): [
                slot_idx
                for slot_idx, slot in enumerate(calendar)
                if isinstance(slot, dict) and slot.get("blocked")
            ]
            for agent_id, calendar in enumerate(task["calendars"])
        }
        task["blocked_errand_count"] = sum(
            len(slots)
            for slots in task["blocked_slots_by_agent"].values()
        )
    if not skip_optimal:
        score_task_difficulty(task, difficulty_scorer)
    validate_task(task, skip_optimal=skip_optimal)
    return task


def build_initial_small_tasks() -> list[dict[str, Any]]:
    return [build_task(spec) for spec in INITIAL_SMALL_TASKS]


def build_minimal_cost_ratio_tasks(
    *,
    seed_base: int = 80_000,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = [
        {
            "task_id": f"ratio_m{multiplier}_3meetings_5a_3p_dense",
            "seed": seed_base + multiplier,
            "total_agents": 5,
            "subset_size": 3,
            "density": MINIMAL_RATIO_DENSITY,
            "pref_level": 1,
            "num_meetings": MINIMAL_RATIO_NUM_MEETINGS,
            "errand_cost_multiplier": multiplier,
            "participant_lists": [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
        }
        for multiplier in MINIMAL_RATIO_COST_MULTIPLIERS
    ]
    tasks = [build_task(spec, difficulty_scorer=difficulty_scorer) for spec in specs]
    assign_setting_normalized_difficulty(tasks)
    summary = {
        "setting": "minimal_cost_ratio_v1",
        "purpose": (
            "Probe whether higher errand move costs make agents prefer moving "
            "previously scheduled meetings, involving external agents."
        ),
        "num_tasks": len(tasks),
        "num_meetings": MINIMAL_RATIO_NUM_MEETINGS,
        "density": MINIMAL_RATIO_DENSITY,
        "pref_level": 1,
        "meeting_cost_range": [1, 1],
        "errand_cost_multipliers": MINIMAL_RATIO_COST_MULTIPLIERS,
        "difficulty_scorer": difficulty_scorer,
        "difficulty_score_definition": DIFFICULTY_SCORER_DESCRIPTIONS[difficulty_scorer],
        "tasks": [
            {
                "task_id": task["task_id"],
                "seed": task["seed"],
                "errand_cost_multiplier": task["params"]["errand_cost_multiplier"],
                "optimal_cost": task["optimal"]["cost"],
                "optimal_evictions": task["optimal_evictions"],
                "difficulty_score": task["difficulty_score"],
            }
            for task in tasks
        ],
    }
    return tasks, summary


def read_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_balanced_cost_ratio_tasks(
    source_path: str | Path = BALANCED_UNIFORM_SOURCE,
    *,
    multiplier: int,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_tasks = read_jsonl(source_path)
    tasks = [
        derive_cost_ratio_task(task, multiplier, difficulty_scorer=difficulty_scorer)
        for task in source_tasks
    ]
    assign_setting_normalized_difficulty(tasks)
    summary = {
        "setting": f"balanced_uniform_errand_x{multiplier}_v1",
        "source": str(source_path),
        "purpose": (
            "Derived from balanced_uniform_cost_v1 with identical calendars, "
            "participants, meetings, and difficulty labels, but with errands "
            f"costing {multiplier} times as much as scheduled meetings."
        ),
        "num_tasks": len(tasks),
        "meeting_cost_range": [1, 1],
        "errand_cost_multiplier": multiplier,
        "errand_cost_range": [multiplier, multiplier],
        "difficulty_scorer": difficulty_scorer,
        "difficulty_score_definition": DIFFICULTY_SCORER_DESCRIPTIONS[difficulty_scorer],
        "bucket_counts": {
            bucket: sum(1 for task in tasks if task.get("difficulty") == bucket)
            for bucket in DIFFICULTY_BUCKETS
        },
        "difficulty_score_min": min(float(task["difficulty_score"]) for task in tasks) if tasks else None,
        "difficulty_score_max": max(float(task["difficulty_score"]) for task in tasks) if tasks else None,
    }
    return tasks, summary


def build_initial_prior_meeting_move_tasks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = [
        build_initial_prior_meeting_move_task(multiplier)
        for multiplier in INITIAL_PRIOR_MEETING_MULTIPLIERS
    ]
    summary = {
        "setting": "initial_prior_meeting_move_v1",
        "purpose": (
            "Four tasks with an already scheduled prior meeting and two valid "
            "repairs: move local errands at higher cost, or coordinate with an "
            "external prior-meeting participant to reschedule the prior meeting "
            "at lower cost."
        ),
        "num_tasks": len(tasks),
        "meeting_move_cost": 1,
        "errand_cost_multipliers": INITIAL_PRIOR_MEETING_MULTIPLIERS,
        "tasks": [
            {
                "task_id": task["task_id"],
                "errand_cost_multiplier": task["params"]["errand_cost_multiplier"],
                "optimal_cost": task["optimal"]["cost"],
                "local_errand_repair_cost": task["local_errand_repair"]["cost"],
                "prior_meeting_moves": task["online_optimal_repair"]["prior_meeting_moves"],
                "external_agents_involved": task["online_optimal_repair"]["external_agents_involved"],
            }
            for task in tasks
        ],
    }
    return tasks, summary


def build_initial_prior_meeting_move_task(multiplier: int) -> dict[str, Any]:
    num_agents = 4
    prior_meeting = {"id": 100, "participants": [0, 2], "slot": 0, "cost": 1}
    new_meeting = {"id": 1, "participants": [0, 1, 3], "duration": 1, "cost": 1}
    prior_target_slot = 15
    local_slot = 1
    local_target_slots = {0: 14, 1: 13}
    free_slots_by_agent = {
        0: {local_target_slots[0], prior_target_slot},
        1: {0, local_target_slots[1]},
        2: {prior_target_slot},
        3: {0, local_slot},
    }
    calendars: list[list[Any]] = []
    for agent_id in range(num_agents):
        calendar: list[Any] = []
        errand_idx = 1
        for slot in range(NUM_SLOTS):
            if agent_id in prior_meeting["participants"] and slot == prior_meeting["slot"]:
                calendar.append({"meeting_id": prior_meeting["id"], "cost": prior_meeting["cost"]})
            elif slot in free_slots_by_agent[agent_id]:
                calendar.append(None)
            else:
                calendar.append({
                    "errand_id": (agent_id * NUM_SLOTS) + errand_idx,
                    "cost": multiplier,
                })
                errand_idx += 1
        calendars.append(calendar)

    repair = replay_initial_prior_meeting_repair(
        calendars,
        prior_meeting=prior_meeting,
        new_meeting=new_meeting,
        prior_target_slot=prior_target_slot,
    )
    local_repair = replay_local_errand_repair(
        calendars,
        new_meeting=new_meeting,
        local_slot=local_slot,
        local_target_slots=local_target_slots,
    )
    task = {
        "task_id": f"initial_prior_meeting_move_errandx{multiplier}",
        "seed": 91_000 + multiplier,
        "version": TASK_SCHEMA_VERSION,
        "params": {
            "total_agents": num_agents,
            "subset_size": len(new_meeting["participants"]),
            "num_slots": NUM_SLOTS,
            "num_meetings": 1,
            "duration": DURATION,
            "density": 0.9375,
            "pref_level": 1,
            "errand_cost_multiplier": multiplier,
            "meeting_cost_range": [1, 1],
            "errand_cost_range": [multiplier, multiplier],
        },
        "prior_meetings": [prior_meeting],
        "calendars": calendars,
        "meetings": [new_meeting],
        "witness_solution": {"assignments": {"1": local_slot}, "cost": local_repair["cost"]},
        "optimal": {
            "cost": repair["cost"],
            "assignments": {"1": 0, "100": prior_target_slot},
            "interpretation": "prior_meeting_consistent_repair",
        },
        "greedy": {
            "cost": local_repair["cost"],
            "assignments": {"1": local_slot},
            "interpretation": "local_errand_repair_counterfactual",
        },
        "feasible": True,
        "optimal_evictions": repair["prior_meeting_moves"],
        "participant_slots": len(new_meeting["participants"]),
        "difficulty_scorer": DEFAULT_DIFFICULTY_SCORER,
        "difficulty_score": repair["cost"] / len(new_meeting["participants"]),
        "cost_per_total_agent": repair["cost"] / num_agents,
        "online_optimal_repair": repair,
        "local_errand_repair": local_repair,
        "notes": (
            "Both repairs are valid. The local_errand_repair schedules the new meeting "
            "at slot 1 by moving errands for agents 0 and 1. The online_optimal_repair "
            "reschedules prior meeting 100 from slot 0 to slot 15 for agent 0 "
            "and external agent 2, then schedules the new meeting at slot 0."
        ),
    }
    validate_initial_prior_meeting_move_task(task)
    return task


def replay_local_errand_repair(
    calendars: list[list[Any]],
    *,
    new_meeting: dict[str, Any],
    local_slot: int,
    local_target_slots: dict[int, int],
) -> dict[str, Any]:
    working = deepcopy(calendars)
    actions: list[dict[str, Any]] = []
    total_cost = 0
    for agent_id in new_meeting["participants"]:
        item = working[agent_id][local_slot]
        if item is None:
            continue
        if not isinstance(item, dict) or "errand_id" not in item:
            raise ValueError("expected local repair slot to contain only errands or free slots")
        local_target_slot = local_target_slots[agent_id]
        if working[agent_id][local_target_slot] is not None:
            raise ValueError("local errand target slot must be free")
        working[agent_id][local_slot] = None
        working[agent_id][local_target_slot] = item
        total_cost += int(item["cost"])
        actions.append({
            "type": "reschedule",
            "agent_id": agent_id,
            "moved_kind": "errand",
            "errand_id": item["errand_id"],
            "from_slot": local_slot,
            "to_slot": local_target_slot,
            "cost": item["cost"],
            "justification": "Move the errand to free the selected meeting slot in the repair plan.",
        })

    for agent_id in new_meeting["participants"]:
        if working[agent_id][local_slot] is not None:
            raise ValueError("local slot is not clear after moving errands")
    for agent_id in new_meeting["participants"]:
        working[agent_id][local_slot] = {"meeting_id": new_meeting["id"], "cost": new_meeting["cost"]}
    actions.append({
        "type": "schedule",
        "meeting_id": new_meeting["id"],
        "participants": new_meeting["participants"],
        "slot": local_slot,
        "cost": 0,
    })
    return {
        "cost": total_cost,
        "assignments": {str(new_meeting["id"]): local_slot},
        "actions": actions,
        "errand_moves": sum(1 for action in actions if action["type"] == "reschedule"),
        "final_calendars": working,
    }


def replay_initial_prior_meeting_repair(
    calendars: list[list[Any]],
    *,
    prior_meeting: dict[str, Any],
    new_meeting: dict[str, Any],
    prior_target_slot: int,
) -> dict[str, Any]:
    working = deepcopy(calendars)
    actions: list[dict[str, Any]] = []
    total_cost = 0
    external_agents: set[int] = set()
    new_participants = set(new_meeting["participants"])
    for agent_id in prior_meeting["participants"]:
        item = working[agent_id][prior_meeting["slot"]]
        if not isinstance(item, dict) or item.get("meeting_id") != prior_meeting["id"]:
            raise ValueError("expected prior meeting in required slot")
        if working[agent_id][prior_target_slot] is not None:
            raise ValueError("prior meeting target slot must be free")
        working[agent_id][prior_meeting["slot"]] = None
        working[agent_id][prior_target_slot] = item
        total_cost += int(item["cost"])
        if agent_id not in new_participants:
            external_agents.add(agent_id)
        actions.append({
            "type": "reschedule",
            "agent_id": agent_id,
            "moved_kind": "meeting",
            "meeting_id": prior_meeting["id"],
            "from_slot": prior_meeting["slot"],
            "to_slot": prior_target_slot,
            "cost": item["cost"],
            "justification": "Move the prior meeting to free the selected meeting slot in the repair plan.",
            "external_to_new_meeting": agent_id not in new_participants,
        })

    required_slot = prior_meeting["slot"]
    for agent_id in new_meeting["participants"]:
        if working[agent_id][required_slot] is not None:
            raise ValueError("required slot is not clear after moving prior meeting")
    for agent_id in new_meeting["participants"]:
        working[agent_id][required_slot] = {"meeting_id": new_meeting["id"], "cost": new_meeting["cost"]}
    actions.append({
        "type": "schedule",
        "meeting_id": new_meeting["id"],
        "participants": new_meeting["participants"],
        "slot": required_slot,
        "cost": 0,
    })
    return {
        "cost": total_cost,
        "assignments": {str(prior_meeting["id"]): prior_target_slot, str(new_meeting["id"]): required_slot},
        "actions": actions,
        "prior_meeting_moves": len(prior_meeting["participants"]),
        "external_agents_involved": sorted(external_agents),
        "final_calendars": working,
    }


def validate_initial_prior_meeting_move_task(task: dict[str, Any]) -> None:
    repair = task["online_optimal_repair"]
    local_repair = task["local_errand_repair"]
    multiplier = task["params"]["errand_cost_multiplier"]
    if repair["prior_meeting_moves"] <= 0:
        raise ValueError(f"{task['task_id']}: expected prior meeting moves")
    if not repair["external_agents_involved"]:
        raise ValueError(f"{task['task_id']}: expected external agent involvement")
    if local_repair["cost"] <= repair["cost"]:
        raise ValueError(f"{task['task_id']}: local errand repair should be more expensive than prior meeting repair")
    if local_repair["errand_moves"] <= 0:
        raise ValueError(f"{task['task_id']}: expected local errand moves")
    for calendar in task["calendars"]:
        for slot in calendar:
            if isinstance(slot, dict) and "errand_id" in slot and slot["cost"] != multiplier:
                raise ValueError(f"{task['task_id']}: errand cost mismatch")
            if isinstance(slot, dict) and "meeting_id" in slot and slot["cost"] != 1:
                raise ValueError(f"{task['task_id']}: meeting cost mismatch")
    if any(action["moved_kind"] != "meeting" for action in repair["actions"] if action["type"] == "reschedule"):
        raise ValueError(f"{task['task_id']}: repair should only move prior meetings")


def assign_balanced_errand_costs(
    calendars: list[list[Any]],
    *,
    cost_values: list[int],
    seed: int,
) -> None:
    """Assign a balanced errand-cost multiset within each agent calendar."""
    if not cost_values:
        raise ValueError("cost_values cannot be empty")
    for agent_id, calendar in enumerate(calendars):
        errand_count = sum(1 for slot in calendar if isinstance(slot, dict) and "errand_id" in slot)
        costs = [
            cost_values[index % len(cost_values)]
            for index in range(errand_count)
        ]
        random.Random(seed + agent_id).shuffle(costs)
        cost_iter = iter(costs)
        for slot in calendar:
            if isinstance(slot, dict) and "errand_id" in slot:
                slot["cost"] = next(cost_iter)


def recompute_scenario_solutions(scenario: dict[str, Any], *, skip_optimal: bool = False) -> None:
    assignments = {
        int(meeting_id): int(slot)
        for meeting_id, slot in scenario["witness_solution"]["assignments"].items()
    }
    working = scenario["calendars"]
    witness_cost = 0
    for meeting in scenario["meetings"]:
        working, cost, _per_agent = apply_schedule(working, meeting, assignments[int(meeting["id"])])
        if cost is None:
            raise ValueError("generated witness solution is not feasible after cost assignment")
        witness_cost += cost
    scenario["witness_solution"] = {"cost": witness_cost, "assignments": assignments}
    scenario["optimal"] = (
        {}
        if skip_optimal
        else solve_optimal(scenario["calendars"], scenario["meetings"], scenario["num_slots"])
    )
    scenario["greedy"] = solve_greedy(scenario["calendars"], scenario["meetings"], scenario["num_slots"])
    scenario["feasible"] = True if skip_optimal else scenario["optimal"].get("cost") is not None


def derive_cost_ratio_task(
    source_task: dict[str, Any],
    multiplier: int,
    *,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
) -> dict[str, Any]:
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1")

    task = deepcopy(source_task)
    task["task_id"] = f"{source_task['task_id']}_errandx{multiplier}"
    task["params"]["pref_level"] = 1
    task["params"]["errand_cost_multiplier"] = multiplier
    task["params"]["meeting_cost_range"] = [1, 1]
    task["params"]["errand_cost_range"] = [multiplier, multiplier]
    task["cost_structure"] = {
        "source_task_id": source_task["task_id"],
        "meeting_move_cost": 1,
        "errand_move_cost": multiplier,
        "errand_to_meeting_cost_ratio": multiplier,
    }
    ensure_task_speaker_orders(task)

    for calendar in task["calendars"]:
        for slot in calendar:
            if isinstance(slot, dict):
                if "errand_id" in slot:
                    slot["cost"] = multiplier
                elif "meeting_id" in slot:
                    slot["cost"] = 1

    for meeting in task["meetings"]:
        meeting["cost"] = 1

    witness_assignments = {int(meeting_id): slot for meeting_id, slot in task["witness_solution"]["assignments"].items()}
    witness_cost = 0
    working = task["calendars"]
    for meeting in task["meetings"]:
        working, cost, _per_agent = apply_schedule(working, meeting, witness_assignments[int(meeting["id"])])
        if cost is None:
            raise ValueError(f"{task['task_id']}: witness assignment is infeasible after cost derivation")
        witness_cost += cost
    task["witness_solution"] = {
        "assignments": task["witness_solution"]["assignments"],
        "cost": witness_cost,
    }

    task["optimal"] = solve_optimal(task["calendars"], task["meetings"], task["params"]["num_slots"])
    task["greedy"] = solve_greedy(task["calendars"], task["meetings"], task["params"]["num_slots"])
    task["feasible"] = task["optimal"].get("cost") is not None
    task["optimal_evictions"] = count_evictions_for_assignments(
        task["calendars"],
        task["meetings"],
        task["optimal"]["assignments"],
    )
    score_task_difficulty(task, difficulty_scorer)
    validate_task(task)
    return task


def _task_solution_cost(
    calendars: list[list[Any]],
    meetings: list[dict[str, Any]],
    assignments: dict[str | int, int],
) -> int | None:
    normalized = {int(meeting_id): int(slot) for meeting_id, slot in assignments.items()}
    working = calendars
    total = 0
    for meeting in meetings:
        slot = normalized[int(meeting["id"])]
        working, cost, _per_agent = apply_schedule(working, meeting, slot)
        if cost is None:
            return None
        total += cost
    return total


def _recompute_task_after_calendar_change(
    task: dict[str, Any],
    *,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
) -> bool:
    task["optimal"] = solve_optimal(task["calendars"], task["meetings"], task["params"]["num_slots"])
    task["greedy"] = solve_greedy(task["calendars"], task["meetings"], task["params"]["num_slots"])
    task["feasible"] = task["optimal"].get("cost") is not None
    if not task["feasible"] or task["greedy"].get("cost") is None:
        return False

    original_assignments = task.get("witness_solution", {}).get("assignments", {})
    original_cost = _task_solution_cost(task["calendars"], task["meetings"], original_assignments)
    if original_cost is not None:
        task["witness_solution"] = {
            "assignments": original_assignments,
            "cost": original_cost,
        }
    else:
        task["witness_solution"] = {
            "assignments": task["optimal"]["assignments"],
            "cost": task["optimal"]["cost"],
        }
    task["optimal_evictions"] = count_evictions_for_assignments(
        task["calendars"],
        task["meetings"],
        task["optimal"]["assignments"],
    )
    score_task_difficulty(task, difficulty_scorer)
    validate_task(task)
    return True


def _scaled_blocked_errands_per_agent(max_blocked_errands_per_agent: int, density: float) -> int:
    if max_blocked_errands_per_agent < 0:
        raise ValueError("blocked_errands_per_agent must be >= 0")
    if max_blocked_errands_per_agent == 0:
        return 0
    return max(1, round(max_blocked_errands_per_agent * density * density))


def derive_blocked_errand_task(
    source_task: dict[str, Any],
    *,
    blocked_errands_per_agent: int = DEFAULT_BLOCKED_ERRANDS_PER_AGENT,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
    max_attempts: int = 250,
) -> dict[str, Any]:
    if blocked_errands_per_agent < 0:
        raise ValueError("blocked_errands_per_agent must be >= 0")

    witness_assignments = {
        int(meeting_id): int(slot)
        for meeting_id, slot in source_task["witness_solution"]["assignments"].items()
    }
    witness_slots_by_agent: dict[int, set[int]] = {
        agent_id: set()
        for agent_id in range(source_task["params"]["total_agents"])
    }
    for meeting in source_task["meetings"]:
        slot = witness_assignments[int(meeting["id"])]
        for agent_id in meeting["participants"]:
            witness_slots_by_agent[agent_id].add(slot)

    eligible_slots_by_agent: dict[int, list[int]] = {}
    for agent_id, calendar in enumerate(source_task["calendars"]):
        eligible = [
            slot_idx
            for slot_idx, slot in enumerate(calendar)
            if (
                isinstance(slot, dict)
                and "errand_id" in slot
                and not slot.get("blocked")
                and slot_idx not in witness_slots_by_agent[agent_id]
            )
        ]
        if len(eligible) < blocked_errands_per_agent:
            raise ValueError(
                f"{source_task['task_id']}: agent {agent_id} has only {len(eligible)} "
                f"eligible errands for {blocked_errands_per_agent} blocked errands"
            )
        eligible_slots_by_agent[agent_id] = eligible

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        rng = random.Random(f"blocked:{source_task['task_id']}:{blocked_errands_per_agent}:{attempt}")
        task = deepcopy(source_task)
        ensure_task_speaker_orders(task)
        task["task_id"] = f"{source_task['task_id']}_blocked{blocked_errands_per_agent}"
        task["blocked_source_task_id"] = source_task["task_id"]
        task.pop("config_normalized_difficulty", None)
        task.pop("config_difficulty_degenerate", None)
        task["params"]["blocked_errands_per_agent"] = blocked_errands_per_agent
        blocked_slots_by_agent: dict[str, list[int]] = {}
        for agent_id, eligible in eligible_slots_by_agent.items():
            chosen = sorted(rng.sample(eligible, blocked_errands_per_agent))
            blocked_slots_by_agent[str(agent_id)] = chosen
            for slot_idx in chosen:
                slot = task["calendars"][agent_id][slot_idx]
                if not isinstance(slot, dict) or "errand_id" not in slot:
                    raise ValueError(f"{task['task_id']}: blocked slot selection no longer points at an errand")
                slot["blocked"] = True
        task["blocked_slots_by_agent"] = blocked_slots_by_agent
        task["blocked_errand_count"] = sum(len(slots) for slots in blocked_slots_by_agent.values())
        try:
            if _recompute_task_after_calendar_change(task, difficulty_scorer=difficulty_scorer):
                return task
        except ValueError as exc:
            last_error = exc

    detail = f": {last_error}" if last_error is not None else ""
    raise ValueError(
        f"{source_task['task_id']}: could not derive feasible blocked task "
        f"after {max_attempts} attempts{detail}"
    )


def assign_tertile_difficulty_buckets(tasks: list[dict[str, Any]]) -> None:
    ranked = sorted(
        tasks,
        key=lambda task: (
            difficulty_score_sort_value(task),
            int(task.get("optimal_evictions", 0)),
            int(task["seed"]),
        ),
    )
    partitions = {
        "easy": ranked[: len(ranked) // 3],
        "medium": ranked[len(ranked) // 3: (2 * len(ranked)) // 3],
        "hard": ranked[(2 * len(ranked)) // 3:],
    }
    for bucket, bucket_tasks in partitions.items():
        for task in bucket_tasks:
            task["difficulty"] = bucket


def build_blocked_errand_tasks(
    source_path: str | Path,
    *,
    blocked_errands_per_agent: int = DEFAULT_BLOCKED_ERRANDS_PER_AGENT,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
    setting_prefix: str = "uniform_full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_tasks = read_jsonl(source_path)
    source_tasks = [
        task for task in source_tasks
        if (
            int(task["params"]["total_agents"]) == 5
            and int(task["params"]["subset_size"]) == 3
            and int(task["params"]["num_meetings"]) == FIVE_MEETING_NUM_MEETINGS
        )
    ]
    tasks = [
        derive_blocked_errand_task(
            task,
            blocked_errands_per_agent=_scaled_blocked_errands_per_agent(
                blocked_errands_per_agent,
                float(task["params"]["density"]),
            ),
            difficulty_scorer=difficulty_scorer,
        )
        for task in source_tasks
    ]
    assign_tertile_difficulty_buckets(tasks)
    assign_setting_normalized_difficulty(tasks)
    summary = {
        "setting": f"blocked_errands_{setting_prefix}_5a3p_n{blocked_errands_per_agent}",
        "source": str(source_path),
        "purpose": (
            "Derived from existing uniform full 5-meeting tasks by marking a "
            "fixed number of each agent's initial errands as blocked, then "
            "recomputing feasibility and difficulty under blocked-slot constraints."
        ),
        "num_source_tasks": len(source_tasks),
        "num_tasks": len(tasks),
        "max_blocked_errands_per_agent": blocked_errands_per_agent,
        "blocked_errands_scaled_by_density": True,
        "blocked_errands_per_agent_by_density": {
            str(density): _scaled_blocked_errands_per_agent(blocked_errands_per_agent, density)
            for density in sorted({float(task["params"]["density"]) for task in source_tasks})
        },
        "blocked_errand_count_per_task_by_density": {
            str(density): (
                _scaled_blocked_errands_per_agent(blocked_errands_per_agent, density)
                * int(source_tasks[0]["params"]["total_agents"])
            )
            for density in sorted({float(task["params"]["density"]) for task in source_tasks})
        } if source_tasks else {},
        "difficulty_scorer": difficulty_scorer,
        "difficulty_score_definition": DIFFICULTY_SCORER_DESCRIPTIONS[difficulty_scorer],
        "bucket_counts": {
            bucket: sum(1 for task in tasks if task.get("difficulty") == bucket)
            for bucket in DIFFICULTY_BUCKETS
        },
        "difficulty_score_min": min(float(task["difficulty_score"]) for task in tasks) if tasks else None,
        "difficulty_score_max": max(float(task["difficulty_score"]) for task in tasks) if tasks else None,
    }
    return tasks, summary


def all_difficulty_configs(pref_levels: list[int] | None = None) -> list[dict[str, Any]]:
    pref_levels = pref_levels or PREF_LEVEL_VALUES
    configs: list[dict[str, Any]] = []
    for total_agents in TOTAL_AGENT_VALUES:
        for subset_size in range(2, total_agents + 1):
            for density in DENSITY_VALUES:
                for pref_level in pref_levels:
                    configs.append({
                        "total_agents": total_agents,
                        "subset_size": subset_size,
                        "density": density,
                        "pref_level": pref_level,
                    })
    return configs


def five_meeting_configs(*, preference: bool = False) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for total_agents in FIVE_MEETING_TOTAL_AGENT_VALUES:
        for subset_size in FIVE_MEETING_SUBSET_VALUES_BY_AGENT_COUNT[total_agents]:
            for density in FIVE_MEETING_DENSITY_VALUES:
                config: dict[str, Any] = {
                    "total_agents": total_agents,
                    "subset_size": subset_size,
                    "density": density,
                    "pref_level": 3 if preference else 1,
                    "meeting_cost_level": 1,
                }
                if preference:
                    config["errand_cost_level"] = 1
                    config["errand_cost_values"] = PREFERENCE_ERRAND_COST_VALUES
                configs.append(config)
    return configs


def build_balanced_tasks_for_configs(
    configs: list[dict[str, Any]],
    *,
    num_meetings: int,
    candidates_per_config: int = BALANCED_CANDIDATES_PER_CONFIG,
    selected_per_bucket: int = BALANCED_SELECTED_PER_BUCKET,
    seed_base: int = 120_000,
    setting_name: str,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
    skip_optimal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if skip_optimal:
        raise ValueError("--skip-optimal cannot be used with balanced suites: difficulty bucketing requires scored tasks")
    tasks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "setting": setting_name,
        "num_meetings": num_meetings,
        "difficulty_scorer": difficulty_scorer,
        "difficulty_score_definition": DIFFICULTY_SCORER_DESCRIPTIONS[difficulty_scorer],
        "difficulty_bucket_scope": "config_relative_tertile",
        "candidates_per_config": candidates_per_config,
        "selected_per_bucket": selected_per_bucket,
        "configs": [],
        "bucket_counts": {bucket: 0 for bucket in DIFFICULTY_BUCKETS},
    }
    total_selected_tasks = len(configs) * len(DIFFICULTY_BUCKETS) * selected_per_bucket

    for config_idx, config in enumerate(configs):
        candidates: list[dict[str, Any]] = []
        for candidate_idx in range(candidates_per_config):
            seed = seed_base + (config_idx * 1000) + candidate_idx
            spec = {
                **config,
                "seed": seed,
                "num_meetings": num_meetings,
                "task_id": f"candidate_{config_idx:03d}_{candidate_idx:03d}",
            }
            apply_per_candidate_agent_variation(
                spec,
                candidate_idx=candidate_idx,
                config_idx=config_idx,
            )
            if "blocked_errands_per_agent" not in spec:
                blocked_count = choose_blocked_errands_per_agent(
                    config,
                    candidate_idx=candidate_idx,
                    config_idx=config_idx,
                )
                if blocked_count is not None:
                    spec["blocked_errands_per_agent"] = blocked_count
            candidates.append(build_task(spec, difficulty_scorer=difficulty_scorer))

        scores = [float(task["difficulty_score"]) for task in candidates]
        costs = [int(task["optimal"]["cost"]) for task in candidates]
        evictions = [int(task["optimal_evictions"]) for task in candidates]
        min_score = min(scores)
        max_score = max(scores)
        score_is_degenerate = max_score == min_score
        for task in candidates:
            score = float(task["difficulty_score"])
            task["config_normalized_difficulty"] = normalized_difficulty_value(
                score,
                min_score,
                max_score,
                difficulty_scorer,
            )
            task["config_difficulty_degenerate"] = score_is_degenerate

        ranked = sorted(
            candidates,
            key=lambda task: (
                difficulty_score_sort_value(task, difficulty_scorer),
                task["optimal_evictions"],
                task["seed"],
            ),
        )
        partitions = {
            "easy": ranked[: len(ranked) // 3],
            "medium": ranked[len(ranked) // 3: (2 * len(ranked)) // 3],
            "hard": ranked[(2 * len(ranked)) // 3:],
        }

        selected_for_config: list[dict[str, Any]] = []
        selected_blocked_balance: dict[str, Any] = {}
        for bucket in DIFFICULTY_BUCKETS:
            bucket_tasks = partitions[bucket]
            if len(bucket_tasks) < selected_per_bucket:
                raise ValueError(f"not enough {bucket} candidates for config {config}")
            selected_bucket_tasks, balance_info = select_balanced_bucket_tasks(
                bucket_tasks,
                selected_per_bucket,
                config=config,
                config_idx=config_idx,
                bucket=bucket,
                selected_so_far=tasks + selected_for_config,
                total_selected_tasks=total_selected_tasks,
            )
            selected_blocked_balance[bucket] = balance_info
            for bucket_idx, task in enumerate(selected_bucket_tasks, start=1):
                task = dict(task)
                task["difficulty"] = bucket
                task["task_id"] = task_id_for_balanced_task(
                    len(tasks) + len(selected_for_config) + 1,
                    config,
                    bucket,
                    bucket_idx,
                )
                selected_for_config.append(task)
                summary["bucket_counts"][bucket] += 1

        tasks.extend(selected_for_config)
        summary["configs"].append({
            **config,
            "candidate_optimal_cost_min": min(costs),
            "candidate_optimal_cost_max": max(costs),
            "candidate_optimal_evictions_min": min(evictions),
            "candidate_optimal_evictions_max": max(evictions),
            "candidate_difficulty_score_min": min_score,
            "candidate_difficulty_score_max": max_score,
            "candidate_difficulty_score_degenerate": score_is_degenerate,
            "selected": {
                bucket: selected_per_bucket
                for bucket in DIFFICULTY_BUCKETS
            },
            "selected_blocked_errands_per_agent_balance": selected_blocked_balance,
        })

    summary["total_configs"] = len(configs)
    summary["total_tasks"] = len(tasks)
    assign_setting_normalized_difficulty(tasks)
    summary["difficulty_score_min"] = min(float(task["difficulty_score"]) for task in tasks) if tasks else None
    summary["difficulty_score_max"] = max(float(task["difficulty_score"]) for task in tasks) if tasks else None
    summary["degenerate_config_count"] = sum(
        1
        for config in summary["configs"]
        if config["candidate_difficulty_score_degenerate"]
    )
    return tasks, summary


def feasible_blocked_errands_per_agent_values(config: dict[str, Any]) -> list[int]:
    raw_values = config.get("blocked_errands_per_agent_values", DEFAULT_BLOCKED_ERRANDS_PER_AGENT_VALUES)
    values = [int(value) for value in _as_list(raw_values)]
    if not values:
        return []

    num_slots = int(config.get("num_slots", NUM_SLOTS))
    density = float(config["density"])
    agent_densities = config.get("agent_densities")
    if isinstance(agent_densities, dict):
        density_values = [
            float(agent_densities.get(agent_id, agent_densities.get(str(agent_id), density)))
            for agent_id in range(int(config["total_agents"]))
        ]
    elif isinstance(agent_densities, list):
        density_values = [float(value) for value in agent_densities]
    else:
        density_values = [density]

    # Leave at least one movable errand so the generated witness solution still
    # has non-zero displacement cost when density is positive.
    max_feasible = max(0, min(round(num_slots * value) for value in density_values) - 1)
    return [value for value in values if 0 <= value <= max_feasible]


def choose_blocked_errands_per_agent(
    config: dict[str, Any],
    *,
    candidate_idx: int,
    config_idx: int = 0,
) -> int | None:
    """Deterministically cycle blocked-count candidates before CP-SAT bucketing."""
    feasible_values = feasible_blocked_errands_per_agent_values(config)
    if not feasible_values:
        return None
    return feasible_values[(candidate_idx + config_idx) % len(feasible_values)]


def apply_per_candidate_agent_variation(
    spec: dict[str, Any],
    *,
    candidate_idx: int,
    config_idx: int,
) -> None:
    total_agents = int(spec["total_agents"])
    seed = int(spec["seed"])
    if spec.get("vary_agent_densities") and "agent_densities" not in spec:
        raw_values = spec.get("agent_density_values", spec.get("densities", spec.get("density")))
        density_values = [float(value) for value in _as_list(raw_values)]
        if not density_values:
            raise ValueError("vary_agent_densities requires agent_density_values or density")
        spec["agent_densities"] = _shuffled_repeating_values(
            density_values,
            total_agents,
            seed=seed + 17,
            offset=config_idx + candidate_idx,
        )

    if spec.get("vary_blocked_errands_per_agent") and "blocked_errands_per_agent" not in spec:
        blocked_values = feasible_blocked_errands_per_agent_values(spec)
        if blocked_values:
            spec["blocked_errands_per_agent"] = _shuffled_repeating_values(
                blocked_values,
                total_agents,
                seed=seed + 31,
                offset=config_idx + candidate_idx,
            )


def _shuffled_repeating_values(
    values: list[Any],
    length: int,
    *,
    seed: int,
    offset: int,
) -> list[Any]:
    if not values:
        raise ValueError("values must not be empty")
    repeated = [values[(offset + index) % len(values)] for index in range(length)]
    rng = random.Random(seed)
    rng.shuffle(repeated)
    return repeated


def _blocked_count_key(value: int | None) -> tuple[int, int]:
    return (1, -1) if value is None else (0, int(value))


def _task_blocked_count(task: dict[str, Any]) -> Any:
    value = task.get("params", {}).get("blocked_errands_per_agent")
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return int(value)


def _counter_by_blocked_count(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        key = "none" if _task_blocked_count(task) is None else str(_task_blocked_count(task))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _task_param_values(task: dict[str, Any], key: str) -> list[Any]:
    value = task.get("params", {}).get(key)
    total_agents = int(task.get("params", {}).get("total_agents", 0))
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value] * total_agents


def _counts_for_values(values: list[Any], target_values: list[Any]) -> dict[Any, int]:
    counts = {value: 0 for value in target_values}
    for value in values:
        if value in counts:
            counts[value] += 1
    return counts


def _balance_metrics(counts: dict[Any, int]) -> tuple[int, float]:
    if not counts:
        return (0, 0.0)
    values = list(counts.values())
    ideal = sum(values) / len(values)
    return (
        max(values) - min(values),
        sum((value - ideal) ** 2 for value in values),
    )


def _balance_summary_for_selected(
    selected: list[dict[str, Any]],
    *,
    density_values: list[float],
    blocked_values: list[int],
) -> dict[str, Any]:
    density_counts = _counts_for_values(
        [
            float(value)
            for task in selected
            for value in _task_param_values(task, "agent_densities")
        ],
        density_values,
    ) if density_values else {}
    blocked_counts = _counts_for_values(
        [
            int(value)
            for task in selected
            for value in _task_param_values(task, "blocked_errands_per_agent")
        ],
        blocked_values,
    ) if blocked_values else {}
    density_range, density_sse = _balance_metrics(density_counts)
    blocked_range, blocked_sse = _balance_metrics(blocked_counts)
    return {
        "density_value_counts": {str(key): value for key, value in density_counts.items()},
        "blocked_value_counts": {str(key): value for key, value in blocked_counts.items()},
        "density_count_range": density_range,
        "blocked_count_range": blocked_range,
        "density_count_sse": density_sse,
        "blocked_count_sse": blocked_sse,
    }


def _agent_pair_values(task: dict[str, Any]) -> list[tuple[int, float, int]]:
    densities = [float(value) for value in _task_param_values(task, "agent_densities")]
    blocked = [int(value) for value in _task_param_values(task, "blocked_errands_per_agent")]
    return [
        (agent_id, densities[agent_id], blocked[agent_id])
        for agent_id in range(min(len(densities), len(blocked)))
    ]


def _agent_pair_counts(
    selected: list[dict[str, Any]],
    *,
    total_agents: int,
    pair_values: list[tuple[float, int]],
) -> dict[tuple[int, float, int], int]:
    counts = {
        (agent_id, density, blocked): 0
        for agent_id in range(total_agents)
        for density, blocked in pair_values
    }
    for task in selected:
        for key in _agent_pair_values(task):
            if key in counts:
                counts[key] += 1
    return counts


def _agent_pair_balance_summary(
    selected: list[dict[str, Any]],
    *,
    total_agents: int,
    pair_values: list[tuple[float, int]],
    target_per_agent_pair: float,
) -> dict[str, Any]:
    counts = _agent_pair_counts(
        selected,
        total_agents=total_agents,
        pair_values=pair_values,
    )
    by_agent: dict[int, list[int]] = {agent_id: [] for agent_id in range(total_agents)}
    for (agent_id, _density, _blocked), count in counts.items():
        by_agent[agent_id].append(count)
    per_agent_range = {
        str(agent_id): max(values) - min(values)
        for agent_id, values in by_agent.items()
        if values
    }
    per_agent_sse = {
        str(agent_id): sum((count - target_per_agent_pair) ** 2 for count in values)
        for agent_id, values in by_agent.items()
    }
    return {
        "agent_pair_counts": {
            str(agent_id): {
                f"{density:g}|{blocked}": counts[(agent_id, density, blocked)]
                for density, blocked in pair_values
            }
            for agent_id in range(total_agents)
        },
        "agent_pair_count_range_by_agent": per_agent_range,
        "agent_pair_count_sse_by_agent": per_agent_sse,
        "agent_pair_max_count_range": max(per_agent_range.values()) if per_agent_range else 0,
        "agent_pair_total_sse": sum(per_agent_sse.values()),
        "target_per_agent_pair": target_per_agent_pair,
    }


def select_balanced_varied_bucket_tasks(
    bucket_tasks: list[dict[str, Any]],
    selected_per_bucket: int,
    *,
    config: dict[str, Any],
    selected_so_far: list[dict[str, Any]] | None = None,
    total_selected_tasks: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select ranked bucket tasks while balancing per-agent config exposure."""
    density_values = (
        [float(value) for value in _as_list(config.get("agent_density_values", config.get("density")))]
        if config.get("vary_agent_densities")
        else []
    )
    blocked_values = (
        feasible_blocked_errands_per_agent_values(config)
        if config.get("vary_blocked_errands_per_agent")
        else []
    )
    selected_so_far = selected_so_far or []
    if not density_values and not blocked_values:
        selected = list(bucket_tasks[:selected_per_bucket])
        return selected, {
            "target_sequence": [],
            "available_counts": _counter_by_blocked_count(bucket_tasks),
            "selected_counts": _counter_by_blocked_count(selected),
            "fallback_count": 0,
        }

    total_agents = int(config["total_agents"])
    pair_values = [
        (density, blocked)
        for density in density_values
        for blocked in blocked_values
    ]
    if not pair_values:
        selected = list(bucket_tasks[:selected_per_bucket])
        return selected, {
            "target_sequence": [],
            "available_counts": _counter_by_blocked_count(bucket_tasks),
            "selected_counts": _counter_by_blocked_count(selected),
            "fallback_count": 0,
        }
    target_per_agent_pair = (
        (total_selected_tasks or (len(selected_so_far) + selected_per_bucket))
        / len(pair_values)
    )
    base_counts = _agent_pair_counts(
        selected_so_far,
        total_agents=total_agents,
        pair_values=pair_values,
    )

    best_score: tuple[Any, ...] | None = None
    best_indices: tuple[int, ...] | None = None
    for indices in combinations(range(len(bucket_tasks)), selected_per_bucket):
        candidate = [bucket_tasks[index] for index in indices]
        counts = dict(base_counts)
        for task in candidate:
            for key in _agent_pair_values(task):
                if key in counts:
                    counts[key] += 1
        values_by_agent: dict[int, list[int]] = {agent_id: [] for agent_id in range(total_agents)}
        for (agent_id, _density, _blocked), count in counts.items():
            values_by_agent[agent_id].append(count)
        per_agent_ranges = [
            max(values) - min(values)
            for values in values_by_agent.values()
            if values
        ]
        per_agent_sse = [
            sum((count - target_per_agent_pair) ** 2 for count in values)
            for values in values_by_agent.values()
        ]
        score = (
            max(per_agent_ranges) if per_agent_ranges else 0,
            sum(per_agent_ranges),
            sum(per_agent_sse),
            sum(indices),
            indices,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_indices = indices

    selected = [bucket_tasks[index] for index in best_indices or tuple(range(selected_per_bucket))]
    balance = _balance_summary_for_selected(
        selected,
        density_values=density_values,
        blocked_values=blocked_values,
    )
    agent_pair_balance = _agent_pair_balance_summary(
        selected_so_far + selected,
        total_agents=total_agents,
        pair_values=pair_values,
        target_per_agent_pair=target_per_agent_pair,
    )
    balance.update({
        **agent_pair_balance,
        "target_sequence": [],
        "available_counts": _counter_by_blocked_count(bucket_tasks),
        "selected_counts": _counter_by_blocked_count(selected),
        "fallback_count": 0,
        "selection_strategy": "per_agent_pair_balance",
    })
    return selected, balance


def select_balanced_bucket_tasks(
    bucket_tasks: list[dict[str, Any]],
    selected_per_bucket: int,
    *,
    config: dict[str, Any],
    config_idx: int,
    bucket: str,
    selected_so_far: list[dict[str, Any]] | None = None,
    total_selected_tasks: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select from a CP-SAT bucket while balancing blocked-count exposure.

    ``bucket_tasks`` is already difficulty-ranked. The selector asks for a
    rotating blocked-count sequence, but falls back to the best remaining task
    when the current CP-SAT bucket does not contain the desired count.
    """
    if selected_per_bucket > len(bucket_tasks):
        raise ValueError("selected_per_bucket cannot exceed bucket size")

    if config.get("vary_agent_densities") or config.get("vary_blocked_errands_per_agent"):
        return select_balanced_varied_bucket_tasks(
            bucket_tasks,
            selected_per_bucket,
            config=config,
            selected_so_far=selected_so_far,
            total_selected_tasks=total_selected_tasks,
        )

    blocked_values = feasible_blocked_errands_per_agent_values(config)
    if not blocked_values:
        selected = list(bucket_tasks[:selected_per_bucket])
        return selected, {
            "target_sequence": [],
            "available_counts": _counter_by_blocked_count(bucket_tasks),
            "selected_counts": _counter_by_blocked_count(selected),
            "fallback_count": 0,
        }

    offset = (config_idx + DIFFICULTY_BUCKETS.index(bucket)) % len(blocked_values)
    target_sequence = [
        blocked_values[(offset + index) % len(blocked_values)]
        for index in range(selected_per_bucket)
    ]

    remaining = list(bucket_tasks)
    selected: list[dict[str, Any]] = []
    fallback_count = 0
    for desired_count in target_sequence:
        match_idx = next(
            (
                index
                for index, task in enumerate(remaining)
                if _task_blocked_count(task) == desired_count
            ),
            None,
        )
        if match_idx is None:
            fallback_count += 1
            match_idx = 0
        selected.append(remaining.pop(match_idx))

    return selected, {
        "target_sequence": target_sequence,
        "available_counts": _counter_by_blocked_count(bucket_tasks),
        "selected_counts": _counter_by_blocked_count(selected),
        "fallback_count": fallback_count,
    }


def build_balanced_tasks(
    *,
    candidates_per_config: int = BALANCED_CANDIDATES_PER_CONFIG,
    selected_per_bucket: int = BALANCED_SELECTED_PER_BUCKET,
    seed_base: int = 20_000,
    pref_levels: list[int] | None = None,
    setting_name: str = "balanced_v1",
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
    skip_optimal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if skip_optimal:
        raise ValueError("--skip-optimal cannot be used with balanced suites: difficulty bucketing requires scored tasks")
    pref_levels = pref_levels or BALANCED_PREF_LEVEL_VALUES
    tasks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "setting": setting_name,
        "num_meetings": BALANCED_NUM_MEETINGS,
        "difficulty_scorer": difficulty_scorer,
        "difficulty_score_definition": DIFFICULTY_SCORER_DESCRIPTIONS[difficulty_scorer],
        "difficulty_bucket_scope": "config_relative_tertile",
        "pref_levels": pref_levels,
        "candidates_per_config": candidates_per_config,
        "selected_per_bucket": selected_per_bucket,
        "configs": [],
        "bucket_counts": {bucket: 0 for bucket in DIFFICULTY_BUCKETS},
    }

    configs = all_difficulty_configs(pref_levels)
    total_selected_tasks = len(configs) * len(DIFFICULTY_BUCKETS) * selected_per_bucket
    for config_idx, config in enumerate(configs):
        candidates: list[dict[str, Any]] = []
        for candidate_idx in range(candidates_per_config):
            seed = seed_base + (config_idx * 1000) + candidate_idx
            spec = {
                **config,
                "seed": seed,
                "num_meetings": BALANCED_NUM_MEETINGS,
                "task_id": f"candidate_{config_idx:03d}_{candidate_idx:03d}",
            }
            apply_per_candidate_agent_variation(
                spec,
                candidate_idx=candidate_idx,
                config_idx=config_idx,
            )
            if "blocked_errands_per_agent" not in spec:
                blocked_count = choose_blocked_errands_per_agent(
                    config,
                    candidate_idx=candidate_idx,
                    config_idx=config_idx,
                )
                if blocked_count is not None:
                    spec["blocked_errands_per_agent"] = blocked_count
            task = build_task(spec, difficulty_scorer=difficulty_scorer)
            candidates.append(task)

        scores = [float(task["difficulty_score"]) for task in candidates]
        costs = [int(task["optimal"]["cost"]) for task in candidates]
        evictions = [int(task["optimal_evictions"]) for task in candidates]
        min_score = min(scores)
        max_score = max(scores)
        score_is_degenerate = max_score == min_score
        for task in candidates:
            score = float(task["difficulty_score"])
            task["config_normalized_difficulty"] = normalized_difficulty_value(
                score,
                min_score,
                max_score,
                difficulty_scorer,
            )
            task["config_difficulty_degenerate"] = score_is_degenerate

        ranked = sorted(
            candidates,
            key=lambda task: (
                difficulty_score_sort_value(task, difficulty_scorer),
                task["optimal_evictions"],
                task["seed"],
            ),
        )
        partitions = {
            "easy": ranked[: len(ranked) // 3],
            "medium": ranked[len(ranked) // 3: (2 * len(ranked)) // 3],
            "hard": ranked[(2 * len(ranked)) // 3:],
        }

        selected_for_config: list[dict[str, Any]] = []
        selected_blocked_balance: dict[str, Any] = {}
        for bucket in DIFFICULTY_BUCKETS:
            bucket_tasks = partitions[bucket]
            if len(bucket_tasks) < selected_per_bucket:
                raise ValueError(f"not enough {bucket} candidates for config {config}")
            selected_bucket_tasks, balance_info = select_balanced_bucket_tasks(
                bucket_tasks,
                selected_per_bucket,
                config=config,
                config_idx=config_idx,
                bucket=bucket,
                selected_so_far=tasks + selected_for_config,
                total_selected_tasks=total_selected_tasks,
            )
            selected_blocked_balance[bucket] = balance_info
            for bucket_idx, task in enumerate(selected_bucket_tasks, start=1):
                task = dict(task)
                task["difficulty"] = bucket
                task["task_id"] = task_id_for_balanced_task(
                    len(tasks) + len(selected_for_config) + 1,
                    config,
                    bucket,
                    bucket_idx,
                )
                selected_for_config.append(task)
                summary["bucket_counts"][bucket] += 1

        tasks.extend(selected_for_config)
        summary["configs"].append({
            **config,
            "candidate_optimal_cost_min": min(costs),
            "candidate_optimal_cost_max": max(costs),
            "candidate_optimal_evictions_min": min(evictions),
            "candidate_optimal_evictions_max": max(evictions),
            "candidate_difficulty_score_min": min_score,
            "candidate_difficulty_score_max": max_score,
            "candidate_difficulty_score_degenerate": score_is_degenerate,
            "selected": {
                bucket: selected_per_bucket
                for bucket in DIFFICULTY_BUCKETS
            },
            "selected_blocked_errands_per_agent_balance": selected_blocked_balance,
        })

    summary["total_configs"] = len(configs)
    summary["total_tasks"] = len(tasks)
    assign_setting_normalized_difficulty(tasks)
    summary["difficulty_score_min"] = min(float(task["difficulty_score"]) for task in tasks) if tasks else None
    summary["difficulty_score_max"] = max(float(task["difficulty_score"]) for task in tasks) if tasks else None
    summary["degenerate_config_count"] = sum(
        1
        for config in summary["configs"]
        if config["candidate_difficulty_score_degenerate"]
    )
    return tasks, summary


def _as_list(value: Any, *, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _load_taskgen_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("task generation config must be a mapping")
    return data


def expand_custom_config_cells(raw_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand user-facing task-generation cells into concrete generator configs."""
    expanded: list[dict[str, Any]] = []
    for raw in raw_configs:
        if not isinstance(raw, dict):
            raise ValueError("each task generation config cell must be a mapping")

        total_agents_values = _as_list(raw.get("total_agents"))
        if not total_agents_values:
            raise ValueError("each config cell must define total_agents")

        densities = _as_list(raw.get("densities", raw.get("density")))
        if not densities:
            raise ValueError("each config cell must define density or densities")

        pref_levels = _as_list(raw.get("pref_levels", raw.get("pref_level", 1)))
        num_meetings_values = _as_list(raw.get("num_meetings", NUM_MEETINGS))
        num_slots_values = _as_list(raw.get("num_slots", NUM_SLOTS))
        for total_agents in total_agents_values:
            subset_values = _as_list(
                raw.get("subset_sizes", raw.get("subset_size")),
                default=list(range(2, int(total_agents) + 1)),
            )
            for total_agents_i, subset_size_i, density, pref_level, num_meetings, num_slots in product(
                [int(total_agents)],
                [int(value) for value in subset_values],
                [float(value) for value in densities],
                [int(value) for value in pref_levels],
                [int(value) for value in num_meetings_values],
                [int(value) for value in num_slots_values],
            ):
                if subset_size_i > total_agents_i:
                    raise ValueError(
                        f"subset_size={subset_size_i} cannot exceed total_agents={total_agents_i}"
                    )
                cell = {
                    "total_agents": total_agents_i,
                    "subset_size": subset_size_i,
                    "density": density,
                    "pref_level": pref_level,
                    "num_meetings": num_meetings,
                    "num_slots": num_slots,
                }
                for key in (
                    "errand_cost_multiplier",
                    "errand_cost_level",
                    "errand_cost_values",
                    "meeting_cost_level",
                    "participant_lists",
                    "speaker_orders",
                    "agent_densities",
                    "agent_density_values",
                    "vary_agent_densities",
                    "blocked_errands_per_agent",
                    "blocked_errands_per_agent_values",
                    "vary_blocked_errands_per_agent",
                ):
                    if key in raw:
                        cell[key] = raw[key]
                expanded.append(cell)
    return expanded


def build_tasks_from_config(
    config: dict[str, Any],
    *,
    difficulty_scorer: str = DEFAULT_DIFFICULTY_SCORER,
    skip_optimal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a balanced fixture suite from a researcher-authored YAML/JSON config.

    The config intentionally mirrors the public suite builder: each concrete
    cell generates many candidate tasks, buckets them into easy/medium/hard by
    the configured difficulty scorer, and selects a fixed number per bucket.
    """
    default_keys = {
        "total_agents",
        "subset_size",
        "subset_sizes",
        "density",
        "densities",
        "pref_level",
        "pref_levels",
        "num_meetings",
        "num_slots",
        "errand_cost_multiplier",
        "errand_cost_level",
        "errand_cost_values",
        "meeting_cost_level",
        "agent_densities",
        "agent_density_values",
        "vary_agent_densities",
        "blocked_errands_per_agent",
        "blocked_errands_per_agent_values",
        "vary_blocked_errands_per_agent",
    }
    generation_defaults = {
        key: value
        for key, value in config.items()
        if key in default_keys
    }
    raw_configs = [
        {**generation_defaults, **cell}
        for cell in config.get("configs", [])
    ]
    configs = expand_custom_config_cells(raw_configs)
    if not configs:
        raise ValueError("task generation config must include at least one config cell")
    num_meeting_values = {int(cell["num_meetings"]) for cell in configs}
    if len(num_meeting_values) != 1:
        raise ValueError(
            "all expanded config cells must use the same num_meetings; "
            f"got {sorted(num_meeting_values)}"
        )

    candidates_per_config = int(config.get("candidates_per_config", FULL_DATASET_CANDIDATES_PER_CONFIG))
    selected_per_bucket = int(config.get("selected_per_bucket", FULL_DATASET_SELECTED_PER_BUCKET))
    seed_base = int(config.get("seed_base", 900_000))
    setting_name = str(config.get("setting_name", "custom_calendar_suite"))
    config_difficulty_scorer = str(config.get("difficulty_scorer", difficulty_scorer))
    difficulty_bucket_mode = str(config.get("difficulty_bucket_mode", DEFAULT_DIFFICULTY_BUCKET_MODE))

    if difficulty_bucket_mode == "cp_sat_tertile":
        tasks, summary = build_balanced_tasks_for_configs(
            configs,
            num_meetings=int(config.get("num_meetings", configs[0]["num_meetings"])),
            candidates_per_config=candidates_per_config,
            selected_per_bucket=selected_per_bucket,
            seed_base=seed_base,
            setting_name=setting_name,
            difficulty_scorer=config_difficulty_scorer,
            skip_optimal=skip_optimal,
        )
    else:
        raise ValueError("difficulty_bucket_mode must be 'cp_sat_tertile'")
    summary["source_config"] = {
        "seed_base": seed_base,
        "candidates_per_config": candidates_per_config,
        "selected_per_bucket": selected_per_bucket,
        "expanded_config_count": len(configs),
        "difficulty_bucket_mode": difficulty_bucket_mode,
        "blocked_errands_per_agent_values": config.get(
            "blocked_errands_per_agent_values",
            DEFAULT_BLOCKED_ERRANDS_PER_AGENT_VALUES,
        ),
    }
    if "description" in config:
        summary["description"] = config["description"]
    return tasks, summary


def assign_setting_normalized_difficulty(tasks: list[dict[str, Any]]) -> None:
    if not tasks:
        return
    scores = [float(task["difficulty_score"]) for task in tasks]
    min_score = min(scores)
    max_score = max(scores)
    for task in tasks:
        score = float(task["difficulty_score"])
        task["setting_normalized_difficulty"] = normalized_difficulty_value(
            score,
            min_score,
            max_score,
            str(task.get("difficulty_scorer", DEFAULT_DIFFICULTY_SCORER)),
        )


def task_id_for_balanced_task(index: int, config: dict[str, Any], bucket: str, bucket_idx: int) -> str:
    density_label = str(config["density"]).replace(".", "p")
    return (
        f"b{index:03d}_{bucket}_{config['total_agents']}a_"
        f"{config['subset_size']}p_d{density_label}_c{config['pref_level']}_{bucket_idx}"
    )


def write_jsonl(tasks: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, sort_keys=True))
            f.write("\n")
    return path


def write_json(data: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def count_evictions_for_assignments(
    calendars: list[list[Any]],
    meetings: list[dict[str, Any]],
    assignments: dict[str | int, int],
) -> int:
    working = calendars
    evictions = 0
    normalized = {int(meeting_id): slot for meeting_id, slot in assignments.items()}
    for meeting in meetings:
        slot = normalized[int(meeting["id"])]
        evictions += sum(
            1
            for agent_id in meeting["participants"]
            if isinstance(working[agent_id][slot], dict)
        )
        working, cost, _per_agent = apply_schedule(working, meeting, slot)
        if cost is None:
            raise ValueError("assignment is not feasible while counting evictions")
    return evictions


def validate_task(task: dict[str, Any], *, skip_optimal: bool = False) -> None:
    params = task["params"]
    calendars = task["calendars"]
    meetings = task["meetings"]
    if len(calendars) != params["total_agents"]:
        raise ValueError(f"{task['task_id']}: calendar count does not match total_agents")
    if any(len(calendar) != params["num_slots"] for calendar in calendars):
        raise ValueError(f"{task['task_id']}: calendar length does not match num_slots")
    if len(meetings) != params["num_meetings"]:
        raise ValueError(f"{task['task_id']}: meeting count does not match num_meetings")
    for meeting in meetings:
        participants = meeting["participants"]
        if len(participants) != params["subset_size"]:
            raise ValueError(f"{task['task_id']}: participant count does not match subset_size")
        if any(agent_id < 0 or agent_id >= params["total_agents"] for agent_id in participants):
            raise ValueError(f"{task['task_id']}: participant out of range")
        speaker_order = meeting.get("speaker_order")
        if speaker_order is None:
            raise ValueError(f"{task['task_id']}: missing speaker_order")
        if sorted(speaker_order) != sorted(participants):
            raise ValueError(f"{task['task_id']}: speaker_order must match participants")
        if meeting["duration"] != params["duration"]:
            raise ValueError(f"{task['task_id']}: duration mismatch")
    if params["num_meetings"] > 1:
        participant_counts = {agent_id: 0 for agent_id in range(params["total_agents"])}
        first_speaker_counts = {agent_id: 0 for agent_id in range(params["total_agents"])}
        for meeting in meetings:
            for agent_id in meeting["participants"]:
                participant_counts[agent_id] += 1
            first_speaker_counts[meeting["speaker_order"][0]] += 1
        appearances = params["num_meetings"] * params["subset_size"]
        if appearances % params["total_agents"] == 0:
            expected = appearances // params["total_agents"]
            if any(count != expected for count in participant_counts.values()):
                raise ValueError(
                    f"{task['task_id']}: participation counts are not exactly balanced"
                )
        if appearances % params["total_agents"] == 0:
            expected = appearances // params["total_agents"]
            if all(count == expected for count in participant_counts.values()):
                first_counts = list(first_speaker_counts.values())
                if max(first_counts) - min(first_counts) > 1:
                    raise ValueError(f"{task['task_id']}: first-speaker counts are imbalanced")
    if not task["feasible"]:
        raise ValueError(f"{task['task_id']}: generated task is infeasible")
    if not skip_optimal:
        if task["optimal"].get("cost") is None:
            raise ValueError(f"{task['task_id']}: missing optimal cost")
        if task.get("optimal_evictions") is None:
            raise ValueError(f"{task['task_id']}: missing optimal eviction count")
        if task.get("participant_slots") is None:
            raise ValueError(f"{task['task_id']}: missing participant slot count")
        if task.get("difficulty_scorer") is None:
            raise ValueError(f"{task['task_id']}: missing difficulty scorer")
        if task.get("difficulty_score") is None:
            raise ValueError(f"{task['task_id']}: missing difficulty score")
        if task.get("cost_per_total_agent") is None:
            raise ValueError(f"{task['task_id']}: missing cost_per_total_agent")
    if task["greedy"].get("cost") is None:
        raise ValueError(f"{task['task_id']}: missing greedy cost")
    if task["witness_solution"].get("cost") is None:
        raise ValueError(f"{task['task_id']}: missing witness cost")
    if params["density"] > 0 and task["witness_solution"]["cost"] <= 0:
        raise ValueError(f"{task['task_id']}: witness cost should be positive")
    assignments = {int(k): v for k, v in task["witness_solution"]["assignments"].items()}
    if params["density"] > 0:
        has_witness_errand = any(
            isinstance(calendars[agent_id][assignments[int(meeting["id"])]], dict)
            for meeting in meetings
            for agent_id in meeting["participants"]
        )
        if not has_witness_errand:
            raise ValueError(f"{task['task_id']}: witness assignments have no participant errands")
    blocked_per_agent = params.get("blocked_errands_per_agent")
    if blocked_per_agent is not None:
        for agent_id, calendar in enumerate(calendars):
            expected_blocked = (
                int(blocked_per_agent[agent_id])
                if isinstance(blocked_per_agent, list)
                else int(blocked_per_agent)
            )
            blocked_slots = [
                slot_idx
                for slot_idx, slot in enumerate(calendar)
                if isinstance(slot, dict) and slot.get("blocked")
            ]
            if len(blocked_slots) != expected_blocked:
                raise ValueError(
                    f"{task['task_id']}: agent {agent_id} has {len(blocked_slots)} "
                    f"blocked errands, expected {expected_blocked}"
                )
            if any("errand_id" not in calendar[slot_idx] for slot_idx in blocked_slots):
                raise ValueError(f"{task['task_id']}: blocked slots must be errands")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate calendar task fixtures.")
    parser.add_argument(
        "--config",
        help=(
            "YAML/JSON task-generation config. When set, generates a custom "
            "balanced suite and ignores --suite."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=[
            "initial-small",
            "balanced-v1",
            "balanced-uniform-v1",
            "balanced-variable-v1",
            "balanced-uniform-5meeting-v1",
            "balanced-preference-5meeting-v1",
            "uniform-full",
            "varied-full",
            "uniform-full-blocked",
            "varied-full-blocked",
            "minimal-cost-ratio-v1",
            "initial-prior-meeting-move-v1",
            "derive-balanced-cost-ratios-v1",
        ],
        default="initial-small",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path, relative to games/calendar when run there.",
    )
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--source",
        default=BALANCED_UNIFORM_SOURCE,
        help="Source JSONL for derived suites.",
    )
    parser.add_argument("--candidates-per-config", type=int, default=BALANCED_CANDIDATES_PER_CONFIG)
    parser.add_argument("--selected-per-bucket", type=int, default=BALANCED_SELECTED_PER_BUCKET)
    parser.add_argument("--blocked-errands-per-agent", type=int, default=DEFAULT_BLOCKED_ERRANDS_PER_AGENT)
    parser.add_argument(
        "--difficulty-scorer",
        choices=sorted(DIFFICULTY_SCORERS),
        default=DEFAULT_DIFFICULTY_SCORER,
    )
    parser.add_argument(
        "--skip-optimal",
        action="store_true",
        help="Skip the CP-SAT optimality solver. Tasks will lack optimal/difficulty fields but generate much faster.",
    )
    args = parser.parse_args()

    if args.config:
        config = _load_taskgen_config(args.config)
        tasks, summary = build_tasks_from_config(
            config,
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        output = args.output or config.get("output", "tasks/custom_calendar_suite.jsonl")
        default_summary = config.get("summary_output", str(Path(output).with_suffix("")) + "_summary.json")
    elif args.suite == "initial-small":
        tasks = [build_task(spec, difficulty_scorer=args.difficulty_scorer, skip_optimal=args.skip_optimal) for spec in INITIAL_SMALL_TASKS]
        output = args.output or "tasks/initial_small.jsonl"
        summary = None
    elif args.suite == "minimal-cost-ratio-v1":
        tasks, summary = build_minimal_cost_ratio_tasks(difficulty_scorer=args.difficulty_scorer)
        output = args.output or "tasks/minimal_cost_ratio_v1.jsonl"
        default_summary = "tasks/minimal_cost_ratio_v1_summary.json"
    elif args.suite == "initial-prior-meeting-move-v1":
        tasks, summary = build_initial_prior_meeting_move_tasks()
        output = args.output or "tasks/initial_prior_meeting_move_v1.jsonl"
        default_summary = "tasks/initial_prior_meeting_move_v1_summary.json"
    elif args.suite == "derive-balanced-cost-ratios-v1":
        if args.output or args.summary_output:
            raise ValueError("derive-balanced-cost-ratios-v1 writes three fixed output/summary file pairs")
        for multiplier in COST_RATIO_MULTIPLIERS:
            tasks, summary = build_balanced_cost_ratio_tasks(
                args.source,
                multiplier=multiplier,
                difficulty_scorer=args.difficulty_scorer,
            )
            output = f"tasks/balanced_uniform_errand_x{multiplier}_v1.jsonl"
            summary_output = f"tasks/balanced_uniform_errand_x{multiplier}_v1_summary.json"
            path = write_jsonl(tasks, output)
            summary_path = write_json(summary, summary_output)
            print(f"Wrote {len(tasks)} tasks to {path}")
            print(f"Wrote summary to {summary_path}")
        return 0
    elif args.suite == "balanced-uniform-5meeting-v1":
        tasks, summary = build_balanced_tasks_for_configs(
            five_meeting_configs(preference=False),
            num_meetings=FIVE_MEETING_NUM_MEETINGS,
            candidates_per_config=args.candidates_per_config,
            selected_per_bucket=args.selected_per_bucket,
            seed_base=120_000,
            setting_name="uniform_5meeting_v1",
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        summary["pref_levels"] = [1]
        summary["meeting_cost_range"] = [1, 1]
        summary["errand_cost_range"] = [1, 1]
        output = args.output or "tasks/balanced_uniform_5meeting_v1.jsonl"
        default_summary = "tasks/balanced_uniform_5meeting_v1_summary.json"
    elif args.suite == "balanced-preference-5meeting-v1":
        tasks, summary = build_balanced_tasks_for_configs(
            five_meeting_configs(preference=True),
            num_meetings=FIVE_MEETING_NUM_MEETINGS,
            candidates_per_config=args.candidates_per_config,
            selected_per_bucket=args.selected_per_bucket,
            seed_base=220_000,
            setting_name="preference_5meeting_v1",
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        summary["pref_levels"] = [3]
        summary["meeting_cost_range"] = [1, 1]
        summary["errand_cost_values"] = PREFERENCE_ERRAND_COST_VALUES
        summary["balanced_errand_cost_distribution_per_agent"] = True
        output = args.output or "tasks/balanced_preference_5meeting_v1.jsonl"
        default_summary = "tasks/balanced_preference_5meeting_v1_summary.json"
    elif args.suite == "uniform-full":
        tasks, summary = build_balanced_tasks_for_configs(
            five_meeting_configs(preference=False),
            num_meetings=FIVE_MEETING_NUM_MEETINGS,
            candidates_per_config=FULL_DATASET_CANDIDATES_PER_CONFIG,
            selected_per_bucket=FULL_DATASET_SELECTED_PER_BUCKET,
            seed_base=320_000,
            setting_name="uniform_full",
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        summary["pref_levels"] = [1]
        summary["meeting_cost_range"] = [1, 1]
        summary["errand_cost_range"] = [1, 1]
        output = args.output or "tasks/uniform_full.jsonl"
        default_summary = "tasks/uniform_summary.json"
    elif args.suite == "varied-full":
        tasks, summary = build_balanced_tasks_for_configs(
            five_meeting_configs(preference=True),
            num_meetings=FIVE_MEETING_NUM_MEETINGS,
            candidates_per_config=FULL_DATASET_CANDIDATES_PER_CONFIG,
            selected_per_bucket=FULL_DATASET_SELECTED_PER_BUCKET,
            seed_base=420_000,
            setting_name="varied_full",
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        summary["pref_levels"] = [3]
        summary["meeting_cost_range"] = [1, 1]
        summary["errand_cost_values"] = PREFERENCE_ERRAND_COST_VALUES
        summary["balanced_errand_cost_distribution_per_agent"] = True
        output = args.output or "tasks/varied_full.jsonl"
        default_summary = "tasks/varied_summary.json"
    elif args.suite == "uniform-full-blocked":
        source = "tasks/uniform_full.jsonl" if args.source == BALANCED_UNIFORM_SOURCE else args.source
        tasks, summary = build_blocked_errand_tasks(
            source,
            blocked_errands_per_agent=args.blocked_errands_per_agent,
            difficulty_scorer=args.difficulty_scorer,
            setting_prefix="uniform_full",
        )
        output = args.output or f"tasks/uniform_full_blocked{args.blocked_errands_per_agent}.jsonl"
        default_summary = f"tasks/uniform_full_blocked{args.blocked_errands_per_agent}_summary.json"
    elif args.suite == "varied-full-blocked":
        source = "tasks/varied_full.jsonl" if args.source == BALANCED_UNIFORM_SOURCE else args.source
        tasks, summary = build_blocked_errand_tasks(
            source,
            blocked_errands_per_agent=args.blocked_errands_per_agent,
            difficulty_scorer=args.difficulty_scorer,
            setting_prefix="varied_full",
        )
        output = args.output or f"tasks/varied_full_blocked{args.blocked_errands_per_agent}.jsonl"
        default_summary = f"tasks/varied_full_blocked{args.blocked_errands_per_agent}_summary.json"
    else:
        if args.suite == "balanced-uniform-v1":
            pref_levels = [1]
            setting_name = "uniform_cost_v1"
            default_output = "tasks/balanced_uniform_cost_v1.jsonl"
            default_summary = "tasks/balanced_uniform_cost_v1_summary.json"
        elif args.suite == "balanced-variable-v1":
            pref_levels = [3]
            setting_name = "variable_cost_v1"
            default_output = "tasks/balanced_variable_cost_v1.jsonl"
            default_summary = "tasks/balanced_variable_cost_v1_summary.json"
        else:
            pref_levels = BALANCED_PREF_LEVEL_VALUES
            setting_name = "combined_uniform_variable_cost_v1"
            default_output = "tasks/balanced_v1.jsonl"
            default_summary = "tasks/balanced_v1_summary.json"

        tasks, summary = build_balanced_tasks(
            candidates_per_config=args.candidates_per_config,
            selected_per_bucket=args.selected_per_bucket,
            pref_levels=pref_levels,
            setting_name=setting_name,
            difficulty_scorer=args.difficulty_scorer,
            skip_optimal=args.skip_optimal,
        )
        output = args.output or default_output

    path = write_jsonl(tasks, output)
    print(f"Wrote {len(tasks)} tasks to {path}")
    if summary is not None:
        summary_path = write_json(summary, args.summary_output or default_summary)
        print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
