from collections import Counter

from calendar_game.game import CalendarGame, CalendarGameConfig
from calendar_game.scenario import generate_scenario
from calendar_game.taskgen import build_task, select_balanced_participant_lists


def test_five_agent_three_participant_five_meeting_tasks_balance_participation_and_first_speaker():
    task = build_task({
        "task_id": "balanced_probe",
        "seed": 12345,
        "total_agents": 5,
        "subset_size": 3,
        "density": 0.6,
        "pref_level": 1,
        "num_meetings": 5,
    })

    participation = Counter(
        agent_id
        for meeting in task["meetings"]
        for agent_id in meeting["participants"]
    )
    first_speakers = Counter(meeting["speaker_order"][0] for meeting in task["meetings"])

    assert participation == Counter({0: 3, 1: 3, 2: 3, 3: 3, 4: 3})
    assert first_speakers == Counter({0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
    for meeting in task["meetings"]:
        assert sorted(meeting["speaker_order"]) == meeting["participants"]


def test_two_agent_five_meeting_first_speaker_is_as_balanced_as_possible():
    task = build_task({
        "task_id": "two_agent_probe",
        "seed": 222,
        "total_agents": 2,
        "subset_size": 2,
        "density": 0.6,
        "pref_level": 1,
        "num_meetings": 5,
    })

    first_counts = Counter(meeting["speaker_order"][0] for meeting in task["meetings"])

    assert Counter(
        agent_id
        for meeting in task["meetings"]
        for agent_id in meeting["participants"]
    ) == Counter({0: 5, 1: 5})
    assert max(first_counts.values()) - min(first_counts.values()) == 1


def test_game_uses_speaker_order_for_first_turn_order():
    scenario = generate_scenario(
        seed=7,
        num_agents=3,
        num_slots=8,
        density=0.0,
        pref_level=1,
        num_meetings=1,
        participant_lists=[[0, 1, 2]],
        speaker_orders=[[2, 0, 1]],
    )
    game = CalendarGame(
        CalendarGameConfig(
            seed=7,
            num_agents=3,
            num_slots=8,
            density=0.0,
            pref_level=1,
            num_meetings=1,
            max_turns_per_round=1,
        ),
        dry_run=True,
    )

    trace = game.run_with_scenario(scenario)
    first_turn_agents = [
        event.data["agent_id"]
        for event in trace.events
        if event.type == "turn_start"
        and event.data["phase"] == "CHEAP_TALK"
        and event.data["round"] == 0
        and event.data["turn"] == 0
    ][:3]

    assert first_turn_agents == [2, 0, 1]


def test_game_adds_balanced_speaker_order_for_legacy_scenarios():
    scenario = generate_scenario(
        seed=11,
        num_agents=5,
        num_slots=8,
        density=0.0,
        pref_level=1,
        num_meetings=5,
        participant_lists=[
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ],
    )
    for meeting in scenario["meetings"]:
        meeting.pop("speaker_order", None)
    game = CalendarGame(
        CalendarGameConfig(
            seed=11,
            num_agents=5,
            num_slots=8,
            density=0.0,
            pref_level=1,
            num_meetings=5,
            max_turns_per_round=0,
        ),
        dry_run=True,
    )

    trace = game.run_with_scenario(scenario)
    first_speakers = Counter(
        event.data["speaker_order"][0]
        for event in trace.events
        if event.type == "round_start"
    )

    assert first_speakers == Counter({0: 1, 1: 1, 2: 1, 3: 1, 4: 1})


def test_balanced_participant_lists_are_exact_when_divisible():
    participant_lists = select_balanced_participant_lists(
        seed=99,
        total_agents=5,
        subset_size=3,
        num_meetings=5,
    )

    assert Counter(
        agent_id
        for participants in participant_lists
        for agent_id in participants
    ) == Counter({0: 3, 1: 3, 2: 3, 3: 3, 4: 3})
