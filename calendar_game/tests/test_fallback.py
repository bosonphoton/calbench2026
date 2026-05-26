"""Tests for deterministic fallback scheduling."""

from __future__ import annotations

from types import SimpleNamespace

from calendar_game.calendar import Calendar
from calendar_game.fallback import find_fallback_slot


def _agent(slots: list[dict | None]) -> SimpleNamespace:
    calendar = Calendar(num_slots=len(slots))
    calendar.slots = slots
    return SimpleNamespace(calendar=calendar)


def test_fallback_does_not_displace_blocked_errands():
    agents = [
        _agent([
            {"blocked": True, "errand_id": 1, "cost": 1},
            None,
        ]),
        _agent([
            None,
            {"errand_id": 2, "cost": 1},
        ]),
    ]
    meeting = {"id": 1, "participants": [0, 1], "cost": 1}

    chosen_slot, plan = find_fallback_slot(
        agents,
        meeting,
        num_slots=2,
        meeting_registry={},
        scenario_meetings=[],
    )

    assert chosen_slot == 1
    assert plan == [
        {
            "agent_id": 1,
            "item_id": 2,
            "from_slot": 1,
            "to_slot": 0,
            "is_meeting_cascade": False,
        }
    ]
