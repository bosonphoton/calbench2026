"""Tests for the streaming IncrementalMAPClient."""

from __future__ import annotations

import json
from unittest.mock import ANY

from calendar_game.agents import GameConfig
from calendar_game.clients.imap import IncrementalMAPClient, _local_cost_table


def _make_config(agent_id: int, num_agents: int = 3, num_slots: int = 4) -> GameConfig:
    return GameConfig(
        num_agents=num_agents,
        num_slots=num_slots,
        agent_id=agent_id,
        all_agent_ids=list(range(num_agents)),
    )


def _meeting(participants: list[int] | None = None) -> dict:
    return {"id": 1, "participants": participants or [0, 1, 2], "cost": 1}


def _render(lines: dict[int, str]) -> str:
    return "\n".join(f"Slot {slot}: {content}" for slot, content in sorted(lines.items()))


def _imap_msg(sender: int, payload: dict, meeting_id: int = 1) -> dict:
    return {"from": sender, "meeting_id": meeting_id, "content": json.dumps(payload)}


def test_local_cost_table_treats_errands_as_movable_and_meetings_as_hard():
    client = IncrementalMAPClient()
    client.register(0, _make_config(0))
    client.observe_calendar(_render({
        0: "[FREE]",
        1: "Errand #1 (cost=4)",
        2: "Meeting M9 (cost=1)",
    }))

    assert _local_cost_table(client._slot_items) == {0: 0, 1: 4, 2: None}


def test_local_cost_table_treats_blocked_errands_as_hard():
    client = IncrementalMAPClient()
    client.register(0, _make_config(0))
    client.observe_calendar(_render({
        0: "Blocked #1 (cost=4)",
        1: "Errand #2 (cost=5)",
        2: "[FREE]",
    }))

    assert _local_cost_table(client._slot_items) == {0: None, 1: 5, 2: 0}


def test_initiator_requests_complete_cost_tables():
    client = IncrementalMAPClient()
    client.register(0, _make_config(0))
    client.start_round(_meeting(), _render({0: "[FREE]", 1: "Errand #1 (cost=5)"}), round_num=0)

    result = client.turn([])

    assert {tool["to"] for tool in result.tool_calls} == {1, 2}
    payload = json.loads(result.tool_calls[0]["content"])
    assert payload["imap"] == "cost_request"
    assert payload["slots"] == [0, 1]


def test_responder_returns_complete_cost_table():
    client = IncrementalMAPClient()
    client.register(1, _make_config(1))
    client.start_round(_meeting(), _render({
        0: "Errand #1 (cost=7)",
        1: "[FREE]",
        2: "Meeting M3 (cost=1)",
    }), round_num=0)

    result = client.turn([_imap_msg(0, {"imap": "cost_request", "slots": [0, 1, 2]})])
    payload = json.loads(result.tool_calls[0]["content"])

    assert payload["imap"] == "costs"
    assert payload["costs"] == {"0": 7, "1": 0, "2": None}


def test_initiator_selects_minimum_total_current_insertion_cost():
    client = IncrementalMAPClient()
    client.register(0, _make_config(0))
    client.start_round(_meeting(), _render({
        0: "Errand #1 (cost=8)",
        1: "Errand #2 (cost=1)",
        2: "[FREE]",
    }), round_num=0)
    client.turn([])

    decision = client.turn([
        _imap_msg(1, {"imap": "costs", "costs": {"0": 1, "1": 9, "2": 5}}),
        _imap_msg(2, {"imap": "costs", "costs": {"0": 1, "1": 1, "2": 10}}),
    ])

    # Totals: slot 0 = 10, slot 1 = 11, slot 2 = 15.
    assert client._agreed_slot == 0
    assert len(decision.tool_calls) == 2
    assert all(json.loads(tool["content"])["slot"] == 0 for tool in decision.tool_calls)


def test_decide_schedules_agreed_slot_with_local_reschedule():
    client = IncrementalMAPClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _meeting(participants=[0, 1])
    render = _render({0: "Errand #1 (cost=7)", 1: "[FREE]"})
    client.start_round(meeting, render, round_num=0)
    client.turn([_imap_msg(0, {"imap": "decision", "slot": 0})])

    result = client.decide(meeting, render)

    assert result.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 0, "to_slot": 1, "justification": ANY},
        {"type": "schedule", "meeting_id": 1, "slot": 0},
    ]


def test_decide_does_not_reschedule_blocked_errand_at_agreed_slot():
    client = IncrementalMAPClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _meeting(participants=[0, 1])
    render = _render({0: "Blocked #1 (cost=7)", 1: "[FREE]"})
    client.start_round(meeting, render, round_num=0)
    client.turn([_imap_msg(0, {"imap": "decision", "slot": 0})])

    result = client.decide(meeting, render)

    assert result.tool_calls == []
