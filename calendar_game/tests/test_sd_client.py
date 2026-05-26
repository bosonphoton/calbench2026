"""Tests for the Scheduling Difficulty IL-MAP baseline client."""

from __future__ import annotations

import json
from unittest.mock import ANY

from calendar_game.agents import GameConfig
from calendar_game.clients.sd import MeetingStatus, SDClient


def _make_config(agent_id: int, num_agents: int = 3, num_slots: int = 4) -> GameConfig:
    return GameConfig(
        num_agents=num_agents,
        num_slots=num_slots,
        agent_id=agent_id,
        all_agent_ids=list(range(num_agents)),
        sd_model={0: 1.0, 1: 1.0, 2: 5.0},
    )


def _meeting(meeting_id: int = 1, participants: list[int] | None = None) -> dict:
    return {"id": meeting_id, "participants": participants or [0, 1], "cost": 1}


def _render(lines: dict[int, str]) -> str:
    return "\n".join(f"Slot {slot}: {content}" for slot, content in sorted(lines.items()))


def _sd_msg(sender: int, payload: dict, meeting_id: int = 1) -> dict:
    return {"from": sender, "meeting_id": meeting_id, "content": json.dumps(payload)}


def _payload(tool: dict) -> dict:
    return json.loads(tool["content"])


def test_initiator_sends_propose_for_next_local_slot():
    client = SDClient()
    client.register(0, _make_config(0))
    client.start_round(_meeting(), _render({0: "[FREE]", 1: "[FREE]"}), round_num=0)

    result = client.turn([])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["to"] == 1
    assert _payload(result.tool_calls[0]) == {"sd": "propose", "meeting_id": 1, "slot": 0, "initiator": 0}
    assert client._status(1, 0) == MeetingStatus.PENDING


def test_initiator_skips_blocked_errand_when_proposing():
    client = SDClient()
    client.register(0, _make_config(0))
    client.start_round(_meeting(), _render({
        0: "Blocked #1 (cost=3)",
        1: "[FREE]",
    }), round_num=0)

    result = client.turn([])

    assert _payload(result.tool_calls[0]) == {"sd": "propose", "meeting_id": 1, "slot": 1, "initiator": 0}


def test_responder_rejects_when_slot_has_pending_meeting():
    client = SDClient()
    client.register(1, _make_config(1))
    client.start_round(_meeting(), _render({0: "[FREE]", 1: "[FREE]"}), round_num=0)
    client._set_status(99, 0, MeetingStatus.PENDING)

    result = client.turn([_sd_msg(0, {"sd": "propose", "meeting_id": 1, "slot": 0, "initiator": 0})])

    payload = _payload(result.tool_calls[0])
    assert payload["sd"] == "reply"
    assert payload["reply_status"] == MeetingStatus.IMPOSSIBLE


def test_responder_rejects_blocked_errand_slot():
    client = SDClient()
    client.register(1, _make_config(1))
    client.start_round(_meeting(), _render({
        0: "Blocked #1 (cost=3)",
        1: "[FREE]",
    }), round_num=0)

    result = client.turn([_sd_msg(0, {"sd": "propose", "meeting_id": 1, "slot": 0, "initiator": 0})])

    payload = _payload(result.tool_calls[0])
    assert payload["sd"] == "reply"
    assert payload["reply_status"] == MeetingStatus.IMPOSSIBLE


def test_responder_bumps_lower_difficulty_confirmed_meeting():
    client = SDClient()
    client.register(1, _make_config(1))
    client._known_meetings[7] = [0, 1]
    client.start_round(
        _meeting(2, [1, 2]),
        _render({0: "Meeting M7 (cost=1) participants=[0, 1]", 1: "[FREE]"}),
        round_num=1,
    )

    result = client.turn([_sd_msg(2, {"sd": "propose", "meeting_id": 2, "slot": 0, "initiator": 2}, meeting_id=2)])

    assert _payload(result.tool_calls[0])["reply_status"] == MeetingStatus.PENDING
    assert client._status(7, 0) == MeetingStatus.BUMPED
    assert client._status(2, 0) == MeetingStatus.PENDING


def test_failed_proposal_reverts_tentative_bump_to_confirmed():
    client = SDClient()
    client.register(1, _make_config(1))
    client._known_meetings[7] = [0, 1]
    client.start_round(
        _meeting(2, [1, 2]),
        _render({0: "Meeting M7 (cost=1) participants=[0, 1]", 1: "[FREE]"}),
        round_num=1,
    )
    client.turn([_sd_msg(2, {"sd": "propose", "meeting_id": 2, "slot": 0, "initiator": 2}, meeting_id=2)])

    client.turn([_sd_msg(2, {"sd": "fail", "meeting_id": 2}, meeting_id=2)])

    assert client._status(2, 0) == MeetingStatus.POSSIBLE
    assert client._status(7, 0) == MeetingStatus.CONFIRMED


def test_confirmed_bump_requests_repair_before_decide_moves_old_meeting():
    client = SDClient()
    client.register(1, _make_config(1, num_agents=3, num_slots=3))
    client._known_meetings[7] = [0, 1]
    meeting = _meeting(2, [1, 2])
    render = _render({0: "Meeting M7 (cost=1) participants=[0, 1]", 1: "[FREE]", 2: "[FREE]"})
    client.start_round(meeting, render, round_num=1)
    client.turn([_sd_msg(2, {"sd": "propose", "meeting_id": 2, "slot": 0, "initiator": 2}, meeting_id=2)])

    result = client.turn([_sd_msg(2, {"sd": "confirm", "meeting_id": 2, "slot": 0}, meeting_id=2)])

    assert [_payload(tool) for tool in result.tool_calls] == [
        {"sd": "reschedule_request", "meeting_id": 7, "from_slot": 0}
    ]
    client.turn([_sd_msg(0, {"sd": "confirm_reschedule", "meeting_id": 7, "from_slot": 0, "to_slot": 1})])
    assert client.decide(meeting, render).tool_calls == [
        {"type": "reschedule", "item_id": 7, "from_slot": 0, "to_slot": 1, "justification": ANY},
        {"type": "schedule", "meeting_id": 2, "slot": 0},
    ]


def test_bumped_meeting_initiator_coordinates_repair_via_cheap_talk():
    initiator = SDClient()
    initiator.register(0, _make_config(0, num_agents=3, num_slots=3))
    initiator._known_meetings[7] = [0, 1]
    initiator.observe_calendar(_render({
        0: "Meeting M7 (cost=1) participants=[0, 1]",
        1: "[FREE]",
        2: "[FREE]",
    }))

    proposal = initiator.turn([
        _sd_msg(1, {"sd": "reschedule_request", "meeting_id": 7, "from_slot": 0}, meeting_id=2)
    ])

    assert [_payload(tool) for tool in proposal.tool_calls] == [
        {"sd": "propose_reschedule", "meeting_id": 7, "from_slot": 0, "to_slot": 1, "initiator": 0}
    ]

    confirm = initiator.turn([
        _sd_msg(1, {
            "sd": "reschedule_reply",
            "meeting_id": 7,
            "from_slot": 0,
            "to_slot": 1,
            "reply_status": "PENDING",
            "attendee": 1,
        }, meeting_id=2)
    ])

    assert [_payload(tool) for tool in confirm.tool_calls] == [
        {"sd": "confirm_reschedule", "meeting_id": 7, "from_slot": 0, "to_slot": 1}
    ]
