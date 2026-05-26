"""Tests for the DSMClient scripted protocol client."""

from __future__ import annotations

import json
from unittest.mock import ANY

import pytest

from calendar_game.agents import GameConfig
from calendar_game.clients.dsm import (
    DSMClient,
    PaperDSMClient,
    PrivateDSMClient,
    _D,
    _parse_slot_items,
    _parse_slot_states,
    _score_slot,
    _score_summary,
    _scoring_cost,
)

LOWEST_FEASIBLE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(agent_id: int, num_agents: int = 3, num_slots: int = 8, **kwargs) -> GameConfig:
    return GameConfig(
        num_agents=num_agents,
        num_slots=num_slots,
        agent_id=agent_id,
        all_agent_ids=list(range(num_agents)),
        **kwargs,
    )


def _make_meeting(meeting_id: int = 1, participants: list[int] | None = None) -> dict:
    return {"id": meeting_id, "participants": participants or [0, 1, 2], "cost": 1}


def _render(slot_states: dict[int, str]) -> str:
    """Build a minimal calendar render from a {slot: 'free'|'busy'} dict."""
    width = len(str(max(slot_states.keys())))
    lines = []
    for i in sorted(slot_states.keys()):
        label = str(i).rjust(width)
        content = "[FREE]" if slot_states[i] == "free" else "Errand #1 (cost=10)"
        lines.append(f"Slot {label}: {content}")
    return "\n".join(lines)


def _dsm_msg(sender: int, payload: dict, meeting_id: int = 1) -> dict:
    return {"from": sender, "meeting_id": meeting_id, "content": json.dumps(payload)}


def _render_lines(lines: dict[int, str]) -> str:
    return "\n".join(f"Slot {slot}: {content}" for slot, content in sorted(lines.items()))


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

def test_parse_slot_states_basic():
    render = _render({0: "free", 1: "busy", 2: "free"})
    states = _parse_slot_states(render)
    assert states == {0: "free", 1: "busy", 2: "free"}


def test_parse_slot_items_distinguishes_blocked_errands_from_movable_errands():
    render = _render_lines({
        0: "Blocked #7 (cost=4)",
        1: "Errand #8 (cost=5)",
        2: "[FREE]",
    })

    assert _parse_slot_items(render) == {
        0: {"type": "blocked", "item_id": 7, "cost": 4},
        1: {"type": "errand", "item_id": 8, "cost": 5},
        2: None,
    }


def test_score_slot():
    states = {0: "free", 1: "busy"}
    assert _score_slot(0, states) == _D - 1
    assert _score_slot(1, states) == 0


def test_score_slot_treats_blocked_errand_as_infeasible():
    items = _parse_slot_items(_render_lines({
        0: "Blocked #7 (cost=4)",
        1: "[FREE]",
    }))

    assert _score_slot(0, items) == 0
    assert _score_slot(1, items) == _D - 1


# ---------------------------------------------------------------------------
# Initiator: proposing → waiting_scores → decided
# ---------------------------------------------------------------------------

def test_initiator_sends_proposals_on_first_turn():
    client = DSMClient()
    client.register(0, _make_config(0))
    meeting = _make_meeting(participants=[0, 1, 2])
    render = _render({0: "free", 1: "busy", 2: "free", 3: "free", 4: "free", 5: "busy"})
    client.start_round(meeting, render, round_num=1)

    assert client._role == "initiator"
    result = client.turn([])
    assert len(result.tool_calls) == 2  # one DM per responder (1 and 2)
    recipients = {tc["to"] for tc in result.tool_calls}
    assert recipients == {1, 2}
    # All DMs carry the same proposals payload
    for tc in result.tool_calls:
        msg = json.loads(tc["content"])
        assert msg["dsm"] == "proposals"
        assert msg["slots"] == [0, 2, 3, 4]  # highest local satisfaction first


def test_initiator_waits_and_then_decides_on_all_scores():
    client = DSMClient()
    client.register(0, _make_config(0))
    meeting = _make_meeting(participants=[0, 1, 2])
    # Slots 0 and 2 free for initiator
    render = _render({0: "free", 1: "busy", 2: "free", 3: "busy"})
    client.start_round(meeting, render, round_num=1)

    # Turn 1: initiator sends proposals
    result1 = client.turn([])
    assert client._state == "waiting_scores"
    proposals = json.loads(result1.tool_calls[0]["content"])["slots"]

    # Turn 2: only one responder has replied — still waiting
    partial_scores = {str(s): (_D - 1 if s == proposals[0] else 0) for s in proposals}
    result2 = client.turn([_dsm_msg(1, {"dsm": "scores", "scores": partial_scores})])
    assert client._state == "waiting_scores"
    assert result2.tool_calls == []

    # Turn 3: second responder replies — now assess and announce
    result3 = client.turn([_dsm_msg(2, {"dsm": "scores", "scores": partial_scores})])
    assert client._state == "decided"
    assert client._agreed_slot is not None
    # Decision DMs sent to both responders
    assert len(result3.tool_calls) == 2
    for tc in result3.tool_calls:
        msg = json.loads(tc["content"])
        assert msg["dsm"] == "decision"
        assert msg["slot"] == client._agreed_slot


def test_initiator_prefers_fully_feasible_slot():
    client = DSMClient()
    client.register(0, _make_config(0, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    # Initiator is free at slots 0 and 1
    render = _render({0: "free", 1: "free"})
    client.start_round(meeting, render, round_num=1)

    client.turn([])  # proposals sent: [0, 1]

    # Responder says slot 0 is infeasible, slot 1 is feasible
    scores = {"0": 0, "1": _D - 1}
    client.turn([_dsm_msg(1, {"dsm": "scores", "scores": scores})])

    # Should pick slot 1 (only fully feasible)
    assert client._agreed_slot == 1


def test_initiator_tries_next_batch_when_no_fully_feasible_slot():
    client = DSMClient()
    client.register(0, _make_config(0, num_slots=6))
    meeting = _make_meeting(participants=[0, 1, 2])
    render = _render({0: "free", 1: "free", 2: "free", 3: "free", 4: "free", 5: "free"})
    client.start_round(meeting, render, round_num=1)

    first = client.turn([])
    first_proposals = json.loads(first.tool_calls[0]["content"])["slots"]
    assert first_proposals == [0, 1, 2, 3]

    infeasible_scores = {str(s): 0 for s in first_proposals}
    client.turn([_dsm_msg(1, {"dsm": "scores", "scores": infeasible_scores, "round": 1})])
    second = client.turn([_dsm_msg(2, {"dsm": "scores", "scores": infeasible_scores, "round": 1})])

    assert client._state == "waiting_scores"
    assert len(second.tool_calls) == 2
    for tc in second.tool_calls:
        msg = json.loads(tc["content"])
        assert msg["dsm"] == "proposals"
        assert msg["slots"] == [4, 5]
        assert msg["round"] == 2

    feasible_scores = {"4": 0, "5": _D - 1}
    client.turn([_dsm_msg(1, {"dsm": "scores", "scores": feasible_scores, "round": 2})])
    decision = client.turn([_dsm_msg(2, {"dsm": "scores", "scores": feasible_scores, "round": 2})])

    assert client._state == "decided"
    assert client._agreed_slot == 5
    for tc in decision.tool_calls:
        msg = json.loads(tc["content"])
        assert msg["dsm"] == "decision"
        assert msg["slot"] == 5


def test_initiator_no_free_slots_decides_immediately():
    client = DSMClient()
    client.register(0, _make_config(0))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "busy", 1: "busy"})
    client.start_round(meeting, render, round_num=1)

    result = client.turn([])
    assert client._state == "decided"
    assert client._agreed_slot is None
    assert result.tool_calls == []


def test_initiator_single_participant_decides_without_dms():
    client = DSMClient()
    client.register(0, _make_config(0, num_agents=1))
    meeting = _make_meeting(participants=[0])
    render = _render({0: "free", 1: "busy"})
    client.start_round(meeting, render, round_num=1)

    result = client.turn([])
    assert client._state == "decided"
    assert client._agreed_slot == 0
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Responder: waiting_proposal → scored → decided
# ---------------------------------------------------------------------------

def test_responder_scores_and_returns_scores():
    client = DSMClient()
    client.register(1, _make_config(1))
    meeting = _make_meeting(participants=[0, 1, 2])
    render = _render({0: "free", 1: "busy", 2: "free"})
    client.start_round(meeting, render, round_num=1)

    assert client._role == "responder"

    # Turn 1: no messages yet — pass
    result1 = client.turn([])
    assert result1.tool_calls == []

    # Turn 2: receive proposals from initiator (agent 0)
    proposals = [0, 1, 2]
    result2 = client.turn([_dsm_msg(0, {"dsm": "proposals", "slots": proposals})])
    assert client._state == "scored"
    assert len(result2.tool_calls) == 1
    tc = result2.tool_calls[0]
    assert tc["to"] == 0  # DM sent back to initiator
    scores = json.loads(tc["content"])
    assert scores["dsm"] == "scores"
    assert scores["scores"]["0"] == _D - 1   # slot 0 free → top score
    assert scores["scores"]["1"] == LOWEST_FEASIBLE  # slot 1 has a costly movable errand
    assert scores["scores"]["2"] == _D - 1   # slot 2 free → top score


def test_responder_records_decision():
    client = DSMClient()
    client.register(2, _make_config(2))
    meeting = _make_meeting(participants=[0, 1, 2])
    render = _render({0: "free", 1: "free", 2: "busy"})
    client.start_round(meeting, render, round_num=1)

    # Receive proposal, send scores
    client.turn([_dsm_msg(0, {"dsm": "proposals", "slots": [0, 1]})])
    # Receive decision
    client.turn([_dsm_msg(0, {"dsm": "decision", "slot": 1})])

    assert client._agreed_slot == 1
    assert client._state == "decided"


def test_responder_rescores_new_proposals_after_scoring():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "busy", 1: "free", 2: "free"})
    client.start_round(meeting, render, round_num=1)

    client.turn([_dsm_msg(0, {"dsm": "proposals", "slots": [0], "round": 1})])
    result = client.turn([_dsm_msg(0, {"dsm": "proposals", "slots": [1, 2], "round": 2})])

    assert client._state == "scored"
    assert len(result.tool_calls) == 1
    scores = json.loads(result.tool_calls[0]["content"])
    assert scores["dsm"] == "scores"
    assert scores["round"] == 2
    assert scores["scores"] == {"1": _D - 1, "2": _D - 1}


def test_responder_scores_reflect_displacement_cost():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2, num_slots=4))
    meeting = _make_meeting(participants=[0, 1])
    render = _render_lines({
        0: "[FREE]",
        1: "Errand #1 (cost=1)",
        2: "Errand #2 (cost=10)",
        3: "[FREE]",
    })
    client.start_round(meeting, render, round_num=1)

    result = client.turn([_dsm_msg(0, {"dsm": "proposals", "slots": [0, 1, 2]})])
    scores = json.loads(result.tool_calls[0]["content"])["scores"]

    assert scores["0"] == _D - 1
    assert scores["1"] > scores["2"] > 0
    assert scores["2"] == LOWEST_FEASIBLE


def test_initiator_uses_cost_aware_aggregate_score():
    client = DSMClient()
    client.register(0, _make_config(0, num_agents=2, num_slots=3))
    meeting = _make_meeting(participants=[0, 1])
    render = _render_lines({
        0: "[FREE]",
        1: "[FREE]",
        2: "[FREE]",
    })
    client.start_round(meeting, render, round_num=1)
    client.turn([])

    # Both slots are feasible, but responder strongly prefers slot 1.
    client.turn([_dsm_msg(1, {"dsm": "scores", "scores": {"0": LOWEST_FEASIBLE, "1": _D - 1}, "round": 1})])

    assert client._agreed_slot == 1


def test_initiator_includes_prior_meeting_reschedule_when_coparticipants_are_present():
    client = DSMClient()
    client.register(0, _make_config(0, num_agents=3, num_slots=10))
    client.start_round(_make_meeting(meeting_id=1, participants=[0, 1]), _render({0: "free"}), round_num=1)

    meeting = _make_meeting(meeting_id=2, participants=[0, 1, 2])
    render = _render_lines({
        5: "Meeting M1 (cost=1)",
        8: "[FREE]",
    })
    client.start_round(meeting, render, round_num=2)

    first = client.turn([])
    proposals = json.loads(first.tool_calls[0]["content"])["slots"]
    assert proposals == [8, 5]

    decision = client.turn([_dsm_msg(2, {"dsm": "scores", "scores": {"5": _D - 1, "8": LOWEST_FEASIBLE}, "round": 1})])
    assert len(decision.tool_calls) == 0

    decision = client.turn([_dsm_msg(1, {"dsm": "scores", "scores": {"5": _D - 1, "8": LOWEST_FEASIBLE}, "round": 1})])
    assert len(decision.tool_calls) == 2
    by_recipient = {tc["to"]: json.loads(tc["content"]) for tc in decision.tool_calls}
    assert by_recipient[2]["dsm"] == "decision"
    assert by_recipient[2]["slot"] == 5
    assert by_recipient[2]["displacements"] == []
    assert by_recipient[2]["cleared_slots"] == [5]
    assert by_recipient[1]["dsm"] == "decision"
    assert by_recipient[1]["displacements"] == [{"meeting_id": 1, "from_slot": 5, "to_slot": 8}]


def test_responder_decision_reschedules_prior_meeting_to_requested_target():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2, num_slots=10))
    client.start_round(_make_meeting(meeting_id=1, participants=[0, 1]), _render({0: "free"}), round_num=1)

    meeting = _make_meeting(meeting_id=2, participants=[0, 1])
    render = _render_lines({
        5: "Meeting M1 (cost=1)",
        8: "[FREE]",
    })
    client.start_round(meeting, render, round_num=2)
    client.turn([_dsm_msg(0, {
        "dsm": "decision",
        "slot": 5,
        "displacements": [{"meeting_id": 1, "from_slot": 5, "to_slot": 8}],
    }, meeting_id=2)])

    result = client.decide(meeting, render)
    assert result.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 5, "to_slot": 8, "justification": ANY},
        {"type": "schedule", "meeting_id": 2, "slot": 5},
    ]


def test_voluntary_decide_honors_prior_meeting_reschedule_request():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=3, num_slots=10))
    client.turn([_dsm_msg(0, {"dsm": "reschedule_request", "meeting_id": 1, "from_slot": 5, "to_slot": 8}, meeting_id=2)])

    result = client.voluntary_decide(
        _make_meeting(meeting_id=2, participants=[0, 2]),
        _render_lines({5: "Meeting M1 (cost=1)", 8: "[FREE]"}),
    )
    assert result.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 5, "to_slot": 8, "justification": ANY}
    ]


def test_initiator_expands_responders_for_nonparticipant_meeting_displacement():
    initiator = DSMClient()
    displaced_participant = DSMClient()
    current_responder = DSMClient()
    for agent_id, client in enumerate([initiator, displaced_participant, current_responder]):
        client.register(agent_id, _make_config(agent_id, num_agents=3, num_slots=10))

    meeting1 = _make_meeting(meeting_id=1, participants=[0, 1])
    initial_render = _render_lines({5: "[FREE]", 8: "[FREE]"})
    initiator.start_round(meeting1, initial_render, round_num=1)
    displaced_participant.start_round(meeting1, initial_render, round_num=1)
    initiator.decide(meeting1, initial_render)
    displaced_participant.decide(meeting1, initial_render)

    meeting2 = _make_meeting(meeting_id=2, participants=[0, 2])
    initiator.start_round(meeting2, _render_lines({5: "Meeting M1 (cost=1)", 8: "[FREE]"}), round_num=2)

    proposal = initiator.turn([])
    recipients = {tc["to"] for tc in proposal.tool_calls}
    assert recipients == {1, 2}

    proposal_to_1 = next(tc for tc in proposal.tool_calls if tc["to"] == 1)
    proposal_to_2 = next(tc for tc in proposal.tool_calls if tc["to"] == 2)
    proposal_payload = json.loads(proposal_to_1["content"])
    assert proposal_payload["displacements"]["5"] == [{"meeting_id": 1, "from_slot": 5, "to_slot": 8}]

    score_from_1 = displaced_participant.turn([
        {"from": 0, "meeting_id": 2, "content": proposal_to_1["content"]}
    ])
    score_payload_1 = json.loads(score_from_1.tool_calls[0]["content"])
    assert score_payload_1["scores"]["5"] > 0

    score_from_2 = _dsm_msg(2, {"dsm": "scores", "scores": {"5": _D - 1, "8": LOWEST_FEASIBLE}, "round": 1}, meeting_id=2)
    decision = initiator.turn([
        {"from": 1, "meeting_id": 2, "content": score_from_1.tool_calls[0]["content"]},
        score_from_2,
    ])

    decision_recipients = {tc["to"] for tc in decision.tool_calls}
    assert decision_recipients == {1, 2}
    decision_to_1 = next(tc for tc in decision.tool_calls if tc["to"] == 1)
    displaced_participant.turn([
        {"from": 0, "meeting_id": 2, "content": decision_to_1["content"]}
    ])
    voluntary = displaced_participant.voluntary_decide(
        meeting2,
        _render_lines({5: "Meeting M1 (cost=1)", 8: "[FREE]"}),
    )
    assert voluntary.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 5, "to_slot": 8, "justification": ANY}
    ]


def test_responder_ignores_non_initiator_messages():
    client = DSMClient()
    client.register(2, _make_config(2))
    meeting = _make_meeting(participants=[0, 1, 2])
    render = _render({0: "free"})
    client.start_round(meeting, render, round_num=1)

    # Message from non-initiator (agent 1) — should be ignored
    result = client.turn([_dsm_msg(1, {"dsm": "proposals", "slots": [0]})])
    assert result.tool_calls == []
    assert client._state == "waiting_proposal"


# ---------------------------------------------------------------------------
# decide() / retry_decide()
# ---------------------------------------------------------------------------

def test_decide_uses_agreed_slot():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "free", 1: "free"})
    client.start_round(meeting, render, round_num=1)
    client._agreed_slot = 1

    result = client.decide(meeting, render)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["slot"] == 1


def test_decide_falls_back_when_agreed_slot_busy():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "free", 1: "busy"})
    client.start_round(meeting, render, round_num=1)
    client._agreed_slot = 1  # busy by the time decide() is called

    result = client.decide(meeting, render)
    assert result.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 1, "to_slot": 0, "justification": ANY},
        {"type": "schedule", "meeting_id": 1, "slot": 1},
    ]


def test_decide_reschedules_errand_when_agreed_slot_is_clearable():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "busy", 1: "free"})
    client.start_round(meeting, render, round_num=1)
    client._agreed_slot = 0

    result = client.decide(meeting, render)
    assert result.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 0, "to_slot": 1, "justification": ANY},
        {"type": "schedule", "meeting_id": 1, "slot": 0},
    ]


def test_decide_does_not_reschedule_blocked_errand_at_agreed_slot():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render_lines({
        0: "Blocked #1 (cost=7)",
        1: "[FREE]",
    })
    client.start_round(meeting, render, round_num=1)
    client._agreed_slot = 0

    result = client.decide(meeting, render)
    assert result.tool_calls == [
        {"type": "schedule", "meeting_id": 1, "slot": 1},
    ]


def test_decide_empty_when_no_free_slots():
    client = DSMClient()
    client.register(1, _make_config(1, num_agents=2))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "busy", 1: "busy"})
    client.start_round(meeting, render, round_num=1)

    result = client.decide(meeting, render)
    assert result.tool_calls == []


def test_retry_decide_cycles_free_slots():
    client = DSMClient()
    client.register(0, _make_config(0))
    meeting = _make_meeting(participants=[0, 1])
    render = _render({0: "free", 1: "busy", 2: "free", 3: "free"})
    client.start_round(meeting, render, round_num=1)

    r0 = client.retry_decide(0, 3, "conflict")
    r1 = client.retry_decide(1, 3, "conflict")
    r2 = client.retry_decide(2, 3, "conflict")
    r3 = client.retry_decide(3, 3, "conflict")

    assert r0.tool_calls[0]["slot"] == 0
    assert r1.tool_calls == [
        {"type": "reschedule", "item_id": 1, "from_slot": 1, "to_slot": 0, "justification": ANY},
        {"type": "schedule", "meeting_id": 1, "slot": 1},
    ]
    assert r2.tool_calls[0]["slot"] == 2
    assert r3.tool_calls[0]["slot"] == 3


# ---------------------------------------------------------------------------
# PaperDSMClient: paper-faithful policy/accounting knobs
# ---------------------------------------------------------------------------

def test_scoring_cost_matches_paper_formula():
    assert _scoring_cost(0) == 0
    assert _scoring_cost(_D - 1) == 0
    assert _scoring_cost(1) == _D - 2
    assert _scoring_cost(_D - 2) == 1


def test_score_summary_tracks_availability_flexibility_and_cost():
    summary = _score_summary({0: 0, 1: _D - 1, 2: LOWEST_FEASIBLE})

    assert summary["availability"] == 2
    assert summary["cost"] == _scoring_cost(_D - 1) + _scoring_cost(LOWEST_FEASIBLE)
    assert summary["flexibility"] > 0


def test_paper_dsm_respects_lmin_lmax_offer_bounds():
    client = PaperDSMClient()
    client.register(0, _make_config(
        0,
        num_agents=2,
        num_slots=8,
        dsm_lmin=3,
        dsm_lmax=3,
        dsm_theta=0,
        dsm_beta=5,
    ))
    meeting = _make_meeting(participants=[0, 1])
    client.start_round(meeting, _render({slot: "free" for slot in range(8)}), round_num=1)

    result = client.turn([])
    payload = json.loads(result.tool_calls[0]["content"])

    assert len(payload["slots"]) == 3


def test_paper_dsm_theta_and_beta_tune_offer_size():
    privacy_preserving = PaperDSMClient()
    social_welfare = PaperDSMClient()
    render = _render({slot: "free" for slot in range(8)})
    meeting = _make_meeting(participants=[0, 1])

    privacy_preserving.register(0, _make_config(
        0,
        num_agents=2,
        num_slots=8,
        dsm_lmin=1,
        dsm_lmax=5,
        dsm_theta=10,
        dsm_beta=0,
    ))
    social_welfare.register(0, _make_config(
        0,
        num_agents=2,
        num_slots=8,
        dsm_lmin=1,
        dsm_lmax=5,
        dsm_theta=0,
        dsm_beta=10,
    ))
    privacy_preserving.start_round(meeting, render, round_num=1)
    social_welfare.start_round(meeting, render, round_num=1)

    private_payload = json.loads(privacy_preserving.turn([]).tool_calls[0]["content"])
    welfare_payload = json.loads(social_welfare.turn([]).tool_calls[0]["content"])

    assert len(private_payload["slots"]) == 1
    assert len(welfare_payload["slots"]) > len(private_payload["slots"])


def test_paper_dsm_stops_when_current_batch_has_feasible_agreement():
    client = PaperDSMClient()
    client.register(0, _make_config(
        0,
        num_agents=2,
        num_slots=6,
        dsm_lmin=1,
        dsm_lmax=2,
        dsm_theta=0,
        dsm_beta=0,
    ))
    meeting = _make_meeting(participants=[0, 1])
    client.start_round(meeting, _render({slot: "free" for slot in range(6)}), round_num=1)

    proposal = client.turn([])
    proposed = json.loads(proposal.tool_calls[0]["content"])["slots"]
    decision = client.turn([_dsm_msg(1, {
        "dsm": "scores",
        "scores": {str(plan_id): _D - 1 for plan_id in proposed},
        "round": 1,
    })])

    assert client._state == "decided"
    assert all(json.loads(tc["content"])["dsm"] == "decision" for tc in decision.tool_calls)


def test_paper_dsm_reward_payload_is_monotone_and_top_score_is_free():
    client = PaperDSMClient()

    assert client._reward_for_score(_D - 1, availability=3, offer_count=3) == 0
    assert client._reward_for_score(LOWEST_FEASIBLE, availability=3, offer_count=3) > client._reward_for_score(
        _D - 2,
        availability=3,
        offer_count=3,
    )


def test_private_dsm_preset_uses_small_high_privacy_offer_sets():
    client = PrivateDSMClient()
    client.register(0, _make_config(0, num_agents=2, num_slots=8))
    client.start_round(_make_meeting(participants=[0, 1]), _render({slot: "free" for slot in range(8)}), round_num=1)

    result = client.turn([])
    payload = json.loads(result.tool_calls[0]["content"])

    assert client._theta() > client._beta()
    assert client._lmin() == 1
    assert client._lmax() == 2
    assert len(payload["slots"]) == 1
