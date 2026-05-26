"""
Tests for calendar_game/prompts.py
"""
import pytest
from calendar_game.prompts import (
    PROMPT_VARIANTS_DIR,
    build_system_prompt,
    build_dspy_system_prompt,
    make_dspy_system_prompt_builder,
    build_round_start_message,
    build_turn_message,
    build_decision_message,
    build_retry_message,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GAME_CONFIG = {
    "num_agents": 3,
    "num_slots": 8,
    "agent_id": 1,
    "all_agent_ids": [1, 2, 3],
    "dm_cap": 5,
    "decision_retries": 2,
}

MEETING = {
    "id": 42,
    "participants": [1, 2, 3],
    "duration": 1,
}

CALENDAR_RENDER = "slot 0: free\nslot 1: errand(id=10)\nslot 2: free"


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_contains_fields():
    out = build_system_prompt(GAME_CONFIG)
    assert str(GAME_CONFIG["agent_id"]) in out
    assert str(GAME_CONFIG["decision_retries"]) in out
    assert "dm" in out
    assert "schedule" in out
    assert "reschedule" in out


def test_build_system_prompt_contains_negotiation_strategy():
    out = build_system_prompt(GAME_CONFIG)
    assert "NEGOTIATION STRATEGY" in out
    assert "Do not move them casually" in out
    assert "pushback to make them move their calendar" in out
    assert "Prefer slots that are free for you" in out
    assert "mutually workable" in out


def test_build_system_prompt_requires_reschedule_justification():
    out = build_system_prompt(GAME_CONFIG)
    assert '"justification"' in out
    assert "why this commitment must be moved" in out
    assert "preserving it" in out


def test_build_system_prompt_warns_not_to_reschedule_blocked_slots():
    out = build_system_prompt(GAME_CONFIG)
    assert "Do not reschedule a blocked slot/item" in out
    assert 'Slots rendered as "Blocked #..." are immovable' in out
    assert "attempt to move them will be rejected" in out


def test_build_system_prompt_warns_not_to_schedule_blocked_slots():
    out = build_system_prompt(GAME_CONFIG)
    assert 'Do not propose or agree to a slot rendered as "Blocked #..."' in out
    assert "cannot be used for scheduling, and cannot be moved or freed" in out
    assert 'Never schedule into a slot rendered as "Blocked #..."' in out
    assert "will be rejected" in out


def test_build_dspy_system_prompt_has_separate_optimized_policy():
    out = build_dspy_system_prompt(GAME_CONFIG)
    assert "DSPY-OPTIMIZED NEGOTIATION POLICY" in out
    assert "fewest total reschedules" in out
    assert "NEGOTIATION STRATEGY" in out


def test_make_dspy_system_prompt_builder_uses_named_variant(tmp_path):
    out = make_dspy_system_prompt_builder("dspy_optimized_v1.md")(GAME_CONFIG)
    assert "DSPY-OPTIMIZED NEGOTIATION POLICY" in out


def test_build_dspy_system_prompt_can_use_sibling_variant_dir():
    variant_dir = PROMPT_VARIANTS_DIR.with_name("prompt_variants_test")
    variant_dir.mkdir(exist_ok=True)
    variant_path = variant_dir / "unit_test_variant.md"
    variant_path.write_text("UNIT TEST DSPY VARIANT\n", encoding="utf-8")
    try:
        out = build_dspy_system_prompt(
            GAME_CONFIG,
            "unit_test_variant.md",
            variant_dir="prompt_variants_test",
        )
    finally:
        variant_path.unlink(missing_ok=True)
        variant_dir.rmdir()
    assert "UNIT TEST DSPY VARIANT" in out


def test_build_system_prompt_all_agent_ids():
    out = build_system_prompt(GAME_CONFIG)
    for aid in GAME_CONFIG["all_agent_ids"]:
        assert str(aid) in out


# ---------------------------------------------------------------------------
# build_round_start_message
# ---------------------------------------------------------------------------

def test_build_round_start_contains_meeting():
    out = build_round_start_message(MEETING, CALENDAR_RENDER, round_num=3)
    assert str(MEETING["id"]) in out
    assert "3" in out  # round number
    assert CALENDAR_RENDER in out
    assert "CHEAP_TALK" in out


def test_build_round_start_contains_participants():
    out = build_round_start_message(MEETING, CALENDAR_RENDER, round_num=1)
    for pid in MEETING["participants"]:
        assert str(pid) in out


def test_build_round_start_contains_incurred_penalty():
    out = build_round_start_message(MEETING, CALENDAR_RENDER, round_num=1, incurred_penalty=37)
    assert "YOUR PENALTY SO FAR" in out
    assert "37 total penalty points" in out


def test_build_round_start_contains_safety_turn_counter():
    out = build_round_start_message(
        MEETING,
        CALENDAR_RENDER,
        round_num=1,
        turn_index=0,
        max_turns_per_round=15,
    )
    assert "safety turn counter: turn 1 of 15" in out
    assert "14 turn(s) remain" in out


def test_build_round_start_reminds_reschedule_is_last_resort():
    out = build_round_start_message(MEETING, CALENDAR_RENDER, round_num=1)
    assert "rescheduling errands or prior meetings as a last resort" in out
    assert "mutually low-displacement slot" in out


def test_build_round_start_warns_not_to_propose_blocked_slots():
    out = build_round_start_message(MEETING, CALENDAR_RENDER, round_num=1)
    assert 'Never propose or agree to a slot rendered as "Blocked #..."' in out


# ---------------------------------------------------------------------------
# build_turn_message
# ---------------------------------------------------------------------------

def test_build_turn_message_with_messages():
    messages = [
        {"from": 2, "meeting_id": 42, "content": "Let's use slot 3."},
        {"from": 3, "meeting_id": 42, "content": "Agreed, slot 3 works."},
    ]
    out = build_turn_message(messages)
    assert "2" in out
    assert "3" in out
    assert "Let's use slot 3." in out
    assert "Agreed, slot 3 works." in out
    # Numbered
    assert "[1]" in out
    assert "[2]" in out


def test_build_turn_message_multiple_messages():
    messages = [
        {"from": 2, "meeting_id": 42, "content": "How about slot 1?"},
        {"from": 3, "meeting_id": 42, "content": "Slot 1 is busy for me."},
        {"from": 4, "meeting_id": 42, "content": "Slot 5 works for me."},
    ]
    out = build_turn_message(messages)
    assert "How about slot 1?" in out
    assert "Slot 1 is busy for me." in out
    assert "Slot 5 works for me." in out
    assert "[1]" in out
    assert "[2]" in out
    assert "[3]" in out


def test_build_turn_message_empty():
    out = build_turn_message([])
    # Should not list any messages
    assert "[1]" not in out
    # Should signal that agent may pass
    assert "[]" in out or "pass" in out.lower() or "no new messages" in out.lower()


def test_build_turn_message_final_turn_budget():
    out = build_turn_message([], turn_index=14, max_turns_per_round=15)
    assert "safety turn counter: turn 15 of 15" in out
    assert "final CHEAP_TALK turn" in out


# ---------------------------------------------------------------------------
# build_decision_message
# ---------------------------------------------------------------------------

def test_build_decision_no_inbox():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert "inbox" not in out.lower()


def test_build_decision_contains_meeting_and_calendar():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert str(MEETING["id"]) in out
    assert CALENDAR_RENDER in out


def test_build_decision_phase_signal():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert "DECISION" in out


def test_build_decision_discourages_avoidable_reschedules():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert "do not add avoidable reschedules" in out


def test_build_decision_requires_reschedule_justification():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert '"justification"' in out
    assert "human user's existing commitment needs to move" in out


def test_build_decision_warns_not_to_schedule_blocked_slots():
    out = build_decision_message(MEETING, CALENDAR_RENDER)
    assert "must be [FREE]" in out
    assert 'Never schedule meeting 42 into a slot rendered as "Blocked #..."' in out
    assert "do not try to move the blocked item" in out


# ---------------------------------------------------------------------------
# build_retry_message
# ---------------------------------------------------------------------------

def test_build_retry_message_header():
    out = build_retry_message(attempt=2, max_attempts=3, conflict="Slot 4 is already occupied.")
    assert "[RETRY 2/3]" in out


def test_build_retry_message_conflict_verbatim():
    conflict = "Slot 4 is already occupied by meeting id=7."
    out = build_retry_message(attempt=1, max_attempts=2, conflict=conflict)
    assert conflict in out


def test_build_retry_message_requires_reschedule_justification():
    out = build_retry_message(attempt=1, max_attempts=2, conflict="missing justification")
    assert '"justification"' in out
    assert '"thinking" and "actions"' in out


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------

def test_all_builders_return_str():
    assert isinstance(build_system_prompt(GAME_CONFIG), str)
    assert isinstance(build_round_start_message(MEETING, CALENDAR_RENDER, 1), str)
    assert isinstance(build_turn_message([]), str)
    assert isinstance(build_turn_message([{"from": 2, "meeting_id": 1, "content": "hi"}]), str)
    assert isinstance(build_decision_message(MEETING, CALENDAR_RENDER), str)
    assert isinstance(build_retry_message(1, 3, "conflict text"), str)
    # None must never be returned
    assert build_system_prompt(GAME_CONFIG) is not None
    assert build_round_start_message(MEETING, CALENDAR_RENDER, 1) is not None
    assert build_turn_message([]) is not None
    assert build_decision_message(MEETING, CALENDAR_RENDER) is not None
    assert build_retry_message(1, 3, "x") is not None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_build_system_prompt_deterministic():
    out1 = build_system_prompt(GAME_CONFIG)
    out2 = build_system_prompt(GAME_CONFIG)
    assert out1 == out2
