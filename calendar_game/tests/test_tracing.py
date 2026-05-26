"""Integration tests for event log and tracing in the calendar scheduling game."""

from __future__ import annotations

import pytest

from calendar_game.game import CalendarGame, CalendarGameConfig
from calendar_game.calendar import Calendar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_dry(seed=42, num_meetings=1, **kwargs):
    config = CalendarGameConfig(seed=seed, num_meetings=num_meetings, **kwargs)
    game = CalendarGame(config, dry_run=True)
    return game.run()


def events_of_type(trace, event_type):
    return [e for e in _normalize_events(trace) if e["type"] == event_type]


def data_of_type(trace, event_type):
    return [e["data"] for e in _normalize_events(trace) if e["type"] == event_type]


def _event_as_dict(event):
    """Convert a GameEvent (Pydantic model or dict) to a plain dict."""
    if isinstance(event, dict):
        return event
    return {"type": event.type, "data": event.data}


def _normalize_events(trace):
    """Return trace.events as a list of plain dicts."""
    return [_event_as_dict(e) for e in trace.events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_event_seq_monotonic():
    """EventLog stores events; list order is consistent and non-repeating."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    assert len(events) > 0, "Expected at least one event"

    # game_start is first, game_end is last
    assert events[0]["type"] == "game_start"
    assert events[-1]["type"] == "game_end"

    # If events have a seq field in data, it must be strictly increasing
    seqs = [e["data"].get("seq") for e in events if "seq" in e["data"]]
    if seqs:
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"seq not strictly increasing at index {i}"
    else:
        # Verify round_start precedes the first turn_start within each round
        round_seen: dict[int, bool] = {}
        for event in events:
            etype = event["type"]
            rnd = event["data"].get("round")
            if rnd is None:
                continue
            if etype == "round_start":
                round_seen[rnd] = True
            elif etype == "turn_start":
                assert round_seen.get(rnd), (
                    f"turn_start for round {rnd} appeared before round_start"
                )


def test_turn_index_correct():
    """turn_start data['turn'] starts at 0 per round and increments; resets at new round."""
    trace = run_dry(seed=42, num_meetings=2)
    events = _normalize_events(trace)

    # Group turn_start events by round
    rounds: dict[int, list[int]] = {}
    last_round = -1
    for event in events:
        if event["type"] == "round_start":
            last_round = event["data"]["round"]
        elif event["type"] == "turn_start":
            rnd = event["data"]["round"]
            turns = rounds.setdefault(rnd, [])
            turns.append(event["data"]["turn"])

    assert len(rounds) >= 2, "Expected at least 2 rounds for num_meetings=2"

    for rnd, turn_indices in rounds.items():
        # First turn_start in each round should have turn=0
        assert turn_indices[0] == 0, (
            f"First turn_start of round {rnd} has turn={turn_indices[0]}, expected 0"
        )
        # Turns must be non-decreasing (same turn can appear for multiple agents)
        for i in range(1, len(turn_indices)):
            assert turn_indices[i] >= turn_indices[i - 1], (
                f"turn index went backwards in round {rnd} at position {i}"
            )


def test_raw_api_response_logged():
    """turn_end and decide_end events always include 'raw_api_response' key."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    for event in events:
        if event["type"] in ("turn_end", "decide_end"):
            assert "raw_api_response" in event["data"], (
                f"{event['type']} event is missing 'raw_api_response' key"
            )


def test_calendar_render_in_turn_start():
    """turn_start events have a non-empty calendar_render with at least one 'Slot' line."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    turn_starts = [e for e in events if e["type"] == "turn_start"]
    assert len(turn_starts) > 0, "Expected at least one turn_start event"

    for event in turn_starts:
        render = event["data"].get("calendar_render", "")
        assert isinstance(render, str) and len(render) > 0, (
            "calendar_render should be a non-empty string"
        )
        assert "Slot" in render, "calendar_render should contain 'Slot'"
        # Must contain at least one of the known slot content types
        has_known_content = (
            "[FREE]" in render
            or "Errand" in render
            or "Meeting" in render
        )
        assert has_known_content, (
            f"calendar_render does not contain [FREE], Errand, or Meeting:\n{render}"
        )


def test_replay_reconstructs_state():
    """calendar_render_after in the last batch_applied matches final_state calendars."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    num_slots = trace.config.num_slots if hasattr(trace.config, "num_slots") else 16
    meeting_participants = {
        int(e["data"]["meeting"]["id"]): list(e["data"]["meeting"].get("participants", []))
        for e in events
        if e["type"] == "round_start" and "meeting" in e["data"]
    }

    for agent_id, slots in enumerate(trace.final_state["calendars"]):
        # Reconstruct calendar from final_state
        cal = Calendar(num_slots=len(slots))
        cal.slots = list(slots)
        cal.meeting_participants = meeting_participants
        expected_render = cal.render()

        # Find last batch_applied or fallback_applied for this agent
        # (fallback_applied uses calendar_renders_after[str(agent_id)])
        last_render: str | None = None
        for e in events:
            if e["type"] == "batch_applied" and e["data"].get("agent_id") == agent_id:
                last_render = e["data"]["calendar_render_after"]
            elif e["type"] == "fallback_applied":
                renders = e["data"].get("calendar_renders_after", {})
                if str(agent_id) in renders:
                    last_render = renders[str(agent_id)]
        if last_render is not None:
            assert last_render == expected_render, (
                f"calendar_render_after for agent {agent_id} does not match final_state"
            )


def test_resolution_event_present():
    """2-meeting game has exactly 2 resolution events with required fields."""
    trace = run_dry(seed=42, num_meetings=2)
    events = _normalize_events(trace)

    resolutions = [e for e in events if e["type"] == "resolution"]
    assert len(resolutions) == 2, (
        f"Expected 2 resolution events, got {len(resolutions)}"
    )

    required_fields = {"meeting_id", "per_agent_slot", "coordinated", "slot_conflicts"}
    for event in resolutions:
        for field in required_fields:
            assert field in event["data"], (
                f"resolution event missing field '{field}'"
            )


def test_game_start_and_end_bookend():
    """game_start is first event, game_end is last and has metric fields."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    assert events[0]["type"] == "game_start", "First event must be game_start"
    assert events[-1]["type"] == "game_end", "Last event must be game_end"

    end_data = events[-1]["data"]
    for field in ("coordination_rate", "efficiency", "fairness", "meetings_scheduled"):
        assert field in end_data, f"game_end event missing field '{field}'"


def test_dm_sent_events_logged():
    """ScriptedClient sends DMs; at least one dm_sent event appears with required fields."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    dm_events = [e for e in events if e["type"] == "dm_sent"]
    assert len(dm_events) >= 1, "Expected at least one dm_sent event"

    required_fields = {"from_agent", "to_agent", "meeting_id", "content"}
    for event in dm_events:
        for field in required_fields:
            assert field in event["data"], (
                f"dm_sent event missing field '{field}'"
            )


def test_decide_start_has_snapshot_render():
    """decide_start events have a non-empty calendar_snapshot_render containing 'Slot'."""
    trace = run_dry(seed=42)
    events = _normalize_events(trace)

    decide_starts = [e for e in events if e["type"] == "decide_start"]
    assert len(decide_starts) > 0, "Expected at least one decide_start event"

    for event in decide_starts:
        render = event["data"].get("calendar_snapshot_render", "")
        assert isinstance(render, str) and len(render) > 0, (
            "calendar_snapshot_render should be a non-empty string"
        )
        assert "Slot" in render, (
            f"calendar_snapshot_render should contain 'Slot':\n{render}"
        )
