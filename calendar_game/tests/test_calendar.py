"""Unit tests for Calendar and batch validation/application logic."""

import pytest

from calendar_game.calendar import Calendar, apply_batch, validate_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_calendar(num_slots: int = 8) -> Calendar:
    return Calendar(num_slots)


def errand(errand_id: int, cost: int) -> dict:
    return {"errand_id": errand_id, "cost": cost}


def meeting(meeting_id: int, cost: int = 1) -> dict:
    return {"meeting_id": meeting_id, "cost": cost}


def reschedule(item_id: int, from_slot: int, to_slot: int) -> dict:
    return {
        "type": "reschedule",
        "item_id": item_id,
        "from_slot": from_slot,
        "to_slot": to_slot,
        "justification": "Required to free the agreed meeting slot.",
    }


# ---------------------------------------------------------------------------
# Snapshot isolation
# ---------------------------------------------------------------------------

def test_snapshot_isolation():
    """Mutating a snapshot must not affect the original calendar."""
    cal = make_calendar(4)
    cal.place(0, errand(1, 3))
    cal.place(2, meeting(1))

    snap = cal.snapshot()

    # Mutate the snapshot
    snap.place(0, None)
    snap.place(2, errand(99, 5))
    snap.place(3, meeting(5))

    # Original must be unchanged
    assert cal.get(0) == {"errand_id": 1, "cost": 3}
    assert cal.get(2) == meeting(1)
    assert cal.get(3) is None


# ---------------------------------------------------------------------------
# render() determinism
# ---------------------------------------------------------------------------

def test_render_deterministic():
    """Same calendar state must always produce the identical render() string."""
    cal = make_calendar(4)
    cal.place(1, errand(3, 2))
    cal.place(2, meeting(1, 2))

    first = cal.render()
    second = cal.render()
    assert first == second


def test_render_format():
    """Check that free, errand, and meeting slots are rendered correctly."""
    cal = make_calendar(4)
    cal.place(1, errand(3, 2))
    cal.place(2, meeting(1, 3))

    rendered = cal.render()
    lines = rendered.splitlines()

    assert "[FREE]" in lines[0]          # slot 0 is free
    assert "Errand #3" in lines[1]       # slot 1 has errand
    assert "cost=2" in lines[1]
    assert "Meeting M1" in lines[2]      # slot 2 has meeting
    assert "cost=3" in lines[2]          # meeting has eviction cost
    assert "[FREE]" in lines[3]          # slot 3 is free


def test_render_blocked_errand():
    """Blocked errands should be visible as blocked slots in calendar renders."""
    cal = make_calendar(2)
    cal.place(0, {"errand_id": 3, "cost": 2, "blocked": True})

    rendered = cal.render()

    assert "Blocked #3" in rendered
    assert "cost=2" in rendered


def test_render_meeting_participants_when_known():
    """Known meeting participants are shown without breaking the cost format."""
    cal = make_calendar(2)
    cal.place(0, meeting(100, 1))
    cal.meeting_participants = {100: [0, 2]}

    rendered = cal.render()

    assert "Meeting M100 (cost=1)" in rendered
    assert "participants=[0, 2]" in rendered


# ---------------------------------------------------------------------------
# Atomic batch — valid case
# ---------------------------------------------------------------------------

def test_atomic_batch_valid():
    """
    Reschedule errand from slot 3 to slot 5 (free), then schedule meeting at
    slot 3 (freed by the reschedule). Assert both changes are applied.
    """
    cal = make_calendar(8)
    cal.place(3, errand(7, 4))
    # slot 5 is free

    actions = [
        reschedule(7, 3, 5),
        {"type": "schedule", "meeting_id": 2, "slot": 3, "cost": 1},
    ]

    ok, reason = validate_batch(cal, actions)
    assert ok, reason

    apply_batch(cal, actions)

    assert cal.get(3) == meeting(2, 1)
    assert cal.get(5) == {"errand_id": 7, "cost": 4}


# ---------------------------------------------------------------------------
# Order irrelevance
# ---------------------------------------------------------------------------

def test_atomic_batch_order_irrelevant():
    """
    Submitting the same batch in reversed order must produce identical results.
    """
    def build_cal():
        cal = make_calendar(8)
        cal.place(3, errand(7, 4))
        return cal

    actions = [
        reschedule(7, 3, 5),
        {"type": "schedule", "meeting_id": 2, "slot": 3, "cost": 1},
    ]

    cal_a = build_cal()
    cal_b = build_cal()

    ok_a, _ = validate_batch(cal_a, actions)
    ok_b, _ = validate_batch(cal_b, list(reversed(actions)))
    assert ok_a and ok_b

    apply_batch(cal_a, actions)
    apply_batch(cal_b, list(reversed(actions)))

    assert cal_a.slots == cal_b.slots


# ---------------------------------------------------------------------------
# Conflict: two actions targeting the same slot
# ---------------------------------------------------------------------------

def test_atomic_batch_conflict_two_targets():
    """Two actions both targeting slot 5 must cause validation to fail."""
    cal = make_calendar(8)
    cal.place(1, errand(1, 1))
    cal.place(2, errand(2, 1))
    # slot 5 is free

    actions = [
        reschedule(1, 1, 5),
        reschedule(2, 2, 5),
        {"type": "schedule", "meeting_id": 3, "slot": 0, "cost": 1},
    ]

    ok, reason = validate_batch(cal, actions)
    assert not ok
    assert "5" in reason  # conflict message should mention slot 5


# ---------------------------------------------------------------------------
# Wrong item_id in reschedule
# ---------------------------------------------------------------------------

def test_atomic_batch_wrong_item_id():
    """
    A reschedule claiming an item_id that doesn't match from_slot contents
    must fail validation.
    """
    cal = make_calendar(8)
    cal.place(3, errand(7, 4))

    actions = [
        # Wrong item_id: 99 != 7
        reschedule(99, 3, 5),
        {"type": "schedule", "meeting_id": 2, "slot": 3, "cost": 1},
    ]

    ok, reason = validate_batch(cal, actions)
    assert not ok
    assert "99" in reason or "3" in reason  # should mention the mismatch


def test_validate_batch_rejects_blocked_reschedule():
    """Blocked errands are hard stops and cannot be moved by a reschedule."""
    cal = make_calendar(4)
    cal.place(1, {"errand_id": 7, "cost": 1, "blocked": True})

    ok, reason = validate_batch(cal, [
        reschedule(7, 1, 2),
        {"type": "schedule", "meeting_id": 1, "slot": 1, "cost": 1},
    ])

    assert not ok
    assert "blocked" in reason


def test_validate_batch_rejects_malformed_actions():
    """Malformed model actions should be rejected, not crash validation."""
    cal = make_calendar(4)
    cal.place(1, errand(10, 1))

    ok, reason = validate_batch(cal, [None])  # type: ignore[list-item]
    assert not ok
    assert "not an object" in reason

    ok, reason = validate_batch(cal, [{"type": "schedule", "meeting_id": 1}])
    assert not ok
    assert "missing required field 'slot'" in reason

    ok, reason = validate_batch(cal, [{"type": "reschedule", "item_id": 10, "from_slot": 1}])
    assert not ok
    assert "missing required field 'to_slot'" in reason

    ok, reason = validate_batch(cal, [{"type": "reschedule", "item_id": 10, "from_slot": 1, "to_slot": 2}])
    assert not ok
    assert "missing required field 'justification'" in reason

    ok, reason = validate_batch(cal, [
        {"type": "reschedule", "item_id": 10, "from_slot": 1, "to_slot": 2, "justification": "  "}
    ])
    assert not ok
    assert "justification must be a non-empty string" in reason


# ---------------------------------------------------------------------------
# Chain reschedule
# ---------------------------------------------------------------------------

def test_atomic_batch_chain():
    """
    Reschedule A→B and reschedule C→A in the same batch.
    A is freed by the first reschedule, so C→A should be valid.
    """
    cal = make_calendar(8)
    cal.place(1, errand(10, 1))   # A = slot 1, item_id 10
    cal.place(3, errand(20, 2))   # C = slot 3, item_id 20
    # slot 2 (B) is free

    actions = [
        reschedule(10, 1, 2),  # A→B
        reschedule(20, 3, 1),  # C→A
        {"type": "schedule", "meeting_id": 5, "slot": 0, "cost": 2},
    ]

    ok, reason = validate_batch(cal, actions)
    assert ok, reason

    apply_batch(cal, actions)

    assert cal.get(0) == meeting(5, 2)
    assert cal.get(1) == {"errand_id": 20, "cost": 2}
    assert cal.get(2) == {"errand_id": 10, "cost": 1}
    assert cal.get(3) is None


# ---------------------------------------------------------------------------
# No partial application on invalid batch
# ---------------------------------------------------------------------------

def test_apply_batch_no_partial():
    """
    An invalid batch must leave the calendar in an identical state.
    validate_batch alone must not mutate the calendar.
    """
    cal = make_calendar(8)
    cal.place(1, errand(1, 1))
    cal.place(2, errand(2, 1))

    original_slots = list(cal.slots)

    # Two reschedules targeting the same slot — invalid
    actions = [
        reschedule(1, 1, 5),
        reschedule(2, 2, 5),
        {"type": "schedule", "meeting_id": 3, "slot": 0, "cost": 1},
    ]

    ok, _ = validate_batch(cal, actions)
    assert not ok

    # Calendar must be completely unchanged after a failed validate_batch call
    assert cal.slots == original_slots
