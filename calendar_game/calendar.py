"""Calendar data structure and batch validation/application logic."""

from __future__ import annotations

import copy
from typing import NotRequired, TypedDict, Union


class Errand(TypedDict):
    errand_id: int
    cost: int
    blocked: NotRequired[bool]


class Meeting(TypedDict):
    meeting_id: int
    cost: int


# A slot is either None (free), an Errand dict, or a Meeting dict
Slot = Union[Errand, Meeting, None]


class Calendar:
    """A fixed-length sequence of slots representing an agent's schedule."""

    def __init__(self, num_slots: int) -> None:
        self.num_slots: int = num_slots
        self.slots: list[Slot] = [None] * num_slots
        self.meeting_participants: dict[int, list[int]] = {}

    def is_free(self, slot: int) -> bool:
        """Return True if the slot is unoccupied."""
        return self.slots[slot] is None

    def get(self, slot: int) -> Slot:
        """Return the raw slot value."""
        return self.slots[slot]

    def place(self, slot: int, item: Slot) -> None:
        """Set slots[slot] = item (overwrites whatever was there)."""
        self.slots[slot] = item

    def move(self, from_slot: int, to_slot: int) -> None:
        """Move slots[from_slot] to slots[to_slot]. to_slot must be free."""
        if not self.is_free(to_slot):
            raise ValueError(
                f"Cannot move to slot {to_slot}: slot is not free (contains {self.slots[to_slot]!r})"
            )
        self.slots[to_slot] = self.slots[from_slot]
        self.slots[from_slot] = None

    def snapshot(self) -> "Calendar":
        """Return a deep copy; mutations to the copy do not affect the original."""
        new_cal = Calendar(self.num_slots)
        new_cal.slots = copy.deepcopy(self.slots)
        new_cal.meeting_participants = copy.deepcopy(self.meeting_participants)
        return new_cal

    def render(self) -> str:
        """Return a human-readable multi-line string for LLM context."""
        width = len(str(self.num_slots - 1))
        lines: list[str] = []
        for i, slot in enumerate(self.slots):
            label = str(i).rjust(width)
            if slot is None:
                content = "[FREE]"
            elif "meeting_id" in slot:
                meeting_id = int(slot["meeting_id"])
                content = f"Meeting M{meeting_id} (cost={slot['cost']})"
                participants = self.meeting_participants.get(meeting_id)
                if participants is not None:
                    participants_str = ", ".join(str(p) for p in participants)
                    content += f" participants=[{participants_str}]"
            else:
                if slot.get("blocked"):
                    content = f"Blocked #{slot['errand_id']} (cost={slot['cost']})"
                else:
                    content = f"Errand #{slot['errand_id']} (cost={slot['cost']})"
            lines.append(f"Slot {label}: {content}")
        return "\n".join(lines)


def _item_matches(slot_value: Slot, item_id: int) -> bool:
    """Return True if slot_value matches the claimed item_id (errand or meeting)."""
    if isinstance(slot_value, dict):
        return (slot_value.get("errand_id") == item_id or
                slot_value.get("meeting_id") == item_id)
    return False


def validate_batch(calendar: Calendar, actions: list[dict], require_schedule: bool = True) -> tuple[bool, str]:
    """
    Validate a batch of schedule/reschedule actions as a transaction.

    Returns (True, "") if the batch is globally consistent, or (False, reason)
    if it is not. Nothing is applied.

    Validation rules:
    1. Each reschedule from_slot must contain the claimed item_id.
    2. No two actions share the same to_slot target.
    3. All to_slot targets must be free on the calendar OR freed by another
       reschedule in this batch (i.e. appearing as a from_slot).
    4. Exactly one "schedule" action is allowed per batch.
    5. The schedule slot must be free after all reschedules are applied.
    6. Each reschedule must include a non-empty justification.
    """
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            return False, f"Action {index} is not an object: {action!r}."

    reschedules = [a for a in actions if a.get("type") == "reschedule"]
    schedules = [a for a in actions if a.get("type") == "schedule"]

    for action in schedules:
        for key in ("meeting_id", "slot"):
            if key not in action:
                return False, f"Schedule action missing required field {key!r}: {action!r}."
        if not isinstance(action["slot"], int):
            return False, f"Schedule slot must be an integer: {action!r}."

    for action in reschedules:
        for key in ("item_id", "from_slot", "to_slot", "justification"):
            if key not in action:
                return False, f"Reschedule action missing required field {key!r}: {action!r}."
        if not isinstance(action["from_slot"], int) or not isinstance(action["to_slot"], int):
            return False, f"Reschedule slots must be integers: {action!r}."
        if not isinstance(action["justification"], str) or not action["justification"].strip():
            return False, f"Reschedule justification must be a non-empty string: {action!r}."

    # Bounds check all slot references before anything else
    all_slots = (
        [a["from_slot"] for a in reschedules]
        + [a["to_slot"] for a in reschedules]
        + [a["slot"] for a in schedules]
    )
    for s in all_slots:
        if s < 0 or s >= calendar.num_slots:
            return False, f"Slot {s} is out of range (calendar has {calendar.num_slots} slots)."

    # Rule 4: exactly one schedule (skipped for voluntary reschedule-only phase)
    if require_schedule and len(schedules) != 1:
        return False, f"Expected exactly 1 schedule action, got {len(schedules)}."

    schedule = schedules[0] if schedules else None

    # Rule 1: each reschedule from_slot must contain the claimed item_id
    freed_slots: set[int] = set()
    for action in reschedules:
        from_slot = action["from_slot"]
        item_id = action["item_id"]
        slot_value = calendar.get(from_slot)
        if not _item_matches(slot_value, item_id):
            return (
                False,
                f"Reschedule claims item_id={item_id} at slot {from_slot}, "
                f"but slot contains {slot_value!r}.",
            )
        if isinstance(slot_value, dict) and slot_value.get("blocked"):
            return (
                False,
                f"Reschedule attempts to move blocked item_id={item_id} at slot {from_slot}.",
            )
        freed_slots.add(from_slot)

    # Collect all target slots
    all_targets: list[int] = [a["to_slot"] for a in reschedules] + ([schedule["slot"]] if schedule else [])

    # Rule 2: no two actions share the same target slot
    seen_targets: set[int] = set()
    for target in all_targets:
        if target in seen_targets:
            return False, f"Two or more actions target slot {target}."
        seen_targets.add(target)

    # Rule 3 & 5: each target must be free on the calendar or freed by another
    # reschedule in this batch
    for action in reschedules:
        target = action["to_slot"]
        if not calendar.is_free(target) and target not in freed_slots:
            return (
                False,
                f"Reschedule targets slot {target}, which is occupied by "
                f"{calendar.get(target)!r} and not freed by this batch.",
            )

    if schedule is not None:
        schedule_slot = schedule["slot"]
        if not calendar.is_free(schedule_slot) and schedule_slot not in freed_slots:
            return (
                False,
                f"Schedule targets slot {schedule_slot}, which is occupied by "
                f"{calendar.get(schedule_slot)!r} and not freed by this batch.",
            )

    return True, ""


def apply_batch(calendar: Calendar, actions: list[dict]) -> None:
    """
    Apply a validated batch atomically.

    Caller MUST call validate_batch first and confirm it returned (True, "").
    All reschedules are cleared first, then all targets are written, then the
    schedule is applied — this avoids ordering dependencies within the batch.
    """
    reschedules = [a for a in actions if a.get("type") == "reschedule"]
    schedules = [a for a in actions if a.get("type") == "schedule"]

    # Collect moved items before clearing any slots
    moved: list[tuple[int, Slot]] = []
    for action in reschedules:
        from_slot = action["from_slot"]
        to_slot = action["to_slot"]
        moved.append((to_slot, calendar.get(from_slot)))

    # Clear all from_slots
    for action in reschedules:
        calendar.place(action["from_slot"], None)

    # Write all to_slots
    for to_slot, item in moved:
        calendar.place(to_slot, item)

    # Apply the schedule if present
    if schedules:
        schedule = schedules[0]
        calendar.place(schedule["slot"], {"meeting_id": schedule["meeting_id"], "cost": schedule["cost"]})
