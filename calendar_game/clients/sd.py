"""Scheduling Difficulty (SD) IL-MAP baseline client.

This client implements Modi and Veloso's scheduling-difficulty bumping rule as
an event-driven cheap-talk protocol. It preserves local calendar privacy: agents
only exchange proposal status, confirmations, failures, and reschedule requests.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult
from calendar_game.clients.dsm import _apply_actions_to_items, _parse_slot_items, _schedule_actions


class MeetingStatus(StrEnum):
    POSSIBLE = "POSSIBLE"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    BUMPED = "BUMPED"
    IMPOSSIBLE = "IMPOSSIBLE"


def _empty_turn() -> TurnResult:
    return TurnResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _empty_decide() -> DecideResult:
    return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _parse_sd(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(payload, dict) and "sd" in payload:
        return payload
    return None


class SDClient(BaseClient):
    """Asynchronous Scheduling Difficulty baseline for incremental MAP."""

    def __init__(self) -> None:
        self.agent_id: int = -1
        self.game_config: GameConfig | None = None
        self.meeting: dict | None = None
        self._slot_items: dict[int, dict[str, int] | None] = {}
        self._known_meetings: dict[int, list[int]] = {}
        self._meeting_costs: dict[int, int] = {}
        self._sd_model: dict[int, float] = {}
        self._statuses: dict[int, dict[int, MeetingStatus]] = {}
        self._initiator_id: int = -1
        self._role: str = "responder"
        self._state: str = "idle"
        self._expected_responders: set[int] = set()
        self._tried_slots: set[int] = set()
        self._active_slot: int | None = None
        self._replies_received: dict[int, MeetingStatus] = {}
        self._agreed_slot: int | None = None
        self._active_round: bool = False
        self._bumped_by: dict[tuple[int, int], int] = {}
        self._agreed_reschedule_targets: dict[int, int] = {}
        self._pending_reschedule_requests: dict[int, tuple[int, int]] = {}
        self._repair_states: dict[int, dict[str, Any]] = {}
        self._agreed_bump_moves: dict[int, tuple[int, int]] = {}

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self.game_config = game_config
        self._sd_model = self._load_sd_model(game_config)
        for prior in getattr(game_config, "dsm_prior_meetings", []):
            try:
                meeting_id = int(prior["id"])
                self._known_meetings[meeting_id] = [int(p) for p in prior.get("participants", [])]
                self._meeting_costs[meeting_id] = int(prior.get("cost", 1))
                if prior.get("slot") is not None:
                    self._set_status(meeting_id, int(prior["slot"]), MeetingStatus.CONFIRMED)
            except (KeyError, TypeError, ValueError):
                continue

    def observe_calendar(self, calendar_render: str) -> None:
        self._slot_items = _parse_slot_items(calendar_render)
        for slot, item in self._slot_items.items():
            if isinstance(item, dict) and item.get("type") == "meeting":
                meeting_id = int(item["item_id"])
                if self._status(meeting_id, slot) == MeetingStatus.POSSIBLE:
                    self._set_status(meeting_id, slot, MeetingStatus.CONFIRMED)

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self.meeting = meeting
        meeting_id = int(meeting["id"])
        participants = [int(p) for p in meeting.get("participants", [])]
        self._known_meetings[meeting_id] = participants
        self._meeting_costs[meeting_id] = int(meeting.get("cost", 1))
        self.observe_calendar(calendar_render)
        self._initiator_id = min(participants) if participants else self.agent_id
        self._role = "initiator" if self.agent_id == self._initiator_id else "responder"
        self._expected_responders = {p for p in participants if p != self._initiator_id}
        self._tried_slots = set()
        self._active_slot = None
        self._replies_received = {}
        self._agreed_slot = None
        self._agreed_reschedule_targets = {}
        self._active_round = True
        self._state = "proposing" if self._role == "initiator" else "waiting_propose"

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        outgoing: list[dict] = []
        for msg in messages:
            outgoing.extend(self._handle_message(msg))

        if self._active_round and self._role == "initiator":
            outgoing.extend(self._initiator_progress())
        outgoing.extend(self._repair_progress())

        return TurnResult(tool_calls=outgoing, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        if self._agreed_slot is None:
            return _empty_decide()
        slot_items = _parse_slot_items(calendar_render)
        reschedule_targets = dict(self._agreed_reschedule_targets)
        for meeting_id, (_from_slot, to_slot) in self._agreed_bump_moves.items():
            reschedule_targets[meeting_id] = to_slot
        for meeting_id, (_from_slot, to_slot) in self._pending_reschedule_requests.items():
            reschedule_targets[meeting_id] = to_slot
        actions = _schedule_actions(meeting, self._agreed_slot, slot_items, reschedule_targets)
        if not actions:
            return _empty_decide()
        self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
        self._active_round = False
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        if self.meeting is None:
            return _empty_decide()
        slot_items = self._slot_items
        candidates = [slot for slot in sorted(slot_items) if self._slot_locally_available(self.meeting, slot, allow_bump=False)]
        if attempt >= len(candidates):
            return _empty_decide()
        actions = _schedule_actions(self.meeting, candidates[attempt], slot_items)
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        slot_items = _parse_slot_items(calendar_render)
        actions: list[dict] = []
        for meeting_id, (from_slot, to_slot) in sorted(self._pending_reschedule_requests.items()):
            item = slot_items.get(from_slot)
            if (
                isinstance(item, dict)
                and item.get("type") == "meeting"
                and int(item.get("item_id", -1)) == meeting_id
                and slot_items.get(to_slot) is None
            ):
                actions.append({
                    "type": "reschedule",
                    "item_id": meeting_id,
                    "from_slot": from_slot,
                    "to_slot": to_slot,
                    "justification": "Move the prior meeting to honor the requested coordination commitment.",
                })
                self._set_status(meeting_id, from_slot, MeetingStatus.IMPOSSIBLE)
                self._set_status(meeting_id, to_slot, MeetingStatus.CONFIRMED)
        self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def _handle_message(self, msg: dict) -> list[dict]:
        parsed = _parse_sd(msg.get("content", ""))
        if not parsed:
            return []
        kind = parsed.get("sd")
        if kind == "propose":
            return self._receive_propose(parsed, int(msg.get("from", -1)))
        if kind == "reply" and self._role == "initiator":
            self._receive_reply(parsed, int(msg.get("from", -1)))
            return []
        if kind == "confirm":
            return self._receive_confirm(parsed)
        if kind == "fail":
            self._cleanup_resolved(int(parsed.get("meeting_id", -1)))
            return []
        if kind == "reschedule":
            return self._receive_reschedule(parsed)
        if kind == "reschedule_request":
            return self._receive_reschedule_request(parsed)
        if kind == "propose_reschedule":
            return self._receive_propose_reschedule(parsed, int(msg.get("from", -1)))
        if kind == "reschedule_reply":
            self._receive_reschedule_reply(parsed, int(msg.get("from", -1)))
            return []
        if kind == "confirm_reschedule":
            return self._receive_confirm_reschedule(parsed)
        if kind == "fail_reschedule":
            return self._receive_fail_reschedule(parsed)
        return []

    def _initiator_progress(self) -> list[dict]:
        assert self.meeting is not None
        if self._state == "proposing":
            return self._send_next_proposal()
        if self._state != "waiting_replies":
            return self._repair_progress()
        if self._replies_received.keys() < self._expected_responders:
            return self._repair_progress()

        if all(status == MeetingStatus.PENDING for status in self._replies_received.values()):
            meeting_id = int(self.meeting["id"])
            assert self._active_slot is not None
            self._set_status(meeting_id, self._active_slot, MeetingStatus.CONFIRMED)
            self._agreed_slot = self._active_slot
            self._cleanup_resolved(meeting_id, keep_slot=self._active_slot)
            messages = [
                self._dm(to, {"sd": "confirm", "meeting_id": meeting_id, "slot": self._active_slot})
                for to in sorted(self._expected_responders)
            ]
            messages.extend(self._trigger_bumped_reschedules(meeting_id, self._active_slot))
            self._state = "decided"
            messages.extend(self._repair_progress())
            return messages

        self._cleanup_resolved(int(self.meeting["id"]))
        self._state = "proposing"
        return self._send_next_proposal()

    def _send_next_proposal(self) -> list[dict]:
        assert self.meeting is not None
        slot = self._next_free_timeslot(self.meeting)
        meeting_id = int(self.meeting["id"])
        if slot is None:
            self._cleanup_resolved(meeting_id)
            self._state = "failed"
            return [
                self._dm(to, {"sd": "fail", "meeting_id": meeting_id})
                for to in sorted(self._expected_responders)
            ]

        self._tried_slots.add(slot)
        self._active_slot = slot
        self._replies_received = {}
        self._set_pending_with_local_bump(self.meeting, slot)
        self._state = "waiting_replies"
        if not self._expected_responders:
            return []
        return [
            self._dm(to, {"sd": "propose", "meeting_id": meeting_id, "slot": slot, "initiator": self.agent_id})
            for to in sorted(self._expected_responders)
        ]

    def _receive_propose(self, payload: dict, sender: int) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            slot = int(payload["slot"])
        except (KeyError, TypeError, ValueError):
            return []
        initiator = int(payload.get("initiator", sender))
        meeting = self._meeting_for(meeting_id)
        status = MeetingStatus.PENDING if self._accept_proposal(meeting, slot) else MeetingStatus.IMPOSSIBLE
        return [self._dm(initiator, {
            "sd": "reply",
            "meeting_id": meeting_id,
            "slot": slot,
            "reply_status": status.value,
            "attendee": self.agent_id,
        })]

    def _receive_reply(self, payload: dict, sender: int) -> None:
        if self.meeting is None:
            return
        try:
            meeting_id = int(payload["meeting_id"])
            slot = int(payload["slot"])
        except (KeyError, TypeError, ValueError):
            return
        if meeting_id != int(self.meeting["id"]) or slot != self._active_slot or sender not in self._expected_responders:
            return
        try:
            status = MeetingStatus(str(payload.get("reply_status")))
        except ValueError:
            status = MeetingStatus.IMPOSSIBLE
        self._replies_received[sender] = status

    def _receive_confirm(self, payload: dict) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            slot = int(payload["slot"])
        except (KeyError, TypeError, ValueError):
            return []
        self._set_status(meeting_id, slot, MeetingStatus.CONFIRMED)
        self._agreed_slot = slot
        self._cleanup_resolved(meeting_id, keep_slot=slot)
        return self._trigger_bumped_reschedules(meeting_id, slot)

    def _receive_reschedule(self, payload: dict) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            from_slot = int(payload["from_slot"])
            to_slot = int(payload["to_slot"])
        except (KeyError, TypeError, ValueError):
            return []
        self._pending_reschedule_requests[meeting_id] = (from_slot, to_slot)
        self._set_status(meeting_id, from_slot, MeetingStatus.IMPOSSIBLE)
        self._set_status(meeting_id, to_slot, MeetingStatus.CONFIRMED)
        return []

    def _receive_reschedule_request(self, payload: dict) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            from_slot = int(payload["from_slot"])
        except (KeyError, TypeError, ValueError):
            return []
        if self.agent_id != self._initiator_for(meeting_id):
            return [self._dm(self._initiator_for(meeting_id), {
                "sd": "reschedule_request",
                "meeting_id": meeting_id,
                "from_slot": from_slot,
            })]
        self._start_repair(meeting_id, from_slot)
        return self._repair_progress()

    def _receive_propose_reschedule(self, payload: dict, sender: int) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            from_slot = int(payload["from_slot"])
            to_slot = int(payload["to_slot"])
        except (KeyError, TypeError, ValueError):
            return []
        if sender != self._initiator_for(meeting_id):
            return []
        accepted = self._can_apply_bump_move(meeting_id, from_slot, to_slot)
        status = MeetingStatus.PENDING if accepted else MeetingStatus.IMPOSSIBLE
        if accepted:
            self._set_status(meeting_id, from_slot, MeetingStatus.PENDING)
        return [self._dm(sender, {
            "sd": "reschedule_reply",
            "meeting_id": meeting_id,
            "from_slot": from_slot,
            "to_slot": to_slot,
            "reply_status": status.value,
            "attendee": self.agent_id,
        })]

    def _receive_reschedule_reply(self, payload: dict, sender: int) -> None:
        try:
            meeting_id = int(payload["meeting_id"])
            to_slot = int(payload["to_slot"])
            status = MeetingStatus(str(payload.get("reply_status")))
        except (KeyError, TypeError, ValueError):
            return
        state = self._repair_states.get(meeting_id)
        if (
            not state
            or state.get("state") != "waiting_replies"
            or int(state.get("active_slot", -1)) != to_slot
            or sender not in state.get("expected", set())
        ):
            return
        state.setdefault("replies", {})[sender] = status

    def _receive_confirm_reschedule(self, payload: dict) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            from_slot = int(payload["from_slot"])
            to_slot = int(payload["to_slot"])
        except (KeyError, TypeError, ValueError):
            return []
        self._agreed_bump_moves[meeting_id] = (from_slot, to_slot)
        self._pending_reschedule_requests[meeting_id] = (from_slot, to_slot)
        self._set_status(meeting_id, from_slot, MeetingStatus.IMPOSSIBLE)
        self._set_status(meeting_id, to_slot, MeetingStatus.CONFIRMED)
        return []

    def _receive_fail_reschedule(self, payload: dict) -> list[dict]:
        try:
            meeting_id = int(payload["meeting_id"])
            from_slot = int(payload["from_slot"])
        except (KeyError, TypeError, ValueError):
            return []
        if self._status(meeting_id, from_slot) == MeetingStatus.PENDING:
            self._set_status(meeting_id, from_slot, MeetingStatus.CONFIRMED)
        return []

    def _accept_proposal(self, meeting: dict, slot: int) -> bool:
        if self._pending_meeting_at(slot) is not None:
            return False
        item = self._slot_items.get(slot)
        if item is None:
            self._set_status(int(meeting["id"]), slot, MeetingStatus.PENDING)
            return True
        if isinstance(item, dict) and item.get("type") == "errand":
            self._set_status(int(meeting["id"]), slot, MeetingStatus.PENDING)
            return True
        if isinstance(item, dict) and item.get("type") == "meeting":
            old_id = int(item["item_id"])
            if self._bumping_rule(meeting, self._meeting_for(old_id)):
                self._set_status(old_id, slot, MeetingStatus.BUMPED)
                self._bumped_by[(old_id, slot)] = int(meeting["id"])
                self._set_status(int(meeting["id"]), slot, MeetingStatus.PENDING)
                return True
        return False

    def _set_pending_with_local_bump(self, meeting: dict, slot: int) -> None:
        item = self._slot_items.get(slot)
        if isinstance(item, dict) and item.get("type") == "meeting":
            old_id = int(item["item_id"])
            if self._bumping_rule(meeting, self._meeting_for(old_id)):
                self._set_status(old_id, slot, MeetingStatus.BUMPED)
                self._bumped_by[(old_id, slot)] = int(meeting["id"])
                target = self._first_free_slot_except(slot)
                if target is not None:
                    self._agreed_reschedule_targets[old_id] = target
        self._set_status(int(meeting["id"]), slot, MeetingStatus.PENDING)

    def _trigger_bumped_reschedules(self, new_meeting_id: int, slot: int) -> list[dict]:
        messages: list[dict] = []
        for (old_id, old_slot), bumper_id in list(self._bumped_by.items()):
            if bumper_id != new_meeting_id or old_slot != slot:
                continue
            initiator = self._initiator_for(old_id)
            if initiator == self.agent_id:
                self._start_repair(old_id, old_slot)
                messages.extend(self._repair_progress())
            else:
                messages.append(self._dm(initiator, {
                    "sd": "reschedule_request",
                    "meeting_id": old_id,
                    "from_slot": old_slot,
                }))
        return messages

    def _start_repair(self, meeting_id: int, from_slot: int) -> None:
        self._repair_states[meeting_id] = {
            "from_slot": from_slot,
            "tried_slots": set(),
            "active_slot": None,
            "expected": {
                p for p in self._known_meetings.get(meeting_id, [])
                if p != self.agent_id
            },
            "replies": {},
            "state": "proposing",
        }

    def _repair_progress(self) -> list[dict]:
        messages: list[dict] = []
        for meeting_id in sorted(self._repair_states):
            messages.extend(self._repair_progress_one(meeting_id))
        return messages

    def _repair_progress_one(self, meeting_id: int) -> list[dict]:
        state = self._repair_states.get(meeting_id)
        if not state or state.get("state") in {"confirmed", "failed"}:
            return []
        if state.get("state") == "proposing":
            return self._send_next_repair_proposal(meeting_id, state)
        if state.get("state") != "waiting_replies":
            return []
        expected: set[int] = state.get("expected", set())
        replies: dict[int, MeetingStatus] = state.get("replies", {})
        if replies.keys() < expected:
            return []
        from_slot = int(state["from_slot"])
        to_slot = int(state["active_slot"])
        if all(status == MeetingStatus.PENDING for status in replies.values()):
            self._agreed_bump_moves[meeting_id] = (from_slot, to_slot)
            self._agreed_reschedule_targets[meeting_id] = to_slot
            self._set_status(meeting_id, from_slot, MeetingStatus.IMPOSSIBLE)
            self._set_status(meeting_id, to_slot, MeetingStatus.CONFIRMED)
            state["state"] = "confirmed"
            return [
                self._dm(participant, {
                    "sd": "confirm_reschedule",
                    "meeting_id": meeting_id,
                    "from_slot": from_slot,
                    "to_slot": to_slot,
                })
                for participant in sorted(expected)
            ]
        self._set_status(meeting_id, to_slot, MeetingStatus.POSSIBLE)
        state["state"] = "proposing"
        return self._send_next_repair_proposal(meeting_id, state)

    def _send_next_repair_proposal(self, meeting_id: int, state: dict[str, Any]) -> list[dict]:
        from_slot = int(state["from_slot"])
        target = self._next_repair_target(from_slot, state["tried_slots"])
        expected: set[int] = state.get("expected", set())
        if target is None:
            state["state"] = "failed"
            return [
                self._dm(participant, {
                    "sd": "fail_reschedule",
                    "meeting_id": meeting_id,
                    "from_slot": from_slot,
                })
                for participant in sorted(expected)
            ]
        state["tried_slots"].add(target)
        state["active_slot"] = target
        state["replies"] = {}
        self._set_status(meeting_id, target, MeetingStatus.PENDING)
        state["state"] = "waiting_replies"
        if not expected:
            return self._repair_progress_one(meeting_id)
        return [
            self._dm(participant, {
                "sd": "propose_reschedule",
                "meeting_id": meeting_id,
                "from_slot": from_slot,
                "to_slot": target,
                "initiator": self.agent_id,
            })
            for participant in sorted(expected)
        ]

    def _next_repair_target(self, from_slot: int, tried_slots: set[int]) -> int | None:
        return next(
            (
                slot
                for slot, item in sorted(self._slot_items.items())
                if slot != from_slot and slot not in tried_slots and item is None
            ),
            None,
        )

    def _can_apply_bump_move(self, meeting_id: int, from_slot: int, to_slot: int) -> bool:
        item = self._slot_items.get(from_slot)
        return (
            isinstance(item, dict)
            and item.get("type") == "meeting"
            and int(item.get("item_id", -1)) == meeting_id
            and self._slot_items.get(to_slot) is None
        )

    def _initiator_for(self, meeting_id: int) -> int:
        participants = self._known_meetings.get(meeting_id, [self.agent_id])
        return min(participants) if participants else self.agent_id

    def _cleanup_resolved(self, meeting_id: int, keep_slot: int | None = None) -> None:
        for slot, status in list(self._statuses.get(meeting_id, {}).items()):
            if status == MeetingStatus.PENDING and slot != keep_slot:
                self._set_status(meeting_id, slot, MeetingStatus.POSSIBLE)
                for (old_id, old_slot), bumper_id in list(self._bumped_by.items()):
                    if bumper_id == meeting_id and old_slot == slot:
                        self._set_status(old_id, old_slot, MeetingStatus.CONFIRMED)
                        del self._bumped_by[(old_id, old_slot)]

    def _next_free_timeslot(self, meeting: dict) -> int | None:
        for slot in sorted(self._slot_items):
            if slot not in self._tried_slots and self._slot_locally_available(meeting, slot, allow_bump=True):
                return slot
        return None

    def _slot_locally_available(self, meeting: dict, slot: int, allow_bump: bool) -> bool:
        if self._pending_meeting_at(slot) is not None:
            return False
        item = self._slot_items.get(slot)
        if item is None:
            return True
        if isinstance(item, dict) and item.get("type") == "errand":
            return self._first_free_slot_except(slot) is not None
        if allow_bump and isinstance(item, dict) and item.get("type") == "meeting":
            return self._bumping_rule(meeting, self._meeting_for(int(item["item_id"])))
        return False

    def _pending_meeting_at(self, slot: int) -> int | None:
        for meeting_id, slots in self._statuses.items():
            if slots.get(slot) == MeetingStatus.PENDING:
                return meeting_id
        return None

    def _bumping_rule(self, new_meeting: dict, old_meeting: dict) -> bool:
        return self._difficulty(old_meeting) < self._difficulty(new_meeting)

    def _difficulty(self, meeting: dict) -> float:
        participants = self._known_meetings.get(int(meeting["id"]), meeting.get("participants", []))
        return sum(self._sd_model.get(int(agent_id), 1.0) for agent_id in participants if int(agent_id) != self.agent_id)

    def _meeting_for(self, meeting_id: int) -> dict:
        return {
            "id": meeting_id,
            "participants": self._known_meetings.get(meeting_id, [self.agent_id]),
            "cost": self._meeting_costs.get(meeting_id, 1),
        }

    def _status(self, meeting_id: int, slot: int) -> MeetingStatus:
        return self._statuses.get(meeting_id, {}).get(slot, MeetingStatus.POSSIBLE)

    def _set_status(self, meeting_id: int, slot: int, status: MeetingStatus) -> None:
        self._statuses.setdefault(meeting_id, {})[slot] = status

    def _first_free_slot_except(self, slot: int) -> int | None:
        return next((s for s, item in sorted(self._slot_items.items()) if s != slot and item is None), None)

    def _dm(self, to: int, payload: dict) -> dict:
        return {
            "type": "dm",
            "to": to,
            "meeting_id": payload.get("meeting_id", self.meeting.get("id") if self.meeting else -1),
            "content": json.dumps(payload),
        }

    def _load_sd_model(self, game_config: GameConfig) -> dict[int, float]:
        raw = (
            getattr(game_config, "sd_model", None)
            or getattr(game_config, "scheduling_difficulty_model", None)
            or getattr(game_config, "scheduling_difficulty", None)
        )
        if isinstance(raw, dict):
            return {int(agent_id): float(value) for agent_id, value in raw.items()}
        if isinstance(raw, list):
            return {agent_id: float(value) for agent_id, value in enumerate(raw)}
        return {agent_id: 1.0 for agent_id in game_config.all_agent_ids}
