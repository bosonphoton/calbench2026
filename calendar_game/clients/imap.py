"""Streaming exact incremental MAP client.

This client is a complete-information baseline for the next incoming meeting,
not an oracle over the future stream. For each meeting, the initiator asks all
participants for their current per-slot insertion costs, chooses the feasible
slot with minimum total current displacement cost, and announces that slot.
"""

from __future__ import annotations

import json
from typing import Any

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult
from calendar_game.clients.dsm import _apply_actions_to_items, _parse_slot_items, _schedule_actions


def _empty_turn() -> TurnResult:
    return TurnResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _empty_decide() -> DecideResult:
    return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _parse_imap(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(payload, dict) and "imap" in payload:
        return payload
    return None


def _local_insertion_cost(slot: int, slot_items: dict[int, dict[str, int] | None]) -> int | None:
    """Cost to insert the current meeting into slot on one local calendar.

    This exact baseline supports the current benchmark's local movable errands.
    Existing meetings are treated as hard commitments for this streaming solver;
    globally coordinated re-bumping of prior meetings is a different dynamic
    DCOP repair problem.
    """
    item = slot_items.get(slot)
    if item is None:
        return 0
    if not isinstance(item, dict) or item.get("type") != "errand":
        return None
    has_target = any(other != slot and value is None for other, value in slot_items.items())
    if not has_target:
        return None
    return max(0, int(item.get("cost", 0)))


def _local_cost_table(slot_items: dict[int, dict[str, int] | None]) -> dict[int, int | None]:
    return {slot: _local_insertion_cost(slot, slot_items) for slot in sorted(slot_items)}


class IncrementalMAPClient(BaseClient):
    """Exact current-meeting insertion baseline for streaming MAP."""

    def __init__(self) -> None:
        self.agent_id: int = -1
        self.game_config: GameConfig | None = None
        self.meeting: dict | None = None
        self._slot_items: dict[int, dict[str, int] | None] = {}
        self._initiator_id: int = -1
        self._role: str = "responder"
        self._state: str = "idle"
        self._expected_responders: set[int] = set()
        self._costs_received: dict[int, dict[int, int | None]] = {}
        self._agreed_slot: int | None = None
        self._active_round: bool = False

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self.game_config = game_config

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self.meeting = meeting
        self._slot_items = _parse_slot_items(calendar_render)
        participants = list(meeting.get("participants", []))
        self._initiator_id = min(participants) if participants else self.agent_id
        self._role = "initiator" if self.agent_id == self._initiator_id else "responder"
        self._expected_responders = {p for p in participants if p != self._initiator_id}
        self._costs_received = {}
        self._agreed_slot = None
        self._active_round = True
        self._state = "requesting" if self._role == "initiator" else "waiting_request"

    def observe_calendar(self, calendar_render: str) -> None:
        self._slot_items = _parse_slot_items(calendar_render)

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        if not self._active_round:
            return _empty_turn()
        if self._role == "initiator":
            return self._initiator_turn(messages)
        return self._responder_turn(messages)

    def _initiator_turn(self, messages: list[dict]) -> TurnResult:
        assert self.meeting is not None
        if self._state == "requesting":
            if not self._expected_responders:
                self._agreed_slot = self._best_slot({})
                self._state = "decided"
                return _empty_turn()
            payload = {
                "imap": "cost_request",
                "meeting_id": self.meeting["id"],
                "slots": sorted(self._slot_items),
            }
            self._state = "waiting_costs"
            return TurnResult(
                tool_calls=[
                    {
                        "type": "dm",
                        "to": responder,
                        "meeting_id": self.meeting["id"],
                        "content": json.dumps(payload),
                    }
                    for responder in sorted(self._expected_responders)
                ],
                text=None,
                thinking=None,
                usage=None,
                latency_ms=None,
                raw=None,
            )

        if self._state != "waiting_costs":
            return _empty_turn()

        for msg in messages:
            parsed = _parse_imap(msg.get("content", ""))
            if not parsed or parsed.get("imap") != "costs" or msg.get("from") not in self._expected_responders:
                continue
            raw_costs = parsed.get("costs", {})
            costs: dict[int, int | None] = {}
            for key, value in raw_costs.items():
                slot = int(key)
                costs[slot] = None if value is None else int(value)
            self._costs_received[int(msg["from"])] = costs

        if self._costs_received.keys() < self._expected_responders:
            return _empty_turn()

        self._agreed_slot = self._best_slot(self._costs_received)
        self._state = "decided"
        if self._agreed_slot is None:
            return _empty_turn()
        decision = {"imap": "decision", "meeting_id": self.meeting["id"], "slot": self._agreed_slot}
        return TurnResult(
            tool_calls=[
                {
                    "type": "dm",
                    "to": responder,
                    "meeting_id": self.meeting["id"],
                    "content": json.dumps(decision),
                }
                for responder in sorted(self._expected_responders)
            ],
            text=None,
            thinking=None,
            usage=None,
            latency_ms=None,
            raw=None,
        )

    def _best_slot(self, responder_costs: dict[int, dict[int, int | None]]) -> int | None:
        own_costs = _local_cost_table(self._slot_items)
        best: tuple[int, int] | None = None
        for slot, own_cost in own_costs.items():
            if own_cost is None:
                continue
            total = own_cost
            feasible = True
            for responder in self._expected_responders:
                cost = responder_costs.get(responder, {}).get(slot)
                if cost is None:
                    feasible = False
                    break
                total += cost
            if feasible and (best is None or (total, slot) < best):
                best = (total, slot)
        return None if best is None else best[1]

    def _responder_turn(self, messages: list[dict]) -> TurnResult:
        assert self.meeting is not None
        for msg in messages:
            if msg.get("from") != self._initiator_id:
                continue
            parsed = _parse_imap(msg.get("content", ""))
            if not parsed:
                continue
            if parsed.get("imap") == "decision":
                self._agreed_slot = int(parsed["slot"]) if parsed.get("slot") is not None else None
                self._state = "decided"
                return _empty_turn()
            if parsed.get("imap") != "cost_request":
                continue
            costs = _local_cost_table(self._slot_items)
            payload = {
                "imap": "costs",
                "meeting_id": self.meeting["id"],
                "costs": {str(slot): cost for slot, cost in costs.items()},
            }
            self._state = "scored"
            return TurnResult(
                tool_calls=[{
                    "type": "dm",
                    "to": self._initiator_id,
                    "meeting_id": self.meeting["id"],
                    "content": json.dumps(payload),
                }],
                text=None,
                thinking=None,
                usage=None,
                latency_ms=None,
                raw=None,
            )
        return _empty_turn()

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        if self._agreed_slot is None:
            return _empty_decide()
        slot_items = _parse_slot_items(calendar_render)
        actions = _schedule_actions(meeting, self._agreed_slot, slot_items)
        if not actions:
            return _empty_decide()
        self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
        self._active_round = False
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        if self.meeting is None:
            return _empty_decide()
        costs = _local_cost_table(self._slot_items)
        feasible_slots = [slot for slot, cost in costs.items() if cost is not None]
        if attempt >= len(feasible_slots):
            return _empty_decide()
        actions = _schedule_actions(self.meeting, feasible_slots[attempt], self._slot_items)
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)
