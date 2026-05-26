"""DSM (Distributed Score-based Multi-round) scripted client.

Implements a simplified version of the protocol from:
  Farhadi & Jennings (2021) "A Faithful Mechanism for Incremental Multi-Agent
  Agreement Problems with Self-Interested and Privacy-Preserving Agents"

Simplifications vs. the full paper:
  - No convenience points / reward function (agents play the socially-optimal
    heuristic strategy directly: score truthfully based on calendar availability)
  - Multiple DSM sub-rounds per meeting (proposal → score → assess, repeated
    until a fully feasible slot is found or initiator-free slots are exhausted)
  - Cost-aware satisfaction levels: 0 = infeasible, D-1 = no displacement,
    lower positive scores = feasible but increasingly costly displacement

Protocol per meeting:
  Initiator (lowest-id participant):
    turn 1  → send up to NUM_PROPOSALS untried free slots to all responders
    turn 2+ → collect score DMs; once all received, announce a fully-feasible
              slot if one exists, otherwise try the next untried batch
  Responders:
    turn 1  → wait for proposals DM
    turn 2  → score each proposed slot, DM scores back to initiator
    turn 3+ → rescore newer proposal DMs until a decision DM arrives

  decide() → both roles schedule the agreed slot.
"""

from __future__ import annotations

import json
import re
from typing import Any

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult

# Number of satisfaction levels (0 = infeasible, D-1 = fully satisfactory).
# With D=12, feasible scores 1..11 map cleanly to displacement costs 10..0.
_D = 12
# Maximum slots the initiator proposes per round
_NUM_PROPOSALS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_turn() -> TurnResult:
    return TurnResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _empty_decide() -> DecideResult:
    return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)


def _parse_slot_states(calendar_render: str) -> dict[int, str]:
    """Return {slot_index: "free" | "busy"} from a calendar render string."""
    return {
        slot: "free" if item is None else "busy"
        for slot, item in _parse_slot_items(calendar_render).items()
    }


def _parse_slot_items(calendar_render: str) -> dict[int, dict[str, int] | None]:
    """Return slot contents parsed from Calendar.render()."""
    items: dict[int, dict[str, int] | None] = {}
    for line in calendar_render.splitlines():
        if "Slot" not in line or ":" not in line:
            continue
        try:
            slot_part, content = line.split(":", 1)
            slot_idx = int(slot_part.split("Slot")[1].strip())
            if "[FREE]" in content:
                items[slot_idx] = None
                continue
            blocked = re.search(r"\bBlocked(?: Errand)? #(\d+) \(cost=(\d+)\)", content)
            if blocked:
                items[slot_idx] = {
                    "type": "blocked",
                    "item_id": int(blocked.group(1)),
                    "cost": int(blocked.group(2)),
                }
                continue
            errand = re.search(r"\bErrand #(\d+) \(cost=(\d+)\)", content)
            if errand:
                items[slot_idx] = {"type": "errand", "item_id": int(errand.group(1)), "cost": int(errand.group(2))}
                continue
            meeting = re.search(r"Meeting M(\d+) \(cost=(\d+)\)", content)
            if meeting:
                items[slot_idx] = {"type": "meeting", "item_id": int(meeting.group(1)), "cost": int(meeting.group(2))}
                continue
            items[slot_idx] = {"type": "unknown", "item_id": -1, "cost": 0}
        except (IndexError, ValueError):
            continue
    return items


def _free_slots(slot_states: dict[int, str]) -> list[int]:
    return sorted(s for s, state in slot_states.items() if state == "free")


def _schedulable_slots(slot_items: dict[int, dict[str, int] | None]) -> list[int]:
    return sorted(s for s in slot_items if _is_schedulable(s, slot_items))


def _is_schedulable(slot: int, slot_items: dict[int, Any]) -> bool:
    """Return whether this agent can locally schedule into slot this turn."""
    item = slot_items.get(slot)
    if item is None or item == "free":
        return True
    if item == "busy" or not isinstance(item, dict):
        return False
    if item.get("type") not in {"errand", "meeting"}:
        return False
    return any(other != slot and value is None for other, value in slot_items.items())


def _score_proposed_slot(slot: int, slot_items: dict[int, Any], displacements: list[dict] | None) -> int:
    item = slot_items.get(slot)
    if not isinstance(item, dict) or item.get("type") != "meeting":
        return _score_slot(slot, slot_items)

    for displacement in displacements or []:
        try:
            if (
                int(displacement["meeting_id"]) == int(item["item_id"])
                and int(displacement["from_slot"]) == slot
                and _displacement_target_clearable(int(displacement["to_slot"]), slot_items, displacements)
            ):
                return _score_from_cost(_slot_displacement_cost(slot, slot_items))
        except (KeyError, TypeError, ValueError):
            continue
    return 0


def _displacement_target_clearable(
    target_slot: int,
    slot_items: dict[int, Any],
    displacements: list[dict] | None,
    cleared_slots: set[int] | None = None,
) -> bool:
    if cleared_slots and target_slot in cleared_slots:
        return True
    if slot_items.get(target_slot) is None:
        return True
    target_item = slot_items.get(target_slot)
    if not isinstance(target_item, dict) or target_item.get("type") != "meeting":
        return False
    target_meeting_id = int(target_item.get("item_id", -1))
    for displacement in displacements or []:
        try:
            if (
                int(displacement["meeting_id"]) == target_meeting_id
                and int(displacement["from_slot"]) == target_slot
            ):
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _score_slot(slot: int, slot_states: dict[int, Any]) -> int:
    """DSM score: 0 if infeasible, otherwise higher is lower local cost."""
    if not _is_schedulable(slot, slot_states):
        return 0
    return _score_from_cost(_slot_displacement_cost(slot, slot_states))


def _score_from_cost(cost: int) -> int:
    """Map displacement cost to DSM satisfaction, preserving 0 for infeasible."""
    return max(1, (_D - 1) - max(0, cost))


def _scoring_cost(score: int, score_levels: int = _D) -> int:
    """Paper DSM scoring cost C(s) = (D - s - 1) sign(s)."""
    score = max(0, min(score_levels - 1, int(score)))
    if score <= 0:
        return 0
    return max(0, score_levels - score - 1)


def _score_summary(scores: dict[int, int], score_levels: int = _D) -> dict[str, int]:
    """Return paper DSM availability, flexibility, and scoring cost for a score vector."""
    positive_scores = [max(0, min(score_levels - 1, int(s))) for s in scores.values() if int(s) > 0]
    availability = len(positive_scores)
    base = availability + 1
    counts = {score: positive_scores.count(score) for score in range(1, score_levels)}
    flexibility = sum(score * (base ** (score - 1)) * counts[score] for score in range(1, score_levels))
    return {
        "availability": availability,
        "flexibility": flexibility,
        "cost": sum(_scoring_cost(score, score_levels) for score in positive_scores),
    }


def _slot_displacement_cost(slot: int, slot_items: dict[int, Any]) -> int:
    item = slot_items.get(slot)
    if item is None or item == "free":
        return 0
    if isinstance(item, dict):
        return max(0, int(item.get("cost", _D - 2)))
    return _D - 2


def _local_plan_cost(
    slot: int,
    slot_items: dict[int, Any],
    displacements: list[dict] | None,
    include_current_slot: bool,
    cleared_slots: set[int] | None = None,
) -> int | None:
    """Return local cost for this proposal, or None if the chain is infeasible."""
    cost = 0
    counted_from_slots: set[int] = set()
    if include_current_slot:
        item = slot_items.get(slot)
        if item is None:
            pass
        elif isinstance(item, dict) and item.get("type") == "errand":
            cost += _slot_displacement_cost(slot, slot_items)
            counted_from_slots.add(slot)
        elif isinstance(item, dict) and item.get("type") == "meeting":
            if not any(
                int(move.get("from_slot", -1)) == slot
                and int(move.get("meeting_id", -1)) == int(item.get("item_id", -2))
                for move in displacements or []
            ):
                return None
        else:
            return None

    for move in displacements or []:
        try:
            meeting_id = int(move["meeting_id"])
            from_slot = int(move["from_slot"])
            to_slot = int(move["to_slot"])
        except (KeyError, TypeError, ValueError):
            return None
        item = slot_items.get(from_slot)
        if (
            not isinstance(item, dict)
            or item.get("type") != "meeting"
            or int(item.get("item_id", -1)) != meeting_id
        ):
            return None
        if not _displacement_target_clearable(to_slot, slot_items, displacements, cleared_slots):
            return None
        if from_slot not in counted_from_slots:
            cost += _slot_displacement_cost(from_slot, slot_items)
            counted_from_slots.add(from_slot)
    return cost


def _score_plan(
    slot: int,
    slot_items: dict[int, Any],
    displacements: list[dict] | None,
    include_current_slot: bool,
    cleared_slots: set[int] | None = None,
) -> int:
    cost = _local_plan_cost(slot, slot_items, displacements, include_current_slot, cleared_slots)
    if cost is None:
        return 0
    return _score_from_cost(cost)


def _schedule_actions(
    meeting: dict,
    slot: int,
    slot_items: dict[int, dict[str, int] | None],
    requested_targets: dict[int, int] | None = None,
    requested_reschedules: list[dict] | None = None,
) -> list[dict]:
    """Build schedule/reschedule actions needed to place meeting in slot."""
    item = slot_items.get(slot)
    if item is None:
        return [{"type": "schedule", "meeting_id": meeting["id"], "slot": slot}]
    if not isinstance(item, dict) or item.get("type") not in {"errand", "meeting"}:
        return []
    requested_targets = requested_targets or {}
    requested_reschedules = requested_reschedules or []
    if requested_reschedules:
        actions: list[dict] = []
        for move in requested_reschedules:
            try:
                meeting_id = int(move["meeting_id"])
                from_slot = int(move["from_slot"])
                to_slot = int(move["to_slot"])
            except (KeyError, TypeError, ValueError):
                return []
            from_item = slot_items.get(from_slot)
            if (
                not isinstance(from_item, dict)
                or from_item.get("type") != "meeting"
                or int(from_item.get("item_id", -1)) != meeting_id
                or not _displacement_target_clearable(to_slot, slot_items, requested_reschedules)
            ):
                return []
            actions.append({
                "type": "reschedule",
                "item_id": meeting_id,
                "from_slot": from_slot,
                "to_slot": to_slot,
                "justification": "Move the prior meeting to honor the negotiated scheduling plan.",
            })
        actions.append({"type": "schedule", "meeting_id": meeting["id"], "slot": slot})
        return actions

    target = requested_targets.get(item["item_id"])
    if item.get("type") == "meeting" and target is None:
        return []
    if target is not None and slot_items.get(target) is not None:
        return []
    if target is None:
        target = next((s for s, value in slot_items.items() if s != slot and value is None), None)
    if target is None:
        return []
    return [
        {
            "type": "reschedule",
            "item_id": item["item_id"],
            "from_slot": slot,
            "to_slot": target,
            "justification": "Free the selected meeting slot after coordination selected it.",
        },
        {"type": "schedule", "meeting_id": meeting["id"], "slot": slot},
    ]


def _apply_actions_to_items(
    slot_items: dict[int, dict[str, int] | None],
    actions: list[dict],
    meeting: dict,
) -> dict[int, dict[str, int] | None]:
    updated = dict(slot_items)
    moved: list[tuple[int, dict[str, int] | None]] = []
    for action in actions:
        if action.get("type") != "reschedule":
            continue
        from_slot = int(action["from_slot"])
        to_slot = int(action["to_slot"])
        moved.append((to_slot, updated.get(from_slot)))
        updated[from_slot] = None
    for to_slot, item in moved:
        updated[to_slot] = item
    for action in actions:
        if action.get("type") == "schedule":
            updated[int(action["slot"])] = {
                "type": "meeting",
                "item_id": int(action["meeting_id"]),
                "cost": int(meeting.get("cost", 1)),
            }
    return updated


def _parse_dsm(content: str) -> dict[str, Any] | None:
    """Parse a DSM protocol JSON message; return None if not DSM."""
    try:
        msg = json.loads(content)
        if isinstance(msg, dict) and "dsm" in msg:
            return msg
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# DSMClient
# ---------------------------------------------------------------------------

class DSMClient(BaseClient):
    """Scripted client implementing the DSM negotiation protocol."""

    def __init__(self) -> None:
        self.agent_id: int = -1
        self.game_config: GameConfig | None = None
        # Per-round state (reset by start_round)
        self.meeting: dict | None = None
        self._slot_states: dict[int, str] = {}
        self._slot_items: dict[int, dict[str, int] | None] = {}
        self._role: str = "responder"          # "initiator" or "responder"
        self._state: str = "idle"
        self._initiator_id: int = -1
        self._expected_responders: set[int] = set()
        self._proposals: list[int] = []
        self._tried_slots: set[int] = set()
        self._proposal_round: int = 0
        self._scores_received: dict[int, dict[int, int]] = {}  # responder → {slot: score}
        self._proposal_plans: dict[int, dict] = {}
        self._proposal_displacements: dict[int, list[dict]] = {}
        self._proposal_responders: set[int] = set()
        self._agreed_slot: int | None = None
        self._best_slot: int | None = None
        self._best_plan_id: int | None = None
        self._best_score: int = -1
        self._best_rank: tuple[int, int, int, int] | None = None
        self._best_displacements: list[dict] = []
        self._best_feasible_slot: int | None = None
        self._best_feasible_plan_id: int | None = None
        self._best_feasible_score: int = -1
        self._best_feasible_rank: tuple[int, int, int, int] | None = None
        self._best_feasible_displacements: list[dict] = []
        self._known_meetings: dict[int, list[int]] = {}
        self._agreed_reschedule_targets: dict[int, int] = {}
        self._agreed_reschedule_moves: list[dict] = []
        self._pending_reschedule_requests: dict[int, tuple[int, int]] = {}
        self._pending_reschedule_moves: list[dict] = []
        self._active_round: bool = False

    # ------------------------------------------------------------------
    # BaseClient interface
    # ------------------------------------------------------------------

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self.game_config = game_config
        for prior in getattr(game_config, "dsm_prior_meetings", []):
            try:
                self._known_meetings[int(prior["id"])] = [int(p) for p in prior.get("participants", [])]
            except (KeyError, TypeError, ValueError):
                continue

    def observe_calendar(self, calendar_render: str) -> None:
        self._slot_items = _parse_slot_items(calendar_render)
        self._slot_states = _parse_slot_states(calendar_render)

    def _num_proposals(self) -> int:
        if self.game_config is None:
            return _NUM_PROPOSALS
        return max(1, int(getattr(self.game_config, "dsm_num_proposals", _NUM_PROPOSALS)))

    def _cascade_depth(self) -> int:
        if self.game_config is None:
            return 1
        return max(0, int(getattr(self.game_config, "dsm_cascade_depth", 1)))

    def _displacement_targets(self) -> int:
        if self.game_config is None:
            return 4
        return max(1, int(getattr(self.game_config, "dsm_displacement_targets", 4)))

    def _exhaustive_search(self) -> bool:
        return bool(getattr(self.game_config, "dsm_exhaustive_search", True))

    def _stop_on_perfect(self) -> bool:
        return bool(getattr(self.game_config, "dsm_stop_on_perfect", True))

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self.meeting = meeting
        self._known_meetings[int(meeting["id"])] = list(meeting.get("participants", []))
        self._slot_items = _parse_slot_items(calendar_render)
        self._slot_states = _parse_slot_states(calendar_render)
        participants: list[int] = meeting.get("participants", [])
        self._initiator_id = min(participants) if participants else self.agent_id
        self._role = "initiator" if self.agent_id == self._initiator_id else "responder"
        self._expected_responders = {p for p in participants if p != self._initiator_id}
        self._proposals = []
        self._tried_slots = set()
        self._proposal_round = 0
        self._scores_received = {}
        self._proposal_plans = {}
        self._proposal_displacements = {}
        self._proposal_responders = set()
        self._agreed_slot = None
        self._best_slot = None
        self._best_plan_id = None
        self._best_score = -1
        self._best_rank = None
        self._best_displacements = []
        self._best_feasible_slot = None
        self._best_feasible_plan_id = None
        self._best_feasible_score = -1
        self._best_feasible_rank = None
        self._best_feasible_displacements = []
        self._agreed_reschedule_targets = {}
        self._agreed_reschedule_moves = []
        self._pending_reschedule_moves = []
        self._active_round = True
        self._state = "proposing" if self._role == "initiator" else "waiting_proposal"

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        self._record_reschedule_requests(messages)
        if self._should_handle_as_displacement_responder(messages):
            return self._responder_turn(messages)
        if self._active_round and self._role == "initiator":
            return self._initiator_turn(messages)
        if self._active_round:
            return self._responder_turn(messages)
        return _empty_turn()

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        slot_items = _parse_slot_items(calendar_render)
        slot = self._agreed_slot

        # Fall back to first locally schedulable slot if agreement was not reached
        # or the agreed slot is no longer locally clearable.
        if slot is None or not _is_schedulable(slot, slot_items):
            schedulable = _schedulable_slots(slot_items)
            slot = schedulable[0] if schedulable else None

        if slot is None:
            return _empty_decide()

        actions = _schedule_actions(
            meeting,
            slot,
            slot_items,
            self._agreed_reschedule_targets,
            self._agreed_reschedule_moves,
        )
        if not actions:
            return _empty_decide()
        self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
        self._active_round = False

        return DecideResult(
            tool_calls=actions,
            text=None, thinking=None, usage=None, latency_ms=None, raw=None,
        )

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        if self.meeting is None:
            return _empty_decide()
        schedulable = _schedulable_slots(self._slot_items)
        if attempt < len(schedulable):
            actions = _schedule_actions(
                self.meeting,
                schedulable[attempt],
                self._slot_items,
                self._agreed_reschedule_targets,
                self._agreed_reschedule_moves,
            )
            return DecideResult(
                tool_calls=actions,
                text=None, thinking=None, usage=None, latency_ms=None, raw=None,
            )
        return _empty_decide()

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        slot_items = _parse_slot_items(calendar_render)
        self._slot_items = slot_items
        actions: list[dict] = []
        if self._pending_reschedule_moves:
            for move in self._pending_reschedule_moves:
                try:
                    meeting_id = int(move["meeting_id"])
                    from_slot = int(move["from_slot"])
                    to_slot = int(move["to_slot"])
                except (KeyError, TypeError, ValueError):
                    continue
                item = slot_items.get(from_slot)
                if (
                    isinstance(item, dict)
                    and item.get("type") == "meeting"
                    and int(item.get("item_id", -1)) == meeting_id
                    and _displacement_target_clearable(to_slot, slot_items, self._pending_reschedule_moves)
                ):
                    actions.append({
                        "type": "reschedule",
                        "item_id": meeting_id,
                        "from_slot": from_slot,
                        "to_slot": to_slot,
                        "justification": "Move the prior meeting to honor the negotiated scheduling plan.",
                    })
            self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
            return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

        for meeting_id, (from_slot, to_slot) in sorted(self._pending_reschedule_requests.items()):
            item = slot_items.get(from_slot)
            if (
                isinstance(item, dict)
                and item.get("type") == "meeting"
                and item.get("item_id") == meeting_id
                and slot_items.get(to_slot) is None
            ):
                actions.append({
                    "type": "reschedule",
                    "item_id": meeting_id,
                    "from_slot": from_slot,
                    "to_slot": to_slot,
                    "justification": "Move the prior meeting to honor the requested coordination commitment.",
                })
        self._slot_items = _apply_actions_to_items(slot_items, actions, meeting)
        return DecideResult(tool_calls=actions, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    # ------------------------------------------------------------------
    # Initiator turns
    # ------------------------------------------------------------------

    def _initiator_turn(self, messages: list[dict]) -> TurnResult:
        assert self.meeting is not None

        if self._state == "proposing":
            self._proposals = self._next_proposals()
            self._proposal_displacements = {
                plan_id: self._proposal_plans[plan_id]["displacements"]
                for plan_id in self._proposals
            }
            self._proposal_responders = self._responders_for_proposals()
            self._scores_received = {}

            # No responders or no free slots: decide immediately without messaging
            if not self._proposal_responders or not self._proposals:
                if self._proposals:
                    plan = self._proposal_plans[self._proposals[0]]
                    self._agreed_slot = plan["slot"]
                    self._agreed_reschedule_moves = plan["displacements"]
                else:
                    self._agreed_slot = self._best_slot
                self._state = "decided"
                return _empty_turn()

            self._proposal_round += 1
            self._tried_slots.update(self._proposals)
            tool_calls = [
                {
                    "type": "dm",
                    "to": r,
                    "meeting_id": self.meeting["id"],
                    "content": json.dumps(self._proposal_payload_for(r)),
                }
                for r in self._proposal_responders
            ]
            self._state = "waiting_scores"
            return TurnResult(tool_calls=tool_calls, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

        if self._state == "waiting_scores":
            # Accumulate score DMs from responders
            for msg in messages:
                parsed = _parse_dsm(msg["content"])
                if parsed and parsed.get("dsm") == "scores" and msg["from"] in self._proposal_responders:
                    score_round = parsed.get("round")
                    if score_round is not None and score_round != self._proposal_round:
                        continue
                    raw_scores: dict = parsed.get("scores", {})
                    self._scores_received[msg["from"]] = {int(k): int(v) for k, v in raw_scores.items()}

            if self._scores_received.keys() < self._proposal_responders:
                return _empty_turn()  # still waiting for more responders

            # All scores in: remember the best feasible slot seen so far, then
            # keep exploring to reduce realized displacement cost.
            self._assess()
            should_continue = (
                self._exhaustive_search()
                and (not self._stop_on_perfect() or not self._best_feasible_is_perfect())
                and self._has_untried_free_slots()
            )
            if should_continue:
                self._state = "proposing"
                return self._initiator_turn([])

            self._agreed_slot = self._best_feasible_slot if self._best_feasible_slot is not None else self._best_slot

            displacements = (
                self._best_feasible_displacements
                if self._best_feasible_slot is not None
                else self._best_displacements
            )
            for displacement in displacements:
                self._agreed_reschedule_targets[displacement["meeting_id"]] = displacement["to_slot"]
            self._agreed_reschedule_moves = displacements

            decision = {
                "dsm": "decision",
                "slot": self._agreed_slot,
                "plan_id": self._best_feasible_plan_id if self._best_feasible_slot is not None else self._best_plan_id,
                "participants": self.meeting.get("participants", []),
                "displacements": displacements,
            }
            decision_responders = self._responders_for_displacements(displacements)
            tool_calls = [
                {
                    "type": "dm",
                    "to": r,
                    "meeting_id": self.meeting["id"],
                    "content": json.dumps(self._decision_payload_for(r, decision)),
                }
                for r in decision_responders
            ]
            self._state = "decided"
            return TurnResult(tool_calls=tool_calls, text=None, thinking=None, usage=None, latency_ms=None, raw=None)

        return _empty_turn()

    def _next_proposals(self) -> list[int]:
        """Return the next untried batch of locally schedulable agreement ids."""
        candidates: list[int] = []
        plans: dict[int, dict] = {}
        for slot in _schedulable_slots(self._slot_items):
            if not self._slot_can_be_proposed(slot):
                continue
            for alt_idx, displacements in enumerate(self._meeting_displacement_plans(slot)):
                plan_id = self._plan_id(slot, alt_idx)
                if plan_id in self._tried_slots:
                    continue
                plans[plan_id] = {"id": plan_id, "slot": slot, "displacements": displacements}
                candidates.append(plan_id)
        self._proposal_plans.update(plans)

        def local_value(plan_id: int) -> int:
            plan = self._proposal_plans[plan_id]
            return _score_plan(plan["slot"], self._slot_items, plan["displacements"], True)

        return sorted(candidates, key=lambda plan_id: (-local_value(plan_id), self._proposal_plans[plan_id]["slot"], plan_id))[:self._num_proposals()]

    def _has_untried_free_slots(self) -> bool:
        return any(
            plan_id not in self._tried_slots
            for slot in _schedulable_slots(self._slot_items)
            if self._slot_can_be_proposed(slot)
            for plan_id in self._plan_ids_for_slot(slot)
        )

    def _plan_ids_for_slot(self, slot: int) -> list[int]:
        return [
            self._plan_id(slot, alt_idx)
            for alt_idx, _ in enumerate(self._meeting_displacement_plans(slot))
        ]

    def _plan_id(self, slot: int, alt_idx: int) -> int:
        if alt_idx == 0:
            return slot
        return ((slot + 1) * 1000) + alt_idx

    def _slot_can_be_proposed(self, slot: int) -> bool:
        item = self._slot_items.get(slot)
        if not isinstance(item, dict) or item.get("type") != "meeting":
            return True
        return bool(self._meeting_displacements(slot))

    def _responders_for_proposals(self) -> set[int]:
        displacements = [
            displacement
            for moves in self._proposal_displacements.values()
            for displacement in moves
        ]
        return self._responders_for_displacements(displacements)

    def _responders_for_displacements(self, displacements: list[dict]) -> set[int]:
        responders = set(self._expected_responders)
        current = set(self.meeting.get("participants", [])) if self.meeting else set()
        for displacement in displacements:
            for participant in self._known_meetings.get(displacement["meeting_id"], []):
                if participant != self.agent_id and participant not in current:
                    responders.add(participant)
        return responders

    def _responders_for_slot(self, slot: int) -> set[int]:
        return self._responders_for_displacements(self._proposal_displacements.get(slot, []))

    def _responders_for_plan(self, plan_id: int) -> set[int]:
        return self._responders_for_displacements(self._proposal_displacements.get(plan_id, []))

    def _proposal_payload_for(self, recipient: int) -> dict:
        plans = []
        displacements_by_plan: dict[str, list[dict]] = {}
        cleared_by_plan: dict[str, list[int]] = {}
        for plan_id in self._proposals:
            plan = self._proposal_plans[plan_id]
            visible, cleared = self._visible_displacements_for(recipient, plan["displacements"])
            plans.append({"id": plan_id, "slot": plan["slot"], "displacements": visible, "cleared_slots": cleared})
            if visible:
                displacements_by_plan[str(plan_id)] = visible
            if cleared:
                cleared_by_plan[str(plan_id)] = cleared
        return {
            "dsm": "proposals",
            "slots": self._proposals,
            "plans": plans,
            "round": self._proposal_round,
            "participants": (
                self.meeting.get("participants", [])
                if self.meeting and recipient in set(self.meeting.get("participants", []))
                else []
            ),
            "displacements": displacements_by_plan,
            "cleared_slots": cleared_by_plan,
        }

    def _decision_payload_for(self, recipient: int, decision: dict) -> dict:
        visible, cleared = self._visible_displacements_for(recipient, decision.get("displacements", []))
        payload = dict(decision)
        if self.meeting and recipient not in set(self.meeting.get("participants", [])):
            payload["participants"] = []
        payload["displacements"] = visible
        if cleared:
            payload["cleared_slots"] = cleared
        return payload

    def _visible_displacements_for(self, recipient: int, displacements: list[dict]) -> tuple[list[dict], list[int]]:
        visible: list[dict] = []
        hidden_cleared: list[int] = []
        for move in displacements:
            participants = set(self._known_meetings.get(int(move["meeting_id"]), []))
            if recipient in participants:
                visible.append(move)
            else:
                hidden_cleared.append(int(move["from_slot"]))
        return visible, sorted(set(hidden_cleared))

    def _best_feasible_is_perfect(self) -> bool:
        if self._best_feasible_slot is None:
            return False
        required_responders = self._responders_for_displacements(self._best_feasible_displacements)
        return self._best_feasible_score >= (_D - 1) * (len(required_responders) + 1)

    def _plan_rank(self, plan_id: int, total: int) -> tuple[int, int, int, int]:
        plan = self._proposal_plans[plan_id]
        required_count = len(self._responders_for_plan(plan_id)) + 1
        satisfaction_loss = ((_D - 1) * required_count) - total
        return (satisfaction_loss, required_count, plan["slot"], plan_id)

    def _assess(self) -> int | None:
        """Return a fully-feasible slot if the current batch has one."""
        # Aggregate scores across responders + initiator's own score
        agg: dict[int, int] = {}
        for plan_id in self._proposals:
            plan = self._proposal_plans[plan_id]
            total = _score_plan(plan["slot"], self._slot_items, plan["displacements"], True)
            for responder in self._responders_for_plan(plan_id):
                total += self._scores_received.get(responder, {}).get(plan_id, 0)
            agg[plan_id] = total
            rank = self._plan_rank(plan_id, total)
            if self._best_rank is None or rank < self._best_rank:
                self._best_score = total
                self._best_rank = rank
                self._best_slot = plan["slot"]
                self._best_plan_id = plan_id
                self._best_displacements = plan["displacements"]

        # Prefer slots that are feasible for everyone (score > 0 for all agents)
        def is_fully_feasible(plan_id: int) -> bool:
            plan = self._proposal_plans[plan_id]
            if _score_plan(plan["slot"], self._slot_items, plan["displacements"], True) == 0:
                return False
            return all(self._scores_received.get(r, {}).get(plan_id, 0) > 0 for r in self._responders_for_plan(plan_id))

        feasible = [plan_id for plan_id in self._proposals if is_fully_feasible(plan_id)]
        if not feasible:
            return None
        plan_id = min(feasible, key=lambda candidate: self._plan_rank(candidate, agg[candidate]))
        plan = self._proposal_plans[plan_id]
        rank = self._plan_rank(plan_id, agg[plan_id])
        if self._best_feasible_rank is None or rank < self._best_feasible_rank:
            self._best_feasible_score = agg[plan_id]
            self._best_feasible_rank = rank
            self._best_feasible_slot = plan["slot"]
            self._best_feasible_plan_id = plan_id
            self._best_feasible_displacements = plan["displacements"]
        return plan["slot"]

    def _meeting_displacements(self, slot: int | None) -> list[dict]:
        plans = self._meeting_displacement_plans(slot)
        return plans[0] if plans else []

    def _meeting_displacement_plans(self, slot: int | None) -> list[list[dict]]:
        if self.meeting is None or slot is None:
            return [[]]
        item = self._slot_items.get(slot)
        if not isinstance(item, dict) or item.get("type") != "meeting":
            return [[]]
        return self._displacement_chains(slot, self._cascade_depth())[:self._displacement_targets()]

    def _best_displacement_chain(
        self,
        from_slot: int,
        remaining_depth: int,
        blocked_slots: set[int] | None = None,
    ) -> list[dict] | None:
        if remaining_depth <= 0:
            return None
        blocked_slots = set(blocked_slots or set())
        item = self._slot_items.get(from_slot)
        if not isinstance(item, dict) or item.get("type") != "meeting":
            return None
        meeting_id = int(item.get("item_id", -1))
        if meeting_id not in self._known_meetings:
            return None

        candidates: list[list[dict]] = []
        next_blocked = blocked_slots | {from_slot}
        for to_slot in sorted(self._slot_items):
            if to_slot in next_blocked:
                continue
            target_item = self._slot_items.get(to_slot)
            move = {"meeting_id": meeting_id, "from_slot": int(from_slot), "to_slot": int(to_slot)}
            if target_item is None:
                candidates.append([move])
                continue
            if isinstance(target_item, dict) and target_item.get("type") == "meeting":
                tail = self._best_displacement_chain(to_slot, remaining_depth - 1, next_blocked)
                if tail:
                    candidates.append([move, *tail])
        if not candidates:
            return None

        def plan_cost(plan: list[dict]) -> tuple[int, list[int]]:
            cost = sum(_slot_displacement_cost(int(move["from_slot"]), self._slot_items) for move in plan)
            targets = [int(move["to_slot"]) for move in plan]
            return cost, targets

        return min(candidates, key=plan_cost)

    def _displacement_chains(
        self,
        from_slot: int,
        remaining_depth: int,
        blocked_slots: set[int] | None = None,
    ) -> list[list[dict]]:
        if remaining_depth <= 0:
            return []
        blocked_slots = set(blocked_slots or set())
        item = self._slot_items.get(from_slot)
        if not isinstance(item, dict) or item.get("type") != "meeting":
            return []
        meeting_id = int(item.get("item_id", -1))
        if meeting_id not in self._known_meetings:
            return []

        chains: list[list[dict]] = []
        next_blocked = blocked_slots | {from_slot}
        for to_slot in sorted(self._slot_items):
            if to_slot in next_blocked:
                continue
            target_item = self._slot_items.get(to_slot)
            move = {"meeting_id": meeting_id, "from_slot": int(from_slot), "to_slot": int(to_slot)}
            if target_item is None:
                chains.append([move])
                continue
            if isinstance(target_item, dict) and target_item.get("type") == "meeting":
                for tail in self._displacement_chains(to_slot, remaining_depth - 1, next_blocked):
                    chains.append([move, *tail])

        def plan_cost(plan: list[dict]) -> tuple[int, int, list[int]]:
            cost = sum(_slot_displacement_cost(int(move["from_slot"]), self._slot_items) for move in plan)
            return cost, len(plan), [int(move["to_slot"]) for move in plan]

        deduped: dict[tuple[tuple[int, int, int], ...], list[dict]] = {}
        for chain in chains:
            key = tuple((int(move["meeting_id"]), int(move["from_slot"]), int(move["to_slot"])) for move in chain)
            deduped[key] = chain
        return sorted(deduped.values(), key=plan_cost)

    # ------------------------------------------------------------------
    # Responder turns
    # ------------------------------------------------------------------

    def _responder_turn(self, messages: list[dict]) -> TurnResult:
        if self._active_round and self._state not in {"waiting_proposal", "scored"}:
            return _empty_turn()

        if self._state in {"waiting_proposal", "scored"} or not self._active_round:
            for msg in messages:
                parsed = _parse_dsm(msg["content"])
                if not parsed:
                    continue
                if self._active_round and msg["from"] != self._initiator_id:
                    continue
                if parsed.get("dsm") == "decision":
                    for displacement in parsed.get("displacements", []):
                        try:
                            meeting_id = int(displacement["meeting_id"])
                            self._agreed_reschedule_targets[meeting_id] = int(displacement["to_slot"])
                            self._pending_reschedule_requests[meeting_id] = (
                                int(displacement["from_slot"]),
                                int(displacement["to_slot"]),
                            )
                            self._pending_reschedule_moves.append(displacement)
                        except (KeyError, TypeError, ValueError):
                            continue
                    participants = self._proposal_participants(parsed)
                    if self.agent_id in participants and parsed.get("slot") is not None:
                        self._agreed_slot = int(parsed["slot"])
                    self._state = "decided"
                    return _empty_turn()
                if parsed.get("dsm") != "proposals":
                    continue

                proposed_slots: list[int] = parsed.get("slots", [])
                raw_displacements = parsed.get("displacements", {})
                raw_cleared = parsed.get("cleared_slots", {})
                plan_lookup = {
                    int(plan.get("id")): plan
                    for plan in parsed.get("plans", [])
                    if isinstance(plan, dict) and plan.get("id") is not None
                }
                scores = {
                    str(plan_id): self._score_incoming_proposal_slot(
                        int(plan_lookup.get(plan_id, {}).get("slot", plan_id)),
                        parsed,
                        plan_lookup.get(plan_id, {}).get("displacements", raw_displacements.get(str(plan_id), [])),
                        set(plan_lookup.get(plan_id, {}).get("cleared_slots", raw_cleared.get(str(plan_id), []))),
                    )
                    for plan_id in proposed_slots
                }
                payload = {"dsm": "scores", "scores": scores}
                if parsed.get("round") is not None:
                    payload["round"] = parsed["round"]
                payload_json = json.dumps(payload)
                self._state = "scored"
                return TurnResult(
                    tool_calls=[{
                        "type": "dm",
                        "to": int(msg["from"]),
                        "meeting_id": int(msg.get("meeting_id", parsed.get("meeting_id", 0))),
                        "content": payload_json,
                    }],
                    text=None, thinking=None, usage=None, latency_ms=None, raw=None,
                )

        return _empty_turn()

    def _record_reschedule_requests(self, messages: list[dict]) -> None:
        for msg in messages:
            parsed = _parse_dsm(msg.get("content", ""))
            if not parsed or parsed.get("dsm") != "reschedule_request":
                continue
            try:
                meeting_id = int(parsed["meeting_id"])
                self._pending_reschedule_requests[meeting_id] = (
                    int(parsed["from_slot"]),
                    int(parsed["to_slot"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _should_handle_as_displacement_responder(self, messages: list[dict]) -> bool:
        active_meeting_id = self.meeting.get("id") if self.meeting else None
        for msg in messages:
            parsed = _parse_dsm(msg.get("content", ""))
            if not parsed or parsed.get("dsm") not in {"proposals", "decision"}:
                continue
            if msg.get("meeting_id") != active_meeting_id:
                return True
        return False

    def _score_incoming_proposal_slot(
        self,
        slot: int,
        proposal: dict,
        displacements: list[dict],
        cleared_slots: set[int] | None = None,
    ) -> int:
        current_participants = self._proposal_participants(proposal)
        if self.agent_id in current_participants:
            return _score_plan(slot, self._slot_items, displacements, True, cleared_slots)

        if not displacements:
            return _D - 1
        return _score_plan(slot, self._slot_items, displacements, False, cleared_slots)

    def _proposal_participants(self, proposal: dict) -> set[int]:
        raw_participants = proposal.get("participants")
        if raw_participants is None and self._active_round and self.meeting:
            raw_participants = self.meeting.get("participants", [])
        try:
            return {int(p) for p in raw_participants or []}
        except (TypeError, ValueError):
            return set()


class PaperDSMClient(DSMClient):
    """Tunable DSM variant closer to the paper protocol.

    This client keeps the calendar-specific plan/displacement machinery from
    DSMClient, but changes the policy layer: offer sizes are chosen from
    Lmin/Lmax using beta/theta tradeoffs, feasible offers are not exhaustively
    revisited, scoring costs are charged, and selected offers carry approximate
    convenience-point rewards.
    """

    def __init__(self) -> None:
        super().__init__()
        self._point_budget: int = 100
        self._last_reward_payload: dict[int, dict[str, int]] = {}
        self._applied_reward_rounds: set[tuple[int, int]] = set()

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        super().register(agent_id, game_config)
        self._point_budget = max(0, int(getattr(game_config, "dsm_initial_budget", 100)))

    def _lmin(self) -> int:
        return max(1, int(getattr(self.game_config, "dsm_lmin", 1)))

    def _lmax(self) -> int:
        configured = getattr(self.game_config, "dsm_lmax", None)
        if configured is None:
            configured = getattr(self.game_config, "dsm_num_proposals", _NUM_PROPOSALS)
        return max(self._lmin(), int(configured))

    def _beta(self) -> float:
        return max(0.0, float(getattr(self.game_config, "dsm_beta", 1.0)))

    def _theta(self) -> float:
        return max(0.0, float(getattr(self.game_config, "dsm_theta", 0.0)))

    def _social_welfare_weight(self) -> float:
        return max(0.0, float(getattr(self.game_config, "dsm_social_welfare_weight", 1.0)))

    def _privacy_unit_cost(self) -> float:
        return max(0.0, float(getattr(self.game_config, "dsm_privacy_unit_cost", 1.0)))

    def _exhaustive_search(self) -> bool:
        return False

    def _next_proposals(self) -> list[int]:
        candidates: list[int] = []
        plans: dict[int, dict] = {}
        for slot in _schedulable_slots(self._slot_items):
            if not self._slot_can_be_proposed(slot):
                continue
            for alt_idx, displacements in enumerate(self._meeting_displacement_plans(slot)):
                plan_id = self._plan_id(slot, alt_idx)
                if plan_id in self._tried_slots:
                    continue
                plans[plan_id] = {"id": plan_id, "slot": slot, "displacements": displacements}
                candidates.append(plan_id)
        self._proposal_plans.update(plans)

        def local_value(plan_id: int) -> int:
            plan = self._proposal_plans[plan_id]
            return _score_plan(plan["slot"], self._slot_items, plan["displacements"], True)

        ordered = sorted(
            candidates,
            key=lambda plan_id: (-local_value(plan_id), self._proposal_plans[plan_id]["slot"], plan_id),
        )
        return ordered[:self._paper_offer_size(ordered)]

    def _paper_offer_size(self, ordered_candidates: list[int]) -> int:
        """Choose L in [Lmin, Lmax] using the paper's beta/theta tradeoff."""
        remaining = len(ordered_candidates)
        if remaining == 0:
            return 0
        lower = min(self._lmin(), remaining)
        upper = min(self._lmax(), remaining)
        if upper <= lower:
            return lower

        p_all = self._estimated_joint_feasibility()
        best_l = lower
        best_utility = float("-inf")
        for offer_size in range(lower, upper + 1):
            values = [self._initiator_value(plan_id) for plan_id in ordered_candidates[:offer_size]]
            avg_value = sum(values) / max(1, len(values))
            fail_prob = (1.0 - p_all) ** offer_size
            success_prob = 1.0 - fail_prob
            privacy_cost = self._theta() * self._privacy_unit_cost() * offer_size
            expected_extra_round_cost = self._beta() * fail_prob
            social_bonus = self._social_welfare_weight() * success_prob
            utility = (success_prob * avg_value) + social_bonus - privacy_cost - expected_extra_round_cost
            if utility > best_utility:
                best_utility = utility
                best_l = offer_size
        return best_l

    def _estimated_joint_feasibility(self) -> float:
        busy = sum(1 for item in self._slot_items.values() if item is not None)
        total = max(1, len(self._slot_items))
        density = busy / total
        responder_count = max(1, len(self._expected_responders))
        single_responder_free = max(0.05, min(0.95, 1.0 - density))
        return max(0.01, min(0.95, single_responder_free ** responder_count))

    def _initiator_value(self, plan_id: int) -> float:
        plan = self._proposal_plans[plan_id]
        return _score_plan(plan["slot"], self._slot_items, plan["displacements"], True) / max(1, _D - 1)

    def _assess(self) -> int | None:
        selected = super()._assess()
        if selected is not None and self._best_feasible_plan_id is not None:
            self._last_reward_payload = self._reward_payload_for_plan(self._best_feasible_plan_id)
            self._apply_initiator_point_transfer(self._best_feasible_plan_id)
        return selected

    def _reward_payload_for_plan(self, plan_id: int) -> dict[int, dict[str, int]]:
        rewards: dict[int, dict[str, int]] = {}
        offer_count = len(self._proposals)
        for responder in self._responders_for_plan(plan_id):
            scores = self._scores_received.get(responder, {})
            summary = _score_summary(scores)
            selected_score = int(scores.get(plan_id, 0))
            reward = self._reward_for_score(selected_score, summary["availability"], offer_count)
            rewards[responder] = {
                "selected_score": selected_score,
                "availability": summary["availability"],
                "flexibility": summary["flexibility"],
                "scoring_cost": summary["cost"],
                "reward": reward,
            }
        return rewards

    def _reward_for_score(self, selected_score: int, availability: int, offer_count: int) -> int:
        if selected_score <= 0 or selected_score >= _D - 1:
            return 0
        dissatisfaction = (_D - 1) - selected_score
        flexibility_bonus = max(0, min(availability, offer_count) - 1)
        return dissatisfaction + flexibility_bonus

    def _apply_initiator_point_transfer(self, plan_id: int) -> None:
        collected = 0
        paid = 0
        for responder in self._responders_for_plan(plan_id):
            scores = self._scores_received.get(responder, {})
            collected += _score_summary(scores)["cost"]
            paid += self._last_reward_payload.get(responder, {}).get("reward", 0)
        self._point_budget = max(0, self._point_budget + collected - paid)

    def _decision_payload_for(self, recipient: int, decision: dict) -> dict:
        payload = super()._decision_payload_for(recipient, decision)
        reward = self._last_reward_payload.get(recipient)
        if reward is not None:
            payload["reward"] = reward
        return payload

    def _responder_turn(self, messages: list[dict]) -> TurnResult:
        for msg in messages:
            parsed = _parse_dsm(msg.get("content", ""))
            if not parsed or parsed.get("dsm") != "decision":
                continue
            round_key = (int(msg.get("meeting_id", -1)), int(parsed.get("plan_id", -1)))
            if round_key in self._applied_reward_rounds:
                continue
            reward = parsed.get("reward")
            if isinstance(reward, dict):
                self._point_budget = max(
                    0,
                    self._point_budget
                    - int(reward.get("scoring_cost", 0))
                    + int(reward.get("reward", 0)),
                )
                self._applied_reward_rounds.add(round_key)

        result = super()._responder_turn(messages)
        for tool in result.tool_calls:
            if not isinstance(tool, dict) or tool.get("type") != "dm":
                continue
            parsed = _parse_dsm(tool.get("content", ""))
            if parsed and parsed.get("dsm") == "scores":
                raw_scores = {int(k): int(v) for k, v in parsed.get("scores", {}).items()}
                # Hold the score vector locally for observability. The paper's
                # final-round cost/refund distinction is reflected in the
                # reward payload once the initiator selects an agreement.
                self._last_score_summary = _score_summary(raw_scores)
        return result

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        if self._agreed_slot is None:
            return _empty_decide()
        return super().decide(meeting, calendar_render)


class PrivateDSMClient(PaperDSMClient):
    """High-privacy DSM preset based on the paper's beta/theta tradeoff.

    This preset favors small proposal sets and low leakage over social-welfare
    exploration: high theta, low beta, Lmin=1, Lmax=2, and shallow displacement
    search. Use PaperDSMClient directly for explicit knob sweeps.
    """

    def _lmin(self) -> int:
        return 1

    def _lmax(self) -> int:
        return 2

    def _beta(self) -> float:
        return 0.25

    def _theta(self) -> float:
        return 10.0

    def _social_welfare_weight(self) -> float:
        return 0.25

    def _cascade_depth(self) -> int:
        return 1

    def _displacement_targets(self) -> int:
        return 2
