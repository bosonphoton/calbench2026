"""Legacy deterministic fallback scheduler for uncoordinated rounds.

When agents fail to coordinate, this module finds the minimum-cost slot
assignment using backtracking with branch-and-bound (depth limit = max_depth).

Note: this search is exponential in displacement depth and did not resolve
dense blocked-slot benchmark cases in reasonable time. It is retained for
legacy/debug runs only; benchmark comparisons should set enable_fallback:
false in their experiment YAMLs.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calendar_game.agents import Agent

Slot = dict[str, int] | None
_Action = dict  # {"agent_id", "item_id", "from_slot", "to_slot", "is_meeting_cascade"}


class FallbackImpossible(Exception):
    def __init__(self, meeting_id: int) -> None:
        self.meeting_id = meeting_id
        super().__init__(f"No feasible displacement plan for meeting {meeting_id}")


class FallbackDepthExceeded(Exception):
    def __init__(self, meeting_id: int, depth: int) -> None:
        self.meeting_id = meeting_id
        self.depth = depth
        super().__init__(
            f"Fallback depth limit {depth} hit for meeting {meeting_id}; "
            "no feasible plan found within depth bound"
        )


def _item_id(item: dict) -> int:
    return item.get("errand_id") or item.get("meeting_id")  # type: ignore[return-value]


def _apply_actions(vcals: list[list[Slot]], actions: list[_Action]) -> None:
    """Apply a list of move actions to vcals in-place."""
    for a in actions:
        aid, fs, ts = a["agent_id"], a["from_slot"], a["to_slot"]
        vcals[aid][ts] = vcals[aid][fs]
        vcals[aid][fs] = None


def _free_slot(
    vcals: list[list[Slot]],
    agent_id: int,
    slot: int,
    exclude: frozenset[int],
    num_slots: int,
    depth: int,
    max_depth: int,
    depth_hit: list[bool],
) -> tuple[int, list[_Action]] | None:
    """
    Return (cost, actions) to make vcals[agent_id][slot] free, or None if infeasible.
    Uses backtracking: tries every candidate target slot T, cascading when T is occupied.
    Actions are NOT applied to vcals; caller applies them if needed.

    depth_hit[0] is set to True whenever a cascade is skipped due to max_depth.
    """
    item = vcals[agent_id][slot]
    if item is None:
        return 0, []
    if item.get("blocked"):
        return None

    item_cost = item["cost"]
    iid = _item_id(item)
    best_cost = math.inf
    best_actions: list[_Action] | None = None

    for T in range(num_slots):
        if T == slot or T in exclude:
            continue
        occupant = vcals[agent_id][T]

        if occupant is None:
            # Direct move — cost is just item_cost; all free T share this cost, take first.
            return item_cost, [{"agent_id": agent_id, "item_id": iid,
                                 "from_slot": slot, "to_slot": T,
                                 "is_meeting_cascade": False}]

        # T is occupied; cascade if depth allows.
        if depth >= max_depth:
            depth_hit[0] = True
            continue

        # Recurse: free T first, then move item there.
        branch = deepcopy(vcals)
        sub = _free_slot(branch, agent_id, T,
                         exclude | frozenset([slot]),
                         num_slots, depth + 1, max_depth, depth_hit)
        if sub is None:
            continue
        sub_cost, sub_actions = sub
        total = item_cost + sub_cost
        if total < best_cost:
            best_cost = total
            best_actions = sub_actions + [{"agent_id": agent_id, "item_id": iid,
                                            "from_slot": slot, "to_slot": T,
                                            "is_meeting_cascade": False}]

    return (best_cost, best_actions) if best_actions is not None else None


def _free_registered_meeting(
    vcals: list[list[Slot]],
    meeting_id: int,
    meeting_cost: int,
    m_participants: list[int],
    from_slot: int,
    num_slots: int,
    max_depth: int,
    depth_hit: list[bool],
) -> tuple[int, list[_Action]] | None:
    """
    Find the minimum-cost target slot T to move registered meeting `meeting_id`
    from `from_slot` to T across ALL m_participants simultaneously.
    Returns (cost, actions) or None.
    """
    best_cost = math.inf
    best_actions: list[_Action] | None = None

    for T in range(num_slots):
        if T == from_slot:
            continue

        T_cost = 0
        T_actions: list[_Action] = []
        T_vcals = deepcopy(vcals)
        T_feasible = True

        for Q in m_participants:
            # Free T on Q's calendar if needed.
            if T_vcals[Q][T] is not None:
                sub = _free_slot(T_vcals, Q, T,
                                 frozenset([from_slot]),
                                 num_slots, 0, max_depth, depth_hit)
                if sub is None:
                    T_feasible = False
                    break
                sub_cost, sub_actions = sub
                _apply_actions(T_vcals, sub_actions)
                T_cost += sub_cost
                T_actions.extend(sub_actions)

            # Move meeting from from_slot to T on Q's calendar.
            T_cost += meeting_cost
            T_actions.append({
                "agent_id": Q, "item_id": meeting_id,
                "from_slot": from_slot, "to_slot": T,
                "is_meeting_cascade": True,
            })
            T_vcals[Q][T] = {"meeting_id": meeting_id, "cost": meeting_cost}
            T_vcals[Q][from_slot] = None

        if T_feasible and T_cost < best_cost:
            best_cost = T_cost
            best_actions = T_actions

    return (best_cost, best_actions) if best_actions is not None else None


def find_fallback_slot(
    agents: list[Agent],
    meeting: dict,
    num_slots: int,
    meeting_registry: dict[int, int],
    scenario_meetings: list[dict],
    max_depth: int = 9,
) -> tuple[int, list[_Action]]:
    """
    Find the minimum-cost slot for `meeting` and return (chosen_slot, displacement_plan).

    displacement_plan entries: {"agent_id", "item_id", "from_slot", "to_slot",
                                "is_meeting_cascade" [, "meeting_id" for cascades]}

    Registered shared meetings are displaced coordinately (all participants move to same T).
    Uses backtracking with branch-and-bound; depth limited to max_depth per cascade chain.

    Raises:
        FallbackImpossible   — no feasible slot found within depth bound
        FallbackDepthExceeded — depth limit was hit and prevented finding a solution
    """
    vcals: list[list[Slot]] = [list(agent.calendar.slots) for agent in agents]

    # Build spec map for registered meetings
    reg_specs: dict[int, list[int]] = {}
    for mid in meeting_registry:
        spec = next((m for m in scenario_meetings if m["id"] == mid), None)
        if spec:
            reg_specs[mid] = list(spec["participants"])

    participants: list[int] = meeting["participants"]
    depth_hit: list[bool] = [False]

    def lower_bound(S: int) -> int:
        lb = 0
        seen_mids: set[int] = set()
        for P in participants:
            item = vcals[P][S]
            if not isinstance(item, dict):
                continue
            if item.get("blocked"):
                return math.inf
            mid = item.get("meeting_id")
            if mid is not None and mid in reg_specs:
                if mid not in seen_mids:
                    seen_mids.add(mid)
                    lb += item["cost"] * len(reg_specs[mid])
            else:
                lb += item["cost"]
        return lb

    candidates = sorted(range(num_slots), key=lower_bound)

    best_slot: int | None = None
    best_cost: int = math.inf
    best_plan: list[_Action] = []

    for S in candidates:
        lb = lower_bound(S)
        if lb >= best_cost:
            continue  # lower-bound prune

        branch_vcals = deepcopy(vcals)
        plan: list[_Action] = []
        total_cost = 0
        feasible = True
        handled_mids: set[int] = set()

        for P in participants:
            item = branch_vcals[P][S]
            if item is None:
                continue  # already free for this participant

            mid = item.get("meeting_id")

            if mid is not None and mid in reg_specs and mid not in handled_mids:
                # Registered shared meeting — coordinated displacement.
                handled_mids.add(mid)
                result = _free_registered_meeting(
                    branch_vcals, mid, item["cost"], reg_specs[mid],
                    S, num_slots, max_depth, depth_hit,
                )
                if result is None:
                    feasible = False
                    break
                r_cost, r_actions = result
                if total_cost + r_cost >= best_cost:
                    feasible = False
                    break
                _apply_actions(branch_vcals, r_actions)
                total_cost += r_cost
                plan.extend(r_actions)

            elif mid is not None and mid in reg_specs:
                # Already handled above (another participant of same meeting).
                pass

            else:
                # Errand or non-registered meeting — unilateral displacement.
                sub = _free_slot(branch_vcals, P, S, frozenset(),
                                 num_slots, 0, max_depth, depth_hit)
                if sub is None:
                    feasible = False
                    break
                sub_cost, sub_actions = sub
                if total_cost + sub_cost >= best_cost:
                    feasible = False
                    break
                _apply_actions(branch_vcals, sub_actions)
                total_cost += sub_cost
                plan.extend(sub_actions)

        if feasible and total_cost < best_cost:
            best_cost = total_cost
            best_slot = S
            best_plan = plan

    if best_slot is None:
        if depth_hit[0]:
            raise FallbackDepthExceeded(meeting["id"], max_depth)
        raise FallbackImpossible(meeting["id"])

    return best_slot, best_plan
