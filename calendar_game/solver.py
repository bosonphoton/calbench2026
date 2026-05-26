"""Optimal and greedy solvers for calendar scenarios."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - dependency is declared in pyproject.
    cp_model = None

Slot = dict[str, int] | None


def _slot_cost(calendars: list[list[Slot]], participants: list[int], slot: int) -> int | None:
    total = 0
    for agent_id in participants:
        value = calendars[agent_id][slot]
        if value is None:
            continue
        if not isinstance(value, dict):
            return None
        if value.get("blocked"):
            return None
        has_absorbing_slot = any(v is None for i, v in enumerate(calendars[agent_id]) if i != slot)
        if not has_absorbing_slot:
            return None
        total += int(value["cost"])
    return total


def _apply_assignment(
    calendars: list[list[Slot]], meeting_id: int, participants: list[int], slot: int, meeting_cost: int = 1
) -> int | None:
    cost = _slot_cost(calendars, participants, slot)
    if cost is None:
        return None

    for agent_id in participants:
        value = calendars[agent_id][slot]
        if isinstance(value, dict):
            target = next((i for i, v in enumerate(calendars[agent_id]) if i != slot and v is None), None)
            if target is None:
                return None
            calendars[agent_id][target] = value
        calendars[agent_id][slot] = {"meeting_id": meeting_id, "cost": meeting_cost}
    return cost


def solve_greedy(calendars: list[list[Slot]], meetings: list[dict[str, Any]], num_slots: int) -> dict[str, Any]:
    """Schedule meetings in order using the first clearable slot."""
    working = deepcopy(calendars)
    total = 0
    assignments: dict[int, int] = {}
    for meeting in meetings:
        for slot in range(num_slots):
            cost = _apply_assignment(working, int(meeting["id"]), meeting["participants"], slot, meeting.get("cost", 1))
            if cost is None:
                continue
            total += cost
            assignments[int(meeting["id"])] = slot
            break
        else:
            return {"cost": None, "assignments": assignments, "failed_meeting": meeting["id"]}
    return {"cost": total, "assignments": assignments}


def solve_optimal(calendars: list[list[Slot]], meetings: list[dict[str, Any]], num_slots: int) -> dict[str, Any]:
    """Solve the minimum-cost assignment.

    OR-Tools CP-SAT is used to choose one slot per meeting while preventing an
    agent from having two fixed meetings in the same slot. The clearability cost
    for each meeting-slot pair is computed against the initial calendars; this
    is exact for the generated MVP scenarios where meetings have reserved common
    slots, and keeps the model small.
    """
    if cp_model is None:
        return _solve_optimal_backtracking(calendars, meetings, num_slots)

    model = cp_model.CpModel()
    variables: dict[tuple[int, int], Any] = {}
    costs: dict[tuple[int, int], int] = {}

    for meeting in meetings:
        meeting_id = int(meeting["id"])
        possible = []
        for slot in range(num_slots):
            cost = _slot_cost(calendars, meeting["participants"], slot)
            if cost is None:
                continue
            var = model.NewBoolVar(f"m{meeting_id}_s{slot}")
            variables[(meeting_id, slot)] = var
            costs[(meeting_id, slot)] = cost
            possible.append(var)
        if not possible:
            return {"cost": None, "assignments": {}, "failed_meeting": meeting_id}
        model.AddExactlyOne(possible)

    for agent_id in range(len(calendars)):
        for slot in range(num_slots):
            same_slot = [
                variables[(int(meeting["id"]), slot)]
                for meeting in meetings
                if agent_id in meeting["participants"] and (int(meeting["id"]), slot) in variables
            ]
            if len(same_slot) > 1:
                model.AddAtMostOne(same_slot)

    model.Minimize(sum(costs[key] * var for key, var in variables.items()))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"cost": None, "assignments": {}}

    assignments: dict[int, int] = {}
    total = 0
    for (meeting_id, slot), var in variables.items():
        if solver.BooleanValue(var):
            assignments[meeting_id] = slot
            total += costs[(meeting_id, slot)]
    return {"cost": total, "assignments": assignments}


def count_feasible_assignments_cp_sat(
    calendars: list[list[Slot]],
    meetings: list[dict[str, Any]],
    num_slots: int,
    *,
    require_distinct_slots: bool = True,
) -> int:
    """Count feasible meeting-slot assignments with the CP-SAT model.

    This counts assignments, not costs. When ``require_distinct_slots`` is true,
    each slot can be used by at most one meeting, matching the task generator's
    ordered-permutation denominator.
    """
    if not meetings:
        return 1
    if cp_model is None:
        return _count_feasible_assignments_backtracking(
            calendars,
            meetings,
            num_slots,
            require_distinct_slots=require_distinct_slots,
        )

    model = cp_model.CpModel()
    variables: dict[tuple[int, int], Any] = {}

    for meeting in meetings:
        meeting_id = int(meeting["id"])
        possible = []
        for slot in range(num_slots):
            if _slot_cost(calendars, meeting["participants"], slot) is None:
                continue
            var = model.NewBoolVar(f"m{meeting_id}_s{slot}")
            variables[(meeting_id, slot)] = var
            possible.append(var)
        if not possible:
            return 0
        model.AddExactlyOne(possible)

    if require_distinct_slots:
        for slot in range(num_slots):
            same_slot = [
                variables[(int(meeting["id"]), slot)]
                for meeting in meetings
                if (int(meeting["id"]), slot) in variables
            ]
            if len(same_slot) > 1:
                model.AddAtMostOne(same_slot)

    for agent_id in range(len(calendars)):
        for slot in range(num_slots):
            same_agent_slot = [
                variables[(int(meeting["id"]), slot)]
                for meeting in meetings
                if agent_id in meeting["participants"] and (int(meeting["id"]), slot) in variables
            ]
            if len(same_agent_slot) > 1:
                model.AddAtMostOne(same_agent_slot)

    class _SolutionCounter(cp_model.CpSolverSolutionCallback):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def OnSolutionCallback(self) -> None:
            self.count += 1

    solver = cp_model.CpSolver()
    counter = _SolutionCounter()
    solver.SearchForAllSolutions(model, counter)
    return counter.count


def _count_feasible_assignments_backtracking(
    calendars: list[list[Slot]],
    meetings: list[dict[str, Any]],
    num_slots: int,
    *,
    require_distinct_slots: bool,
) -> int:
    feasible_slots_by_meeting: list[list[int]] = []
    for meeting in meetings:
        feasible_slots = [
            slot
            for slot in range(num_slots)
            if _slot_cost(calendars, meeting["participants"], slot) is not None
        ]
        if not feasible_slots:
            return 0
        feasible_slots_by_meeting.append(feasible_slots)

    memo: dict[tuple[int, int, tuple[int, ...]], int] = {}

    def walk(meeting_idx: int, used_slots: int, occupied_by_agent: tuple[int, ...]) -> int:
        if meeting_idx == len(meetings):
            return 1
        key = (meeting_idx, used_slots, occupied_by_agent)
        if key in memo:
            return memo[key]
        meeting = meetings[meeting_idx]
        total = 0
        for slot in feasible_slots_by_meeting[meeting_idx]:
            bit = 1 << slot
            if require_distinct_slots and used_slots & bit:
                continue
            if any(occupied_by_agent[agent_id] & bit for agent_id in meeting["participants"]):
                continue
            next_occupied = list(occupied_by_agent)
            for agent_id in meeting["participants"]:
                next_occupied[agent_id] |= bit
            total += walk(meeting_idx + 1, used_slots | bit, tuple(next_occupied))
        memo[key] = total
        return total

    return walk(0, 0, tuple(0 for _ in calendars))


def _solve_optimal_backtracking(calendars: list[list[Slot]], meetings: list[dict[str, Any]], num_slots: int) -> dict[str, Any]:
    best_cost: int | None = None
    best_assignments: dict[int, int] = {}

    def walk(idx: int, working: list[list[Slot]], cost_so_far: int, assignments: dict[int, int]) -> None:
        nonlocal best_cost, best_assignments
        if best_cost is not None and cost_so_far >= best_cost:
            return
        if idx == len(meetings):
            best_cost = cost_so_far
            best_assignments = dict(assignments)
            return

        meeting = meetings[idx]
        for slot in range(num_slots):
            next_calendars = deepcopy(working)
            cost = _apply_assignment(next_calendars, int(meeting["id"]), meeting["participants"], slot)
            if cost is None:
                continue
            assignments[int(meeting["id"])] = slot
            walk(idx + 1, next_calendars, cost_so_far + cost, assignments)
            assignments.pop(int(meeting["id"]), None)

    walk(0, deepcopy(calendars), 0, {})
    return {"cost": best_cost, "assignments": best_assignments}


def apply_schedule(calendars: list[list[Slot]], meeting: dict[str, Any], slot: int) -> tuple[list[list[Slot]], int | None, list[int]]:
    """Apply a realized schedule and return new calendars, cost, per-agent cost."""
    working = deepcopy(calendars)
    per_agent = [0 for _ in working]
    total_cost = _slot_cost(working, meeting["participants"], slot)
    if total_cost is None:
        return calendars, None, per_agent

    for agent_id in meeting["participants"]:
        value = working[agent_id][slot]
        if isinstance(value, dict):
            per_agent[agent_id] += int(value["cost"])
            target = next((i for i, v in enumerate(working[agent_id]) if i != slot and v is None), None)
            if target is None:
                return calendars, None, per_agent
            working[agent_id][target] = value
        working[agent_id][slot] = {"meeting_id": meeting["id"], "cost": meeting.get("cost", 1)}
    return working, total_cost, per_agent


def cost_by_agent_for_assignments(
    calendars: list[list[Slot]],
    meetings: list[dict[str, Any]],
    assignments: dict[str | int, int],
) -> tuple[int | None, list[int]]:
    """Replay assignments and return total plus per-agent displacement cost.

    The oracle solver returns the globally optimal meeting-slot assignment.
    Replaying that assignment against the initial calendars tells us which
    agents bear the oracle displacement burden.
    """
    working = deepcopy(calendars)
    per_agent = [0 for _ in working]
    total = 0
    normalized = {int(meeting_id): int(slot) for meeting_id, slot in assignments.items()}
    for meeting in meetings:
        meeting_id = int(meeting["id"])
        if meeting_id not in normalized:
            return None, per_agent
        working, cost, meeting_per_agent = apply_schedule(
            working,
            meeting,
            normalized[meeting_id],
        )
        if cost is None:
            return None, per_agent
        total += cost
        for agent_id, agent_cost in enumerate(meeting_per_agent):
            per_agent[agent_id] += int(agent_cost)
    return total, per_agent
