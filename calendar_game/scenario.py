"""Scenario generation for the calendar scheduling benchmark."""

from __future__ import annotations

import random
from typing import Any

Slot = dict[str, int] | None
ScenarioDict = dict[str, Any]


def generate_scenario(
    seed: int | None,
    num_agents: int,
    num_slots: int,
    density: float,
    pref_level: int,
    num_meetings: int,
    participant_lists: list[list[int]] | None = None,
    speaker_orders: list[list[int]] | None = None,
    force_witness_errand: bool = True,
    skip_optimal: bool = False,
    errand_cost_multiplier: int = 1,
    errand_cost_level: int | None = None,
    meeting_cost_level: int = 1,
    agent_densities: list[float] | dict[int | str, float] | None = None,
    blocked_errands_per_agent: int | list[int] | dict[int | str, int] | None = None,
) -> ScenarioDict:
    """Generate a scenario with guaranteed feasible meeting assignments.

    Calendars are generated backward from a hidden witness solution. Meeting
    markers are not placed in the visible calendars; errands may occupy witness
    slots, with protected absorbing slots kept free so the witness remains
    feasible.
    """
    if not 0 <= density <= 1:
        raise ValueError("density must be between 0 and 1")
    density_by_agent = _normalize_agent_densities(agent_densities, num_agents, density)
    if num_meetings > num_slots:
        raise ValueError("num_meetings cannot exceed num_slots for one-slot MVP meetings")
    if pref_level < 1:
        raise ValueError("pref_level must be >= 1")
    if errand_cost_multiplier < 1:
        raise ValueError("errand_cost_multiplier must be >= 1")
    if meeting_cost_level < 1:
        raise ValueError("meeting_cost_level must be >= 1")
    blocked_by_agent = _normalize_blocked_errands_per_agent(
        blocked_errands_per_agent,
        num_agents,
    )
    _errand_cost_level = errand_cost_level if errand_cost_level is not None else pref_level
    if _errand_cost_level < 1:
        raise ValueError("errand_cost_level must be >= 1")

    rng = random.Random(seed)
    if participant_lists is None:
        participant_lists = [list(range(num_agents)) for _ in range(num_meetings)]
    if len(participant_lists) != num_meetings:
        raise ValueError("participant_lists length must match num_meetings")
    if speaker_orders is not None and len(speaker_orders) != num_meetings:
        raise ValueError("speaker_orders length must match num_meetings")

    meetings = []
    for index, participants in enumerate(participant_lists):
        if not participants:
            raise ValueError("meeting participants cannot be empty")
        if any(agent_id < 0 or agent_id >= num_agents for agent_id in participants):
            raise ValueError("meeting participant out of range")
        normalized_participants = sorted(set(participants))
        speaker_order = (
            list(speaker_orders[index])
            if speaker_orders is not None
            else list(normalized_participants)
        )
        if sorted(speaker_order) != normalized_participants:
            raise ValueError("speaker_order must be a permutation of meeting participants")
        meetings.append({
            "id": index + 1,
            "participants": normalized_participants,
            "speaker_order": speaker_order,
            "duration": 1,
            "cost": rng.randint(1, meeting_cost_level),
        })

    assignments: dict[int, int] = {}
    used_by_agent: dict[int, set[int]] = {agent_id: set() for agent_id in range(num_agents)}
    for meeting in meetings:
        candidates = [
            slot for slot in range(num_slots)
            if all(slot not in used_by_agent[agent_id] for agent_id in meeting["participants"])
        ]
        if not candidates:
            raise ValueError("could not choose non-conflicting witness slots")
        slot = rng.choice(candidates)
        assignments[meeting["id"]] = slot
        for agent_id in meeting["participants"]:
            used_by_agent[agent_id].add(slot)

    # Keep enough absorbing slots free for displaced errands, but allow the
    # chosen meeting slots themselves to contain movable errands. That preserves
    # known feasibility while making optimal cost meaningful at high density.
    calendars: list[list[Slot]] = []
    for agent_id in range(num_agents):
        calendar: list[Slot] = [None] * num_slots
        witness_slots = {
            assignments[meeting["id"]]
            for meeting in meetings
            if agent_id in meeting["participants"]
        }
        # Blocked errands are sampled first from non-witness slots. Density is
        # then applied to the remaining capacity after reserving witness meeting
        # slots and blocked slots.
        k = len(witness_slots)
        blocked_target = blocked_by_agent[agent_id]
        remaining_capacity = num_slots - k - blocked_target
        if remaining_capacity < 0:
            raise ValueError(
                "blocked_errands_per_agent cannot exceed non-witness capacity "
                f"for agent {agent_id}"
            )
        movable_target = round(remaining_capacity * density_by_agent[agent_id])
        if force_witness_errand and density_by_agent[agent_id] > 0 and movable_target <= 0:
            raise ValueError(
                "density and blocked_errands_per_agent must leave at least one movable errand "
                f"for agent {agent_id} when force_witness_errand is enabled"
            )
        num_absorbing = min(k, movable_target)
        non_witness = [slot for slot in range(num_slots) if slot not in witness_slots]
        blocked_slots: set[int] = set()
        if blocked_target > 0:
            if len(non_witness) < blocked_target:
                raise ValueError(
                    f"could not place {blocked_target} blocked errands for agent {agent_id}"
                )
            blocked_slots = set(rng.sample(non_witness, blocked_target))

        absorbing_candidates = [slot for slot in non_witness if slot not in blocked_slots]
        if num_absorbing > len(absorbing_candidates):
            raise ValueError("could not reserve absorbing slots")
        absorbing_slots = set(rng.sample(absorbing_candidates, num_absorbing))
        fillable_slots = [slot for slot in range(num_slots) if slot not in absorbing_slots]

        forced_slots: set[int] = set()
        if force_witness_errand and movable_target > 0:
            forced_slots = set(sorted(witness_slots)[:movable_target])

        remaining_slots = [
            slot
            for slot in fillable_slots
            if slot not in forced_slots and slot not in blocked_slots
        ]
        chosen_slots = list(forced_slots)
        chosen_slots.extend(rng.sample(remaining_slots, movable_target - len(chosen_slots)))

        for errand_id, slot in enumerate(sorted(blocked_slots | set(chosen_slots)), start=1):
            calendar[slot] = {
                "errand_id": (agent_id * num_slots) + errand_id,
                "cost": rng.randint(1, _errand_cost_level) * errand_cost_multiplier,
            }
            if slot in blocked_slots:
                calendar[slot]["blocked"] = True
        calendars.append(calendar)

    scenario = {
        "seed": seed,
        "num_agents": num_agents,
        "num_slots": num_slots,
        "density": density,
        "agent_densities": density_by_agent,
        "errand_cost_multiplier": errand_cost_multiplier,
        "blocked_errands_per_agent": (
            blocked_by_agent[0]
            if len(set(blocked_by_agent)) == 1
            else blocked_by_agent
        ),
        "calendars": calendars,
        "meetings": meetings,
        "witness_solution": {"cost": None, "assignments": assignments},
        "optimal": {"cost": None, "assignments": {}},
    }
    from calendar_game.solver import apply_schedule, solve_greedy, solve_optimal

    working = calendars
    witness_cost = 0
    for meeting in meetings:
        working, cost, _per_agent = apply_schedule(working, meeting, assignments[meeting["id"]])
        if cost is None:
            raise ValueError("generated witness solution is not feasible")
        witness_cost += cost
    scenario["witness_solution"] = {"cost": witness_cost, "assignments": assignments}
    scenario["optimal"] = {} if skip_optimal else solve_optimal(calendars, meetings, num_slots)
    scenario["greedy"] = solve_greedy(calendars, meetings, num_slots)
    scenario["feasible"] = True if skip_optimal else scenario["optimal"].get("cost") is not None
    return scenario


def _normalize_agent_densities(
    agent_densities: list[float] | dict[int | str, float] | None,
    num_agents: int,
    default_density: float,
) -> list[float]:
    if agent_densities is None:
        return [float(default_density)] * num_agents
    if isinstance(agent_densities, dict):
        densities = [
            float(agent_densities.get(agent_id, agent_densities.get(str(agent_id), default_density)))
            for agent_id in range(num_agents)
        ]
    else:
        if len(agent_densities) != num_agents:
            raise ValueError("agent_densities length must match num_agents")
        densities = [float(value) for value in agent_densities]
    if any(value < 0 or value > 1 for value in densities):
        raise ValueError("agent_densities values must be between 0 and 1")
    return densities


def _normalize_blocked_errands_per_agent(
    blocked_errands_per_agent: int | list[int] | dict[int | str, int] | None,
    num_agents: int,
) -> list[int]:
    if blocked_errands_per_agent is None:
        return [0] * num_agents
    if isinstance(blocked_errands_per_agent, dict):
        counts = [
            int(blocked_errands_per_agent.get(agent_id, blocked_errands_per_agent.get(str(agent_id), 0)))
            for agent_id in range(num_agents)
        ]
    elif isinstance(blocked_errands_per_agent, list):
        if len(blocked_errands_per_agent) != num_agents:
            raise ValueError("blocked_errands_per_agent length must match num_agents")
        counts = [int(value) for value in blocked_errands_per_agent]
    else:
        counts = [int(blocked_errands_per_agent)] * num_agents
    if any(value < 0 for value in counts):
        raise ValueError("blocked_errands_per_agent values must be >= 0")
    return counts
