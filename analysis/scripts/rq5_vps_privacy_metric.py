"""RQ5: protocol-semantic VPS privacy leakage for calendar traces.

This script implements an offline, auditable protocol-semantic adaptation of the
Valuation of Possible States (VPS) privacy metric for calendar scheduling traces.
It scores only typed protocol messages emitted by scripted baselines
(DSM/IMAP/SD-MAP). Free-form natural-language DMs are intentionally ignored here;
LLM agents are scored with reflection-calibrated VPS or optional post-hoc judge
VPS instead.

Private state:
    Whether each slot/agreement is locally feasible for a target agent.

Belief:
    For each observer j and target i, Bel[j,i,k] is j's belief that slot k is
    feasible for i in a given meeting round.

Default ideal belief:
    Secrecy: the target wants the observer's posterior to remain at the prior.
    With this default, VPS loss is weighted belief movement caused by visible
    negotiation messages.

Blocked slots:
    Under cost weighting, blocked errands receive max_weight. They are hard
    stops, so learning about one is treated as maximally sensitive calendar
    state rather than as a low-cost movable errand.

This is not an online mechanism component. It replays already-run traces and
scores leakage from protocol messages that were visible in the trace.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_OUT_DIR = "analysis/outputs/rq5_vps_privacy_metric"
DSM_SCORE_LEVELS = 12
DEFAULT_GAME_VPS_FLOOR = 5.0


@dataclass
class BeliefState:
    prior: list[float]
    posterior: list[float]
    weights: list[float]
    observations: int = 0


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    if root_candidate.exists():
        return root_candidate
    return Path.cwd() / candidate


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_paths(inputs: Iterable[str], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = _resolve(raw, root=root)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif any(char in raw for char in "*?[]"):
            paths.extend(sorted(Path().glob(raw)))
        else:
            paths.append(path)
    return [
        path for path in paths
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
        and "_reports" not in path.parts
        and "_index" not in path.parts
    ]


def _parse_dsm(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and "dsm" in parsed else None


def _parse_sd(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and "sd" in parsed else None


def _parse_imap(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and "imap" in parsed else None


def _num_agents(trace: dict[str, Any]) -> int:
    config = trace.get("config") or {}
    return int(config.get("num_agents") or len(config.get("agents") or []))


def _num_slots(trace: dict[str, Any]) -> int:
    for event in trace.get("events", []):
        if event.get("type") == "game_start":
            data = event.get("data") or {}
            if data.get("num_slots") is not None:
                return int(data["num_slots"])
    final_calendars = (trace.get("final_state") or {}).get("calendars") or []
    return len(final_calendars[0]) if final_calendars else 0


def _participants_by_round(trace: dict[str, Any]) -> dict[int, set[int]]:
    participants: dict[int, set[int]] = {}
    for event in trace.get("events", []):
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        meeting = data.get("meeting") or {}
        try:
            round_idx = int(data.get("round"))
        except (TypeError, ValueError):
            continue
        participants[round_idx] = {int(agent_id) for agent_id in meeting.get("participants", [])}
    return participants


def _calendar_by_round_agent(trace: dict[str, Any]) -> dict[tuple[int, int], str]:
    """Return the first visible calendar_render for each round/agent."""
    calendars: dict[tuple[int, int], str] = {}
    for event in trace.get("events", []):
        data = event.get("data") or {}
        if "calendar_render" not in data:
            continue
        try:
            round_idx = int(data.get("round", -1))
            agent_id = int(data.get("agent_id"))
        except (TypeError, ValueError):
            continue
        if round_idx >= 0:
            calendars.setdefault((round_idx, agent_id), str(data.get("calendar_render") or ""))
    return calendars


def _slot_items_from_render(calendar_render: str, *, num_slots: int) -> list[dict[str, Any] | None]:
    items: list[dict[str, Any] | None] = [None for _ in range(num_slots)]
    for line in calendar_render.splitlines():
        if "Slot" not in line or ":" not in line:
            continue
        try:
            slot_part, content = line.split(":", 1)
            slot = int(slot_part.split("Slot")[1].strip())
        except (IndexError, ValueError):
            continue
        if not 0 <= slot < num_slots:
            continue
        if "[FREE]" in content:
            items[slot] = None
            continue
        blocked = re.search(r"\bBlocked(?: Errand)? #(\d+) \(cost=(\d+)\)", content)
        if blocked:
            items[slot] = {
                "type": "blocked",
                "cost": int(blocked.group(2)),
                "blocked": True,
            }
            continue
        errand = re.search(r"\bErrand #(\d+) \(cost=(\d+)\)", content)
        if errand:
            items[slot] = {"type": "errand", "cost": int(errand.group(2))}
            continue
        meeting = re.search(r"Meeting M(\d+) \(cost=(\d+)\)", content)
        if meeting:
            items[slot] = {"type": "meeting", "cost": int(meeting.group(2))}
            continue
        items[slot] = {"type": "unknown", "cost": 0}
    return items


def _slot_weights(
    calendar_render: str,
    *,
    num_slots: int,
    mode: str,
    max_weight: float,
) -> list[float]:
    if mode == "uniform" or not calendar_render:
        return [1.0 for _ in range(num_slots)]
    items = _slot_items_from_render(calendar_render, num_slots=num_slots)
    weights: list[float] = []
    for item in items:
        if item is None:
            weights.append(1.0)
        elif item.get("blocked"):
            weights.append(max_weight)
        else:
            cost = max(0.0, float(item.get("cost", 0)))
            weights.append(min(max_weight, 1.0 + cost ** 0.5))
    return weights


def _prior_vector(value: float, *, num_slots: int) -> list[float]:
    clipped = min(1.0, max(0.0, value))
    return [clipped for _ in range(num_slots)]


def _distance(a: list[float], b: list[float], weights: list[float]) -> float:
    return sum(weight * abs(left - right) for left, right, weight in zip(a, b, weights))


def _weight_sum(weights: list[float]) -> float:
    return sum(max(0.0, float(weight)) for weight in weights)


def _nudge(current: float, evidence: float, strength: float) -> float:
    strength = min(1.0, max(0.0, strength))
    return (1.0 - strength) * current + strength * evidence


def _plan_slots(payload: dict[str, Any]) -> dict[int, int]:
    slots: dict[int, int] = {}
    for plan in payload.get("plans") or []:
        if not isinstance(plan, dict):
            continue
        try:
            slots[int(plan["id"])] = int(plan["slot"])
        except (KeyError, TypeError, ValueError):
            continue
    # Older DSM payloads only contain slots. In that case plan_id == slot.
    for raw_slot in payload.get("slots") or []:
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        slots.setdefault(slot, slot)
    return slots


def _empty_belief(
    trace: dict[str, Any],
    *,
    round_idx: int,
    target: int,
    observer: int,
    prior: float,
    weight_mode: str,
    max_weight: float,
    calendar_by_round_agent: dict[tuple[int, int], str],
) -> BeliefState:
    num_slots = _num_slots(trace)
    calendar_render = calendar_by_round_agent.get((round_idx, target), "")
    return BeliefState(
        prior=_prior_vector(prior, num_slots=num_slots),
        posterior=_prior_vector(prior, num_slots=num_slots),
        weights=_slot_weights(calendar_render, num_slots=num_slots, mode=weight_mode, max_weight=max_weight),
    )


def _update_slot(state: BeliefState, slot: int, evidence: float, strength: float) -> None:
    if 0 <= slot < len(state.posterior):
        state.posterior[slot] = _nudge(state.posterior[slot], evidence, strength)
        state.observations += 1


def _rows_for_trace(
    trace_path: Path,
    *,
    prior: float,
    weight_mode: str,
    max_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = _load_json(trace_path)
    game_id = str(trace.get("game_id") or trace_path.stem)
    num_agents = _num_agents(trace)
    calendar_by_round_agent = _calendar_by_round_agent(trace)
    participants_by_round = _participants_by_round(trace)
    beliefs: dict[tuple[int, int, int], BeliefState] = {}
    proposal_maps: dict[tuple[int, int, int], dict[int, int]] = {}
    imap_request_slots: dict[tuple[int, int, int], set[int]] = {}
    evidence_rows: list[dict[str, Any]] = []

    def belief(round_idx: int, target: int, observer: int) -> BeliefState:
        key = (round_idx, target, observer)
        if key not in beliefs:
            beliefs[key] = _empty_belief(
                trace,
                round_idx=round_idx,
                target=target,
                observer=observer,
                prior=prior,
                weight_mode=weight_mode,
                max_weight=max_weight,
                calendar_by_round_agent=calendar_by_round_agent,
            )
        return beliefs[key]

    def add_evidence(
        *,
        event_index: int,
        round_idx: int,
        target: int,
        observer: int,
        slot: int,
        evidence: float,
        strength: float,
        source: str,
    ) -> None:
        state = belief(round_idx, target, observer)
        before = state.posterior[slot] if 0 <= slot < len(state.posterior) else None
        _update_slot(state, slot, evidence, strength)
        after = state.posterior[slot] if 0 <= slot < len(state.posterior) else None
        evidence_rows.append({
            "trace_path": str(trace_path),
            "game_id": game_id,
            "event_index": event_index,
            "round": round_idx,
            "target_agent": target,
            "observer_agent": observer,
            "slot": slot,
            "source": source,
            "evidence": evidence,
            "strength": strength,
            "belief_before": before,
            "belief_after": after,
        })

    for event_index, event in enumerate(trace.get("events", [])):
        if event.get("type") != "dm_sent":
            continue
        data = event.get("data") or {}
        try:
            round_idx = int(data.get("round"))
            sender = int(data.get("from_agent"))
            recipient = int(data.get("to_agent"))
        except (TypeError, ValueError):
            continue
        content = str(data.get("content") or "")
        parsed = _parse_dsm(content)
        if parsed:
            kind = parsed.get("dsm")
            if kind == "proposals":
                plan_slots = _plan_slots(parsed)
                proposal_maps[(round_idx, sender, recipient)] = plan_slots
                for slot in sorted(set(plan_slots.values())):
                    add_evidence(
                        event_index=event_index,
                        round_idx=round_idx,
                        target=sender,
                        observer=recipient,
                        slot=slot,
                        evidence=1.0,
                        strength=1.0,
                        source="dsm_proposal",
                    )
            elif kind == "scores":
                plan_slots = proposal_maps.get((round_idx, recipient, sender), {})
                raw_scores = parsed.get("scores") or {}
                for raw_plan_id, raw_score in raw_scores.items():
                    try:
                        plan_id = int(raw_plan_id)
                        score = int(raw_score)
                    except (TypeError, ValueError):
                        continue
                    slot = plan_slots.get(plan_id, plan_id)
                    if score <= 0:
                        evidence = 0.0
                    else:
                        evidence = min(1.0, max(0.0, score / (DSM_SCORE_LEVELS - 1)))
                    add_evidence(
                        event_index=event_index,
                        round_idx=round_idx,
                        target=sender,
                        observer=recipient,
                        slot=slot,
                        evidence=evidence,
                        strength=1.0,
                        source="dsm_score",
                    )
            elif kind == "decision":
                try:
                    slot = int(parsed.get("slot"))
                except (TypeError, ValueError):
                    continue
                add_evidence(
                    event_index=event_index,
                    round_idx=round_idx,
                    target=sender,
                    observer=recipient,
                    slot=slot,
                    evidence=1.0,
                    strength=1.0,
                    source="dsm_decision",
                )
            continue

        parsed_imap = _parse_imap(content)
        if parsed_imap:
            kind = parsed_imap.get("imap")
            if kind == "cost_request":
                slots: set[int] = set()
                for raw_slot in parsed_imap.get("slots") or []:
                    try:
                        slots.add(int(raw_slot))
                    except (TypeError, ValueError):
                        continue
                imap_request_slots[(round_idx, sender, recipient)] = slots
            elif kind == "costs":
                requested_slots = imap_request_slots.get((round_idx, recipient, sender), set())
                raw_costs = parsed_imap.get("costs") or {}
                for raw_slot, raw_cost in raw_costs.items():
                    try:
                        slot = int(raw_slot)
                    except (TypeError, ValueError):
                        continue
                    if requested_slots and slot not in requested_slots:
                        continue
                    evidence = 0.0 if raw_cost is None else 1.0
                    add_evidence(
                        event_index=event_index,
                        round_idx=round_idx,
                        target=sender,
                        observer=recipient,
                        slot=slot,
                        evidence=evidence,
                        strength=1.0,
                        source="imap_cost",
                    )
            elif kind == "decision":
                try:
                    slot = int(parsed_imap.get("slot"))
                except (TypeError, ValueError):
                    continue
                add_evidence(
                    event_index=event_index,
                    round_idx=round_idx,
                    target=sender,
                    observer=recipient,
                    slot=slot,
                    evidence=1.0,
                    strength=1.0,
                    source="imap_decision",
                )
            continue

        parsed_sd = _parse_sd(content)
        if parsed_sd:
            kind = parsed_sd.get("sd")
            # propose: initiator reveals slot S is locally available to them.
            # Strength < 1 because local availability doesn't guarantee global feasibility
            # (initiator may be bumping an existing meeting).
            if kind == "propose":
                try:
                    slot = int(parsed_sd["slot"])
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    add_evidence(
                        event_index=event_index,
                        round_idx=round_idx,
                        target=sender,
                        observer=recipient,
                        slot=slot,
                        evidence=0.85,
                        strength=0.70,
                        source="sd_propose",
                    )
            # reply: attendee reveals hard feasibility (PENDING=feasible, IMPOSSIBLE=not feasible).
            elif kind == "reply":
                try:
                    slot = int(parsed_sd["slot"])
                    status = str(parsed_sd.get("reply_status", ""))
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if status in ("PENDING", "IMPOSSIBLE"):
                        add_evidence(
                            event_index=event_index,
                            round_idx=round_idx,
                            target=sender,
                            observer=recipient,
                            slot=slot,
                            evidence=1.0 if status == "PENDING" else 0.0,
                            strength=1.0,
                            source="sd_reply",
                        )
            # propose_reschedule: same semantics as propose, but for to_slot.
            elif kind == "propose_reschedule":
                try:
                    slot = int(parsed_sd["to_slot"])
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    add_evidence(
                        event_index=event_index,
                        round_idx=round_idx,
                        target=sender,
                        observer=recipient,
                        slot=slot,
                        evidence=0.85,
                        strength=0.70,
                        source="sd_propose_reschedule",
                    )
            # reschedule_reply: hard feasibility for to_slot.
            elif kind == "reschedule_reply":
                try:
                    slot = int(parsed_sd["to_slot"])
                    status = str(parsed_sd.get("reply_status", ""))
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if status in ("PENDING", "IMPOSSIBLE"):
                        add_evidence(
                            event_index=event_index,
                            round_idx=round_idx,
                            target=sender,
                            observer=recipient,
                            slot=slot,
                            evidence=1.0 if status == "PENDING" else 0.0,
                            strength=1.0,
                            source="sd_reschedule_reply",
                        )
            continue

        # Free-form language is intentionally ignored by protocol-semantic VPS.
        # LLM traces are scored with reflection-calibrated VPS or the optional
        # post-hoc LLM judge pipeline instead.

    # Materialize every ordered target/observer pair for each meeting round so
    # protocol-semantic VPS reports include zero-leak pairs instead of silently
    # dropping them from downstream averages and excess-VPS attribution.
    for round_idx in sorted(participants_by_round):
        for target in range(num_agents):
            for observer in range(num_agents):
                if target != observer:
                    belief(round_idx, target, observer)

    pair_rows: list[dict[str, Any]] = []
    for (round_idx, target, observer), state in sorted(beliefs.items()):
        prior_distance = 0.0
        posterior_distance = _distance(state.posterior, state.prior, state.weights)
        vps_loss = posterior_distance - prior_distance
        weight_sum = _weight_sum(state.weights)
        participants = participants_by_round.get(round_idx, set())
        pair_rows.append({
            "trace_path": str(trace_path),
            "game_id": game_id,
            "round": round_idx,
            "target_agent": target,
            "observer_agent": observer,
            "target_is_participant": target in participants,
            "observer_is_participant": observer in participants,
            "num_agents": num_agents,
            "num_slots": len(state.posterior),
            "observations": state.observations,
            "prior_distance_to_ideal": prior_distance,
            "posterior_distance_to_ideal": posterior_distance,
            "vps_loss": vps_loss,
            "vps_loss_per_slot": vps_loss / len(state.posterior) if state.posterior else 0.0,
            "weight_sum": weight_sum,
            "vps_loss_per_weight": vps_loss / weight_sum if weight_sum else 0.0,
            "weight_mode": weight_mode,
            "prior": prior,
        })
    return evidence_rows, pair_rows


def _game_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_game.setdefault(
            (row["trace_path"], row["game_id"], str(row.get("weight_mode") or "")),
            [],
        ).append(row)
    rows: list[dict[str, Any]] = []
    for (trace_path, game_id, weight_mode), game_pair_rows in sorted(by_game.items()):
        total = sum(float(row["vps_loss"]) for row in game_pair_rows)
        weight_sum = sum(float(row.get("weight_sum") or 0.0) for row in game_pair_rows)
        participant_rows = [
            row for row in game_pair_rows
            if row["target_is_participant"] and row["observer_is_participant"]
        ]
        participant_total = sum(
            float(row["vps_loss"])
            for row in participant_rows
        )
        participant_weight_sum = sum(float(row.get("weight_sum") or 0.0) for row in participant_rows)
        rows.append({
            "trace_path": trace_path,
            "game_id": game_id,
            "weight_mode": weight_mode,
            "pair_round_count": len(game_pair_rows),
            "vps_loss_total": total,
            "vps_loss_mean": total / len(game_pair_rows) if game_pair_rows else 0.0,
            "weight_sum": weight_sum,
            "vps_loss_per_weight": total / weight_sum if weight_sum else 0.0,
            "participant_pair_vps_loss_total": participant_total,
            "participant_pair_vps_loss_mean": (
                participant_total / len(participant_rows) if participant_rows else 0.0
            ),
            "participant_pair_weight_sum": participant_weight_sum,
            "participant_pair_vps_loss_per_weight": (
                participant_total / participant_weight_sum if participant_weight_sum else 0.0
            ),
            "observation_count": sum(int(row["observations"]) for row in game_pair_rows),
        })
    return rows


def _target_game_rows(pair_rows: list[dict[str, Any]], *, vps_floor: float) -> list[dict[str, Any]]:
    by_target: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_target.setdefault(
            (
                row["trace_path"],
                row["game_id"],
                str(row.get("weight_mode") or ""),
                int(row["target_agent"]),
            ),
            [],
        ).append(row)

    rows: list[dict[str, Any]] = []
    for (trace_path, game_id, weight_mode, target_agent), target_rows in sorted(by_target.items()):
        total = sum(float(row["vps_loss"]) for row in target_rows)
        participant_rows = [
            row for row in target_rows
            if row["target_is_participant"] and row["observer_is_participant"]
        ]
        participant_total = sum(float(row["vps_loss"]) for row in participant_rows)
        rows.append({
            "trace_path": trace_path,
            "game_id": game_id,
            "weight_mode": weight_mode,
            "target_agent": target_agent,
            "pair_round_count": len(target_rows),
            "vps_loss_total": total,
            "vps_floor": vps_floor,
            "excess_vps_loss_total": max(0.0, total - vps_floor),
            "participant_pair_round_count": len(participant_rows),
            "participant_pair_vps_loss_total": participant_total,
            "participant_pair_excess_vps_loss_total": max(0.0, participant_total - vps_floor),
            "observation_count": sum(int(row["observations"]) for row in target_rows),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _summary_for_mode(
    game_rows: list[dict[str, Any]],
    *,
    trace_count: int,
    evidence_count: int,
    pair_round_count: int,
    prior: float,
    weight_mode: str,
    max_weight: float,
) -> dict[str, Any]:
    mode_rows = [row for row in game_rows if row.get("weight_mode") == weight_mode]
    total = sum(float(row["vps_loss_total"]) for row in mode_rows)
    weight_sum = sum(float(row.get("weight_sum") or 0.0) for row in mode_rows)
    participant_total = sum(float(row.get("participant_pair_vps_loss_total") or 0.0) for row in mode_rows)
    participant_weight_sum = sum(float(row.get("participant_pair_weight_sum") or 0.0) for row in mode_rows)
    return {
        "trace_count": trace_count,
        "game_count": len(mode_rows),
        "evidence_count": evidence_count,
        "pair_round_count": pair_round_count,
        "vps_loss_total": total,
        "vps_loss_mean_per_game": total / len(mode_rows) if mode_rows else 0.0,
        "weight_sum": weight_sum,
        "vps_loss_per_weight": total / weight_sum if weight_sum else 0.0,
        "participant_pair_vps_loss_total": participant_total,
        "participant_pair_vps_loss_per_weight": (
            participant_total / participant_weight_sum if participant_weight_sum else 0.0
        ),
        "prior": prior,
        "weight_mode": weight_mode,
        "max_weight": max_weight,
        "vps_implementation": "protocol_semantic",
    }


def main() -> int:
    root = _calendar_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Trace JSON files, directories, or globs.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prior", type=float, default=0.5, help="Prior feasibility belief for every slot.")
    parser.add_argument(
        "--weight-mode",
        choices=["both", "uniform", "cost"],
        default="both",
        help="Which VPS weighting to compute. Default computes and reports both.",
    )
    parser.add_argument("--max-weight", type=float, default=32.0)
    parser.add_argument(
        "--game-vps-floor",
        type=float,
        default=DEFAULT_GAME_VPS_FLOOR,
        help="Unavoidable slot-equivalent VPS floor subtracted from every game-target total.",
    )
    args = parser.parse_args()

    trace_paths = _trace_paths(args.traces, root=root)
    evidence_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    weight_modes = ["uniform", "cost"] if args.weight_mode == "both" else [args.weight_mode]
    for trace_path in trace_paths:
        for idx, weight_mode in enumerate(weight_modes):
            trace_evidence_rows, trace_pair_rows = _rows_for_trace(
                trace_path,
                prior=args.prior,
                weight_mode=weight_mode,
                max_weight=args.max_weight,
            )
            if idx == 0:
                evidence_rows.extend(trace_evidence_rows)
            pair_rows.extend(trace_pair_rows)

    game_rows = _game_rows(pair_rows)
    target_game_rows = _target_game_rows(pair_rows, vps_floor=args.game_vps_floor)
    out_dir = _resolve(args.out_dir, root=root)
    _write_csv(out_dir / "belief_evidence.csv", evidence_rows)
    _write_csv(out_dir / "pair_round_vps.csv", pair_rows)
    _write_csv(out_dir / "game_summary.csv", game_rows)
    _write_csv(out_dir / "game_target_summary.csv", target_game_rows)
    summaries_by_mode = {
        mode: _summary_for_mode(
            game_rows,
            trace_count=len(trace_paths),
            evidence_count=len(evidence_rows),
            pair_round_count=sum(1 for row in pair_rows if row.get("weight_mode") == mode),
            prior=args.prior,
            weight_mode=mode,
            max_weight=args.max_weight,
        )
        for mode in weight_modes
    }
    primary = summaries_by_mode.get("uniform") or next(iter(summaries_by_mode.values()), {})
    uniform = summaries_by_mode.get("uniform", {})
    cost = summaries_by_mode.get("cost", {})
    summary = {
        **primary,
        "game_count_total_rows": len(game_rows),
        "game_target_count_total_rows": len(target_game_rows),
        "pair_round_count_total_rows": len(pair_rows),
        "game_vps_floor": args.game_vps_floor,
        "weight_mode": "both" if len(weight_modes) > 1 else weight_modes[0],
        "reported_weight_modes": weight_modes,
        "summaries_by_mode": summaries_by_mode,
        "uniform_vps_loss_total": uniform.get("vps_loss_total"),
        "uniform_vps_loss_mean_per_game": uniform.get("vps_loss_mean_per_game"),
        "uniform_vps_loss_per_weight": uniform.get("vps_loss_per_weight"),
        "cost_weighted_vps_loss_total": cost.get("vps_loss_total"),
        "cost_weighted_vps_loss_mean_per_game": cost.get("vps_loss_mean_per_game"),
        "cost_weighted_vps_loss_per_weight": cost.get("vps_loss_per_weight"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"traces: {len(trace_paths)}")
    print(f"games: {len(game_rows)}")
    print(f"belief evidence rows: {len(evidence_rows)}")
    print(f"pair-round rows: {len(pair_rows)}")
    for mode in weight_modes:
        mode_summary = summaries_by_mode[mode]
        print(
            f"{mode} VPS loss total: {mode_summary['vps_loss_total']:.3f} "
            f"(per weight: {mode_summary['vps_loss_per_weight']:.6f})"
        )
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
