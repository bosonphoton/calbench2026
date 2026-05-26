"""Calendar-specific adapter for game-agnostic OpenSkill ratings."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from a2a_engine.ratings import MetricSpec, RatingEvent, RatingParticipant


CALENDAR_RATING_METRICS = [
    MetricSpec(name="coordination_ratio", higher_is_better=True, weight=0.40),
    MetricSpec(name="excess_cost", higher_is_better=False, weight=0.35),
    MetricSpec(name="excess_vps", higher_is_better=False, weight=0.25),
]

SCORE_MARGIN_RATING_VARIANT = "score_margin_v1"

CALENDAR_SCORE_MARGIN_RATING_METRICS = [
    MetricSpec(name="coordination_ratio", higher_is_better=True, weight=0.40, score_margin=0.1),
    MetricSpec(name="excess_cost", higher_is_better=False, weight=0.35, score_margin=10.0),
    MetricSpec(name="excess_vps", higher_is_better=False, weight=0.25, score_margin=5.0),
]

CALENDAR_RATING_VARIANT_METRICS = {
    "default": CALENDAR_RATING_METRICS,
    SCORE_MARGIN_RATING_VARIANT: CALENDAR_SCORE_MARGIN_RATING_METRICS,
}


def _as_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _paper_game_excess_cost(trace: dict[str, Any]) -> float | None:
    metrics = trace.get("metrics") or {}
    scheduled_total = _as_float(metrics.get("total_scheduled_only_excess_burden"))
    if scheduled_total is not None:
        return scheduled_total
    per_agent_scheduled = _list_metric(
        trace,
        "per_agent_scheduled_only_excess_burden",
        "per_agent_coordination_cost",
    )
    if per_agent_scheduled:
        values = [_as_float(value) for value in per_agent_scheduled]
        if all(value is not None for value in values):
            return sum(float(value) for value in values)
    realized = _as_float(metrics.get("realized_cost"))
    optimal = _as_float(metrics.get("scheduled_only_optimal_cost"))
    if optimal is None:
        optimal = _as_float(metrics.get("optimal_cost"))
    if realized is None or optimal is None:
        return None
    return realized - optimal


def _legacy_full_task_excess_cost(trace: dict[str, Any]) -> float | None:
    metrics = trace.get("metrics") or {}
    realized = _as_float(metrics.get("realized_cost"))
    optimal = _as_float(metrics.get("optimal_cost"))
    if realized is None or optimal is None:
        return None
    return realized - optimal


def _started_at(trace: dict[str, Any]) -> datetime:
    raw = trace.get("started_at") or trace.get("ended_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _agent_model(trace: dict[str, Any], agent_id: int) -> str:
    agents = (trace.get("config") or {}).get("agents") or []
    if agent_id < len(agents):
        spec = agents[agent_id] or {}
        if spec.get("rating_player_id"):
            return str(spec["rating_player_id"])
        return str(spec.get("model") or spec.get("type") or f"agent_{agent_id}")
    final_state = trace.get("final_state") or {}
    models = final_state.get("agent_models") or []
    if agent_id < len(models):
        return str(models[agent_id])
    return f"agent_{agent_id}"


def _num_agents(trace: dict[str, Any]) -> int:
    config = trace.get("config") or {}
    metrics = trace.get("metrics") or {}
    return int(config.get("num_agents") or metrics.get("num_agents") or len(config.get("agents") or []))


def _contribution_by_agent(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = (trace.get("final_state") or {}).get("contribution_scores") or []
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            agent_id = int(row.get("agent_id"))
        except (TypeError, ValueError):
            continue
        out[agent_id] = row
    return out


def _list_metric(trace: dict[str, Any], *names: str) -> list[Any]:
    metrics = trace.get("metrics") or {}
    final_state = trace.get("final_state") or {}
    for name in names:
        value = metrics.get(name)
        if isinstance(value, list):
            return value
        value = final_state.get(name)
        if isinstance(value, list):
            return value
    return []


def _task_setting(trace: dict[str, Any], task_scenario: dict[str, Any] | None = None) -> str:
    config = trace.get("config") or {}
    metrics = trace.get("metrics") or {}
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            config.get("task_path"),
            metrics.get("task_path"),
            config.get("experiment_name"),
            config.get("task_id"),
            metrics.get("task_id"),
        )
    )
    if "varied" in haystack:
        return "varied"
    if "uniform" in haystack:
        return "uniform"
    if task_scenario:
        for calendar in task_scenario.get("calendars") or []:
            for slot in calendar:
                if isinstance(slot, dict) and int(slot.get("cost", 1)) not in {0, 1}:
                    return "varied"
    return "uniform"


def _repo_calendar_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _task_path_candidates(raw_path: object, repo_root: str | Path | None = None) -> list[Path]:
    if not raw_path:
        return []
    path = Path(str(raw_path))
    candidates = [path]
    roots = []
    if repo_root is not None:
        roots.append(Path(repo_root))
    roots.extend([Path.cwd(), _repo_calendar_root(), _repo_calendar_root().parents[1]])
    for root in roots:
        candidates.append(root / path)
        candidates.append(root / "games" / "calendar" / path)
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else candidate.resolve()
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def load_task_scenario_for_trace(
    trace: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load the task scenario referenced by a trace, when the local task file is available."""

    config = trace.get("config") or {}
    metrics = trace.get("metrics") or {}
    task_id = config.get("task_id") or metrics.get("task_id")
    if not task_id:
        return None
    task_path = config.get("task_path") or metrics.get("task_path")
    for path in _task_path_candidates(task_path, repo_root=repo_root):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("task_id")) == str(task_id):
                    return row
    return None


def _errand_cost(cost: object) -> float:
    value = _as_float(cost)
    return value if value is not None else 1.0


def _errand_cost_by_id(task_scenario: dict[str, Any] | None, trace: dict[str, Any]) -> dict[int, float]:
    lookup: dict[int, float] = {}
    calendar_sources = []
    if task_scenario:
        calendar_sources.append(task_scenario.get("calendars") or [])
    calendar_sources.append((trace.get("final_state") or {}).get("calendars") or [])
    for calendars in calendar_sources:
        for calendar in calendars:
            for slot in calendar or []:
                if not isinstance(slot, dict) or "errand_id" not in slot:
                    continue
                try:
                    lookup[int(slot["errand_id"])] = float(slot.get("cost", 1))
                except (TypeError, ValueError):
                    continue
    return lookup


def _realized_errand_units_by_agent(
    trace: dict[str, Any],
    *,
    task_scenario: dict[str, Any] | None,
) -> list[float]:
    num_agents = _num_agents(trace)
    totals = [0.0 for _ in range(num_agents)]
    errand_costs = _errand_cost_by_id(task_scenario, trace)
    for event in trace.get("events") or []:
        if event.get("type") != "batch_applied":
            continue
        data = event.get("data") or {}
        try:
            agent_id = int(data.get("agent_id"))
        except (TypeError, ValueError):
            continue
        if agent_id < 0 or agent_id >= num_agents:
            continue
        for action in data.get("actions") or []:
            if not isinstance(action, dict) or action.get("type") != "reschedule":
                continue
            try:
                item_id = int(action.get("item_id"))
            except (TypeError, ValueError):
                continue
            if item_id in errand_costs:
                totals[agent_id] += _errand_cost(errand_costs[item_id])
    return totals


def _oracle_errand_units_by_agent(
    trace: dict[str, Any],
    *,
    task_scenario: dict[str, Any] | None,
) -> list[float] | None:
    if not task_scenario:
        return None
    num_agents = _num_agents(trace)
    calendars = task_scenario.get("calendars") or []
    meetings = task_scenario.get("meetings") or []
    assignments = (trace.get("metrics") or {}).get("scheduled_only_oracle_assignments") or {}
    try:
        normalized_assignments = {int(meeting_id): int(slot) for meeting_id, slot in assignments.items()}
    except (TypeError, ValueError):
        return None
    if not normalized_assignments:
        return [0.0 for _ in range(num_agents)]

    working = deepcopy(calendars)
    totals = [0.0 for _ in range(num_agents)]
    for meeting in meetings:
        try:
            meeting_id = int(meeting["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if meeting_id not in normalized_assignments:
            continue
        slot = normalized_assignments[meeting_id]
        for agent_id in meeting.get("participants") or []:
            try:
                agent_idx = int(agent_id)
            except (TypeError, ValueError):
                continue
            if agent_idx < 0 or agent_idx >= len(working) or slot < 0 or slot >= len(working[agent_idx]):
                continue
            value = working[agent_idx][slot]
            if isinstance(value, dict):
                if "errand_id" in value and not value.get("blocked"):
                    totals[agent_idx] += _errand_cost(value.get("cost", 1))
                target = next((i for i, v in enumerate(working[agent_idx]) if i != slot and v is None), None)
                if target is not None:
                    working[agent_idx][target] = value
            working[agent_idx][slot] = {"meeting_id": meeting_id, "cost": meeting.get("cost", 1)}
    return totals


def _score_margin_excess_cost_by_agent(
    trace: dict[str, Any],
    *,
    task_scenario: dict[str, Any] | None,
) -> list[float] | None:
    oracle = _oracle_errand_units_by_agent(trace, task_scenario=task_scenario)
    if oracle is None:
        return None
    realized = _realized_errand_units_by_agent(trace, task_scenario=task_scenario)
    return [
        realized[i] - oracle[i]
        for i in range(min(len(realized), len(oracle)))
    ]


def extract_calendar_rating_event(
    trace: dict[str, Any],
    *,
    source_path: str | None = None,
    excess_vps_by_agent: dict[int, float] | None = None,
    rating_variant: str = "default",
    task_scenario: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> RatingEvent | None:
    """Extract one seat-level FFA rating event from a calendar trace."""

    num_agents = _num_agents(trace)
    if num_agents <= 1:
        return None

    participants = [
        RatingParticipant(
            participant_id=f"agent_{agent_id}",
            player_id=_agent_model(trace, agent_id),
            metadata={"agent_id": agent_id},
        )
        for agent_id in range(num_agents)
    ]
    contributions = _contribution_by_agent(trace)
    per_agent_excess = _list_metric(
        trace,
        "per_agent_scheduled_only_excess_burden",
        "per_agent_coordination_cost",
        "per_agent_excess_burden",
        "per_agent_cost",
    )
    raw_per_agent_excess = list(per_agent_excess)
    variant_metadata: dict[str, Any] = {}
    if rating_variant == SCORE_MARGIN_RATING_VARIANT:
        task_scenario = task_scenario or load_task_scenario_for_trace(trace, repo_root=repo_root)
        setting = _task_setting(trace, task_scenario)
        score_margin_excess = _score_margin_excess_cost_by_agent(
            trace,
            task_scenario=task_scenario,
        )
        if score_margin_excess is not None:
            per_agent_excess = score_margin_excess
        variant_metadata = {
            "rating_variant": rating_variant,
            "score_margin_setting": setting,
            "metric_score_margins": {
                "coordination_ratio": 0.1,
                "excess_cost": 10.0,
                "excess_vps": 5.0,
            },
        }

    metric_scores: dict[str, dict[str, float]] = {
        "coordination_ratio": {},
        "excess_cost": {},
    }
    raw_metric_scores: dict[str, dict[str, float]] = {
        "coordination_ratio": {},
        "excess_cost": {},
    }
    for agent_id in range(num_agents):
        participant_id = f"agent_{agent_id}"
        contribution = contributions.get(agent_id, {})
        coordination = _as_float(contribution.get("task_success"))
        if coordination is None:
            coordination = _as_float(contribution.get("coordination_rate"))
        if coordination is None:
            coordination = _as_float((trace.get("metrics") or {}).get("coordination_rate"))
        if coordination is not None:
            metric_scores["coordination_ratio"][participant_id] = coordination
            raw_metric_scores["coordination_ratio"][participant_id] = coordination
        if agent_id < len(per_agent_excess):
            excess_cost = _as_float(per_agent_excess[agent_id])
            if excess_cost is not None:
                metric_scores["excess_cost"][participant_id] = excess_cost
        if agent_id < len(raw_per_agent_excess):
            raw_excess_cost = _as_float(raw_per_agent_excess[agent_id])
            if raw_excess_cost is not None:
                raw_metric_scores["excess_cost"][participant_id] = raw_excess_cost

    if excess_vps_by_agent is not None:
        metric_scores["excess_vps"] = {}
        raw_metric_scores["excess_vps"] = {}
        for agent_id in range(num_agents):
            value = excess_vps_by_agent.get(agent_id, 0.0)
            if math.isfinite(float(value)):
                metric_scores["excess_vps"][f"agent_{agent_id}"] = max(0.0, float(value))
                raw_metric_scores["excess_vps"][f"agent_{agent_id}"] = max(0.0, float(value))

    metric_scores = {
        name: scores
        for name, scores in metric_scores.items()
        if len(scores) >= 2
    }
    raw_metric_scores = {
        name: scores
        for name, scores in raw_metric_scores.items()
        if len(scores) >= 2
    }
    if not metric_scores:
        return None

    config = trace.get("config") or {}
    raw_game_metric_scores: dict[str, float] = {}
    paper_game_excess_cost = _paper_game_excess_cost(trace)
    if paper_game_excess_cost is not None:
        raw_game_metric_scores["excess_cost"] = paper_game_excess_cost
    legacy_full_task_excess_cost = _legacy_full_task_excess_cost(trace)
    if legacy_full_task_excess_cost is not None:
        raw_game_metric_scores["legacy_full_task_excess_cost"] = legacy_full_task_excess_cost
    return RatingEvent(
        game_id=str(trace.get("game_id") or Path(source_path or "").stem),
        game_name=str(config.get("game_name") or "calendar"),
        participants=participants,
        metric_scores=metric_scores,
        timestamp=_started_at(trace),
        source_path=source_path,
        metadata={
            "experiment_name": config.get("experiment_name"),
            "experiment_run_id": config.get("experiment_run_id"),
            "task_id": config.get("task_id") or (trace.get("metrics") or {}).get("task_id"),
            "raw_metric_scores": raw_metric_scores,
            "raw_game_metric_scores": raw_game_metric_scores,
            **variant_metadata,
        },
    )


def load_calendar_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_target_vps_from_pair_csv(
    path: str | Path,
    *,
    weight_mode: str | None = "uniform",
    participants_only: bool = True,
    value_column: str = "vps_loss",
) -> dict[str, dict[int, float]]:
    """Group pair-round VPS rows by leaked-about target agent.

    Returns {game_id: {target_agent: total_vps_loss}}. The target grouping
    measures how much information about that agent's calendar moved observers'
    beliefs.
    """

    grouped: dict[str, dict[int, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if weight_mode and (row.get("weight_mode") or "").strip().lower() != weight_mode:
                continue
            if participants_only:
                if str(row.get("target_is_participant")).lower() != "true":
                    continue
                if str(row.get("observer_is_participant")).lower() != "true":
                    continue
            game_id = row.get("game_id")
            if not game_id:
                continue
            try:
                target_agent = int(row["target_agent"])
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(game_id, {})
            grouped[game_id][target_agent] = grouped[game_id].get(target_agent, 0.0) + value
    return grouped


def load_target_vps_from_game_target_csv(
    path: str | Path,
    *,
    value_column: str = "excess_calibrated_vps_loss_total",
    weight_mode: str | None = "uniform",
) -> dict[str, dict[int, float]]:
    """Load {game_id: {target_agent: vps_loss}} from a game-target summary CSV."""

    grouped: dict[str, dict[int, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_weight_mode = (row.get("weight_mode") or "").strip().lower()
            if weight_mode and row_weight_mode and row_weight_mode != weight_mode:
                continue
            game_id = row.get("game_id")
            if not game_id:
                continue
            try:
                target_agent = int(row["target_agent"])
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(game_id, {})
            grouped[game_id][target_agent] = max(0.0, value)
    return grouped
