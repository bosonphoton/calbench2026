"""Calendar scatter: excess VPS vs excess cost.

By default this script follows ``TRACE_SELECTION.md`` and uses only the
canonical CalBench 90-game local trace mirror. Passing ``--include-live`` adds
the extended live-matchmaking traces and writes outputs with a ``with_live``
suffix so those artifacts are never mistaken for the canonical set.

It penalizes unscheduled meetings by the task-derived worst-complete gap, so
failed meetings cannot make excess cost look artificially low. For model points,
the initial mixed5 run and the Qwen/DeepSeek arena-entry runs each contribute
all five model seats. Baselines are shown as reference lines instead of points:
SD-MAP for the high-privacy static
baseline and IMAP for the low-cost static baseline. When available, the
canonical Paper-DSM theta sweep is overlaid as a static-policy frontier.

Usage:

    uv run python analysis/scripts/canonical_uniform_excess_vps_scatter.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CALENDAR_ROOT.parents[1]
DEFAULT_THETA_SWEEP_SUMMARY = (
    CALENDAR_ROOT / "analysis" / "outputs" / "canonical_dsm_theta_sweep" / "summary.csv"
)


def _default_trace_root() -> Path:
    env_root = os.environ.get("CALBENCH_TRACE_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    candidates = [
        CALENDAR_ROOT / "shared-traces" / "calbench-mixed",
        CALENDAR_ROOT.parent / "shared-traces" / "calbench-mixed",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_TRACE_ROOT = _default_trace_root()

sys.path.insert(0, str(CALENDAR_ROOT))

from calendar_game.solver import _slot_cost, cost_by_agent_for_assignments, solve_optimal  # noqa: E402

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    cp_model = None


MODEL_RUN = "mixed5-calbench90-001"
ENTRY_RUNS = (
    "qwen36-plus-arena-entry-reflection-002",
    "deepseek-v4-pro-arena-entry-reflection-002",
)
LIVE_MATCHMAKING_DIR = "live-matchmaking"
BASELINE_RUNS = {
    "baseline-imap-calbench-mixed-001": "IMAP",
    "baseline-dsm-private-calbench-mixed-001": "DSM-private",
    "baseline-dsm-welfare-calbench-mixed-001": "DSM-welfare",
    "baseline-sd-map-calbench-mixed-001": "SD-MAP",
}

MODEL_LABELS = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "publishers/google/models/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "publishers/google/models/gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "meta/llama-4-maverick-17b-128e-instruct-maas": "Llama 4 Maverick",
    "qwen/qwen3.6-plus": "Qwen 3.6 Plus",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
}

MODEL_STYLE = {
    "Claude Sonnet 4.6": {"color": "#8b5cf6", "marker": "v", "short": "Claude"},
    "DeepSeek V4 Pro": {"color": "#0891b2", "marker": "<", "short": "DeepSeek"},
    "Gemini 3.1 Pro": {"color": "#111827", "marker": "X", "short": "Gemini Pro"},
    "Gemini 3 Flash": {"color": "#6b7280", "marker": "P", "short": "Gemini Flash"},
    "GPT-5.4 Mini": {"color": "#f59e0b", "marker": "*", "short": "GPT-5.4 Mini"},
    "Llama 4 Maverick": {"color": "#65a30d", "marker": "h", "short": "Llama"},
    "Qwen 3.6 Plus": {"color": "#db2777", "marker": ">", "short": "Qwen"},
}

BASELINE_STYLE = {
    "IMAP": {"color": "#1f77b4", "marker": "o"},
    "DSM-private": {"color": "#2ca02c", "marker": "s"},
    "DSM-welfare": {"color": "#d62728", "marker": "^"},
    "SD-MAP": {"color": "#9467bd", "marker": "D"},
}

STATIC_FRONTIER_STYLE = {"color": "#0f766e", "marker": "o"}

LABEL_OFFSETS = {
    "DSM-private": (8, -10),
    "SD-MAP": (8, 7),
    "GPT-5.4 Mini": (8, -17),
    "Gemini 3 Flash": (8, 12),
    "Claude Sonnet 4.6": (8, -19),
    "Llama 4 Maverick": (8, 12),
    "Gemini 3.1 Pro": (8, 8),
    "Qwen 3.6 Plus": (8, -13),
    "IMAP": (8, 8),
    "DeepSeek V4 Pro": (8, 8),
    "DSM-welfare": (-74, 8),
}


@dataclass
class PointAccumulator:
    label: str
    series: str
    excess_vps: list[float]
    excess_cost: list[float]
    excluded_incomplete: int = 0

    def add(self, *, vps: float, cost: float) -> None:
        if math.isfinite(vps) and math.isfinite(cost):
            self.excess_vps.append(max(0.0, vps))
            self.excess_cost.append(cost)

    def row(self) -> dict[str, Any]:
        if not self.excess_vps:
            return {
                "label": self.label,
                "series": self.series,
                "n": 0,
                "incomplete": self.excluded_incomplete,
                "excess_vps_mean": "",
                "excess_cost_mean": "",
            }
        return {
            "label": self.label,
            "series": self.series,
            "n": len(self.excess_vps),
            "incomplete": self.excluded_incomplete,
            "excess_vps_mean": fmean(self.excess_vps),
            "excess_cost_mean": fmean(self.excess_cost),
        }


_TASK_BODY_CACHE: dict[str, tuple[list[list[dict[str, Any] | None]], list[dict[str, Any]], int]] = {}
_WORST_COMPLETE_CACHE: dict[str, tuple[float, list[float]]] = {}
_OPTIMAL_AGENT_COST_CACHE: dict[str, list[float]] = {}
_SCHEDULED_ORACLE_AGENT_COST_CACHE: dict[tuple[str, tuple[int, ...]], list[float]] = {}


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("*.json")
        if not path.name.endswith(".metadata.json")
        and "_reports" not in path.parts
        and "_index" not in path.parts
    )


def _is_uniform_trace(trace: dict[str, Any]) -> bool:
    return _trace_matches_setting(trace, "uniform")


def _trace_matches_setting(trace: dict[str, Any], setting: str) -> bool:
    config = trace.get("config") or {}
    experiment_name = str(config.get("experiment_name") or "")
    task_id = str(config.get("task_id") or "")
    if setting == "uniform":
        return "uniform" in experiment_name or "_c1_" in task_id
    if setting == "varied":
        return "varied" in experiment_name or "_c3_" in task_id
    raise ValueError(f"unknown setting: {setting}")


def _is_complete_trace(trace: dict[str, Any]) -> bool:
    coordination_rate = float((trace.get("metrics") or {}).get("coordination_rate") or 0.0)
    return coordination_rate >= 1.0 - 1e-9


def _load_reflection_vps(run_dir: Path) -> dict[tuple[str, int], float]:
    path = run_dir / "_reports" / "reflection_vps_excess" / "game_target_summary.csv"
    return _load_game_target_values(path, "excess_calibrated_vps_loss_total")


def _load_protocol_vps(run_dir: Path) -> dict[tuple[str, int], float]:
    path = run_dir / "_reports" / "vps" / "game_target_summary.csv"
    return _load_game_target_values(path, "excess_vps_loss_total", weight_mode="uniform")


def _load_game_target_values(
    path: Path,
    value_column: str,
    *,
    weight_mode: str | None = None,
) -> dict[tuple[str, int], float]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[tuple[str, int], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if weight_mode is not None and (row.get("weight_mode") or "").lower() != weight_mode:
                continue
            game_id = row.get("game_id")
            if not game_id:
                continue
            out[(game_id, int(row["target_agent"]))] = float(row[value_column])
    return out


def _trace_excess_cost(trace: dict[str, Any]) -> float:
    metrics = trace.get("metrics") or {}
    realized = float(metrics.get("realized_cost") or 0.0)
    optimal = float(metrics.get("optimal_cost") or 0.0)
    return realized - optimal


def _vps_for_game(trace: dict[str, Any], vps_total: float) -> float:
    _ = trace
    return vps_total


def _task_id(trace: dict[str, Any]) -> str:
    config = trace.get("config") or {}
    return str(config.get("task_id") or (trace.get("metrics") or {}).get("task_id") or trace.get("game_id"))


def _task_body(trace: dict[str, Any]) -> tuple[list[list[dict[str, Any] | None]], list[dict[str, Any]], int]:
    task_id = _task_id(trace)
    if task_id in _TASK_BODY_CACHE:
        return _TASK_BODY_CACHE[task_id]
    config = trace.get("config") or {}
    calendars = config.get("calendars") or trace.get("initial_calendars")
    meetings = config.get("meetings")
    num_slots = int(config.get("num_slots") or 0)
    if calendars is None or meetings is None or not num_slots:
        # Trace configs normally omit full task bodies; fall back to task_path.
        task_path = config.get("task_path")
        if task_path:
            resolved = CALENDAR_ROOT / str(task_path)
            with resolved.open(encoding="utf-8") as handle:
                for line in handle:
                    task = json.loads(line)
                    if str(task.get("task_id")) == task_id:
                        calendars = task["calendars"]
                        meetings = task["meetings"]
                        num_slots = int((task.get("params") or {}).get("num_slots") or task.get("num_slots") or 0)
                        break
    if calendars is None or meetings is None or not num_slots:
        raise ValueError(f"cannot resolve task body for {task_id}")
    _TASK_BODY_CACHE[task_id] = (calendars, meetings, num_slots)
    return _TASK_BODY_CACHE[task_id]


def _worst_complete_costs(trace: dict[str, Any]) -> tuple[float, list[float]]:
    task_id = _task_id(trace)
    if task_id not in _WORST_COMPLETE_CACHE:
        calendars, meetings, num_slots = _task_body(trace)
        _WORST_COMPLETE_CACHE[task_id] = _solve_worst_complete_costs(calendars, meetings, num_slots)
    return _WORST_COMPLETE_CACHE[task_id]


def _solve_worst_complete_costs(
    calendars: list[list[dict[str, Any] | None]],
    meetings: list[dict[str, Any]],
    num_slots: int,
) -> tuple[float, list[float]]:
    if cp_model is None:
        return _solve_worst_complete_costs_backtracking(calendars, meetings, num_slots)

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
            costs[(meeting_id, slot)] = int(cost)
            possible.append(var)
        if not possible:
            raise ValueError(f"meeting {meeting_id} has no feasible slot")
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

    model.Maximize(sum(costs[key] * var for key, var in variables.items()))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError("no complete feasible assignment")
    assignments = {
        meeting_id: slot
        for (meeting_id, slot), var in variables.items()
        if solver.BooleanValue(var)
    }
    total, per_agent = cost_by_agent_for_assignments(calendars, meetings, assignments)
    if total is None:
        raise ValueError("could not replay worst complete assignment")
    return float(total), [float(value) for value in per_agent]


def _solve_worst_complete_costs_backtracking(
    calendars: list[list[dict[str, Any] | None]],
    meetings: list[dict[str, Any]],
    num_slots: int,
) -> tuple[float, list[float]]:
    feasible_slots_by_meeting: list[list[tuple[int, int]]] = []
    for meeting in meetings:
        feasible: list[tuple[int, int]] = []
        for slot in range(num_slots):
            cost = _slot_cost(calendars, meeting["participants"], slot)
            if cost is not None:
                feasible.append((slot, int(cost)))
        if not feasible:
            raise ValueError(f"meeting {meeting['id']} has no feasible slot")
        feasible_slots_by_meeting.append(feasible)

    best: float | None = None
    best_assignments: dict[int, int] = {}

    def walk(meeting_idx: int, occupied_by_agent: tuple[int, ...], cost_so_far: int, assignments: dict[int, int]) -> None:
        nonlocal best, best_assignments
        if meeting_idx == len(meetings):
            if best is None or cost_so_far > best:
                best = cost_so_far
                best_assignments = dict(assignments)
            return
        meeting = meetings[meeting_idx]
        meeting_id = int(meeting["id"])
        for slot, cost in feasible_slots_by_meeting[meeting_idx]:
            bit = 1 << slot
            if any(occupied_by_agent[agent_id] & bit for agent_id in meeting["participants"]):
                continue
            next_occupied = list(occupied_by_agent)
            for agent_id in meeting["participants"]:
                next_occupied[agent_id] |= bit
            assignments[meeting_id] = slot
            walk(meeting_idx + 1, tuple(next_occupied), cost_so_far + cost, assignments)
            assignments.pop(meeting_id, None)

    walk(0, tuple(0 for _ in calendars), 0, {})
    if best is None:
        raise ValueError("no complete feasible assignment")
    total, per_agent = cost_by_agent_for_assignments(calendars, meetings, best_assignments)
    if total is None:
        raise ValueError("could not replay worst complete assignment")
    return float(total), [float(value) for value in per_agent]


def _optimal_agent_costs(trace: dict[str, Any]) -> list[float]:
    task_id = _task_id(trace)
    if task_id not in _OPTIMAL_AGENT_COST_CACHE:
        calendars, meetings, _num_slots = _task_body(trace)
        assignments = (trace.get("metrics") or {}).get("oracle_assignments") or None
        if assignments is None:
            task_path = (trace.get("config") or {}).get("task_path")
            if task_path:
                with (CALENDAR_ROOT / str(task_path)).open(encoding="utf-8") as handle:
                    for line in handle:
                        task = json.loads(line)
                        if str(task.get("task_id")) == task_id:
                            assignments = (task.get("optimal") or {}).get("assignments")
                            break
        if assignments is None:
            assignments = solve_optimal(calendars, meetings, _num_slots).get("assignments", {})
        _total, per_agent = cost_by_agent_for_assignments(calendars, meetings, assignments)
        _OPTIMAL_AGENT_COST_CACHE[task_id] = [float(value) for value in per_agent]
    return _OPTIMAL_AGENT_COST_CACHE[task_id]


def _scheduled_meeting_ids(trace: dict[str, Any]) -> set[int]:
    metrics = trace.get("metrics") or {}
    raw_ids = metrics.get("scheduled_meeting_ids")
    if raw_ids is not None:
        return {int(value) for value in raw_ids}
    return {
        int(row["meeting_id"])
        for row in (trace.get("final_state") or {}).get("round_outcomes") or []
        if row.get("coordinated")
    }


def _scheduled_oracle_agent_costs(trace: dict[str, Any], scheduled_ids: set[int]) -> list[float]:
    task_id = _task_id(trace)
    key = (task_id, tuple(sorted(scheduled_ids)))
    if key not in _SCHEDULED_ORACLE_AGENT_COST_CACHE:
        calendars, meetings, num_slots = _task_body(trace)
        subset = [meeting for meeting in meetings if int(meeting["id"]) in scheduled_ids]
        if not subset:
            _SCHEDULED_ORACLE_AGENT_COST_CACHE[key] = [0.0 for _ in calendars]
        else:
            optimal = solve_optimal(calendars, subset, num_slots)
            _total, per_agent = cost_by_agent_for_assignments(
                calendars,
                subset,
                optimal.get("assignments", {}),
            )
            _SCHEDULED_ORACLE_AGENT_COST_CACHE[key] = [float(value) for value in per_agent]
    return _SCHEDULED_ORACLE_AGENT_COST_CACHE[key]


def _agent_participant_meetings(trace: dict[str, Any], agent_id: int) -> list[dict[str, Any]]:
    _calendars, meetings, _num_slots = _task_body(trace)
    return [meeting for meeting in meetings if agent_id in meeting.get("participants", [])]


def _agent_point_cost(trace: dict[str, Any], agent_id: int, cost_mode: str) -> float:
    participant_meetings = _agent_participant_meetings(trace, agent_id)
    if cost_mode == "trace":
        return _trace_excess_cost(trace)

    final_state = trace.get("final_state") or {}
    metrics = trace.get("metrics") or {}
    realized_costs = (
        metrics.get("per_agent_realized_cost")
        or final_state.get("per_agent_cost")
        or [0.0 for _ in range(int((trace.get("config") or {}).get("num_agents") or 0))]
    )
    scheduled_ids = _scheduled_meeting_ids(trace)
    scheduled_oracle = _scheduled_oracle_agent_costs(trace, scheduled_ids)
    scheduled_excess = float(realized_costs[agent_id]) - float(scheduled_oracle[agent_id])

    if cost_mode == "scheduled-only":
        return scheduled_excess
    if cost_mode == "complete-only":
        return scheduled_excess if _is_complete_trace(trace) else float("nan")
    if cost_mode == "adjusted":
        _worst_total, worst_agent_costs = _worst_complete_costs(trace)
        oracle_agent_costs = _optimal_agent_costs(trace)
        agent_failure_penalty = max(
            0.0,
            float(worst_agent_costs[agent_id]) - float(oracle_agent_costs[agent_id]),
        )
        missed_ids = {
            int(meeting["id"])
            for meeting in participant_meetings
            if int(meeting["id"]) not in scheduled_ids
        }
        adjusted_excess = scheduled_excess + len(missed_ids) * agent_failure_penalty
        return adjusted_excess / max(len(participant_meetings), 1)
    raise ValueError(f"unknown cost_mode: {cost_mode}")


def _accumulator(
    points: dict[str, PointAccumulator],
    label: str,
    series: str,
) -> PointAccumulator:
    if label not in points:
        points[label] = PointAccumulator(label=label, series=series, excess_vps=[], excess_cost=[])
    return points[label]


def _add_model_trace(
    points: dict[str, PointAccumulator],
    trace: dict[str, Any],
    *,
    vps_values: dict[tuple[str, int], float],
    path: Path,
    cost_mode: str,
    setting: str,
    agent_filter: set[int] | None = None,
    forced_label: str | None = None,
) -> None:
    if not _trace_matches_setting(trace, setting):
        return
    game_id = str(trace.get("game_id") or path.stem)
    agents = (trace.get("config") or {}).get("agents") or []
    for agent_id, agent in enumerate(agents):
        if agent_filter is not None and agent_id not in agent_filter:
            continue
        raw_model = str(agent.get("model") or agent.get("type") or f"agent_{agent_id}")
        label = forced_label or MODEL_LABELS.get(raw_model, raw_model)
        accumulator = _accumulator(points, label, "model")
        if not _is_complete_trace(trace):
            accumulator.excluded_incomplete += 1
        accumulator.add(
            vps=_vps_for_game(trace, vps_values.get((game_id, agent_id), 0.0)),
            cost=_agent_point_cost(trace, agent_id, cost_mode),
        )


def _add_model_run(
    points: dict[str, PointAccumulator],
    run_dir: Path,
    *,
    cost_mode: str,
    setting: str,
    agent_filter: set[int] | None = None,
    forced_label: str | None = None,
) -> None:
    vps_values = _load_reflection_vps(run_dir)
    for path in _trace_paths(run_dir):
        trace = json.loads(path.read_text(encoding="utf-8"))
        _add_model_trace(
            points,
            trace,
            vps_values=vps_values,
            path=path,
            cost_mode=cost_mode,
            setting=setting,
            agent_filter=agent_filter,
            forced_label=forced_label,
        )


def collect_points(
    trace_root: Path,
    *,
    cost_mode: str,
    setting: str,
    include_baselines: bool = True,
) -> list[dict[str, Any]]:
    points: dict[str, PointAccumulator] = {}

    _add_model_run(points, trace_root / MODEL_RUN, cost_mode=cost_mode, setting=setting)

    for run_id in ENTRY_RUNS:
        _add_model_run(points, trace_root / run_id, cost_mode=cost_mode, setting=setting)

    if include_baselines:
        for run_id, label in BASELINE_RUNS.items():
            run_dir = trace_root / run_id
            vps_values = _load_protocol_vps(run_dir)
            for path in _trace_paths(run_dir):
                trace = json.loads(path.read_text(encoding="utf-8"))
                if not _trace_matches_setting(trace, setting):
                    continue
                game_id = str(trace.get("game_id") or path.stem)
                for agent_id in range(int((trace.get("config") or {}).get("num_agents") or 0)):
                    accumulator = _accumulator(points, label, "baseline")
                    if not _is_complete_trace(trace):
                        accumulator.excluded_incomplete += 1
                    accumulator.add(
                        vps=_vps_for_game(trace, vps_values.get((game_id, agent_id), 0.0)),
                        cost=_agent_point_cost(trace, agent_id, cost_mode),
                    )

    return sorted(
        (point.row() for point in points.values()),
        key=lambda row: (
            row["series"] != "baseline",
            float(row["excess_vps_mean"]) if row["excess_vps_mean"] != "" else math.inf,
            row["label"],
        ),
    )


def collect_points_with_live(
    trace_root: Path,
    *,
    cost_mode: str,
    setting: str,
    include_baselines: bool = True,
) -> list[dict[str, Any]]:
    # Rebuild canonical points from raw traces instead of summary replication,
    # so live and canonical samples have the same weighting.
    points_by_key: dict[str, PointAccumulator] = {}
    _add_model_run(points_by_key, trace_root / MODEL_RUN, cost_mode=cost_mode, setting=setting)
    for run_id in ENTRY_RUNS:
        _add_model_run(points_by_key, trace_root / run_id, cost_mode=cost_mode, setting=setting)
    live_root = trace_root / LIVE_MATCHMAKING_DIR
    for run_dir in sorted(path for path in live_root.iterdir() if path.is_dir()):
        _add_model_run(points_by_key, run_dir, cost_mode=cost_mode, setting=setting)

    # Add baselines after model aggregation so reference lines are unchanged.
    if include_baselines:
        for run_id, label in BASELINE_RUNS.items():
            run_dir = trace_root / run_id
            vps_values = _load_protocol_vps(run_dir)
            for path in _trace_paths(run_dir):
                trace = json.loads(path.read_text(encoding="utf-8"))
                if not _trace_matches_setting(trace, setting):
                    continue
                game_id = str(trace.get("game_id") or path.stem)
                for agent_id in range(int((trace.get("config") or {}).get("num_agents") or 0)):
                    accumulator = _accumulator(points_by_key, label, "baseline")
                    if not _is_complete_trace(trace):
                        accumulator.excluded_incomplete += 1
                    accumulator.add(
                        vps=_vps_for_game(trace, vps_values.get((game_id, agent_id), 0.0)),
                        cost=_agent_point_cost(trace, agent_id, cost_mode),
                    )

    return sorted(
        (point.row() for point in points_by_key.values()),
        key=lambda row: (
            row["series"] != "baseline",
            float(row["excess_vps_mean"]) if row["excess_vps_mean"] != "" else math.inf,
            row["label"],
        ),
    )


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_static_frontier(path: Path, setting: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("setting") != setting:
                continue
            try:
                rows.append({
                    "theta": float(row["theta"]),
                    "excess_vps_mean": float(row["excess_vps_mean"]),
                    "excess_cost_mean": float(row["adjusted_excess_cost_mean"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda item: item["excess_vps_mean"])


def _row_value(rows: list[dict[str, Any]], label: str, key: str) -> float:
    for row in rows:
        if row["label"] == label and row.get(key) != "":
            return float(row[key])
    raise ValueError(f"missing {key} for {label}")


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated rows for a two-objective minimization plot."""
    frontier: list[dict[str, Any]] = []
    best_y = math.inf
    for row in sorted(rows, key=lambda item: (float(item["excess_vps_mean"]), float(item["excess_cost_mean"]))):
        y = float(row["excess_cost_mean"])
        if y < best_y - 1e-12:
            frontier.append(row)
            best_y = y
    return frontier


def _axis_bounds(setting: str, *, benefit_axes: bool = False) -> tuple[float, float, float, float]:
    if setting == "uniform":
        bounds = (-0.6, 6.0, -0.05, 0.8)
    elif setting == "varied":
        bounds = (-0.5, 5.0, -4.0, 80.0)
    else:
        raise ValueError(f"unknown setting: {setting}")
    if not benefit_axes:
        return bounds
    x_left, x_right, y_bottom, y_top = bounds
    return -x_right, -x_left, -y_top, -y_bottom


def _plot_xy(row: dict[str, Any], *, benefit_axes: bool) -> tuple[float, float]:
    x = float(row["excess_vps_mean"])
    y = float(row["excess_cost_mean"])
    if benefit_axes:
        return -x, -y
    return x, y


def _dynamic_benefit_bounds(
    rows: list[dict[str, Any]],
    *,
    privacy_x: float | None,
    welfare_y: float | None,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x, y = _plot_xy(row, benefit_axes=True)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if privacy_x is not None and math.isfinite(privacy_x):
        xs.append(-privacy_x)
    if welfare_y is not None and math.isfinite(welfare_y):
        ys.append(-welfare_y)
    if not xs or not ys:
        return _axis_bounds("uniform", benefit_axes=True)

    x_min = min(xs)
    x_max = max(max(xs), 0.25)
    y_min = min(ys)
    y_max = max(max(ys), 0.05)
    x_pad = max((x_max - x_min) * 0.08, 0.35)
    y_pad = max((y_max - y_min) * 0.10, 0.05)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _paper_figure_paths(
    *,
    setting: str,
    benefit_axes: bool,
    include_live: bool,
    show_baselines: bool,
) -> tuple[Path, Path]:
    figures = CALENDAR_ROOT / "paper" / "figures"
    if benefit_axes and not include_live and setting == "uniform":
        stem = "protocol_scatter_uniform_excess_vps_without_live_benefit_axes"
    elif benefit_axes and not include_live and setting == "varied" and show_baselines:
        stem = "protocol_scatter_varied_excess_vps_with_baselines_benefit_axes"
    else:
        stem = f"protocol_scatter_{setting}_excess_vps"
        if include_live:
            stem += "_with_live"
        else:
            stem += "_without_live"
        if not show_baselines:
            stem += "_models_only"
        if benefit_axes:
            stem += "_benefit_axes"
    return figures / f"{stem}.png", figures / f"{stem}.pdf"


def plot(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    reference_rows: list[dict[str, Any]],
    static_frontier_rows: list[dict[str, Any]],
    setting: str,
    include_live: bool,
    show_baselines: bool,
    write_paper_copy: bool,
    benefit_axes: bool,
) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    plotted_rows = [
        row
        for row in rows
        if row["series"] == "model" and row["excess_vps_mean"] != "" and row["excess_cost_mean"] != ""
    ]
    privacy_x: float | None = None
    welfare_y: float | None = None
    if show_baselines:
        privacy_x = _row_value(reference_rows, "SD-MAP", "excess_vps_mean")
        welfare_y = _row_value(rows, "IMAP", "excess_cost_mean")
    if benefit_axes:
        x_left, x_right, y_bottom, y_top = _dynamic_benefit_bounds(
            [*plotted_rows, *static_frontier_rows],
            privacy_x=privacy_x,
            welfare_y=welfare_y,
        )
    else:
        x_left, x_right, y_bottom, y_top = _axis_bounds(setting, benefit_axes=False)

    if show_baselines:
        privacy_plot_x = -privacy_x if benefit_axes else privacy_x
        welfare_plot_y = -welfare_y if benefit_axes else welfare_y

        ax.axvline(
            privacy_plot_x,
            color=BASELINE_STYLE["SD-MAP"]["color"],
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            label=(
                f"SD-MAP privacy score = {-privacy_x:.2f}"
                if benefit_axes
                else f"SD-MAP excess VPS = {privacy_x:.2f}"
            ),
            zorder=1,
        )
        ax.axhline(
            welfare_plot_y,
            color=BASELINE_STYLE["IMAP"]["color"],
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            label=(
                f"IMAP cost score = {-welfare_y:.2f}"
                if benefit_axes
                else f"IMAP excess cost = {welfare_y:.2f}"
            ),
            zorder=1,
        )

    if show_baselines and len(static_frontier_rows) >= 2:
        ax.plot(
            [_plot_xy(row, benefit_axes=benefit_axes)[0] for row in static_frontier_rows],
            [_plot_xy(row, benefit_axes=benefit_axes)[1] for row in static_frontier_rows],
            color=STATIC_FRONTIER_STYLE["color"],
            marker=STATIC_FRONTIER_STYLE["marker"],
            markersize=5.5,
            linewidth=2.0,
            alpha=0.86,
            label="Static-policy baseline (DSM)",
            zorder=3,
        )
        for row in static_frontier_rows:
            x, y = _plot_xy(row, benefit_axes=benefit_axes)
            if y > y_top or x < x_left or x > x_right:
                continue
            ax.annotate(
                f"{row['theta']:.2f}",
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7.8,
                color=STATIC_FRONTIER_STYLE["color"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
                zorder=4,
            )
        for row in static_frontier_rows:
            x, y = _plot_xy(row, benefit_axes=benefit_axes)
            outside_top = y > y_top
            outside_bottom = y < y_bottom
            if not (outside_top or outside_bottom) or x < x_left or x > x_right:
                continue
            clipped_y = y_top if outside_top else y_bottom
            marker = "^" if outside_top else "v"
            arrow = "↑" if outside_top else "↓"
            ax.scatter(
                [x],
                [clipped_y],
                marker=marker,
                s=82,
                facecolors=STATIC_FRONTIER_STYLE["color"],
                edgecolors="#064e3b",
                linewidths=0.9,
                clip_on=False,
                zorder=7,
            )
            ax.annotate(
                f"{row['theta']:.2f} {arrow}",
                (x, clipped_y),
                xytext=(8, -12),
                textcoords="offset points",
                fontsize=8,
                color=STATIC_FRONTIER_STYLE["color"],
                weight="bold",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
                zorder=8,
            )

    frontier_rows = _pareto_frontier(plotted_rows)
    if len(frontier_rows) >= 2:
        ax.plot(
            [_plot_xy(row, benefit_axes=benefit_axes)[0] for row in frontier_rows],
            [_plot_xy(row, benefit_axes=benefit_axes)[1] for row in frontier_rows],
            color="#111827",
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            label="LLM Pareto frontier",
            zorder=2,
        )

    for row in plotted_rows:
        label = row["label"]
        x, y = _plot_xy(row, benefit_axes=benefit_axes)
        style = MODEL_STYLE.get(label, {"color": "#374151", "marker": "X", "short": label})
        ax.scatter(
            [x],
            [y],
            marker=style["marker"],
            s=150,
            facecolors="white",
            edgecolors=style["color"],
            linewidths=2.4,
            zorder=5,
        )
        color = style["color"]
        text = style.get("short", label)
        ax.annotate(
            text,
            (x, y),
            xytext=LABEL_OFFSETS.get(label, (8, 6)),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
            zorder=6,
        )

    ax.set_title(f"{setting.title()} Games: Cost Efficiency vs Privacy", fontsize=15, fontweight="bold")
    if benefit_axes:
        ax.set_xlabel("Privacy score (-excess VPS; mean per game-agent)", fontsize=12)
        ax.set_ylabel("Cost score (-excess cost; mean per game-agent)", fontsize=12)
    else:
        ax.set_xlabel("Excess VPS privacy loss (mean per game-agent)", fontsize=12)
        ax.set_ylabel("Scheduled-only excess cost (mean per game-agent)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(bottom=y_bottom, top=y_top)

    if show_baselines and privacy_x is not None:
        privacy_plot_x = -privacy_x if benefit_axes else privacy_x
        ax.text(
            privacy_plot_x - 0.08,
            y_top - 0.08 * (y_top - y_bottom),
            "SD-MAP",
            color=BASELINE_STYLE["SD-MAP"]["color"],
            fontsize=9,
            weight="bold",
            ha="right",
            va="center",
        )
    if show_baselines:
        welfare_plot_y = -welfare_y if benefit_axes and welfare_y is not None else (welfare_y or 0.0)
        ax.text(
            x_right - 0.05 * (x_right - x_left),
            max(y_bottom + 0.02 * (y_top - y_bottom), welfare_plot_y - 0.04 * (y_top - y_bottom)),
            "IMAP",
            color=BASELINE_STYLE["IMAP"]["color"],
            fontsize=9,
            weight="bold",
            ha="right",
            va="center",
        )

    model_patch = plt.Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="white",
        markeredgecolor="#374151", markeredgewidth=2.0, markersize=9, label="LLM model seats"
    )
    ax.legend(handles=[model_patch, *ax.lines], loc="upper right", fontsize=9)

    if include_live:
        stem = f"{setting}_excess_vps_scatter_with_live"
    else:
        stem = f"canonical_{setting}_excess_vps_scatter"
    if not show_baselines:
        stem += "_models_only"
    if benefit_axes:
        stem += "_benefit_axes"
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")

    paper_copy = None
    if write_paper_copy:
        paper_png, paper_pdf = _paper_figure_paths(
            setting=setting,
            benefit_axes=benefit_axes,
            include_live=include_live,
            show_baselines=show_baselines,
        )
        paper_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_png, dpi=180, bbox_inches="tight")
        fig.savefig(paper_pdf, bbox_inches="tight")
        paper_copy = paper_png

    return png, pdf, paper_copy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a canonical-only or with-live directory based on the selected flags.",
    )
    parser.add_argument("--setting", choices=["uniform", "varied"], default="uniform")
    parser.add_argument(
        "--paper-copy",
        action="store_true",
        help="Also write publication-style PNG and PDF figure outputs.",
    )
    parser.add_argument(
        "--benefit-axes",
        action="store_true",
        help="Plot negative VPS and negative excess cost so up and right is better.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=["scheduled-only", "adjusted", "complete-only", "trace"],
        default="adjusted",
        help=(
            "Cost aggregation. scheduled-only uses realized minus scheduled-only oracle cost; "
            "adjusted adds missed_meetings * (worst_complete_cost - optimal_cost); "
            "complete-only excludes incomplete traces; trace uses ordinary realized - optimal for all traces."
        ),
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Also include all live-matchmaking traces from shared-traces/calbench-mixed/live-matchmaking.",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Plot model points only and omit baseline rows/reference lines.",
    )
    parser.add_argument(
        "--theta-sweep-summary",
        type=Path,
        default=DEFAULT_THETA_SWEEP_SUMMARY,
        help="Canonical DSM theta sweep summary to overlay as the static-policy frontier.",
    )
    args = parser.parse_args()

    include_baselines = not args.no_baselines
    out_dir = args.out_dir
    if out_dir is None:
        if args.include_live:
            name = f"{args.setting}_excess_vps_scatter_with_live"
        else:
            name = f"canonical_{args.setting}_excess_vps_scatter"
        if not include_baselines:
            name += "_models_only"
        out_dir = CALENDAR_ROOT / "analysis" / "outputs" / name

    rows = (
        collect_points_with_live(
            args.trace_root,
            cost_mode=args.cost_mode,
            setting=args.setting,
            include_baselines=include_baselines,
        )
        if args.include_live
        else collect_points(
            args.trace_root,
            cost_mode=args.cost_mode,
            setting=args.setting,
            include_baselines=include_baselines,
        )
    )
    reference_rows = rows
    static_frontier_rows = (
        load_static_frontier(args.theta_sweep_summary, args.setting)
        if include_baselines
        else []
    )
    summary = write_summary(rows, out_dir)
    png, pdf, paper_copy = plot(
        rows,
        out_dir,
        reference_rows=reference_rows,
        static_frontier_rows=static_frontier_rows,
        setting=args.setting,
        include_live=args.include_live,
        show_baselines=include_baselines,
        write_paper_copy=args.paper_copy,
        benefit_axes=args.benefit_axes,
    )

    print(f"summary={summary}")
    print(f"png={png}")
    print(f"pdf={pdf}")
    if paper_copy is not None:
        print(f"paper_copy={paper_copy}")
    print()
    print(f"{'Label':<20} {'Series':<8} {'N':>4} {'Incomplete':>10} {'Excess VPS':>11} {'Excess cost':>12}")
    print("-" * 73)
    for row in rows:
        vps = row["excess_vps_mean"]
        cost = row["excess_cost_mean"]
        vps_text = f"{float(vps):.3f}" if vps != "" else "NA"
        cost_text = f"{float(cost):.3f}" if cost != "" else "NA"
        print(
            f"{row['label']:<20} {row['series']:<8} {row['n']:>4} "
            f"{row['incomplete']:>10} {vps_text:>11} {cost_text:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
