"""Plot basic per-model statistics for the canonical CalBench mixed cohort.

The cohort follows ``TRACE_SELECTION.md``:

* ``mixed5-calbench90-001`` contributes all five mixed-model seats.
* Qwen and DeepSeek arena-entry runs contribute only entrant agent 0.
* Live-matchmaking, smoke-test, and baseline runs are excluded.

Outputs:

* trace_agent_metrics.csv
* failure_reasons.csv
* decision_failure_causes.csv
* affected_failed_participants_with_cause.csv
* trace_game_metrics.csv
* meeting_index_metrics.csv
* summary_by_model_difficulty.csv
* coordination_ratio_by_difficulty.png
* messaging_efficiency_by_excess_cost.png
* messaging_efficiency_by_difficulty.png
* excess_cost_by_difficulty_model.png
* message_channel_mix_overall.png
* fairness_by_difficulty_model.png
* fairness_by_setting_model.png
* failure_rate_by_meeting_index.png
* excess_cost_by_meeting_index.png
* excess_cost_by_success_count.png
* failure_reason_by_model.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CALENDAR_ROOT.parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "shared-traces" / "calbench-mixed"
DEFAULT_OUT_DIR = CALENDAR_ROOT / "analysis" / "outputs" / "calbench_mixed_model_performance"

sys.path.insert(0, str(CALENDAR_ROOT))

from calendar_game.solver import _slot_cost  # noqa: E402

MODEL_RUN = "mixed5-calbench90-001"
ENTRY_RUNS: dict[str, tuple[str, int]] = {
    "qwen36-plus-arena-entry-reflection-002": ("Qwen3.6 Plus", 0),
    "deepseek-v4-pro-arena-entry-reflection-002": ("DeepSeek V4 Pro", 0),
    "deepseek-v4-pro-arena-entry-reflection-002-varied": ("DeepSeek V4 Pro", 0),
}

MODEL_LABELS = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "publishers/google/models/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "publishers/google/models/gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "meta/llama-4-maverick-17b-128e-instruct-maas": "Llama 4 Maverick",
    "qwen/qwen3.6-plus": "Qwen3.6 Plus",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
}

MODEL_ORDER = [
    "Claude Sonnet 4.6",
    "Gemini 3.1 Pro",
    "Gemini 3 Flash",
    "GPT-5.4 Mini",
    "Llama 4 Maverick",
    "Qwen3.6 Plus",
    "DeepSeek V4 Pro",
]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
SETTING_ORDER = ["uniform", "varied"]

MODEL_COLORS = {
    "Claude Sonnet 4.6": "#7c3aed",
    "Gemini 3.1 Pro": "#111827",
    "Gemini 3 Flash": "#6b7280",
    "GPT-5.4 Mini": "#d97706",
    "Llama 4 Maverick": "#65a30d",
    "Qwen3.6 Plus": "#db2777",
    "DeepSeek V4 Pro": "#0891b2",
}

MESSAGE_EVENT_TYPES = {
    "dm_sent",
    "groupchat_sent",
    "participant_groupchat_sent",
    "all_groupchat_sent",
}

FAILURE_REASON_DESCRIPTIONS = {
    "missing schedule action": (
        "The decision batch validator saw zero valid schedule actions for this meeting after parsing, "
        "filtering to the current meeting, and any retry attempts."
    ),
    "blocked/unmovable slot": (
        "A participant attempted to move a blocked calendar item, or the failed outcome recorded a "
        "blocked-slot violation."
    ),
    "occupied slot not freed": (
        "A schedule or reschedule target was already occupied and was not freed by another action in "
        "the same batch."
    ),
    "participants chose different slots": (
        "Participants submitted valid schedules, but not all of them chose the same slot."
    ),
    "prior-meeting inconsistency": (
        "A reschedule of a previously scheduled meeting would have left its participants inconsistent."
    ),
    "double-booking conflict": (
        "A participant's batch would have produced an overlapping meeting conflict."
    ),
    "decision batch rejected": (
        "At least one participant's decision batch was rejected, leaving null final slots; inspect the "
        "validator detail for the exact rejection."
    ),
    "unknown/other": "The trace did not expose enough structured detail to classify the failed meeting.",
}

DECISION_FAILURE_DESCRIPTIONS = {
    "empty/no schedule action": (
        "The final rejected decision batch contained no valid schedule action for this meeting. This can "
        "come from empty actions, malformed/truncated JSON that parsed to no actions, or a retry that "
        "dropped the schedule action."
    ),
    "wrong number of schedule actions": (
        "The final rejected decision batch contained more or fewer than one schedule action."
    ),
    "tried to move blocked item": "The batch attempted to reschedule a blocked item, which is illegal.",
    "scheduled occupied slot": "The batch scheduled the meeting into an occupied slot that it did not free.",
    "rescheduled into occupied slot": "The batch moved an item into an occupied slot that it did not free.",
    "malformed action": "A tool call was missing a required schedule/reschedule field.",
    "slot out of range": "A schedule or reschedule action referenced a slot outside the calendar bounds.",
    "duplicate target slot": "Two or more actions in the batch targeted the same destination slot.",
    "missing justification": "A reschedule action lacked the required non-empty justification.",
    "wrong item id/from slot": "A reschedule action claimed an item was at a slot where it was not present.",
    "other rejection": "The validator rejected the batch for another reason; inspect the conflict text.",
}

_TASK_CACHE: dict[str, dict[str, Any]] | None = None


def _run_specs(trace_root: Path) -> list[tuple[str, str | None, int | None]]:
    run_specs: list[tuple[str, str | None, int | None]] = [(MODEL_RUN, None, None)]
    for run_name, (label, entrant_agent_id) in ENTRY_RUNS.items():
        if (trace_root / run_name).is_dir():
            run_specs.append((run_name, label, entrant_agent_id))
    return run_specs


def _iter_selected_traces(trace_root: Path) -> Iterable[tuple[str, str | None, int | None, Path, dict[str, Any]]]:
    """Yield traces from the canonical cohort, deduping optional split entrant runs."""
    seen_entry_game_ids: set[tuple[str, str, str]] = set()
    for run_name, entrant_label, entrant_agent_id in _run_specs(trace_root):
        run_dir = trace_root / run_name
        if not run_dir.is_dir():
            continue
        for path in _trace_paths(run_dir):
            trace = _load_trace(path)
            if trace is None:
                continue
            difficulty = _difficulty(path, trace)
            if difficulty not in DIFFICULTY_ORDER:
                continue
            setting = _setting(path, trace) or "unknown"
            task_id = _task_id(trace)
            if entrant_label:
                dedupe_key = (entrant_label, task_id, setting)
                if dedupe_key in seen_entry_game_ids:
                    continue
                seen_entry_game_ids.add(dedupe_key)
            yield run_name, entrant_label, entrant_agent_id, path, trace


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("*.json")
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
        and "_reports" not in path.parts
        and "_index" not in path.parts
        and not path.name.startswith("openskill_rating_snapshot")
    )


def _load_trace(path: Path) -> dict[str, Any] | None:
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(trace, dict) or not isinstance(trace.get("events"), list):
        return None
    if not isinstance(trace.get("metrics"), dict) or not isinstance(trace.get("config"), dict):
        return None
    return trace


def _difficulty(path: Path, trace: dict[str, Any]) -> str | None:
    metrics = trace.get("metrics") or {}
    value = metrics.get("difficulty") or (trace.get("config") or {}).get("difficulty")
    if isinstance(value, str) and value.lower() in DIFFICULTY_ORDER:
        return value.lower()
    haystack = " ".join(
        str(value)
        for value in (
            path,
            trace.get("game_id"),
            (trace.get("config") or {}).get("experiment_name"),
            (trace.get("config") or {}).get("experiment_run_id"),
            (trace.get("config") or {}).get("task_id"),
        )
        if value is not None
    )
    match = re.search(r"\b(?:b\d+_)?(easy|medium|hard)_", haystack, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _setting(path: Path, trace: dict[str, Any]) -> str | None:
    haystack = " ".join(
        str(value).lower()
        for value in (
            path,
            (trace.get("config") or {}).get("experiment_name"),
            (trace.get("config") or {}).get("task_path"),
            (trace.get("config") or {}).get("task_id"),
        )
        if value is not None
    )
    if "varied" in haystack:
        return "varied"
    if "uniform" in haystack or "_c1_" in haystack:
        return "uniform"
    return None


def _model_label(raw: str | None) -> str:
    return MODEL_LABELS.get(str(raw), str(raw or "unknown"))


def _task_id(trace: dict[str, Any]) -> str:
    return str((trace.get("metrics") or {}).get("task_id") or (trace.get("config") or {}).get("task_id") or "")


def _load_tasks() -> dict[str, dict[str, Any]]:
    global _TASK_CACHE
    if _TASK_CACHE is not None:
        return _TASK_CACHE
    tasks: dict[str, dict[str, Any]] = {}
    for path in (CALENDAR_ROOT / "tasks").glob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                task = json.loads(line)
                tid = task.get("task_id")
                if tid is not None:
                    tasks.setdefault(str(tid), task)
                    tasks[f"{path.name}:{tid}"] = task
                    tasks[f"tasks/{path.name}:{tid}"] = task
    _TASK_CACHE = tasks
    return tasks


def _task_body(trace: dict[str, Any]) -> dict[str, Any] | None:
    task_id = _task_id(trace)
    tasks = _load_tasks()
    if task_id in tasks:
        return tasks[task_id]
    task_path = (trace.get("config") or {}).get("task_path")
    if task_path:
        key = f"{Path(str(task_path)).name}:{task_id}"
        return tasks.get(key) or tasks.get(f"{task_path}:{task_id}")
    return None


def _float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def _median(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(median(clean)) if clean else math.nan


def _ci95(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return math.nan
    mean = sum(clean) / len(clean)
    stdev = math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))
    return 1.96 * stdev / math.sqrt(len(clean))


def _round_meeting_ids(trace: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for event in trace.get("events") or []:
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        meeting = data.get("meeting") or {}
        if meeting.get("id") is not None:
            out[int(data.get("round", len(out)))] = int(meeting["id"])
    return out


def _message_counts_by_round_agent(trace: dict[str, Any]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for event in trace.get("events") or []:
        if event.get("type") not in MESSAGE_EVENT_TYPES:
            continue
        data = event.get("data") or {}
        if data.get("round") is None:
            continue
        agent_id = data.get("from_agent", data.get("agent_id"))
        if agent_id is None:
            continue
        counts[(int(data["round"]), int(agent_id))] += 1
    return counts


def _reject_descriptions_by_round(trace: dict[str, Any]) -> dict[int, list[str]]:
    descriptions: dict[int, list[str]] = defaultdict(list)
    for event in trace.get("events") or []:
        if event.get("type") != "batch_rejected":
            continue
        data = event.get("data") or {}
        if data.get("round") is None:
            continue
        text = str(data.get("conflict_description") or "").strip()
        if text:
            descriptions[int(data["round"])].append(text)
    return descriptions


def _meeting_messages_and_excess(trace: dict[str, Any]) -> dict[tuple[int, int], dict[str, float]]:
    """Return per-agent metrics limited to coordinated meeting rounds."""
    round_to_meeting = _round_meeting_ids(trace)
    messages = _message_counts_by_round_agent(trace)
    outcomes = (trace.get("final_state") or {}).get("round_outcomes") or []
    scheduled_ids = {int(row.get("meeting_id")) for row in outcomes if row.get("coordinated")}
    scheduled_only_excess = [
        _float(value, 0.0)
        for value in (trace.get("metrics") or {}).get("per_agent_scheduled_only_excess_burden") or []
    ]
    per_agent_cost = [
        max(0.0, _float(value, 0.0))
        for value in (trace.get("metrics") or {}).get("per_agent_realized_cost") or []
    ]
    participant_counts: Counter[int] = Counter()
    for outcome in outcomes:
        if not outcome.get("coordinated"):
            continue
        for raw_agent in (outcome.get("per_agent_slot") or {}).keys():
            participant_counts[int(raw_agent)] += 1

    rows: dict[tuple[int, int], dict[str, float]] = {}
    for round_idx, meeting_id in round_to_meeting.items():
        if meeting_id not in scheduled_ids:
            continue
        outcome = next(
            (row for row in outcomes if int(row.get("meeting_id", -1)) == meeting_id),
            None,
        )
        if not outcome:
            continue
        for raw_agent in (outcome.get("per_agent_slot") or {}).keys():
            agent_id = int(raw_agent)
            divisor = max(participant_counts[agent_id], 1)
            rows[(meeting_id, agent_id)] = {
                "messages": float(messages.get((round_idx, agent_id), 0)),
                "scheduled_excess_cost": (
                    scheduled_only_excess[agent_id] / divisor
                    if agent_id < len(scheduled_only_excess)
                    else 0.0
                ),
                "scheduled_realized_cost": (
                    per_agent_cost[agent_id] / divisor
                    if agent_id < len(per_agent_cost)
                    else 0.0
                ),
            }
    return rows


def _paper_fairness_costs(trace: dict[str, Any]) -> list[float]:
    """Paper fairness cost components: |e_g,i - mean_j e_g,j| for each agent."""
    metrics = trace.get("metrics") or {}
    final_state = trace.get("final_state") or {}
    excess_burdens = (
        metrics.get("per_agent_scheduled_only_excess_burden")
        or final_state.get("per_agent_scheduled_only_excess_burden")
        or metrics.get("per_agent_excess_burden")
        or final_state.get("per_agent_excess_burden")
        or []
    )
    values = [_float(value) for value in excess_burdens]
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return []
    mean_excess = sum(clean) / len(clean)
    return [abs(value - mean_excess) if math.isfinite(value) else math.nan for value in values]


def _paper_fairness_cost_for_agent(trace: dict[str, Any], agent_id: int) -> float:
    costs = _paper_fairness_costs(trace)
    return costs[agent_id] if 0 <= agent_id < len(costs) else math.nan


def _paper_fairness_cost_for_game(trace: dict[str, Any]) -> float:
    return _mean(_paper_fairness_costs(trace))


def _classify_failure(outcome: dict[str, Any], descriptions: list[str]) -> tuple[str, str]:
    text = " ".join(descriptions).lower()
    slots = outcome.get("per_agent_slot") or {}
    non_null_slots = {slot for slot in slots.values() if slot is not None}
    has_missing = any(slot is None for slot in slots.values())

    if outcome.get("blocked_slot_violations") or "reschedule attempts to move blocked" in text:
        return "blocked/unmovable slot", "; ".join(descriptions)
    if "occupied" in text or "not freed" in text:
        return "occupied slot not freed", "; ".join(descriptions)
    if "expected exactly 1 schedule action, got 0" in text:
        return "missing schedule action", "; ".join(descriptions)
    if outcome.get("consistency_violated_meeting_ids"):
        return "prior-meeting inconsistency", "; ".join(descriptions)
    if outcome.get("slot_conflicts"):
        return "double-booking conflict", "; ".join(descriptions)
    if len(non_null_slots) > 1:
        return "participants chose different slots", "; ".join(descriptions)
    if has_missing:
        return "decision batch rejected", "; ".join(descriptions)
    return "unknown/other", "; ".join(descriptions)


def _classify_decision_rejection(conflict: str) -> str:
    text = conflict.lower()
    if "expected exactly 1 schedule action, got 0" in text:
        return "empty/no schedule action"
    if "expected exactly 1 schedule action" in text:
        return "wrong number of schedule actions"
    if "attempts to move blocked" in text:
        return "tried to move blocked item"
    if "schedule targets slot" in text and "occupied" in text:
        return "scheduled occupied slot"
    if "reschedule targets slot" in text and "occupied" in text:
        return "rescheduled into occupied slot"
    if "missing required field" in text:
        return "malformed action"
    if "out of range" in text:
        return "slot out of range"
    if "two or more actions target" in text:
        return "duplicate target slot"
    if "non-empty" in text:
        return "missing justification"
    if "claims item_id" in text:
        return "wrong item id/from slot"
    return "other rejection"


def _selected_agent_ids(trace: dict[str, Any], *, entrant_agent_id: int | None) -> list[int]:
    agents = (trace.get("config") or {}).get("agents") or []
    if entrant_agent_id is not None:
        return [entrant_agent_id] if entrant_agent_id < len(agents) else []
    selected: list[int] = []
    for agent_id, agent in enumerate(agents):
        if _model_label(agent.get("model") or agent.get("type")) in MODEL_ORDER:
            selected.append(agent_id)
    return selected


def collect_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for run_name, entrant_label, entrant_agent_id, path, trace in _iter_selected_traces(trace_root):
            difficulty = _difficulty(path, trace)
            if difficulty not in DIFFICULTY_ORDER:
                continue
            setting = _setting(path, trace) or "unknown"
            game_id = str(trace.get("game_id") or path.stem)
            task_id = _task_id(trace)
            metrics = trace.get("metrics") or {}
            realized_cost = _float(metrics.get("realized_cost"))
            optimal_cost = _float(metrics.get("optimal_cost"))
            legacy_trace_excess_cost = (
                realized_cost - optimal_cost
                if math.isfinite(realized_cost) and math.isfinite(optimal_cost)
                else math.nan
            )

            selected_agent_ids = set(_selected_agent_ids(trace, entrant_agent_id=entrant_agent_id))
            agents = (trace.get("config") or {}).get("agents") or []
            contribution_by_agent = {
                int(row["agent_id"]): row
                for row in (trace.get("final_state") or {}).get("contribution_scores") or []
                if row.get("agent_id") is not None
            }
            meeting_rows = _meeting_messages_and_excess(trace)
            meeting_rows_by_agent: dict[int, list[dict[str, float]]] = defaultdict(list)
            for (_meeting_id, agent_id), row in meeting_rows.items():
                meeting_rows_by_agent[agent_id].append(row)

            for agent_id in sorted(selected_agent_ids):
                agent = agents[agent_id] if agent_id < len(agents) else {}
                raw_model = str(agent.get("model") or agent.get("type") or "")
                model = entrant_label or _model_label(raw_model)
                contribution = contribution_by_agent.get(agent_id, {})
                coordinated = _float(contribution.get("coordinated_participant_rounds"), 0.0)
                participant_rounds = _float(contribution.get("participant_rounds"), 0.0)
                messages_sent = _float(contribution.get("messages_sent"), 0.0)
                scheduled_meeting_rows = meeting_rows_by_agent.get(agent_id, [])
                scheduled_messages = sum(row["messages"] for row in scheduled_meeting_rows)
                scheduled_excess = sum(row["scheduled_excess_cost"] for row in scheduled_meeting_rows)
                scheduled_only_excess_burden = _float(contribution.get("scheduled_only_excess_burden"))
                if not math.isfinite(scheduled_only_excess_burden):
                    metric_values = metrics.get("per_agent_scheduled_only_excess_burden") or []
                    if agent_id < len(metric_values):
                        scheduled_only_excess_burden = _float(metric_values[agent_id])
                task_success = coordinated / participant_rounds if participant_rounds > 0 else math.nan
                messages_per_scheduled_meeting = messages_sent / max(coordinated, 1.0)
                trace_rows.append(
                    {
                        "run": run_name,
                        "trace_path": str(path),
                        "game_id": game_id,
                        "task_id": task_id,
                        "setting": setting,
                        "difficulty": difficulty,
                        "agent_id": agent_id,
                        "model": model,
                        "raw_model": raw_model,
                        "participant_rounds": participant_rounds,
                        "coordinated_participant_rounds": coordinated,
                        "task_success": task_success,
                        "coordination_rate": task_success,
                        "messages_sent": messages_sent,
                        "messages_per_participant_round": messages_sent / participant_rounds if participant_rounds > 0 else math.nan,
                        "scheduled_meetings_for_agent": float(len(scheduled_meeting_rows)),
                        "scheduled_messages_sent": scheduled_messages,
                        "messages_for_efficiency": messages_sent,
                        "messages_per_successful_meeting": (
                            messages_per_scheduled_meeting
                            if math.isfinite(messages_per_scheduled_meeting)
                            else math.nan
                        ),
                        "messages_per_scheduled_meeting": (
                            messages_per_scheduled_meeting
                            if math.isfinite(messages_per_scheduled_meeting)
                            else math.nan
                        ),
                        "communication_efficiency": (
                            messages_per_scheduled_meeting
                            if math.isfinite(messages_per_scheduled_meeting)
                            else math.nan
                        ),
                        "scheduled_only_excess_cost": scheduled_excess,
                        "scheduled_excess_cost_per_successful_meeting": (
                            scheduled_excess / len(scheduled_meeting_rows)
                            if scheduled_meeting_rows
                            else math.nan
                        ),
                        "scheduled_only_excess_burden": scheduled_only_excess_burden,
                        "coordination_cost": scheduled_only_excess_burden,
                        "trace_excess_cost": legacy_trace_excess_cost,
                        "legacy_full_task_excess_cost": legacy_trace_excess_cost,
                        "fairness_cost": _paper_fairness_cost_for_agent(trace, agent_id),
                        "communication_share": _float(contribution.get("communication_share")),
                        "contribution_score": _float(contribution.get("contribution_score")),
                    }
                )

            round_to_meeting = _round_meeting_ids(trace)
            reject_descriptions = _reject_descriptions_by_round(trace)
            round_by_meeting = {meeting_id: round_idx for round_idx, meeting_id in round_to_meeting.items()}
            for outcome in (trace.get("final_state") or {}).get("round_outcomes") or []:
                if outcome.get("coordinated"):
                    continue
                meeting_id = int(outcome.get("meeting_id", -1))
                round_idx = round_by_meeting.get(meeting_id, meeting_id - 1)
                reason, detail = _classify_failure(outcome, reject_descriptions.get(round_idx, []))
                for raw_agent in (outcome.get("per_agent_slot") or {}).keys():
                    agent_id = int(raw_agent)
                    if agent_id not in selected_agent_ids:
                        continue
                    agent = agents[agent_id] if agent_id < len(agents) else {}
                    failure_rows.append(
                        {
                            "run": run_name,
                            "trace_path": str(path),
                            "game_id": game_id,
                            "task_id": task_id,
                            "setting": setting,
                            "difficulty": difficulty,
                            "meeting_id": meeting_id,
                            "agent_id": agent_id,
                            "model": entrant_label or _model_label(agent.get("model") or agent.get("type")),
                            "failure_reason": reason,
                            "failure_reason_description": FAILURE_REASON_DESCRIPTIONS.get(reason, ""),
                            "participant_slots": json.dumps(outcome.get("per_agent_slot") or {}, sort_keys=True),
                            "detail": detail,
                        }
                    )
    return trace_rows, failure_rows


def collect_decision_failure_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect causal decision failures and affected selected participants.

    ``failure_reasons.csv`` is participant-centric: every selected participant
    in a failed meeting receives the same meeting-level reason. This function
    records the actual agent-level ``decision_failed`` events that caused
    meeting failure, plus a bridge table showing which selected participants
    were affected by those causal failures.
    """
    causal_rows: list[dict[str, Any]] = []
    affected_rows: list[dict[str, Any]] = []
    seen_entry_game_ids: set[tuple[str, str, str]] = set()

    run_specs: list[tuple[str, str | None, int | None]] = [(MODEL_RUN, None, None)]
    for run_name, (label, entrant_agent_id) in ENTRY_RUNS.items():
        if (trace_root / run_name).is_dir():
            run_specs.append((run_name, label, entrant_agent_id))

    for run_name, entrant_label, entrant_agent_id in run_specs:
        run_dir = trace_root / run_name
        if not run_dir.is_dir():
            continue
        for path in _trace_paths(run_dir):
            trace = _load_trace(path)
            if trace is None:
                continue
            difficulty = _difficulty(path, trace)
            if difficulty not in DIFFICULTY_ORDER:
                continue
            setting = _setting(path, trace) or "unknown"
            game_id = str(trace.get("game_id") or path.stem)
            task_id = str((trace.get("metrics") or {}).get("task_id") or (trace.get("config") or {}).get("task_id") or "")
            if entrant_label:
                dedupe_key = (entrant_label, task_id, setting)
                if dedupe_key in seen_entry_game_ids:
                    continue
                seen_entry_game_ids.add(dedupe_key)

            agents = (trace.get("config") or {}).get("agents") or []
            selected_agent_ids = set(_selected_agent_ids(trace, entrant_agent_id=entrant_agent_id))
            model_by_agent = {
                agent_id: (
                    entrant_label
                    if entrant_label is not None and agent_id == entrant_agent_id
                    else _model_label((agents[agent_id] if agent_id < len(agents) else {}).get("model"))
                )
                for agent_id in range(len(agents))
            }
            round_to_meeting = _round_meeting_ids(trace)
            round_by_meeting = {meeting_id: round_idx for round_idx, meeting_id in round_to_meeting.items()}
            failed_outcomes = {
                int(outcome["meeting_id"]): outcome
                for outcome in (trace.get("final_state") or {}).get("round_outcomes") or []
                if not outcome.get("coordinated") and outcome.get("meeting_id") is not None
            }

            rejections: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            decide_ends: dict[tuple[int, int], dict[str, Any]] = {}
            causal_by_meeting: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for event in trace.get("events") or []:
                data = event.get("data") or {}
                if data.get("phase") != "DECISION":
                    continue
                if event.get("type") == "batch_rejected" and data.get("round") is not None and data.get("agent_id") is not None:
                    rejections[(int(data["round"]), int(data["agent_id"]))].append(data)
                elif event.get("type") == "decide_end" and data.get("round") is not None and data.get("agent_id") is not None:
                    decide_ends[(int(data["round"]), int(data["agent_id"]))] = data
                elif event.get("type") == "decision_failed" and data.get("round") is not None and data.get("agent_id") is not None:
                    round_idx = int(data["round"])
                    agent_id = int(data["agent_id"])
                    meeting_id = round_to_meeting.get(round_idx)
                    if meeting_id not in failed_outcomes:
                        continue
                    last_rejection = (rejections.get((round_idx, agent_id)) or [{}])[-1]
                    decide_end = decide_ends.get((round_idx, agent_id), {})
                    conflict = str(last_rejection.get("conflict_description") or "")
                    subtype = _classify_decision_rejection(conflict)
                    tool_calls = decide_end.get("tool_calls")
                    row = {
                        "run": run_name,
                        "trace_path": str(path),
                        "game_id": game_id,
                        "task_id": task_id,
                        "setting": setting,
                        "difficulty": difficulty,
                        "meeting_id": meeting_id,
                        "round": round_idx,
                        "agent_id": agent_id,
                        "model": model_by_agent.get(agent_id, "unknown"),
                        "selected_agent": agent_id in selected_agent_ids,
                        "decision_failure_reason": subtype,
                        "decision_failure_description": DECISION_FAILURE_DESCRIPTIONS.get(subtype, ""),
                        "validator_conflict": conflict,
                        "final_rejected_actions": json.dumps(last_rejection.get("actions") or []),
                        "initial_tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else "",
                        "initial_text_preview": str(decide_end.get("text") or "").replace("\n", " ")[:240],
                    }
                    causal_rows.append(row)
                    causal_by_meeting[meeting_id].append(row)

            for meeting_id, outcome in failed_outcomes.items():
                participant_agent_ids = [int(agent_id) for agent_id in (outcome.get("per_agent_slot") or {}).keys()]
                meeting_causal_rows = causal_by_meeting.get(meeting_id, [])
                causal_reasons = sorted({str(row["decision_failure_reason"]) for row in meeting_causal_rows})
                causal_models = sorted({str(row["model"]) for row in meeting_causal_rows})
                round_idx = round_by_meeting.get(meeting_id, meeting_id - 1)
                for agent_id in participant_agent_ids:
                    if agent_id not in selected_agent_ids:
                        continue
                    affected_rows.append(
                        {
                            "run": run_name,
                            "trace_path": str(path),
                            "game_id": game_id,
                            "task_id": task_id,
                            "setting": setting,
                            "difficulty": difficulty,
                            "meeting_id": meeting_id,
                            "round": round_idx,
                            "agent_id": agent_id,
                            "model": model_by_agent.get(agent_id, "unknown"),
                            "own_decision_failed": any(row["agent_id"] == agent_id for row in meeting_causal_rows),
                            "causal_decision_failure_reasons": ";".join(causal_reasons) or "no decision_failed event",
                            "causal_decision_failure_descriptions": " | ".join(
                                DECISION_FAILURE_DESCRIPTIONS.get(reason, "") for reason in causal_reasons
                            ),
                            "causal_models": ";".join(causal_models) or "none",
                        }
                    )

    return causal_rows, affected_rows


def _calendar_render_costs(text: str | None) -> dict[int, float]:
    costs: dict[int, float] = {}
    if not text:
        return costs
    for line in text.splitlines():
        slot_match = re.match(r"\s*Slot\s+(\d+):", line)
        cost_match = re.search(r"\(cost=(\d+(?:\.\d+)?)\)", line)
        if slot_match and cost_match:
            costs[int(slot_match.group(1))] = float(cost_match.group(1))
    return costs


def _decision_snapshot_costs(trace: dict[str, Any]) -> dict[tuple[int, int, str], dict[int, float]]:
    snapshots: dict[tuple[int, int, str], dict[int, float]] = {}
    for event in trace.get("events") or []:
        if event.get("type") != "decide_start":
            continue
        data = event.get("data") or {}
        if data.get("round") is None or data.get("agent_id") is None:
            continue
        render = data.get("calendar_snapshot_render") or data.get("calendar_render")
        key = (int(data["round"]), int(data["agent_id"]), str(data.get("phase") or ""))
        snapshots[key] = _calendar_render_costs(str(render or ""))
    return snapshots


def _realized_cost_by_meeting(trace: dict[str, Any]) -> dict[int, float]:
    round_to_meeting = _round_meeting_ids(trace)
    snapshots = _decision_snapshot_costs(trace)
    costs: dict[int, float] = defaultdict(float)
    for event in trace.get("events") or []:
        if event.get("type") != "batch_applied":
            continue
        data = event.get("data") or {}
        if data.get("round") is None or data.get("agent_id") is None:
            continue
        meeting_id = round_to_meeting.get(int(data["round"]))
        if meeting_id is None:
            continue
        phase = str(data.get("phase") or "")
        slot_costs = snapshots.get((int(data["round"]), int(data["agent_id"]), phase), {})
        for action in data.get("actions") or []:
            if isinstance(action, dict) and action.get("type") == "reschedule":
                costs[meeting_id] += slot_costs.get(int(action.get("from_slot", -1)), 0.0)
    return costs


def _oracle_cost_by_meeting(trace: dict[str, Any]) -> dict[int, float]:
    task = _task_body(trace)
    if not task:
        return {}
    calendars = task.get("calendars") or []
    assignments = (trace.get("metrics") or {}).get("scheduled_only_oracle_assignments") or {}
    if not assignments:
        assignments = (task.get("optimal") or {}).get("assignments") or {}
    costs: dict[int, float] = {}
    for meeting in task.get("meetings") or []:
        meeting_id = int(meeting["id"])
        raw_slot = assignments.get(str(meeting_id), assignments.get(meeting_id))
        if raw_slot is None:
            continue
        slot_cost = _slot_cost(calendars, meeting.get("participants") or [], int(raw_slot))
        costs[meeting_id] = float(slot_cost or 0.0)
    return costs


def collect_game_and_meeting_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    game_rows: list[dict[str, Any]] = []
    meeting_rows: list[dict[str, Any]] = []
    for run_name, entrant_label, _entrant_agent_id, path, trace in _iter_selected_traces(trace_root):
        difficulty = _difficulty(path, trace)
        if difficulty not in DIFFICULTY_ORDER:
            continue
        setting = _setting(path, trace) or "unknown"
        metrics = trace.get("metrics") or {}
        game_id = str(trace.get("game_id") or path.stem)
        task_id = _task_id(trace)
        realized = _float(metrics.get("realized_cost"))
        optimal = _float(metrics.get("optimal_cost"))
        trace_excess = realized - optimal if math.isfinite(realized) and math.isfinite(optimal) else math.nan
        scheduled_only_excess = _float(metrics.get("total_scheduled_only_excess_burden"))
        if not math.isfinite(scheduled_only_excess):
            scheduled_only_values = [
                _float(value)
                for value in metrics.get("per_agent_scheduled_only_excess_burden") or []
            ]
            clean_scheduled_only = [value for value in scheduled_only_values if math.isfinite(value)]
            scheduled_only_excess = sum(clean_scheduled_only) if clean_scheduled_only else math.nan
        game_rows.append(
            {
                "run": run_name,
                "cohort_label": entrant_label or "Mixed5 cohort",
                "trace_path": str(path),
                "game_id": game_id,
                "task_id": task_id,
                "setting": setting,
                "difficulty": difficulty,
                "coordination_rate": _float(metrics.get("coordination_rate")),
                "meetings_scheduled": _float(metrics.get("meetings_scheduled")),
                "realized_cost": realized,
                "optimal_cost": optimal,
                "trace_excess_cost": scheduled_only_excess,
                "scheduled_only_excess_cost": scheduled_only_excess,
                "legacy_full_task_excess_cost": trace_excess,
                "fairness_cost": _paper_fairness_cost_for_game(trace),
                "dm_messages": _float(metrics.get("total_dms_sent"), 0.0),
                "participant_groupchat_messages": _float(metrics.get("total_participant_groupchat_messages_sent"), 0.0),
                "all_groupchat_messages": _float(metrics.get("total_all_groupchat_messages_sent"), 0.0),
                "total_cheap_talk_messages": _float(metrics.get("total_cheap_talk_messages"), 0.0),
            }
        )

        realized_by_meeting = _realized_cost_by_meeting(trace)
        oracle_by_meeting = _oracle_cost_by_meeting(trace)
        round_to_meeting = _round_meeting_ids(trace)
        round_by_meeting = {meeting_id: round_idx for round_idx, meeting_id in round_to_meeting.items()}
        for outcome in (trace.get("final_state") or {}).get("round_outcomes") or []:
            meeting_id = int(outcome.get("meeting_id", -1))
            if meeting_id < 0:
                continue
            coordinated = bool(outcome.get("coordinated"))
            realized_meeting_cost = realized_by_meeting.get(meeting_id, 0.0) if coordinated else math.nan
            oracle_meeting_cost = oracle_by_meeting.get(meeting_id, math.nan)
            excess = (
                realized_meeting_cost - oracle_meeting_cost
                if coordinated and math.isfinite(oracle_meeting_cost)
                else math.nan
            )
            meeting_rows.append(
                {
                    "run": run_name,
                    "cohort_label": entrant_label or "Mixed5 cohort",
                    "trace_path": str(path),
                    "game_id": game_id,
                    "task_id": task_id,
                    "setting": setting,
                    "difficulty": difficulty,
                    "meeting_id": meeting_id,
                    "meeting_index": round_by_meeting.get(meeting_id, meeting_id - 1) + 1,
                    "coordinated": coordinated,
                    "failure": not coordinated,
                    "realized_meeting_cost": realized_meeting_cost,
                    "oracle_meeting_cost": oracle_meeting_cost,
                    "meeting_excess_cost": excess,
                }
            )
    return game_rows, meeting_rows


def summarize(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        groups[(str(row["model"]), str(row["difficulty"]))].append(row)

    out: list[dict[str, Any]] = []
    for (model, difficulty), rows in sorted(
        groups.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), DIFFICULTY_ORDER.index(item[0][1])),
    ):
        participant_rounds = sum(_float(row["participant_rounds"], 0.0) for row in rows)
        coordinated = sum(_float(row["coordinated_participant_rounds"], 0.0) for row in rows)
        scheduled_meetings = sum(_float(row["scheduled_meetings_for_agent"], 0.0) for row in rows)
        scheduled_messages = sum(_float(row["scheduled_messages_sent"], 0.0) for row in rows)
        attempted_messages = sum(_float(row["messages_sent"], 0.0) for row in rows)
        scheduled_excess = sum(_float(row["scheduled_only_excess_cost"], 0.0) for row in rows)
        task_success_values = [_float(row["task_success"]) for row in rows]
        messages_per_scheduled_values = [_float(row["messages_per_scheduled_meeting"]) for row in rows]
        coordination_cost_values = [_float(row["coordination_cost"]) for row in rows]
        fairness_cost_values = [_float(row["fairness_cost"]) for row in rows]
        out.append(
            {
                "model": model,
                "difficulty": difficulty,
                "trace_agent_count": len(rows),
                "participant_rounds": participant_rounds,
                "coordinated_participant_rounds": coordinated,
                "task_success": _mean(task_success_values),
                "coordination_ratio": _mean(task_success_values),
                "coordination_ratio_pooled": coordinated / participant_rounds if participant_rounds else math.nan,
                "coordination_ratio_trace_mean": _mean(task_success_values),
                "coordination_ratio_ci95": _ci95(task_success_values),
                "messages_per_successful_meeting": _mean(messages_per_scheduled_values),
                "messages_per_scheduled_meeting": _mean(messages_per_scheduled_values),
                "communication_efficiency": _mean(messages_per_scheduled_values),
                "messages_per_successful_meeting_pooled_scheduled_only": (
                    scheduled_messages / scheduled_meetings if scheduled_meetings else math.nan
                ),
                "messages_per_participant_round_attempted": attempted_messages / participant_rounds if participant_rounds else math.nan,
                "scheduled_only_excess_cost": scheduled_excess,
                "scheduled_excess_cost_per_successful_meeting": scheduled_excess / scheduled_meetings if scheduled_meetings else math.nan,
                "scheduled_only_excess_burden_mean": _mean(coordination_cost_values),
                "coordination_cost": _mean(coordination_cost_values),
                "trace_excess_cost_mean": _mean(coordination_cost_values),
                "legacy_full_task_excess_cost_mean": _mean(_float(row["legacy_full_task_excess_cost"]) for row in rows),
                "fairness_cost_mean": _mean(fairness_cost_values),
                "fairness_cost": _mean(fairness_cost_values),
                "median_trace_messages_per_successful_meeting": _median(
                    _float(row["messages_per_successful_meeting"]) for row in rows
                ),
            }
        )
    return out


def summarize_by_model_setting(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        setting = str(row["setting"])
        if setting in SETTING_ORDER:
            groups[(str(row["model"]), setting)].append(row)

    out: list[dict[str, Any]] = []
    for (model, setting), rows in sorted(
        groups.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), SETTING_ORDER.index(item[0][1])),
    ):
        out.append(
            {
                "model": model,
                "setting": setting,
                "trace_agent_count": len(rows),
                "fairness_cost_mean": _mean(_float(row["fairness_cost"]) for row in rows),
                "fairness_cost_ci95": _ci95(_float(row["fairness_cost"]) for row in rows),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean_label(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=13))


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d1d5db", alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_coordination(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "coordination_ratio_by_difficulty.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True, constrained_layout=True)
    for ax, difficulty in zip(axes, DIFFICULTY_ORDER, strict=True):
        rows = [row for row in summary_rows if row["difficulty"] == difficulty]
        values = [100 * _float(next((row["coordination_ratio"] for row in rows if row["model"] == model), math.nan)) for model in MODEL_ORDER]
        colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), values, color=colors, alpha=0.88)
        ax.set_title(difficulty.title(), fontweight="bold")
        ax.set_xticks(
            range(len(MODEL_ORDER)),
            [_clean_label(model) for model in MODEL_ORDER],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0, 105)
        _style_axes(ax)
        for idx, value in enumerate(values):
            if math.isfinite(value):
                ax.text(idx, value + 1.2, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Task success (%)")
    fig.suptitle("Task Success by Model and Difficulty", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_messages_by_excess(trace_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "messaging_efficiency_by_excess_cost.png"
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    for model in MODEL_ORDER:
        rows = [
            row
            for row in trace_rows
            if row["model"] == model
            and _float(row["scheduled_meetings_for_agent"], 0.0) > 0
            and math.isfinite(_float(row["messages_per_successful_meeting"]))
        ]
        if not rows:
            continue
        x_values = [_float(row["messages_per_successful_meeting"]) for row in rows]
        y_values = [_float(row["scheduled_only_excess_burden"], 0.0) for row in rows]
        ax.scatter(
            x_values,
            y_values,
            s=32,
            alpha=0.58,
            color=MODEL_COLORS[model],
            label=model,
            edgecolors="white",
            linewidths=0.35,
        )
        clean = sorted(
            (x, y)
            for x, y in zip(x_values, y_values, strict=True)
            if math.isfinite(x) and math.isfinite(y)
        )
        if len(clean) >= 2 and len({x for x, _y in clean}) >= 2:
            mean_x = sum(x for x, _y in clean) / len(clean)
            mean_y = sum(y for _x, y in clean) / len(clean)
            denom = sum((x - mean_x) ** 2 for x, _y in clean)
            if denom > 0:
                slope = sum((x - mean_x) * (y - mean_y) for x, y in clean) / denom
                intercept = mean_y - slope * mean_x
                x0, x1 = min(x for x, _y in clean), max(x for x, _y in clean)
                ax.plot(
                    [x0, x1],
                    [intercept + slope * x0, intercept + slope * x1],
                    color=MODEL_COLORS[model],
                    linewidth=1.8,
                    alpha=0.82,
                )
    ax.set_title("Scheduled-Only Excess Burden vs Messages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Messages sent per scheduled meeting")
    ax.set_ylabel("Scheduled-only excess burden")
    _style_axes(ax)
    ax.grid(axis="x", color="#d1d5db", alpha=0.28, linewidth=0.8)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_excess_cost_by_difficulty(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "excess_cost_by_difficulty_model.png"
    fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    width = 0.11
    base_positions = list(range(len(DIFFICULTY_ORDER)))
    offsets = [(idx - (len(MODEL_ORDER) - 1) / 2) * width for idx in range(len(MODEL_ORDER))]
    for model_idx, model in enumerate(MODEL_ORDER):
        values = []
        for difficulty in DIFFICULTY_ORDER:
            row = next((row for row in summary_rows if row["model"] == model and row["difficulty"] == difficulty), None)
            values.append(_float(row["scheduled_only_excess_burden_mean"]) if row else math.nan)
        ax.bar(
            [pos + offsets[model_idx] for pos in base_positions],
            values,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.9,
        )
    ax.set_title("Mean Selected-Agent Excess Burden by Difficulty", fontsize=14, fontweight="bold")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean scheduled-only excess burden")
    ax.set_xticks(base_positions, [difficulty.title() for difficulty in DIFFICULTY_ORDER])
    _style_axes(ax)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_message_channel_mix(game_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "message_channel_mix_overall.png"
    totals = {
        "DM": sum(_float(row["dm_messages"], 0.0) for row in game_rows),
        "Participant groupchat": sum(_float(row["participant_groupchat_messages"], 0.0) for row in game_rows),
        "All groupchat": sum(_float(row["all_groupchat_messages"], 0.0) for row in game_rows),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    colors = ["#2563eb", "#16a34a", "#d97706"]
    labels = list(totals.keys())
    values = [totals[label] for label in labels]
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("Overall Message Channel Mix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Message count")
    _style_axes(ax)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.015 if values else 0, f"{int(value):,}", ha="center", fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_fairness_cost_by_difficulty(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "fairness_by_difficulty_model.png"
    fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    width = 0.11
    base_positions = list(range(len(DIFFICULTY_ORDER)))
    offsets = [(idx - (len(MODEL_ORDER) - 1) / 2) * width for idx in range(len(MODEL_ORDER))]
    for model_idx, model in enumerate(MODEL_ORDER):
        values = []
        for difficulty in DIFFICULTY_ORDER:
            row = next((row for row in summary_rows if row["model"] == model and row["difficulty"] == difficulty), None)
            values.append(_float(row["fairness_cost_mean"]) if row else math.nan)
        ax.bar(
            [pos + offsets[model_idx] for pos in base_positions],
            values,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.9,
        )
    ax.set_title("Paper Fairness Cost by Difficulty", fontsize=14, fontweight="bold")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean |agent excess burden - game mean|")
    ax.set_xticks(base_positions, [difficulty.title() for difficulty in DIFFICULTY_ORDER])
    _style_axes(ax)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_fairness_cost_by_setting(setting_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "fairness_by_setting_model.png"
    fig, axes = plt.subplots(1, len(SETTING_ORDER), figsize=(13.2, 5.0), constrained_layout=True, sharey=False)
    width = 0.72
    for ax, setting in zip(axes, SETTING_ORDER, strict=True):
        values = []
        errors = []
        for model in MODEL_ORDER:
            row = next((row for row in setting_rows if row["model"] == model and row["setting"] == setting), None)
            values.append(_float(row["fairness_cost_mean"]) if row else math.nan)
            errors.append(_float(row["fairness_cost_ci95"]) if row else math.nan)
        positions = list(range(len(MODEL_ORDER)))
        ax.bar(
            positions,
            values,
            yerr=errors,
            capsize=2,
            width=width,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            alpha=0.9,
            linewidth=0.4,
            edgecolor="white",
        )
        finite_tops = [
            value + (error if math.isfinite(error) else 0.0)
            for value, error in zip(values, errors, strict=True)
            if math.isfinite(value)
        ]
        if finite_tops:
            ax.set_ylim(0, max(finite_tops) * 1.14)
        ax.set_title(setting.title(), fontsize=13, fontweight="bold")
        ax.set_xticks(positions, [_clean_label(model) for model in MODEL_ORDER], fontsize=8)
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_ylabel("Mean |agent excess burden - game mean|")
        _style_axes(ax)
    fig.suptitle("Paper Fairness Cost by Setting", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_failure_rate_by_meeting_index(meeting_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "failure_rate_by_meeting_index.png"
    groups: dict[int, list[float]] = defaultdict(list)
    for row in meeting_rows:
        groups[int(row["meeting_index"])].append(1.0 if row["failure"] else 0.0)
    indices = sorted(groups)
    values = [100 * _mean(groups[index]) for index in indices]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(indices, values, marker="o", color="#dc2626", linewidth=2.2)
    ax.set_title("Failure Rate by Meeting Index", fontsize=14, fontweight="bold")
    ax.set_xlabel("Meeting index within game")
    ax.set_ylabel("Failure rate (%)")
    ax.set_xticks(indices)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_excess_cost_by_meeting_index(meeting_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "excess_cost_by_meeting_index.png"
    groups: dict[int, list[float]] = defaultdict(list)
    for row in meeting_rows:
        value = _float(row["meeting_excess_cost"])
        if math.isfinite(value):
            groups[int(row["meeting_index"])].append(value)
    indices = sorted(groups)
    values = [_mean(groups[index]) for index in indices]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(indices, values, marker="o", color="#7c3aed", linewidth=2.2)
    ax.set_title("Scheduled-Meeting Excess Cost by Meeting Index", fontsize=14, fontweight="bold")
    ax.set_xlabel("Meeting index within game")
    ax.set_ylabel("Mean excess cost, successful meetings only")
    ax.set_xticks(indices)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def success_count_rows(meeting_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reindex successful meeting rows by count of successes so far in trace."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in meeting_rows:
        by_trace[str(row["trace_path"])].append(row)

    out: list[dict[str, Any]] = []
    for rows in by_trace.values():
        success_count = 0
        for row in sorted(rows, key=lambda item: int(item["meeting_index"])):
            if not row["coordinated"]:
                continue
            success_count += 1
            next_row = dict(row)
            next_row["success_count"] = success_count
            out.append(next_row)
    return out


def plot_excess_cost_by_success_count(meeting_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "excess_cost_by_success_count.png"
    groups: dict[int, list[float]] = defaultdict(list)
    for row in success_count_rows(meeting_rows):
        value = _float(row["meeting_excess_cost"])
        if math.isfinite(value):
            groups[int(row["success_count"])].append(value)
    indices = sorted(groups)
    values = [_mean(groups[index]) for index in indices]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(indices, values, marker="o", color="#0891b2", linewidth=2.2)
    ax.set_title("Scheduled-Meeting Excess Cost by Success Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Successful meetings so far")
    ax.set_ylabel("Mean excess cost, successful meetings only")
    ax.set_xticks(indices)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_messages_by_difficulty(summary_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "messaging_efficiency_by_difficulty.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True, constrained_layout=True)
    width = 0.11
    base_positions = list(range(len(DIFFICULTY_ORDER)))
    offsets = [(idx - (len(MODEL_ORDER) - 1) / 2) * width for idx in range(len(MODEL_ORDER))]
    panels = [
        (
            axes[0],
            "messages_per_successful_meeting",
            "Reported metric",
            "Messages sent per scheduled meeting",
        ),
        (
            axes[1],
            "messages_per_participant_round_attempted",
            "All attempted participant rounds",
            "Messages sent per participant round",
        ),
    ]
    for ax, metric, title, ylabel in panels:
        for model_idx, model in enumerate(MODEL_ORDER):
            values = []
            for difficulty in DIFFICULTY_ORDER:
                row = next((row for row in summary_rows if row["model"] == model and row["difficulty"] == difficulty), None)
                values.append(_float(row[metric]) if row else math.nan)
            ax.bar(
                [pos + offsets[model_idx] for pos in base_positions],
                values,
                width=width,
                label=model,
                color=MODEL_COLORS[model],
                alpha=0.9,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel(ylabel)
        ax.set_xticks(base_positions, [difficulty.title() for difficulty in DIFFICULTY_ORDER])
        _style_axes(ax)
    axes[0].legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Messaging Efficiency by Difficulty", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_failure_reasons(failure_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "failure_reason_by_model.png"
    reasons = [
        "missing schedule action",
        "blocked/unmovable slot",
        "occupied slot not freed",
        "participants chose different slots",
        "prior-meeting inconsistency",
        "double-booking conflict",
        "decision batch rejected",
        "unknown/other",
    ]
    reason_colors = {
        "missing schedule action": "#ef4444",
        "blocked/unmovable slot": "#f97316",
        "occupied slot not freed": "#f59e0b",
        "participants chose different slots": "#3b82f6",
        "prior-meeting inconsistency": "#8b5cf6",
        "double-booking conflict": "#14b8a6",
        "decision batch rejected": "#a855f7",
        "unknown/other": "#6b7280",
    }
    counts: Counter[tuple[str, str]] = Counter((row["model"], row["failure_reason"]) for row in failure_rows)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    bottoms = [0 for _ in MODEL_ORDER]
    for reason in reasons:
        values = [counts[(model, reason)] for model in MODEL_ORDER]
        ax.bar(
            range(len(MODEL_ORDER)),
            values,
            bottom=bottoms,
            label=reason,
            color=reason_colors[reason],
            alpha=0.9,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_title("Failed Meeting Reasons by Model Seat", fontsize=14, fontweight="bold")
    ax.set_ylabel("Failed participant-meeting count")
    ax.set_xticks(range(len(MODEL_ORDER)), [_clean_label(model) for model in MODEL_ORDER], fontsize=8)
    _style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision_failure_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    lines = ["# CalBench Mixed Model Performance", ""]
    lines.append("Canonical cohort from `TRACE_SELECTION.md`; Qwen and DeepSeek include only entrant agent 0.")
    lines.append("")
    lines.append("## Outputs")
    lines.extend(f"- `{plot.name}`" for plot in plot_paths)
    lines.append("")
    lines.append("## Task Success")
    lines.append("")
    lines.append("| model | easy | medium | hard |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model in MODEL_ORDER:
        values = []
        for difficulty in DIFFICULTY_ORDER:
            row = next((row for row in summary_rows if row["model"] == model and row["difficulty"] == difficulty), None)
            value = _float(row["coordination_ratio"]) if row else math.nan
            values.append("" if not math.isfinite(value) else f"{100 * value:.1f}%")
        lines.append(f"| {model} | {' | '.join(values)} |")
    lines.append("")
    lines.append("## Failure Reasons")
    lines.append("")
    reason_counts = Counter(row["failure_reason"] for row in failure_rows)
    lines.append("| reason | failed participant-meetings | description |")
    lines.append("| --- | ---: | --- |")
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} | {FAILURE_REASON_DESCRIPTIONS.get(reason, '')} |")
    lines.append("")
    lines.append("## Causal Decision Failures")
    lines.append("")
    selected_causal = [row for row in decision_failure_rows if row.get("selected_agent")]
    causal_counts = Counter(row["decision_failure_reason"] for row in decision_failure_rows)
    selected_causal_counts = Counter(row["decision_failure_reason"] for row in selected_causal)
    lines.append("| validator reason | all causal agents | selected model seats | description |")
    lines.append("| --- | ---: | ---: | --- |")
    for reason, count in causal_counts.most_common():
        lines.append(
            f"| {reason} | {count} | {selected_causal_counts.get(reason, 0)} | "
            f"{DECISION_FAILURE_DESCRIPTIONS.get(reason, '')} |"
        )
    lines.append("")
    lines.append("## Suggested Additional Basic Plots")
    lines.append("")
    lines.append("- Scheduled-only excess burden by difficulty and model.")
    lines.append("- Privacy leakage / VPS loss by model and difficulty.")
    lines.append("- Message channel mix by model (`dm`, participant groupchat, all-groupchat).")
    lines.append("- Failure rate by meeting index, to see whether later rounds compound earlier mistakes.")
    lines.append("- Fairness cost / relative excess burden by model.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_rows, failure_rows = collect_rows(trace_root)
    decision_failure_rows, affected_failure_rows = collect_decision_failure_rows(trace_root)
    game_rows, meeting_rows = collect_game_and_meeting_rows(trace_root)
    summary_rows = summarize(trace_rows)
    setting_summary_rows = summarize_by_model_setting(trace_rows)

    write_csv(out_dir / "trace_agent_metrics.csv", trace_rows)
    write_csv(out_dir / "failure_reasons.csv", failure_rows)
    write_csv(out_dir / "decision_failure_causes.csv", decision_failure_rows)
    write_csv(out_dir / "affected_failed_participants_with_cause.csv", affected_failure_rows)
    write_csv(out_dir / "trace_game_metrics.csv", game_rows)
    write_csv(out_dir / "meeting_index_metrics.csv", meeting_rows)
    write_csv(out_dir / "meeting_success_count_metrics.csv", success_count_rows(meeting_rows))
    write_csv(out_dir / "summary_by_model_difficulty.csv", summary_rows)
    write_csv(out_dir / "summary_by_model_setting.csv", setting_summary_rows)

    plot_paths = [
        plot_coordination(summary_rows, out_dir),
        plot_messages_by_excess(trace_rows, out_dir),
        plot_messages_by_difficulty(summary_rows, out_dir),
        plot_excess_cost_by_difficulty(summary_rows, out_dir),
        plot_message_channel_mix(game_rows, out_dir),
        plot_fairness_cost_by_difficulty(summary_rows, out_dir),
        plot_fairness_cost_by_setting(setting_summary_rows, out_dir),
        plot_failure_rate_by_meeting_index(meeting_rows, out_dir),
        plot_excess_cost_by_meeting_index(meeting_rows, out_dir),
        plot_excess_cost_by_success_count(meeting_rows, out_dir),
        plot_failure_reasons(failure_rows, out_dir),
    ]
    write_report(out_dir / "summary.md", summary_rows, failure_rows, decision_failure_rows, plot_paths)

    print(
        f"Wrote {len(trace_rows)} trace-agent rows, {len(failure_rows)} failure rows, "
        f"{len(decision_failure_rows)} causal decision-failure rows, and {len(meeting_rows)} meeting rows to {out_dir}"
    )
    for path in plot_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
