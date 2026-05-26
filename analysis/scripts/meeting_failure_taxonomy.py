"""Classify calendar meeting scheduling failures by resolution and decision batch.

This analysis follows ``TRACE_SELECTION.md`` and separates:

* meeting-level resolution outcomes,
* participant/agent-level decision outcomes, and
* causal failed action or batch validation records.

It also attaches the oracle displacement cost of each meeting's full-oracle
assigned slot so failed meetings can be compared against intrinsic scheduling
difficulty.

Usage:
    cd games/calendar
    uv run python analysis/scripts/meeting_failure_taxonomy.py
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
DEFAULT_OUT_DIR = CALENDAR_ROOT / "analysis" / "outputs" / "meeting_failure_taxonomy"

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

MODEL_COLORS = {
    "Claude Sonnet 4.6": "#7c3aed",
    "Gemini 3.1 Pro": "#111827",
    "Gemini 3 Flash": "#6b7280",
    "GPT-5.4 Mini": "#d97706",
    "Llama 4 Maverick": "#65a30d",
    "Qwen3.6 Plus": "#db2777",
    "DeepSeek V4 Pro": "#0891b2",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]

MEETING_FAILURE_DESCRIPTIONS = {
    "scheduled": "Resolution coordinated successfully.",
    "failed_due_to_decision_validation": (
        "At least one participant exhausted decision-batch validation retries before resolution."
    ),
    "failed_due_to_blocked_slot_violation": (
        "Resolution recorded a meeting on a blocked slot, with no exhausted decision validation event."
    ),
    "failed_due_to_prior_meeting_inconsistency": (
        "A previously scheduled meeting became inconsistent across its participants."
    ),
    "failed_due_to_double_booking_conflict": "Resolution recorded an overlapping meeting conflict.",
    "failed_due_to_slot_mismatch": "Participants placed the target meeting, but not all in the same slot.",
    "failed_due_to_missing_placement_no_decision_failure": (
        "At least one participant had no placement at resolution, but no decision_failed event was recorded."
    ),
    "failed_unknown": "Resolution failed, but the trace did not expose a more specific signal.",
}

VALIDATOR_REASON_DESCRIPTIONS = {
    "no_valid_schedule_action_for_current_meeting": (
        "The final rejected batch had zero valid schedule actions for this meeting after parsing, "
        "meeting-id filtering, and retries."
    ),
    "multiple_schedule_actions": "The final rejected batch had more than one schedule action.",
    "non_object_action": "An action in the batch was not a JSON object.",
    "schedule_missing_required_field": "A schedule action missed meeting_id or slot.",
    "schedule_slot_not_integer": "A schedule action used a non-integer slot.",
    "reschedule_missing_required_field": "A reschedule action missed item_id/from_slot/to_slot/justification.",
    "reschedule_slot_not_integer": "A reschedule action used a non-integer slot.",
    "missing_reschedule_justification": "A reschedule action had an empty or non-string justification.",
    "slot_out_of_range": "A schedule or reschedule referenced a slot outside the calendar bounds.",
    "wrong_item_or_from_slot": "A reschedule claimed an item was in a slot where it was not present.",
    "attempted_blocked_item_reschedule": "A reschedule attempted to move an immovable blocked item.",
    "duplicate_target_slot": "Two or more actions in the batch targeted the same slot.",
    "reschedule_target_occupied_not_freed": "A reschedule target was occupied and not freed by the batch.",
    "schedule_target_occupied_not_freed": "The scheduled meeting slot was occupied and not freed by the batch.",
    "other_validator_rejection": "Validator rejected the batch for another reason.",
}

REPORTING_FAILURE_CLASS_DESCRIPTIONS = {
    "model_tool_call_action_failure_did_not_use_schedule_tool": (
        "The participant exhausted retries with no valid schedule action for the current meeting; "
        "report as a model tool-call/action failure rather than a scheduling-cost failure."
    ),
}

SPECIFIC_FAILURE_REASON_DESCRIPTIONS = {
    "missing_schedule_action": (
        "A participant exhausted decision retries without a valid schedule action for the current meeting."
    ),
    "attempted_blocked_item_reschedule": "A participant tried to reschedule an immovable blocked item.",
    "new_meeting_schedule_slot_occupied": (
        "A participant scheduled the new meeting into an occupied slot it did not free."
    ),
    "reschedule_slot_occupied": "A participant tried to move an item into an occupied slot it did not free.",
    "new_meeting_schedule_mismatch": "All participants placed the new meeting, but not in the same slot.",
    "meeting_reschedule_mismatch": "A previously registered meeting became inconsistent across its participants.",
    "meeting_on_blocked_slot": "Resolution found a meeting marker on a blocked slot.",
    "double_booking_slot_conflict": "Resolution found more than one meeting marker in the same agent slot.",
    "missing_placement_without_decision_failure": (
        "At least one participant did not place the target meeting and no decision_failed event was recorded."
    ),
    "failed_unknown": "Resolution failed without a more specific signal in the trace.",
    "multiple_reasons": "Multiple concrete failure reasons were observed for the same meeting.",
}

PARTICIPANT_ROLE_DESCRIPTIONS = {
    "scheduled_success": "Participant placed the meeting in a coordinated successful resolution.",
    "causal_validation_failure": "Participant exhausted decision validation retries and caused meeting failure.",
    "valid_but_rolled_back": "Participant had a valid staged batch, but it was rolled back because another batch failed.",
    "submitted_mismatched_slot": "Participant placed the meeting, but participants did not agree on one slot.",
    "affected_by_failure": "Participant's meeting failed, but this participant was not identified as causal.",
    "causal_blocked_slot_violation": "Participant is listed in a blocked-slot violation without decision_failed.",
    "unknown_failed_participant_state": "Participant was in a failed meeting and no cleaner role was inferred.",
}

ORACLE_COST_BUCKETS = [
    ("0", 0.0, 0.0),
    ("1", 1.0, 1.0),
    ("2-5", 2.0, 5.0),
    ("6-20", 6.0, 20.0),
    ("21-100", 21.0, 100.0),
    (">100", 100.0, math.inf),
]

_TASK_CACHE: dict[str, dict[str, Any]] | None = None


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
    if not isinstance(trace.get("config"), dict):
        return None
    return trace


def _task_id(trace: dict[str, Any]) -> str:
    return str((trace.get("metrics") or {}).get("task_id") or (trace.get("config") or {}).get("task_id") or "")


def _difficulty(path: Path, trace: dict[str, Any]) -> str | None:
    metrics = trace.get("metrics") or {}
    value = metrics.get("difficulty")
    if isinstance(value, str) and value.lower() in DIFFICULTY_ORDER:
        return value.lower()
    haystack = " ".join(
        str(value)
        for value in (
            path,
            trace.get("game_id"),
            (trace.get("config") or {}).get("experiment_name"),
            (trace.get("config") or {}).get("task_id"),
        )
        if value is not None
    )
    match = re.search(r"\b(?:b\d+_)?(easy|medium|hard)_", haystack, flags=re.I)
    return match.group(1).lower() if match else None


def _setting(path: Path, trace: dict[str, Any]) -> str:
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
    return "unknown"


def _model_label(raw: Any) -> str:
    return MODEL_LABELS.get(str(raw), str(raw or "unknown"))


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
        return tasks.get(f"{Path(str(task_path)).name}:{task_id}") or tasks.get(f"{task_path}:{task_id}")
    return None


def _oracle_meeting_costs(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    task = _task_body(trace)
    if not task:
        return {}
    calendars = task.get("calendars") or []
    assignments = (task.get("optimal") or {}).get("assignments") or {}
    out: dict[int, dict[str, Any]] = {}
    for meeting in task.get("meetings") or []:
        meeting_id = int(meeting["id"])
        raw_slot = assignments.get(str(meeting_id), assignments.get(meeting_id))
        if raw_slot is None:
            continue
        slot = int(raw_slot)
        cost = _slot_cost(calendars, meeting.get("participants") or [], slot)
        out[meeting_id] = {
            "oracle_slot": slot,
            "oracle_meeting_displacement_cost": float(cost or 0.0),
        }
    return out


def _oracle_cost_bucket(cost: float) -> str:
    if not math.isfinite(cost):
        return "unknown"
    for label, low, high in ORACLE_COST_BUCKETS:
        if low == high and cost == low:
            return label
        if low <= cost <= high:
            return label
    return "unknown"


def _iter_selected_traces(trace_root: Path) -> Iterable[tuple[str, str | None, int | None, Path, dict[str, Any]]]:
    seen_entry_game_ids: set[tuple[str, str, str]] = set()
    specs: list[tuple[str, str | None, int | None]] = [(MODEL_RUN, None, None)]
    for run_name, (label, entrant_agent_id) in ENTRY_RUNS.items():
        if (trace_root / run_name).is_dir():
            specs.append((run_name, label, entrant_agent_id))
    for run_name, entrant_label, entrant_agent_id in specs:
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
            setting = _setting(path, trace)
            task_id = _task_id(trace)
            if entrant_label:
                key = (entrant_label, task_id, setting)
                if key in seen_entry_game_ids:
                    continue
                seen_entry_game_ids.add(key)
            yield run_name, entrant_label, entrant_agent_id, path, trace


def _selected_agent_ids(trace: dict[str, Any], entrant_agent_id: int | None) -> set[int]:
    agents = (trace.get("config") or {}).get("agents") or []
    if entrant_agent_id is not None:
        return {entrant_agent_id} if entrant_agent_id < len(agents) else set()
    return {
        agent_id
        for agent_id, agent in enumerate(agents)
        if _model_label(agent.get("model") or agent.get("type")) in MODEL_ORDER
    }


def _round_meetings(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in trace.get("events") or []:
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        meeting = data.get("meeting")
        if isinstance(meeting, dict) and meeting.get("id") is not None:
            out[int(data.get("round", len(out)))] = meeting
    return out


def _resolution_events(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in trace.get("events") or []:
        if event.get("type") != "resolution":
            continue
        data = event.get("data") or {}
        if data.get("round") is not None:
            out[int(data["round"])] = data
    return out


def _events_by_round_agent(trace: dict[str, Any]) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    indexed: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for event in trace.get("events") or []:
        data = event.get("data") or {}
        if data.get("round") is None or data.get("agent_id") is None:
            continue
        key = (int(data["round"]), int(data["agent_id"]))
        indexed[str(event.get("type"))][key].append(data)
    return indexed


def _validator_reason(conflict: str) -> str:
    text = conflict.lower()
    if "action" in text and "is not an object" in text:
        return "non_object_action"
    if "schedule action missing required field" in text:
        return "schedule_missing_required_field"
    if "schedule slot must be an integer" in text:
        return "schedule_slot_not_integer"
    if "reschedule action missing required field" in text:
        return "reschedule_missing_required_field"
    if "reschedule slots must be integers" in text:
        return "reschedule_slot_not_integer"
    if "reschedule justification must be" in text:
        return "missing_reschedule_justification"
    if "out of range" in text:
        return "slot_out_of_range"
    if "expected exactly 1 schedule action, got 0" in text:
        return "no_valid_schedule_action_for_current_meeting"
    if "expected exactly 1 schedule action" in text:
        return "multiple_schedule_actions"
    if "reschedule claims item_id" in text:
        return "wrong_item_or_from_slot"
    if "reschedule attempts to move blocked" in text:
        return "attempted_blocked_item_reschedule"
    if "two or more actions target slot" in text:
        return "duplicate_target_slot"
    if "reschedule targets slot" in text and "occupied" in text:
        return "reschedule_target_occupied_not_freed"
    if "schedule targets slot" in text and "occupied" in text:
        return "schedule_target_occupied_not_freed"
    return "other_validator_rejection"


def _meeting_failure_type(
    resolution: dict[str, Any],
    causal_failures: list[dict[str, Any]],
) -> str:
    if resolution.get("coordinated"):
        return "scheduled"
    if causal_failures:
        return "failed_due_to_decision_validation"
    if resolution.get("blocked_slot_violations"):
        return "failed_due_to_blocked_slot_violation"
    if resolution.get("consistency_violated_meeting_ids"):
        return "failed_due_to_prior_meeting_inconsistency"
    if resolution.get("slot_conflicts"):
        return "failed_due_to_double_booking_conflict"
    slots = list((resolution.get("per_agent_slot") or {}).values())
    non_null = [slot for slot in slots if slot is not None]
    if len(non_null) == len(slots) and len(set(non_null)) > 1:
        return "failed_due_to_slot_mismatch"
    if any(slot is None for slot in slots):
        return "failed_due_to_missing_placement_no_decision_failure"
    return "failed_unknown"


def _slot_mismatch_roles(per_agent_slot: dict[str, Any]) -> set[int]:
    non_null = [int(slot) for slot in per_agent_slot.values() if slot is not None]
    if len(set(non_null)) <= 1:
        return set()
    return {int(agent_id) for agent_id, slot in per_agent_slot.items() if slot is not None}


def _infer_failed_action(reason: str, actions: list[dict[str, Any]], conflict: str) -> tuple[str, str]:
    if reason == "no_valid_schedule_action_for_current_meeting":
        return "batch", ""
    slot_match = re.search(r"slot (\d+)", conflict)
    item_match = re.search(r"item_id=(\d+)", conflict)
    slot = int(slot_match.group(1)) if slot_match else None
    item_id = int(item_match.group(1)) if item_match else None
    candidates: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if reason.startswith("schedule") and action.get("type") == "schedule":
            if slot is None or action.get("slot") == slot:
                candidates.append(action)
        elif reason.startswith("reschedule") and action.get("type") == "reschedule":
            if slot is None or action.get("to_slot") == slot:
                candidates.append(action)
        elif reason == "attempted_blocked_item_reschedule" and action.get("type") == "reschedule":
            if item_id is None or action.get("item_id") == item_id:
                candidates.append(action)
        elif reason == "wrong_item_or_from_slot" and action.get("type") == "reschedule":
            if item_id is None or action.get("item_id") == item_id:
                candidates.append(action)
        elif reason == "duplicate_target_slot":
            target = action.get("slot") if action.get("type") == "schedule" else action.get("to_slot")
            if slot is None or target == slot:
                candidates.append(action)
    if candidates:
        return str(candidates[0].get("type") or "action"), json.dumps(candidates[0], sort_keys=True)
    return "batch", json.dumps(actions, sort_keys=True)


def _reporting_failure_class(final_reason: str) -> str:
    if final_reason == "no_valid_schedule_action_for_current_meeting":
        return "model_tool_call_action_failure_did_not_use_schedule_tool"
    return ""


def _specific_failure_reason(resolution: dict[str, Any], causal_failures: list[dict[str, Any]]) -> str:
    if resolution.get("coordinated"):
        return "scheduled"
    if causal_failures:
        reasons = {
            str(row.get("reporting_failure_class") or row.get("validator_reason") or "other_validator_rejection")
            for row in causal_failures
        }
        reasons = {
            _clean_specific_failure_reason(reason)
            for reason in reasons
        }
        if len(reasons) == 1:
            return next(iter(reasons))
        return "multiple_reasons"
    blocked = resolution.get("blocked_slot_violations") or []
    if blocked:
        kinds = {str(row.get("kind") or "") for row in blocked}
        if "meeting_on_blocked_slot" in kinds:
            return "meeting_on_blocked_slot"
        if "blocked_reschedule" in kinds:
            return "attempted_blocked_item_reschedule"
        return "blocked_slot_violation"
    if resolution.get("consistency_violated_meeting_ids"):
        return "meeting_reschedule_mismatch"
    if resolution.get("slot_conflicts"):
        return "double_booking_slot_conflict"
    slots = list((resolution.get("per_agent_slot") or {}).values())
    non_null = [slot for slot in slots if slot is not None]
    if len(non_null) == len(slots) and len(set(non_null)) > 1:
        return "new_meeting_schedule_mismatch"
    if any(slot is None for slot in slots):
        return "missing_placement_without_decision_failure"
    return "failed_unknown"


def _clean_specific_failure_reason(raw_reason: str) -> str:
    aliases = {
        "model_tool_call_action_failure_did_not_use_schedule_tool": "missing_schedule_action",
        "schedule_target_occupied_not_freed": "new_meeting_schedule_slot_occupied",
        "reschedule_target_occupied_not_freed": "reschedule_slot_occupied",
        "participants_scheduled_different_slots": "new_meeting_schedule_mismatch",
        "prior_meeting_inconsistency": "meeting_reschedule_mismatch",
    }
    return aliases.get(raw_reason, raw_reason)


def collect_taxonomy(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meeting_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    action_failure_rows: list[dict[str, Any]] = []

    for run_name, entrant_label, entrant_agent_id, path, trace in _iter_selected_traces(trace_root):
        game_id = str(trace.get("game_id") or path.stem)
        task_id = _task_id(trace)
        setting = _setting(path, trace)
        difficulty = _difficulty(path, trace) or "unknown"
        agents = (trace.get("config") or {}).get("agents") or []
        model_by_agent = {
            agent_id: (
                entrant_label
                if entrant_label is not None and entrant_agent_id == agent_id
                else _model_label(agent.get("model") or agent.get("type"))
            )
            for agent_id, agent in enumerate(agents)
        }
        selected_agent_ids = _selected_agent_ids(trace, entrant_agent_id)
        rounds = _round_meetings(trace)
        resolutions = _resolution_events(trace)
        indexed = _events_by_round_agent(trace)
        oracle_costs = _oracle_meeting_costs(trace)

        for round_idx, meeting in sorted(rounds.items()):
            resolution = resolutions.get(round_idx)
            if not resolution:
                continue
            meeting_id = int(meeting["id"])
            participants = [int(agent_id) for agent_id in meeting.get("participants") or []]
            per_agent_slot = resolution.get("per_agent_slot") or {}

            causal_failures: list[dict[str, Any]] = []
            causal_by_agent: dict[int, dict[str, Any]] = {}
            for agent_id in participants:
                decision_failed = indexed["decision_failed"].get((round_idx, agent_id), [])
                if not decision_failed:
                    continue
                rejected = indexed["batch_rejected"].get((round_idx, agent_id), [])
                final_rejection = rejected[-1] if rejected else {}
                conflict = str(final_rejection.get("conflict_description") or "")
                reason = _validator_reason(conflict)
                first_conflict = str((rejected[0] if rejected else {}).get("conflict_description") or "")
                first_reason = _validator_reason(first_conflict)
                rejection_reason_sequence = [
                    _validator_reason(str(row.get("conflict_description") or "")) for row in rejected
                ]
                actions = final_rejection.get("actions") or []
                failed_action_type, failed_action_json = _infer_failed_action(reason, actions, conflict)
                record = {
                    "agent_id": agent_id,
                    "model": model_by_agent.get(agent_id, "unknown"),
                    "reporting_failure_class": _reporting_failure_class(reason),
                    "reporting_failure_class_description": REPORTING_FAILURE_CLASS_DESCRIPTIONS.get(
                        _reporting_failure_class(reason),
                        "",
                    ),
                    "first_validator_reason": first_reason,
                    "first_validator_conflict": first_conflict,
                    "validator_reason": reason,
                    "validator_reason_description": VALIDATOR_REASON_DESCRIPTIONS.get(reason, ""),
                    "validator_conflict": conflict,
                    "rejection_reason_sequence": json.dumps(rejection_reason_sequence),
                    "final_rejected_actions": json.dumps(actions, sort_keys=True),
                    "failed_action_type": failed_action_type,
                    "failed_action_json": failed_action_json,
                    "rejected_attempt_count": len(rejected),
                }
                causal_failures.append(record)
                causal_by_agent[agent_id] = record
                action_failure_rows.append(
                    {
                        "run": run_name,
                        "trace_path": str(path),
                        "game_id": game_id,
                        "task_id": task_id,
                        "setting": setting,
                        "difficulty": difficulty,
                        "meeting_index": round_idx + 1,
                        "meeting_id": meeting_id,
                        "agent_id": agent_id,
                        "model": record["model"],
                        "selected_agent": agent_id in selected_agent_ids,
                        **{key: value for key, value in record.items() if key not in {"agent_id", "model"}},
                    }
            )

            failure_type = _meeting_failure_type(resolution, causal_failures)
            specific_failure_reason = _specific_failure_reason(resolution, causal_failures)
            oracle = oracle_costs.get(meeting_id, {})
            oracle_cost = _float(oracle.get("oracle_meeting_displacement_cost"))
            mismatch_agents = _slot_mismatch_roles(per_agent_slot)
            rolled_back_agents = {
                agent_id
                for agent_id in participants
                if indexed["batch_rolled_back"].get((round_idx, agent_id))
            }
            applied_agents = {
                agent_id
                for agent_id in participants
                if indexed["batch_applied"].get((round_idx, agent_id))
            }
            blocked_violation_agents = {
                int(row.get("agent_id"))
                for row in resolution.get("blocked_slot_violations") or []
                if row.get("agent_id") is not None
            }

            meeting_rows.append(
                {
                    "run": run_name,
                    "cohort_label": entrant_label or "Mixed5 cohort",
                    "trace_path": str(path),
                    "game_id": game_id,
                    "task_id": task_id,
                    "setting": setting,
                    "difficulty": difficulty,
                    "meeting_index": round_idx + 1,
                    "meeting_id": meeting_id,
                    "participants": json.dumps(participants),
                    "resolution_coordinated": bool(resolution.get("coordinated")),
                    "meeting_failure_type": failure_type,
                    "meeting_failure_description": MEETING_FAILURE_DESCRIPTIONS.get(failure_type, ""),
                    "specific_failure_reason": specific_failure_reason,
                    "specific_failure_reason_description": SPECIFIC_FAILURE_REASON_DESCRIPTIONS.get(
                        specific_failure_reason,
                        "",
                    ),
                    "causal_agent_ids": json.dumps([row["agent_id"] for row in causal_failures]),
                    "causal_models": json.dumps([row["model"] for row in causal_failures]),
                    "causal_validator_reasons": json.dumps([row["validator_reason"] for row in causal_failures]),
                    "per_agent_slot": json.dumps(per_agent_slot, sort_keys=True),
                    "slot_conflicts": json.dumps(resolution.get("slot_conflicts") or {}, sort_keys=True),
                    "consistency_violated_meeting_ids": json.dumps(
                        resolution.get("consistency_violated_meeting_ids") or []
                    ),
                    "blocked_slot_violations": json.dumps(resolution.get("blocked_slot_violations") or []),
                    "oracle_slot": oracle.get("oracle_slot", ""),
                    "oracle_meeting_displacement_cost": oracle_cost,
                    "oracle_cost_bucket": _oracle_cost_bucket(oracle_cost),
                }
            )

            for agent_id in participants:
                causal = causal_by_agent.get(agent_id)
                slot = per_agent_slot.get(str(agent_id))
                if resolution.get("coordinated"):
                    role = "scheduled_success"
                elif causal is not None:
                    role = "causal_validation_failure"
                elif agent_id in blocked_violation_agents:
                    role = "causal_blocked_slot_violation"
                elif agent_id in rolled_back_agents:
                    role = "valid_but_rolled_back"
                elif agent_id in mismatch_agents:
                    role = "submitted_mismatched_slot"
                elif slot is None:
                    role = "affected_by_failure"
                else:
                    role = "unknown_failed_participant_state"
                rejected = indexed["batch_rejected"].get((round_idx, agent_id), [])
                participant_rows.append(
                    {
                        "run": run_name,
                        "trace_path": str(path),
                        "game_id": game_id,
                        "task_id": task_id,
                        "setting": setting,
                        "difficulty": difficulty,
                        "meeting_index": round_idx + 1,
                        "meeting_id": meeting_id,
                        "agent_id": agent_id,
                        "model": model_by_agent.get(agent_id, "unknown"),
                        "selected_agent": agent_id in selected_agent_ids,
                        "participant_role": role,
                        "participant_role_description": PARTICIPANT_ROLE_DESCRIPTIONS.get(role, ""),
                        "resolution_slot": slot,
                        "meeting_failure_type": failure_type,
                        "specific_failure_reason": specific_failure_reason,
                        "validator_reason": causal.get("validator_reason", "") if causal else "",
                        "validator_reason_description": causal.get("validator_reason_description", "") if causal else "",
                        "validator_conflict": causal.get("validator_conflict", "") if causal else "",
                        "rejected_attempt_count": len(rejected),
                        "applied_actions": json.dumps(
                            [
                                row.get("actions")
                                for row in indexed["batch_applied"].get((round_idx, agent_id), [])
                            ],
                            sort_keys=True,
                        ),
                        "rolled_back_actions": json.dumps(
                            [
                                row.get("actions")
                                for row in indexed["batch_rolled_back"].get((round_idx, agent_id), [])
                            ],
                            sort_keys=True,
                        ),
                        "oracle_meeting_displacement_cost": oracle_cost,
                        "oracle_cost_bucket": _oracle_cost_bucket(oracle_cost),
                    }
                )

    return meeting_rows, participant_rows, action_failure_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_tool_call_action_failure_appendix_rows(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in actions
        if row.get("reporting_failure_class") == "model_tool_call_action_failure_did_not_use_schedule_tool"
    ]
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["setting"]),
            str(row["difficulty"]),
            str(row["first_validator_reason"]),
            str(row["validator_reason"]),
            str(row["rejection_reason_sequence"]),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, grouped in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        model, setting, difficulty, first_reason, final_reason, sequence = key
        meeting_indices = sorted({int(row["meeting_index"]) for row in grouped})
        task_ids = sorted({str(row["task_id"]) for row in grouped})
        out.append(
            {
                "reporting_failure_class": "model_tool_call_action_failure_did_not_use_schedule_tool",
                "model": model,
                "setting": setting,
                "difficulty": difficulty,
                "first_validator_reason": first_reason,
                "final_validator_reason": final_reason,
                "rejection_reason_sequence": sequence,
                "count": len(grouped),
                "selected_agent_count": sum(1 for row in grouped if row["selected_agent"]),
                "meeting_indices": json.dumps(meeting_indices),
                "example_task_ids": json.dumps(task_ids[:8]),
            }
        )
    return out


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d1d5db", alpha=0.45)
    ax.set_axisbelow(True)


def _plot_label(label: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(label.replace("_", " "), width=width))


def specific_failure_reason_key(meetings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in meetings if row["meeting_failure_type"] != "scheduled"]
    counts = Counter(row["specific_failure_reason"] for row in rows)
    out: list[dict[str, Any]] = []
    for idx, (reason, count) in enumerate(counts.most_common()):
        label = chr(ord("A") + idx)
        out.append(
            {
                "label": label,
                "specific_failure_reason": reason,
                "count": count,
                "description": SPECIFIC_FAILURE_REASON_DESCRIPTIONS.get(reason, ""),
            }
        )
    return out


def _reason_label_map(reason_key: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row["specific_failure_reason"]): str(row["label"]) for row in reason_key}


def plot_specific_failure_counts(meetings: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "specific_failure_reason_counts.png"
    reason_key = specific_failure_reason_key(meetings)
    labels = [str(row["label"]) for row in reason_key]
    values = [int(row["count"]) for row in reason_key]
    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    ax.bar(range(len(labels)), values, width=0.52, color="#4f46e5", alpha=0.9)
    ax.set_title("Meeting Failure Reasons", fontsize=8.5, fontweight="bold", pad=3)
    ax.set_ylabel("Failed meetings", fontsize=7.5)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(axis="x", labelrotation=35, pad=0)
    ax.tick_params(axis="y", labelsize=7)
    _style_axes(ax)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.018 if values else 0, str(value), ha="center", fontsize=6.5)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_causal_validator_by_model(actions: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "causal_validator_reason_by_model.png"
    selected = [row for row in actions if row["selected_agent"]]
    reasons = [reason for reason, _count in Counter(row["validator_reason"] for row in selected).most_common()]
    if not reasons:
        reasons = ["none"]
    palette = ["#ef4444", "#f97316", "#f59e0b", "#3b82f6", "#8b5cf6", "#14b8a6", "#6b7280"]
    colors = {reason: palette[idx % len(palette)] for idx, reason in enumerate(reasons)}
    counts = Counter((row["model"], row["validator_reason"]) for row in selected)
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    bottoms = [0 for _ in MODEL_ORDER]
    for reason in reasons:
        values = [counts[(model, reason)] for model in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), values, bottom=bottoms, color=colors[reason], label=reason, alpha=0.9)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_title("Causal Decision Validator Failures by Selected Model Seat", fontsize=14, fontweight="bold")
    ax.set_ylabel("Causal failed batches")
    ax.set_xticks(range(len(MODEL_ORDER)), [model.replace(" ", "\n") for model in MODEL_ORDER], fontsize=8)
    _style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_participant_role_by_model(participants: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "participant_role_by_selected_model.png"
    selected = [row for row in participants if row["selected_agent"]]
    roles = [
        role
        for role, _count in Counter(row["participant_role"] for row in selected if row["participant_role"] != "scheduled_success").most_common()
    ]
    if not roles:
        roles = ["scheduled_success"]
    colors = ["#4f46e5", "#ef4444", "#f59e0b", "#14b8a6", "#8b5cf6", "#6b7280"]
    counts = Counter((row["model"], row["participant_role"]) for row in selected)
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    bottoms = [0 for _ in MODEL_ORDER]
    for idx, role in enumerate(roles):
        values = [counts[(model, role)] for model in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), values, bottom=bottoms, label=role, color=colors[idx % len(colors)], alpha=0.88)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_title("Failed-Meeting Participant Roles by Selected Model Seat", fontsize=14, fontweight="bold")
    ax.set_ylabel("Participant-meeting rows")
    ax.set_xticks(range(len(MODEL_ORDER)), [model.replace(" ", "\n") for model in MODEL_ORDER], fontsize=8)
    _style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_specific_failure_reason_by_difficulty(meetings: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "specific_failure_reason_by_difficulty.png"
    rows = [row for row in meetings if row["meeting_failure_type"] != "scheduled"]
    label_map = _reason_label_map(specific_failure_reason_key(meetings))
    failure_types = [label for label, _count in Counter(row["specific_failure_reason"] for row in rows).most_common()]
    colors = ["#4f46e5", "#ef4444", "#f59e0b", "#14b8a6", "#8b5cf6", "#6b7280"]
    counts = Counter((row["difficulty"], row["specific_failure_reason"]) for row in rows)
    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    bottoms = [0 for _ in DIFFICULTY_ORDER]
    for idx, failure_type in enumerate(failure_types):
        values = [counts[(difficulty, failure_type)] for difficulty in DIFFICULTY_ORDER]
        ax.bar(
            DIFFICULTY_ORDER,
            values,
            width=0.5,
            bottom=bottoms,
            label=label_map.get(failure_type, failure_type),
            color=colors[idx % len(colors)],
            alpha=0.88,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_title("Failure Reasons by Difficulty", fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Difficulty", fontsize=7.5)
    ax.set_ylabel("Failed meetings", fontsize=7.5)
    ax.tick_params(axis="x", labelrotation=35, labelsize=7, pad=0)
    ax.tick_params(axis="y", labelsize=7)
    _style_axes(ax)
    ax.legend(ncol=4, fontsize=6, frameon=False, handlelength=0.8, columnspacing=0.6)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_failure_rate_by_oracle_cost(meetings: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "failure_rate_by_oracle_displacement_cost.png"
    groups: dict[str, list[float]] = defaultdict(list)
    bucket_order = [label for label, _low, _high in ORACLE_COST_BUCKETS]
    for row in meetings:
        bucket = row["oracle_cost_bucket"]
        if bucket == "unknown":
            continue
        groups[bucket].append(0.0 if row["resolution_coordinated"] else 1.0)
    labels = [label for label in bucket_order if groups.get(label)]
    values = [100 * _mean(groups[label]) for label in labels]
    counts = [len(groups[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.bar(labels, values, color="#dc2626", alpha=0.85)
    ax.set_title("Resolution Failure Rate by Oracle Meeting Displacement Cost", fontsize=13, fontweight="bold")
    ax.set_xlabel("Oracle displacement cost for this meeting")
    ax.set_ylabel("Failure rate (%)")
    _style_axes(ax)
    for idx, (value, count) in enumerate(zip(values, counts, strict=True)):
        ax.text(idx, value + max(values) * 0.02 if values else 0, f"{value:.1f}%\nn={count}", ha="center", fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_oracle_cost_by_outcome(meetings: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "oracle_displacement_by_meeting_outcome.png"
    categories = ["scheduled"] + [
        label
        for label, _count in Counter(
            row["meeting_failure_type"] for row in meetings if row["meeting_failure_type"] != "scheduled"
        ).most_common()
    ]
    data = [
        [
            _float(row["oracle_meeting_displacement_cost"])
            for row in meetings
            if row["meeting_failure_type"] == category and math.isfinite(_float(row["oracle_meeting_displacement_cost"]))
        ]
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    ax.boxplot(data, tick_labels=[category.replace("_", "\n") for category in categories], showfliers=False)
    ax.set_title("Oracle Meeting Displacement Cost by Resolution Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Oracle displacement cost for this meeting")
    ax.tick_params(axis="x", labelsize=8)
    _style_axes(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_specific_failure_reason_by_oracle_bucket(meetings: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "specific_failure_reason_by_oracle_cost_bucket.png"
    rows = [row for row in meetings if row["meeting_failure_type"] != "scheduled" and row["oracle_cost_bucket"] != "unknown"]
    label_map = _reason_label_map(specific_failure_reason_key(meetings))
    failure_types = [label for label, _count in Counter(row["specific_failure_reason"] for row in rows).most_common()]
    bucket_order = [label for label, _low, _high in ORACLE_COST_BUCKETS if any(row["oracle_cost_bucket"] == label for row in rows)]
    colors = ["#4f46e5", "#ef4444", "#f59e0b", "#14b8a6", "#8b5cf6", "#6b7280"]
    counts = Counter((row["oracle_cost_bucket"], row["specific_failure_reason"]) for row in rows)
    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    bottoms = [0 for _ in bucket_order]
    for idx, failure_type in enumerate(failure_types):
        values = [counts[(bucket, failure_type)] for bucket in bucket_order]
        ax.bar(
            bucket_order,
            values,
            width=0.5,
            bottom=bottoms,
            label=label_map.get(failure_type, failure_type),
            color=colors[idx % len(colors)],
            alpha=0.88,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_title("Failure Reasons by Oracle Cost", fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Oracle displacement cost", fontsize=7.5)
    ax.set_ylabel("Failed meetings", fontsize=7.5)
    ax.tick_params(axis="x", labelrotation=35, labelsize=7, pad=0)
    ax.tick_params(axis="y", labelsize=7)
    _style_axes(ax)
    ax.legend(ncol=4, fontsize=6, frameon=False, handlelength=0.8, columnspacing=0.6)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    meetings: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    appendix_rows: list[dict[str, Any]],
    reason_key: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Meeting Failure Taxonomy")
    lines.append("")
    lines.append("Resolution failure is the meeting-level signal. Decision validation and rollback events are used for agent-level attribution.")
    lines.append("")
    lines.append("## Outputs")
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    lines.append("")
    lines.append("## Meeting Outcomes")
    lines.append("")
    lines.append("| meeting_failure_type | count | description |")
    lines.append("| --- | ---: | --- |")
    for failure_type, count in Counter(row["meeting_failure_type"] for row in meetings).most_common():
        lines.append(f"| {failure_type} | {count} | {MEETING_FAILURE_DESCRIPTIONS.get(failure_type, '')} |")
    lines.append("")
    lines.append("## Specific Failure Reasons")
    lines.append("")
    lines.append("| label | specific_failure_reason | count | description |")
    lines.append("| --- | --- | ---: | --- |")
    for row in reason_key:
        lines.append(
            f"| {row['label']} | {row['specific_failure_reason']} | {row['count']} | {row['description']} |"
        )
    lines.append("")
    lines.append("## Participant Roles")
    lines.append("")
    lines.append("| participant_role | count | description |")
    lines.append("| --- | ---: | --- |")
    for role, count in Counter(row["participant_role"] for row in participants).most_common():
        lines.append(f"| {role} | {count} | {PARTICIPANT_ROLE_DESCRIPTIONS.get(role, '')} |")
    lines.append("")
    lines.append("## Causal Validator Reasons")
    lines.append("")
    lines.append("| validator_reason | count | selected_agent_count | description |")
    lines.append("| --- | ---: | ---: | --- |")
    reason_counts = Counter(row["validator_reason"] for row in actions)
    selected_counts = Counter(row["validator_reason"] for row in actions if row["selected_agent"])
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} | {selected_counts.get(reason, 0)} | {VALIDATOR_REASON_DESCRIPTIONS.get(reason, '')} |")
    lines.append("")
    lines.append("## Appendix: Model Tool-Call Action Failures")
    lines.append("")
    lines.append(
        "Rows in this appendix correspond to final validator failures with no valid schedule action for the current meeting. "
        "They should be reported as model tool-call/action failures: the participant did not use the schedule tool successfully."
    )
    lines.append("")
    lines.append("| model | setting | difficulty | first_validator_reason | final_validator_reason | count | selected_agent_count |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
    for row in appendix_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["setting"]),
                    str(row["difficulty"]),
                    str(row["first_validator_reason"]),
                    str(row["final_validator_reason"]),
                    str(row["count"]),
                    str(row["selected_agent_count"]),
                ]
            )
            + " |"
        )
    failed = [row for row in meetings if not row["resolution_coordinated"]]
    scheduled = [row for row in meetings if row["resolution_coordinated"]]
    lines.append("")
    lines.append("## Oracle Displacement Check")
    lines.append("")
    lines.append(f"- Scheduled meetings median oracle displacement cost: {_median(_float(row['oracle_meeting_displacement_cost']) for row in scheduled):.2f}")
    lines.append(f"- Failed meetings median oracle displacement cost: {_median(_float(row['oracle_meeting_displacement_cost']) for row in failed):.2f}")
    out_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meetings, participants, actions = collect_taxonomy(trace_root)
    appendix_rows = model_tool_call_action_failure_appendix_rows(actions)
    reason_key = specific_failure_reason_key(meetings)
    write_csv(out_dir / "meeting_failure_taxonomy.csv", meetings)
    write_csv(out_dir / "participant_decision_outcomes.csv", participants)
    write_csv(out_dir / "causal_action_failures.csv", actions)
    write_csv(out_dir / "appendix_model_tool_call_action_failures.csv", appendix_rows)
    write_csv(out_dir / "specific_failure_reason_key.csv", reason_key)

    plots = [
        plot_specific_failure_counts(meetings, out_dir),
        plot_causal_validator_by_model(actions, out_dir),
        plot_participant_role_by_model(participants, out_dir),
        plot_specific_failure_reason_by_difficulty(meetings, out_dir),
        plot_failure_rate_by_oracle_cost(meetings, out_dir),
        plot_oracle_cost_by_outcome(meetings, out_dir),
        plot_specific_failure_reason_by_oracle_bucket(meetings, out_dir),
    ]
    write_report(out_dir, meetings, participants, actions, appendix_rows, reason_key, plots)

    print(
        f"Wrote {len(meetings)} meeting rows, {len(participants)} participant rows, "
        f"{len(actions)} causal action rows, and {len(appendix_rows)} appendix rows to {out_dir}"
    )
    for plot in plots:
        print(plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
