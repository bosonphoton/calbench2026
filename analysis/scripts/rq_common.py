"""Shared trace-loading helpers for calendar research-question analyses."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


MODEL_LABELS: dict[str, str] = {
    "claude-sonnet46": "Claude Sonnet 4.6",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini31pro": "Gemini 3.1 Pro",
    "gemini3flash": "Gemini 3 Flash",
    "gpt54-mini": "GPT-5.4 Mini",
    "llama4-maverick": "Llama 4 Maverick",
    "qwen36-plus": "Qwen3.6 Plus",
}

MODEL_RUNS: dict[str, str] = {
    "Gemini 3.1 Pro": "gemini31pro-benchmark-lite-001",
    "Claude Sonnet 4.6": "claude-sonnet46-benchmark-lite-001",
    "Gemini 3 Flash": "gemini3flash-benchmark-lite-001",
    "DeepSeek V4 Pro": "deepseek-v4-pro-benchmark-lite-002",
    "GPT-5.4 Mini": "gpt54-mini-benchmark-lite-001",
    "Llama 4 Maverick": "llama4-maverick-benchmark-lite-001",
    "Qwen3.6 Plus": "qwen36-plus-benchmark-lite-002",
}

BASELINE_RUNS: dict[str, str] = {
    "Baseline: IMAP": "baseline-imap-benchmark-lite-001",
    "Baseline: DSM-Private": "baseline-dsm-private-benchmark-lite-001",
    "Baseline: DSM-Welfare": "baseline-dsm-welfare-benchmark-lite-001",
    "Baseline: SD-MAP": "baseline-sd-benchmark-lite-001",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_traces_root() -> Path:
    return repo_root().parent / "shared-traces"


def resolve_path(raw: str | Path, *, base: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


def output_dir(name: str) -> Path:
    path = calendar_root() / "analysis" / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_model_runs(traces_root: Path) -> dict[str, list[Path]]:
    """Return the seven unblocked benchmark-lite model run directories."""
    runs: dict[str, list[Path]] = {}
    for label, dirname in MODEL_RUNS.items():
        path = traces_root / dirname
        if path.is_dir():
            runs[label] = [path]
    return runs


def discover_baseline_runs(traces_root: Path) -> dict[str, list[Path]]:
    """Return unblocked non-LLM benchmark-lite baseline run directories."""
    runs: dict[str, list[Path]] = {}
    for label, dirname in BASELINE_RUNS.items():
        path = traces_root / dirname
        if path.is_dir():
            runs[label] = [path]
    return runs


def trace_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*.json"))
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and "_reports" not in path.parts
        and "_index" not in path.parts
        and path.name != "_run_manifest.jsonl"
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_setting(path: Path, trace: dict[str, Any]) -> str | None:
    config = trace.get("config") or {}
    haystack = " ".join(
        str(value).lower()
        for value in (
            path,
            config.get("experiment_name"),
            config.get("experiment_run_id"),
            config.get("task_id"),
            config.get("task_path"),
        )
        if value is not None
    )
    if "varied" in haystack:
        return "varied"
    if "uniform" in haystack:
        return "uniform"
    return None


def is_5a3p(trace: dict[str, Any]) -> bool:
    config = trace.get("config") or {}
    if int(config.get("num_agents") or 0) != 5:
        return False
    participants = config.get("num_participants")
    return participants is None or int(participants) == 3


def task_id(trace: dict[str, Any], path: Path) -> str:
    config = trace.get("config") or {}
    if config.get("task_id"):
        return str(config["task_id"])
    run_id = str(config.get("experiment_run_id") or path.parent.name)
    return re.sub(r"\.\d+$", "", run_id.split(".", 1)[-1])


def load_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in (calendar_root() / "tasks").glob("*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                task = json.loads(line)
                tid = task.get("task_id")
                if tid:
                    tasks.setdefault(str(tid), task)
                    tasks[f"{path.name}:{tid}"] = task
                    tasks[f"tasks/{path.name}:{tid}"] = task
    return tasks


def round_meetings(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in trace.get("events", []):
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        if data.get("meeting"):
            out[int(data.get("round", len(out)))] = data["meeting"]
    return out


def round_outcome_slot(trace: dict[str, Any], meeting_id: int) -> int | None:
    for row in (trace.get("final_state") or {}).get("round_outcomes") or []:
        if int(row.get("meeting_id", -1)) != int(meeting_id) or not row.get("coordinated"):
            continue
        slots = {int(v) for v in (row.get("per_agent_slot") or {}).values()}
        if len(slots) == 1:
            return next(iter(slots))
    return None


def decision_snapshots(trace: dict[str, Any]) -> dict[tuple[int, int], dict[int, dict[str, Any] | None]]:
    snapshots: dict[tuple[int, int], dict[int, dict[str, Any] | None]] = {}
    for event in trace.get("events", []):
        if event.get("type") != "decide_start":
            continue
        data = event.get("data") or {}
        if data.get("calendar_snapshot_render") is None:
            continue
        key = (int(data.get("round", -1)), int(data.get("agent_id", -1)))
        snapshots[key] = parse_calendar_render(str(data["calendar_snapshot_render"]))
    return snapshots


def parse_calendar_render(text: str) -> dict[int, dict[str, Any] | None]:
    slots: dict[int, dict[str, Any] | None] = {}
    for line in text.splitlines():
        match = re.match(r"\s*Slot\s+(\d+):\s*(.*)", line)
        if not match:
            continue
        slot = int(match.group(1))
        body = match.group(2)
        if "[FREE]" in body:
            slots[slot] = None
            continue
        blocked = re.search(r"Blocked(?: Errand)? #(\d+) \(cost=(\d+)\)", body)
        if blocked:
            slots[slot] = {
                "type": "blocked",
                "item_id": int(blocked.group(1)),
                "cost": int(blocked.group(2)),
            }
            continue
        errand = re.search(r"Errand #(\d+) \(cost=(\d+)\)", body)
        if errand:
            slots[slot] = {
                "type": "errand",
                "item_id": int(errand.group(1)),
                "cost": int(errand.group(2)),
            }
            continue
        meeting = re.search(r"Meeting M(\d+) \(cost=(\d+)\)", body)
        if meeting:
            slots[slot] = {
                "type": "meeting",
                "item_id": int(meeting.group(1)),
                "cost": int(meeting.group(2)),
            }
    return slots


def extract_slots(text: str, *, num_slots: int = 16) -> list[int]:
    """Extract slot indices from natural-language scheduling messages."""
    slots: list[int] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        protocol = payload.get("dsm") or payload.get("imap") or payload.get("sd")
        raw_slots: list[Any] = []
        if protocol in {"proposals", "cost_request"} and isinstance(payload.get("slots"), list):
            raw_slots.extend(payload["slots"])
        if protocol in {"propose", "decision", "confirm"} and isinstance(payload.get("slot"), int):
            raw_slots.append(payload["slot"])
        if raw_slots:
            for value in raw_slots:
                if isinstance(value, int) and 0 <= value < num_slots and value not in slots:
                    slots.append(value)
            return slots
    for match in re.finditer(r"\bslots?\s+((?:\d{1,2}\s*(?:,|/|or|and|-)?\s*)+)", text, flags=re.I):
        for value in re.findall(r"\d{1,2}", match.group(1)):
            slot = int(value)
            if 0 <= slot < num_slots and slot not in slots:
                slots.append(slot)
    for match in re.finditer(r"\b(?:at|for|use|using|try|prefer|preferred|finalize|lock in)\s+slot\s*(\d{1,2})\b", text, flags=re.I):
        slot = int(match.group(1))
        if 0 <= slot < num_slots and slot not in slots:
            slots.append(slot)
    return slots


def slot_is_locally_feasible(item: dict[str, Any] | None) -> bool:
    return item is None or item.get("type") == "errand"


def action_costs_by_round(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in trace.get("events", []):
        if event.get("type") == "batch_applied":
            data = event.get("data") or {}
            round_idx = int(data.get("round", -1))
            agent_id = int(data.get("agent_id", -1))
            after = parse_calendar_render(str(data.get("calendar_render_after", "")))
            for action in data.get("actions") or []:
                if action.get("type") != "reschedule":
                    continue
                to_slot = action.get("to_slot")
                item = after.get(int(to_slot)) if isinstance(to_slot, int) else None
                rows.append(
                    {
                        "round": round_idx,
                        "agent_id": agent_id,
                        "from_slot": action.get("from_slot"),
                        "to_slot": to_slot,
                        "item_id": action.get("item_id"),
                        "item_type": (item or {}).get("type"),
                        "cost": float((item or {}).get("cost", 0)),
                        "source": "batch_applied",
                    }
                )
            continue
        if event.get("type") == "fallback_applied":
            data = event.get("data") or {}
            plan = data.get("displacement_plan") or []
            total = float(data.get("fallback_displacement_cost") or 0)
            per_action_cost = total / len(plan) if plan else 0.0
            for action in plan:
                rows.append(
                    {
                        "round": int(data.get("round", -1)),
                        "agent_id": action.get("agent_id"),
                        "from_slot": action.get("from_slot"),
                        "to_slot": action.get("to_slot"),
                        "item_id": action.get("item_id"),
                        "item_type": "meeting" if action.get("is_meeting_cascade") else "errand_or_unknown",
                        "cost": per_action_cost,
                        "source": "fallback_applied",
                    }
                )
    return rows


def optimal_costs_by_round(task: dict[str, Any]) -> list[float]:
    """Return per-meeting oracle costs under the solver's cost definition."""
    calendars = task["calendars"]
    assignments = {int(k): int(v) for k, v in (task.get("optimal") or {}).get("assignments", {}).items()}
    costs: list[float] = []
    for meeting in task.get("meetings") or []:
        slot = assignments.get(int(meeting["id"]))
        if slot is None:
            costs.append(math.nan)
            continue
        cost = 0.0
        for agent_id in meeting["participants"]:
            item = calendars[agent_id][slot]
            if isinstance(item, dict):
                cost += float(item.get("cost", 0))
        costs.append(cost)
    return costs


def mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else math.nan


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def model_order_by_excess(rows: list[dict[str, Any]]) -> list[str]:
    """Order models from lowest to highest paper coordination cost."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        for key in (
            "coordination_cost",
            "scheduled_only_excess_burden",
            "scheduled_only_excess_cost",
            "excess_cost",
        ):
            value = row.get(key)
            if value not in (None, ""):
                values[str(model)].append(float(value))
                break
        else:
            value_added = False
            for realized_key, optimal_key in (
                ("realized_cost", "scheduled_only_optimal_cost"),
                ("metrics_realized_cost", "metrics_scheduled_only_optimal_cost"),
                ("realized_cost", "optimal_cost"),
                ("metrics_realized_cost", "metrics_optimal_cost"),
            ):
                realized = row.get(realized_key)
                optimal = row.get(optimal_key)
                if realized in (None, "") or optimal in (None, ""):
                    continue
                values[str(model)].append(float(realized) - float(optimal))
                value_added = True
                break
            if value_added:
                continue
    if not values:
        return sorted({str(row["model"]) for row in rows if row.get("model")})
    return [
        model
        for model, _ in sorted(
            values.items(),
            key=lambda item: (mean(item[1]), item[0]),
        )
    ]
