"""Analyze whether models disclose/propose their own slots in varied-cost games.

This is a lightweight, deterministic language audit over cheap-talk messages in
the TRACE_SELECTION cohort. It is intended as a follow-up to the fairness result:
does GPT-5.4 Mini reveal/propose fewer own slots or own slot costs?

Usage:
    cd games/calendar
    uv run python analysis/scripts/own_slot_disclosure_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CALENDAR_ROOT.parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "shared-traces" / "calbench-mixed"
DEFAULT_OUT_DIR = CALENDAR_ROOT / "analysis" / "outputs" / "own_slot_disclosure"

sys.path.insert(0, str(CALENDAR_ROOT))


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

MESSAGE_EVENT_TYPES = {
    "dm_sent",
    "groupchat_sent",
    "participant_groupchat_sent",
    "all_groupchat_sent",
}

SLOT_RE = re.compile(r"\bslots?\s*(?:#\s*)?(\d{1,2})\b", re.I)
FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i am|me|my|mine|myself|for me|on my end|my calendar)\b", re.I)
OWN_AVAILABILITY_RE = re.compile(
    r"\b("
    r"free|available|open|clear|completely free|totally free|"
    r"can do|could do|can make|could make|works for me|work for me|"
    r"fine for me|good for me|available for me|free for me"
    r")\b",
    re.I,
)
PROPOSAL_RE = re.compile(
    r"\b("
    r"how about|let'?s|lets|propose|suggest|schedule|lock|go with|"
    r"choose|pick|use|meet at|settle on"
    r")\b",
    re.I,
)
OWN_COST_RE = re.compile(
    r"\b("
    r"cost|costs|cheap|cheaper|expensive|costly|difficult|hard|easy|"
    r"low[- ]?cost|high[- ]?cost|move my|moving my|reschedule my|"
    r"my errand|blocked|immovable|can move|could move|flexible"
    r")\b",
    re.I,
)
EXACT_COST_RE = re.compile(r"\b(cost(?:s)?\s*(?:=|is|of|:)?\s*\$?\d+|\$?\d+\s*cost)\b", re.I)


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
            if trace is None or _setting(path, trace) != "varied":
                continue
            task_id = _task_id(trace)
            if entrant_label:
                key = (entrant_label, task_id, "varied")
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


def _round_participants(trace: dict[str, Any]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for event in trace.get("events") or []:
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        meeting = data.get("meeting") or {}
        if data.get("round") is not None:
            out[int(data["round"])] = {int(agent_id) for agent_id in meeting.get("participants") or []}
    return out


def _slots(text: str) -> set[int]:
    return {int(match.group(1)) for match in SLOT_RE.finditer(text)}


def _classify_message(content: str) -> dict[str, Any]:
    slots = _slots(content)
    has_slot = bool(slots)
    first_person = bool(FIRST_PERSON_RE.search(content))
    availability = bool(OWN_AVAILABILITY_RE.search(content))
    proposal = bool(PROPOSAL_RE.search(content))
    cost = bool(OWN_COST_RE.search(content))
    exact_cost = bool(EXACT_COST_RE.search(content))
    return {
        "slot_mentions": len(slots),
        "slots": json.dumps(sorted(slots)),
        "own_slot_availability_message": bool(has_slot and first_person and availability),
        "own_slot_proposal_message": bool(has_slot and first_person and (availability or proposal)),
        "own_cost_or_constraint_message": bool(has_slot and first_person and cost),
        "exact_own_cost_message": bool(has_slot and first_person and exact_cost),
    }


def collect_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    message_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    for run_name, entrant_label, entrant_agent_id, path, trace in _iter_selected_traces(trace_root):
        agents = (trace.get("config") or {}).get("agents") or []
        model_by_agent = {
            agent_id: (
                entrant_label
                if entrant_label is not None and entrant_agent_id == agent_id
                else _model_label(agent.get("model") or agent.get("type"))
            )
            for agent_id, agent in enumerate(agents)
        }
        selected = _selected_agent_ids(trace, entrant_agent_id)
        participants_by_round = _round_participants(trace)
        task_id = _task_id(trace)
        game_id = str(trace.get("game_id") or path.stem)

        accum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for event in trace.get("events") or []:
            if event.get("type") not in MESSAGE_EVENT_TYPES:
                continue
            data = event.get("data") or {}
            round_idx = data.get("round")
            sender = data.get("from_agent", data.get("agent_id"))
            content = str(data.get("content") or "")
            if round_idx is None or sender is None or int(sender) not in selected:
                continue
            round_idx = int(round_idx)
            sender = int(sender)
            if sender not in participants_by_round.get(round_idx, set()):
                continue
            classified = _classify_message(content)
            accum[sender]["outgoing_messages"] += 1
            accum[sender]["slot_mention_messages"] += 1 if classified["slot_mentions"] else 0
            accum[sender]["own_slot_availability_messages"] += classified["own_slot_availability_message"]
            accum[sender]["own_slot_proposal_messages"] += classified["own_slot_proposal_message"]
            accum[sender]["own_cost_or_constraint_messages"] += classified["own_cost_or_constraint_message"]
            accum[sender]["exact_own_cost_messages"] += classified["exact_own_cost_message"]
            accum[sender]["slot_mentions"] += classified["slot_mentions"]
            message_rows.append(
                {
                    "run": run_name,
                    "trace_path": str(path),
                    "game_id": game_id,
                    "task_id": task_id,
                    "round": round_idx,
                    "agent_id": sender,
                    "model": model_by_agent.get(sender, "unknown"),
                    "event_type": event.get("type"),
                    "channel": data.get("channel", event.get("type")),
                    "content": content,
                    **classified,
                }
            )

        for agent_id in selected:
            row = accum[agent_id]
            outgoing = row["outgoing_messages"]
            agent_rows.append(
                {
                    "run": run_name,
                    "trace_path": str(path),
                    "game_id": game_id,
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "model": model_by_agent.get(agent_id, "unknown"),
                    "outgoing_messages": outgoing,
                    "slot_mention_messages": row["slot_mention_messages"],
                    "own_slot_availability_messages": row["own_slot_availability_messages"],
                    "own_slot_proposal_messages": row["own_slot_proposal_messages"],
                    "own_cost_or_constraint_messages": row["own_cost_or_constraint_messages"],
                    "exact_own_cost_messages": row["exact_own_cost_messages"],
                    "slot_mentions": row["slot_mentions"],
                    "own_slot_proposal_rate": row["own_slot_proposal_messages"] / outgoing if outgoing else math.nan,
                    "own_slot_availability_rate": row["own_slot_availability_messages"] / outgoing if outgoing else math.nan,
                    "own_cost_or_constraint_rate": row["own_cost_or_constraint_messages"] / outgoing if outgoing else math.nan,
                    "exact_own_cost_rate": row["exact_own_cost_messages"] / outgoing if outgoing else math.nan,
                }
            )
    return message_rows, agent_rows


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _signflip_pvalue(diffs: list[float], *, seed: int, n_perm: int = 200000, lower: bool = True) -> tuple[float, float]:
    arr = np.array(diffs, dtype=float)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    batch = 10000
    while done < n_perm:
        size = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(arr)))
        means = (signs * arr).mean(axis=1)
        if lower:
            count += int(np.sum(means <= obs + 1e-12))
        else:
            count += int(np.sum(means >= obs - 1e-12))
        done += size
    return obs, (count + 1) / (n_perm + 1)


def _bootstrap_ci(values: list[float], *, seed: int, n_boot: int = 100000) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_and_test(agent_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_names = [
        "own_slot_proposal_rate",
        "own_slot_availability_rate",
        "own_cost_or_constraint_rate",
        "exact_own_cost_rate",
    ]
    summaries: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        by_model[str(row["model"])].append(row)
    for model in MODEL_ORDER:
        rows = by_model.get(model, [])
        if not rows:
            continue
        summary = {"model": model, "trace_agent_count": len(rows)}
        for metric in metric_names:
            values = _finite(row.get(metric) for row in rows)
            summary[f"{metric}_mean"] = mean(values) if values else math.nan
            summary[f"{metric}_median"] = median(values) if values else math.nan
        for count_name in [
            "outgoing_messages",
            "own_slot_proposal_messages",
            "own_slot_availability_messages",
            "own_cost_or_constraint_messages",
            "exact_own_cost_messages",
        ]:
            summary[f"{count_name}_total"] = sum(float(row.get(count_name, 0.0)) for row in rows)
        summaries.append(summary)

    # Paired task tests: GPT rate minus mean(other six rates), H1: GPT lower.
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in agent_rows:
        by_task[str(row["task_id"])][str(row["model"])] = row
    complete = {task: values for task, values in by_task.items() if all(model in values for model in MODEL_ORDER)}
    tests: list[dict[str, Any]] = []
    for idx, metric in enumerate(metric_names):
        diffs: list[float] = []
        for values in complete.values():
            gpt = float(values["GPT-5.4 Mini"][metric])
            other = mean(float(values[model][metric]) for model in MODEL_ORDER if model != "GPT-5.4 Mini")
            if math.isfinite(gpt) and math.isfinite(other):
                diffs.append(gpt - other)
        if not diffs:
            continue
        obs, p = _signflip_pvalue(diffs, seed=1000 + idx, lower=True)
        lo, hi = _bootstrap_ci(diffs, seed=2000 + idx)
        tests.append(
            {
                "metric": metric,
                "paired_tasks": len(diffs),
                "gpt_minus_other_mean_difference": obs,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "one_sided_permutation_p_gpt_lower": p,
                "negative_task_differences": sum(diff < 0 for diff in diffs),
                "positive_task_differences": sum(diff > 0 for diff in diffs),
            }
        )
    return summaries, tests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    messages, agent_rows = collect_rows(Path(args.trace_root))
    summaries, tests = summarize_and_test(agent_rows)
    write_csv(out_dir / "own_slot_message_rows.csv", messages)
    write_csv(out_dir / "own_slot_trace_agent_metrics.csv", agent_rows)
    write_csv(out_dir / "own_slot_summary_by_model.csv", summaries)
    write_csv(out_dir / "gpt_vs_other_own_slot_tests.csv", tests)
    print(f"Wrote {len(messages)} message rows and {len(agent_rows)} trace-agent rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
