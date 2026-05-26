"""RQ8: proposing a bump where displaced errand is cheaper than avoided one.

This is a cheap-talk heuristic. For each DM, the script extracts slot mentions
in order and compares those slots on the sender's decision snapshot. A message
is counted when it uses bump/avoidance language and mentions an earlier errand
slot with higher cost followed by a later errand slot with lower cost.

Usage:
    cd games/calendar
    uv run python analysis/scripts/rq8_cheaper_errand_bump_proposals.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rq_common import (
    decision_snapshots,
    default_traces_root,
    discover_model_runs,
    extract_slots,
    is_5a3p,
    load_json,
    mean,
    model_order_by_excess,
    output_dir,
    resolve_path,
    task_id,
    trace_files,
    trace_setting,
)


COMPARATIVE_RE = re.compile(
    r"\b(require|requires|required|move|reschedule|commitment|difficult|hard|prefer|instead|rather|avoid|cheaper|easier|lower)\b",
    flags=re.I,
)


def _sender(data: dict) -> int | None:
    for key in ("from_agent", "agent_id", "from_agent_id"):
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def analyze_trace(path: Path, model: str) -> list[dict]:
    trace = load_json(path)
    if not is_5a3p(trace):
        return []
    setting = trace_setting(path, trace)
    if setting != "varied":
        return []
    metrics = trace.get("metrics") or {}
    snapshots = decision_snapshots(trace)
    num_slots = int((trace.get("config") or {}).get("num_slots", 16))
    rows: list[dict] = []
    for event_index, event in enumerate(trace.get("events", [])):
        if event.get("type") != "dm_sent":
            continue
        data = event.get("data") or {}
        content = str(data.get("content", ""))
        if not COMPARATIVE_RE.search(content):
            continue
        round_idx = int(data.get("round", -1))
        sender = _sender(data)
        if sender is None:
            continue
        snapshot = snapshots.get((round_idx, sender), {})
        slots = extract_slots(content, num_slots=num_slots)
        best_pair = None
        for i, avoided_slot in enumerate(slots):
            avoided = snapshot.get(avoided_slot)
            if not avoided or avoided.get("type") != "errand":
                continue
            for proposed_slot in slots[i + 1 :]:
                proposed = snapshot.get(proposed_slot)
                if not proposed or proposed.get("type") != "errand":
                    continue
                avoided_cost = float(avoided.get("cost", 0))
                proposed_cost = float(proposed.get("cost", 0))
                if proposed_cost < avoided_cost:
                    best_pair = (avoided_slot, avoided_cost, proposed_slot, proposed_cost)
                    break
            if best_pair:
                break
        if not best_pair:
            continue
        avoided_slot, avoided_cost, proposed_slot, proposed_cost = best_pair
        rows.append(
            {
                "model": model,
                "setting": setting,
                "game_id": trace.get("game_id") or path.stem,
                "task_id": task_id(trace, path),
                "trace_path": str(path),
                "event_index": event_index,
                "round": round_idx,
                "turn": data.get("turn"),
                "sender": sender,
                "avoided_slot": avoided_slot,
                "avoided_errand_cost": avoided_cost,
                "proposed_slot": proposed_slot,
                "proposed_errand_cost": proposed_cost,
                "cost_delta_avoided_minus_proposed": avoided_cost - proposed_cost,
                "content": content,
                "realized_cost": metrics.get("realized_cost"),
                "optimal_cost": metrics.get("optimal_cost"),
            }
        )
    return rows


def trace_metric_row(path: Path, model: str) -> dict | None:
    trace = load_json(path)
    if not is_5a3p(trace) or trace_setting(path, trace) not in {"uniform", "varied"}:
        return None
    metrics = trace.get("metrics") or {}
    return {
        "model": model,
        "setting": trace_setting(path, trace),
        "realized_cost": metrics.get("realized_cost"),
        "optimal_cost": metrics.get("optimal_cost"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    games: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["model"], row["setting"])
        groups[key].append(row)
        games[key].add(row["game_id"])
    summary: list[dict] = []
    for key, group in sorted(groups.items()):
        model, setting = key
        summary.append(
            {
                "model": model,
                "setting": setting,
                "proposal_messages": len(group),
                "games_with_proposals": len(games[key]),
                "mean_cost_delta": mean([float(r["cost_delta_avoided_minus_proposed"]) for r in group]),
            }
        )
    return summary


def plot(summary: list[dict], metric_rows: list[dict], out: Path) -> None:
    models = model_order_by_excess(metric_rows)
    setting = "varied"
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = list(range(len(models)))
    vals = []
    for model in models:
        row = next((r for r in summary if r["model"] == model and r["setting"] == setting), None)
        vals.append(float(row["proposal_messages"]) if row else 0.0)
    ax.bar(xs, vals, width=0.65, color="#4c78a8")
    ax.set_ylabel("Detected DM count")
    ax.set_title("RQ8: Cheaper Errand Bump Proposals (Varied Only)")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cheaper_errand_bump_proposals.png", dpi=180)
    fig.savefig(out / "cheaper_errand_bump_proposals.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=str(default_traces_root()))
    parser.add_argument("--out", default="rq8_cheaper_errand_bump_proposals")
    args = parser.parse_args()

    rows: list[dict] = []
    metric_rows: list[dict] = []
    for model, run_dirs in discover_model_runs(resolve_path(args.traces_root)).items():
        for run_dir in run_dirs:
            for path in trace_files(run_dir):
                metric = trace_metric_row(path, model)
                if metric is not None:
                    metric_rows.append(metric)
                rows.extend(analyze_trace(path, model))

    out = output_dir(args.out)
    summary = summarize(rows)
    write_csv(out / "proposal_messages.csv", rows)
    write_csv(out / "summary_by_model_setting.csv", summary)
    plot(summary, metric_rows, out)
    print(f"wrote {len(rows)} proposal rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
