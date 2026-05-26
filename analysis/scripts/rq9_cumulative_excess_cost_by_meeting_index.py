"""RQ9: per-meeting-index cumulative excess cost plot, completed games only.

For each completed trace, realized per-round cost is reconstructed from applied
reschedule/fallback actions. Optimal per-round cost is reconstructed by
replaying the task fixture's global oracle assignment. The plotted quantity is
mean cumulative realized cost minus cumulative oracle replay cost at each
meeting index. Prefix values can still be negative because the oracle is global,
but final values are standard full-game excess costs.

Usage:
    cd games/calendar
    uv run python analysis/scripts/rq9_cumulative_excess_cost_by_meeting_index.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rq_common import (
    action_costs_by_round,
    default_traces_root,
    discover_model_runs,
    is_5a3p,
    load_json,
    load_tasks,
    mean,
    model_order_by_excess,
    optimal_costs_by_round,
    output_dir,
    resolve_path,
    task_id,
    trace_files,
    trace_setting,
)


def _coordinated_by_round(trace: dict) -> dict[int, bool]:
    outcomes = (trace.get("final_state") or {}).get("round_outcomes") or []
    by_meeting = {
        int(row.get("meeting_id", -1)): bool(row.get("coordinated"))
        for row in outcomes
    }
    out: dict[int, bool] = {}
    for event in trace.get("events", []):
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        meeting = data.get("meeting") or {}
        if "id" in meeting:
            out[int(data.get("round", len(out)))] = by_meeting.get(int(meeting["id"]), False)
    return out


def analyze_trace(path: Path, model: str, tasks: dict[str, dict]) -> list[dict]:
    trace = load_json(path)
    if not is_5a3p(trace):
        return []
    setting = trace_setting(path, trace)
    if setting not in {"uniform", "varied"}:
        return []
    tid = task_id(trace, path)
    config = trace.get("config") or {}
    task_path = str(config.get("task_path") or "")
    task = tasks.get(f"{task_path}:{tid}") or tasks.get(tid)
    if not task or not (task.get("optimal") or {}).get("assignments"):
        return []
    metrics = trace.get("metrics") or {}
    if int(metrics.get("meetings_scheduled") or 0) < int(config.get("num_meetings") or len(task.get("meetings") or [])):
        return []

    realized_by_round: dict[int, float] = defaultdict(float)
    for row in action_costs_by_round(trace):
        if row.get("source") == "fallback_applied":
            continue
        if int(row.get("round", -1)) >= 0:
            realized_by_round[int(row["round"])] += float(row.get("cost", 0))

    optimal_by_round = optimal_costs_by_round(task)
    rows: list[dict] = []
    cumulative_realized = 0.0
    cumulative_optimal = 0.0
    for idx, optimal_cost in enumerate(optimal_by_round):
        realized_cost = realized_by_round.get(idx, 0.0)
        cumulative_realized += realized_cost
        if not math.isnan(optimal_cost):
            cumulative_optimal += optimal_cost
        meeting_index = idx + 1
        rows.append(
            {
                "model": model,
                "setting": setting,
                "game_id": trace.get("game_id") or path.stem,
                "task_id": tid,
                "trace_path": str(path),
                "meeting_index": meeting_index,
                "round": idx,
                "realized_round_cost": realized_cost,
                "optimal_round_cost": optimal_cost,
                "cumulative_realized_cost": cumulative_realized,
                "cumulative_optimal_cost": cumulative_optimal,
                "cumulative_excess_cost": cumulative_realized - cumulative_optimal,
                "metrics_realized_cost": metrics.get("realized_cost"),
                "metrics_optimal_cost": metrics.get("optimal_cost"),
                "meetings_scheduled": metrics.get("meetings_scheduled"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["setting"], int(row["meeting_index"]))].append(row)
    summary: list[dict] = []
    for (model, setting, meeting_index), group in sorted(groups.items()):
        vals = [float(r["cumulative_excess_cost"]) for r in group]
        summary.append(
            {
                "model": model,
                "setting": setting,
                "meeting_index": meeting_index,
                "traces": len(group),
                "mean_cumulative_excess_cost": mean(vals),
                "median_cumulative_excess_cost": sorted(vals)[len(vals) // 2] if vals else math.nan,
            }
        )
    return summary


def plot(summary: list[dict], rows_for_order: list[dict], out: Path) -> None:
    models = model_order_by_excess(rows_for_order)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, setting in zip(axes, ["uniform", "varied"], strict=True):
        for model in models:
            rows = sorted(
                [r for r in summary if r["model"] == model and r["setting"] == setting],
                key=lambda r: int(r["meeting_index"]),
            )
            if not rows:
                continue
            ax.plot(
                [int(r["meeting_index"]) for r in rows],
                [float(r["mean_cumulative_excess_cost"]) for r in rows],
                marker="o",
                linewidth=1.8,
                label=model,
            )
        ax.set_title(setting.title())
        ax.set_xlabel("Meeting index")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean cumulative excess cost")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("RQ9: Cumulative Excess Cost by Meeting Index (Completed Games Only)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "cumulative_excess_cost_by_meeting_index.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "cumulative_excess_cost_by_meeting_index.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=str(default_traces_root()))
    parser.add_argument("--out", default="rq9_cumulative_excess_cost_by_meeting_index")
    args = parser.parse_args()

    tasks = load_tasks()
    rows: list[dict] = []
    for model, run_dirs in discover_model_runs(resolve_path(args.traces_root)).items():
        for run_dir in run_dirs:
            for path in trace_files(run_dir):
                rows.extend(analyze_trace(path, model, tasks))

    out = output_dir(args.out)
    summary = summarize(rows)
    write_csv(out / "per_trace_meeting_index.csv", rows)
    write_csv(out / "summary_by_model_setting_index.csv", summary)
    plot(summary, rows, out)
    print(f"wrote {len(rows)} trace-index rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
