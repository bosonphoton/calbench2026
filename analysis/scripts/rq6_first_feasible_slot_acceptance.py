"""RQ6: accepting the first feasible slot offered, by model.

The script uses a trace-level heuristic: extract slot mentions from cheap-talk
DMs in order, find the first mentioned slot that is locally feasible for all
participants at decision time (free or blocked only by a movable errand), and
compare it with the coordinated final slot for that meeting.

Usage:
    cd games/calendar
    uv run python analysis/scripts/rq6_first_feasible_slot_acceptance.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rq_common import (
    default_traces_root,
    discover_baseline_runs,
    decision_snapshots,
    discover_model_runs,
    extract_slots,
    is_5a3p,
    load_json,
    mean,
    model_order_by_excess,
    output_dir,
    resolve_path,
    round_meetings,
    round_outcome_slot,
    slot_is_locally_feasible,
    task_id,
    trace_files,
    trace_setting,
)


def _slot_cost(snapshot: dict[int, dict | None], slot: int) -> float | None:
    item = snapshot.get(slot)
    if not slot_is_locally_feasible(item):
        return None
    return float((item or {}).get("cost", 0))


def _joint_feasible_slot_cost(
    snapshots: dict[tuple[int, int], dict[int, dict | None]],
    round_idx: int,
    participants: list[int],
    slot: int,
) -> float | None:
    total = 0.0
    for agent_id in participants:
        cost = _slot_cost(snapshots.get((round_idx, agent_id), {}), slot)
        if cost is None:
            return None
        total += cost
    return total


def _best_joint_feasible_slot(
    snapshots: dict[tuple[int, int], dict[int, dict | None]],
    round_idx: int,
    participants: list[int],
    num_slots: int,
) -> tuple[int | None, float | None]:
    best_slot = None
    best_cost = None
    for slot in range(num_slots):
        cost = _joint_feasible_slot_cost(snapshots, round_idx, participants, slot)
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_slot = slot
            best_cost = cost
    return best_slot, best_cost


def analyze_trace(path: Path, model: str, *, series: str = "model") -> list[dict]:
    trace = load_json(path)
    if not is_5a3p(trace):
        return []
    setting = trace_setting(path, trace)
    if setting not in {"uniform", "varied"}:
        return []
    snapshots = decision_snapshots(trace)
    meetings = round_meetings(trace)
    rows: list[dict] = []
    num_slots = int((trace.get("config") or {}).get("num_slots", 16))
    for round_idx, meeting in meetings.items():
        participants = [int(p) for p in meeting.get("participants", [])]
        first_feasible = None
        first_offered = None
        first_offer_turn = None
        for event in trace.get("events", []):
            if event.get("type") != "dm_sent":
                continue
            data = event.get("data") or {}
            if int(data.get("round", -1)) != round_idx:
                continue
            for slot in extract_slots(str(data.get("content", "")), num_slots=num_slots):
                if first_offered is None:
                    first_offered = slot
                feasible = True
                for agent_id in participants:
                    item = snapshots.get((round_idx, agent_id), {}).get(slot)
                    if not slot_is_locally_feasible(item):
                        feasible = False
                        break
                if feasible:
                    first_feasible = slot
                    first_offer_turn = data.get("turn")
                    break
            if first_feasible is not None:
                break
        final_slot = round_outcome_slot(trace, int(meeting["id"]))
        first_feasible_cost = (
            _joint_feasible_slot_cost(snapshots, round_idx, participants, int(first_feasible))
            if first_feasible is not None
            else None
        )
        best_feasible_slot, best_feasible_cost = _best_joint_feasible_slot(
            snapshots, round_idx, participants, num_slots
        )
        final_slot_cost = (
            _joint_feasible_slot_cost(snapshots, round_idx, participants, int(final_slot))
            if final_slot is not None
            else None
        )
        metrics = trace.get("metrics") or {}
        rows.append(
            {
                "model": model,
                "series": series,
                "setting": setting,
                "game_id": trace.get("game_id") or path.stem,
                "task_id": task_id(trace, path),
                "trace_path": str(path),
                "round": round_idx,
                "meeting_id": int(meeting["id"]),
                "first_offered_slot": first_offered,
                "first_feasible_slot": first_feasible,
                "first_feasible_cost": first_feasible_cost,
                "first_feasible_turn": first_offer_turn,
                "best_feasible_slot": best_feasible_slot,
                "best_feasible_cost": best_feasible_cost,
                "final_slot": final_slot,
                "final_slot_cost": final_slot_cost,
                "coordinated": final_slot is not None,
                "accepted_first_feasible": (
                    first_feasible is not None and final_slot is not None and int(first_feasible) == int(final_slot)
                ),
                "cheaper_feasible_alternative_exists": (
                    first_feasible_cost is not None
                    and best_feasible_cost is not None
                    and float(best_feasible_cost) < float(first_feasible_cost)
                ),
                "first_feasible_excess_over_best": (
                    float(first_feasible_cost) - float(best_feasible_cost)
                    if first_feasible_cost is not None and best_feasible_cost is not None
                    else None
                ),
                "realized_cost": metrics.get("realized_cost"),
                "optimal_cost": metrics.get("optimal_cost"),
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
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["setting"])].append(row)
    out: list[dict] = []
    for (model, setting), group in sorted(groups.items()):
        denom = [r for r in group if r["first_feasible_slot"] is not None and r["coordinated"]]
        out.append(
            {
                "model": model,
                "setting": setting,
                "meetings": len(group),
                "meetings_with_first_feasible_and_coordination": len(denom),
                "accepted_first_feasible_rate": mean([1.0 if r["accepted_first_feasible"] else 0.0 for r in denom]),
                "cheaper_feasible_alternative_rate": mean([
                    1.0 if r["cheaper_feasible_alternative_exists"] else 0.0
                    for r in group
                    if r["first_feasible_slot"] is not None
                ]),
                "accepted_first_feasible_bad_decision_rate": mean([
                    1.0 if r["cheaper_feasible_alternative_exists"] else 0.0
                    for r in group
                    if r["accepted_first_feasible"]
                ]),
                "mean_first_feasible_excess_over_best": mean([
                    float(r["first_feasible_excess_over_best"])
                    for r in group
                    if r["first_feasible_excess_over_best"] is not None
                ]),
                "no_feasible_offer_rate": mean([1.0 if r["first_feasible_slot"] is None else 0.0 for r in group]),
            }
        )
    return out


def plot(summary: list[dict], rows: list[dict], out: Path) -> None:
    model_rows = [row for row in rows if row.get("series") == "model"]
    baseline_labels = sorted({row["model"] for row in rows if row.get("series") == "baseline"})
    models = model_order_by_excess(model_rows) + baseline_labels
    settings = ["uniform", "varied"]
    x = range(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, setting in enumerate(settings):
        values = []
        for model in models:
            row = next((r for r in summary if r["model"] == model and r["setting"] == setting), None)
            value = float(row["accepted_first_feasible_rate"]) if row and row["accepted_first_feasible_rate"] != "" else math.nan
            values.append(value)
        offsets = [i + (idx - 0.5) * width for i in x]
        ax.bar(offsets, values, width=width, label=setting.title())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("RQ6: Accepted First Feasible Offered Slot")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "accepted_first_feasible_rate.png", dpi=180)
    fig.savefig(out / "accepted_first_feasible_rate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, setting in enumerate(settings):
        values = []
        for model in models:
            row = next((r for r in summary if r["model"] == model and r["setting"] == setting), None)
            values.append(float(row["cheaper_feasible_alternative_rate"]) if row else math.nan)
        ax.bar([i + (idx - 0.5) * width for i in x], values, width=width, label=setting.title())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("RQ6+: First Feasible Offer Had Cheaper Feasible Alternative")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "first_feasible_had_cheaper_alternative_rate.png", dpi=180)
    fig.savefig(out / "first_feasible_had_cheaper_alternative_rate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for idx, setting in enumerate(settings):
        values = []
        for model in models:
            row = next((r for r in summary if r["model"] == model and r["setting"] == setting), None)
            values.append(float(row["accepted_first_feasible_bad_decision_rate"]) if row else math.nan)
        ax.bar([i + (idx - 0.5) * width for i in x], values, width=width, label=setting.title())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate among accepted-first-feasible cases")
    ax.set_title("RQ6++: Accepted First Feasible Despite Cheaper Feasible Alternative")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "accepted_first_feasible_bad_decision_rate.png", dpi=180)
    fig.savefig(out / "accepted_first_feasible_bad_decision_rate.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=str(default_traces_root()))
    parser.add_argument("--out", default="rq6_first_feasible_slot_acceptance")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    traces_root = resolve_path(args.traces_root)
    rows: list[dict] = []
    for model, run_dirs in discover_model_runs(traces_root).items():
        for run_dir in run_dirs:
            for path in trace_files(run_dir):
                rows.extend(analyze_trace(path, model))
    if args.include_baselines:
        for model, run_dirs in discover_baseline_runs(traces_root).items():
            for run_dir in run_dirs:
                for path in trace_files(run_dir):
                    rows.extend(analyze_trace(path, model, series="baseline"))

    out = output_dir(args.out)
    summary = summarize(rows)
    write_csv(out / "per_meeting.csv", rows)
    write_csv(out / "summary_by_model_setting.csv", summary)
    plot(summary, rows, out)
    print(f"wrote {len(rows)} meeting rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
