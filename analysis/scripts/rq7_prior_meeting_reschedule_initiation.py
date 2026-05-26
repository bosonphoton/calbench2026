"""RQ7: initiating reschedules of prior multi-agent meetings, by model.

Counts decision/voluntary reschedule actions whose source slot contains a
previously scheduled meeting rather than a local errand.

Usage:
    cd games/calendar
    uv run python analysis/scripts/rq7_prior_meeting_reschedule_initiation.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rq_common import (
    action_costs_by_round,
    default_traces_root,
    discover_model_runs,
    is_5a3p,
    load_json,
    mean,
    model_order_by_excess,
    output_dir,
    resolve_path,
    round_meetings,
    task_id,
    trace_files,
    trace_setting,
)


def analyze_trace(path: Path, model: str) -> list[dict]:
    trace = load_json(path)
    if not is_5a3p(trace):
        return []
    setting = trace_setting(path, trace)
    if setting not in {"uniform", "varied"}:
        return []
    metrics = trace.get("metrics") or {}
    meetings = round_meetings(trace)
    action_rows = action_costs_by_round(trace)
    prior_by_round: dict[int, list[dict]] = defaultdict(list)
    for action in action_rows:
        if action.get("item_type") == "meeting":
            prior_by_round[int(action["round"])].append(action)

    rows: list[dict] = []
    for round_idx, meeting in meetings.items():
        prior = prior_by_round.get(round_idx, [])
        rows.append(
            {
                "model": model,
                "setting": setting,
                "game_id": trace.get("game_id") or path.stem,
                "task_id": task_id(trace, path),
                "trace_path": str(path),
                "round": round_idx,
                "meeting_id": int(meeting["id"]),
                "prior_meeting_reschedule_actions": len(prior),
                "prior_meeting_reschedule_cost": sum(float(row["cost"]) for row in prior),
                "initiated_prior_meeting_reschedule": len(prior) > 0,
                "initiating_agents": ",".join(str(row["agent_id"]) for row in prior),
                "displaced_meeting_ids": ",".join(str(row["item_id"]) for row in prior),
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
    per_game = summarize_games(rows)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_game:
        groups[(row["model"], row["setting"])].append(row)
    summary: list[dict] = []
    for (model, setting), group in sorted(groups.items()):
        summary.append(
            {
                "model": model,
                "setting": setting,
                "games": len(group),
                "mean_prior_meetings_rescheduled_per_game": mean([
                    float(r["prior_meetings_rescheduled"]) for r in group
                ]),
                "games_with_any_prior_meeting_reschedule_rate": mean([
                    1.0 if int(r["prior_meetings_rescheduled"]) > 0 else 0.0 for r in group
                ]),
            }
        )
    return summary


def summarize_games(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["setting"], row["game_id"])].append(row)
    out: list[dict] = []
    for (model, setting, game_id), group in sorted(groups.items()):
        displaced: set[str] = set()
        actions = 0
        cost = 0.0
        for row in group:
            if not row["displaced_meeting_ids"]:
                continue
            actions += int(row["prior_meeting_reschedule_actions"])
            cost += float(row["prior_meeting_reschedule_cost"])
            displaced.update(value for value in str(row["displaced_meeting_ids"]).split(",") if value)
        out.append(
            {
                "model": model,
                "setting": setting,
                "game_id": game_id,
                "task_id": group[0]["task_id"],
                "trace_path": group[0]["trace_path"],
                "prior_meetings_rescheduled": len(displaced),
                "prior_meeting_reschedule_actions": actions,
                "prior_meeting_reschedule_cost": cost,
                "realized_cost": group[0].get("realized_cost"),
                "optimal_cost": group[0].get("optimal_cost"),
            }
        )
    return out


def plot(per_game: list[dict], rows: list[dict], out: Path) -> None:
    models = model_order_by_excess(rows)
    settings = ["uniform", "varied"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.36
    xs = list(range(len(models)))
    ymax = 0.0
    for idx, setting in enumerate(settings):
        vals = []
        setting_rows = [row for row in per_game if row["setting"] == setting]
        for model in models:
            model_rows = [r for r in setting_rows if r["model"] == model]
            value = mean([float(r["prior_meetings_rescheduled"]) for r in model_rows])
            vals.append(value)
            ymax = max(ymax, value)
        ax.bar([x + (idx - 0.5) * width for x in xs], vals, width=width, label=setting.title())
    ax.set_ylabel("Mean prior meetings rescheduled per game")
    ax.set_title("RQ7: Prior Meetings Rescheduled Per Game")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, ymax * 1.2 if ymax else 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "prior_meeting_reschedule_rate.png", dpi=180)
    fig.savefig(out / "prior_meeting_reschedule_rate.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=str(default_traces_root()))
    parser.add_argument("--out", default="rq7_prior_meeting_reschedule_initiation")
    args = parser.parse_args()

    rows: list[dict] = []
    for model, run_dirs in discover_model_runs(resolve_path(args.traces_root)).items():
        for run_dir in run_dirs:
            for path in trace_files(run_dir):
                rows.extend(analyze_trace(path, model))

    out = output_dir(args.out)
    per_game = summarize_games(rows)
    summary = summarize(rows)
    write_csv(out / "per_meeting.csv", rows)
    write_csv(out / "per_game.csv", per_game)
    write_csv(out / "summary_by_model_setting.csv", summary)
    plot(per_game, rows, out)
    print(f"wrote {len(rows)} meeting rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
