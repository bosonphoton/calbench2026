"""Aggregate benchmark-lite metrics by difficulty bucket.

This script scans synced shared-traces benchmark-lite runs and groups traces by
model, setting, and difficulty bucket (easy/medium/hard). It writes:

* trace_metrics.csv: one row per trace
* summary_by_model_setting_difficulty.csv: aggregate metrics
* rankings_by_setting_difficulty.csv: per-bucket model rankings
* rank_stability.csv: easy/medium/hard rank deltas per model and setting
* summary.md: readable report

Example:
    cd games/calendar
    uv run python analysis/scripts/analyze_difficulty_buckets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


MODEL_RUNS: dict[str, str] = {
    "Claude Sonnet 4.6": "claude-sonnet46-benchmark-lite-001",
    "DeepSeek V4 Pro": "deepseek-v4-pro-benchmark-lite-002",
    "Gemini 3 Flash": "gemini3flash-benchmark-lite-001",
    "Gemini 3.1 Pro": "gemini31pro-benchmark-lite-001",
    "GPT-5.4 Mini": "gpt54-mini-benchmark-lite-001",
    "Llama 4 Maverick": "llama4-maverick-benchmark-lite-001",
    "Qwen3.6 Plus": "qwen36-plus-benchmark-lite-002",
}

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
SETTING_ORDER = {"uniform": 0, "varied": 1}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _trace_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*.json"))
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and "_reports" not in path.parts
        and "_index" not in path.parts
    ]


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _field_haystack(path: Path, trace: dict[str, Any]) -> str:
    config = trace.get("config") or {}
    return " ".join(
        str(value)
        for value in [
            path,
            trace.get("game_id"),
            config.get("experiment_name"),
            config.get("experiment_run_id"),
            config.get("task_id"),
            config.get("task_path"),
        ]
        if value is not None
    )


def _difficulty(path: Path, trace: dict[str, Any]) -> str | None:
    match = re.search(r"\b(?:b\d+_)?(easy|medium|hard)_", _field_haystack(path, trace), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _setting(path: Path, trace: dict[str, Any]) -> str | None:
    haystack = _field_haystack(path, trace).lower()
    if "varied" in haystack:
        return "varied"
    if "uniform" in haystack:
        return "uniform"
    return None


def _is_5a3p(trace: dict[str, Any]) -> bool:
    config = trace.get("config") or {}
    if int(config.get("num_agents") or 0) != 5:
        return False
    num_participants = config.get("num_participants")
    return num_participants is None or int(num_participants) == 3


def _dm_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in trace.get("events", []) if event.get("type") == "dm_sent"]


def _vps_by_game_id(run_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = run_dir / "_reports" / "vps" / "game_summary.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            game_id = row.get("game_id")
            mode = row.get("weight_mode")
            if not game_id or not mode:
                continue
            out[game_id][mode] = {
                "vps_loss_mean": _float(row.get("vps_loss_mean")),
                "vps_loss_per_weight": _float(row.get("vps_loss_per_weight")),
                "participant_pair_vps_loss_mean": _float(row.get("participant_pair_vps_loss_mean")),
                "observation_count": _float(row.get("observation_count")),
            }
    return dict(out)


def _float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def _median(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(median(clean)) if clean else math.nan


def _p90(values: Iterable[float]) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    idx = math.ceil(0.9 * len(clean)) - 1
    return clean[min(max(idx, 0), len(clean) - 1)]


def _stdev(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return math.nan
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def _ci95(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return math.nan
    return 1.96 * _stdev(clean) / math.sqrt(len(clean))


def _fmt(value: float, digits: int = 2) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def _per_agent_range(trace: dict[str, Any]) -> float:
    costs = (trace.get("final_state") or {}).get("per_agent_cost") or []
    vals = [float(value) for value in costs if value is not None]
    return max(vals) - min(vals) if vals else math.nan


def _per_agent_max(trace: dict[str, Any]) -> float:
    costs = (trace.get("final_state") or {}).get("per_agent_cost") or []
    vals = [float(value) for value in costs if value is not None]
    return max(vals) if vals else math.nan


def _contribution_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in (trace.get("final_state") or {}).get("contribution_scores") or []
        if isinstance(row, dict)
    ]


def _mean_contribution(trace: dict[str, Any], key: str) -> float:
    vals = [_float(row.get(key)) for row in _contribution_rows(trace)]
    return _mean(vals)


def _paper_fairness_cost(trace: dict[str, Any]) -> float:
    direct = _mean_contribution(trace, "fairness_cost")
    if not math.isnan(direct):
        return direct
    metrics = trace.get("metrics") or {}
    values = [_float(value) for value in metrics.get("per_agent_fairness_cost") or []]
    return _mean(values)


def collect_trace_rows(shared_traces: Path, models: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, run_name in models.items():
        run_dir = shared_traces / run_name
        vps = _vps_by_game_id(run_dir)
        for path in _trace_files(run_dir):
            trace = _load_trace(path)
            if not _is_5a3p(trace):
                continue
            setting = _setting(path, trace)
            difficulty = _difficulty(path, trace)
            if setting not in SETTING_ORDER or difficulty not in DIFFICULTY_ORDER:
                continue
            metrics = trace.get("metrics") or {}
            game_id = str(trace.get("game_id") or path.stem)
            meetings = _float(metrics.get("meetings_scheduled"))
            total_dms = _float(metrics.get("total_dms_sent"))
            total_cheap_talk = _float(metrics.get("total_cheap_talk_messages"))
            realized = _float(metrics.get("realized_cost"))
            optimal = _float(metrics.get("optimal_cost"))
            legacy_full_task_excess = realized - optimal if not math.isnan(realized) and not math.isnan(optimal) else math.nan
            coordination_cost = _mean_contribution(trace, "scheduled_only_excess_burden")
            if math.isnan(coordination_cost):
                coordination_cost = _mean(_float(value) for value in metrics.get("per_agent_scheduled_only_excess_burden") or [])
            task_success = _mean_contribution(trace, "task_success")
            if math.isnan(task_success):
                task_success = _float(metrics.get("coordination_rate"))
            communication_efficiency = _mean_contribution(trace, "messages_per_scheduled_meeting")
            fairness_cost = _paper_fairness_cost(trace)
            cost_ratio = realized / optimal if not math.isnan(realized) and optimal > 0 else math.nan
            failed_meetings = max(0.0, 5.0 - meetings) if not math.isnan(meetings) else math.nan
            full_coord = task_success >= 1.0
            costly_full_coordination = 1.0 if full_coord and not math.isnan(coordination_cost) and coordination_cost > 0 else 0.0
            uniform_vps = vps.get(game_id, {}).get("uniform", {})
            cost_vps = vps.get(game_id, {}).get("cost", {})
            rows.append(
                {
                    "model": model,
                    "run": run_name,
                    "setting": setting,
                    "difficulty": difficulty,
                    "game_id": game_id,
                    "trace_path": str(path),
                    "headline_score": _float(metrics.get("headline_score")),
                    "headline_score_weighted": _float(metrics.get("headline_score_weighted")),
                    "success_score": _float(metrics.get("success_score")),
                    "cost_score": _float(metrics.get("cost_score")),
                    "privacy_score": _float(metrics.get("privacy_score")),
                    "efficiency_score": _float(metrics.get("efficiency_score")),
                    "communication_score": _float(metrics.get("communication_score")),
                    "excess_cost_score": _float(metrics.get("excess_cost_score")),
                    "normalized_communication_cost": _float(metrics.get("normalized_communication_cost")),
                    "normalized_privacy_leakage": _float(metrics.get("normalized_privacy_leakage")),
                    "normalized_excess_cost": _float(metrics.get("normalized_excess_cost")),
                    "coordination_rate": task_success,
                    "task_success": task_success,
                    "coordination_failure_rate": 1.0 - task_success,
                    "meetings_scheduled": meetings,
                    "failed_meetings": failed_meetings,
                    "slot_conflict_rate": _float(metrics.get("slot_conflict_rate")),
                    "realized_cost": realized,
                    "optimal_cost": optimal,
                    "excess_cost": coordination_cost,
                    "coordination_cost": coordination_cost,
                    "legacy_full_task_excess_cost": legacy_full_task_excess,
                    "cost_regret": _float(metrics.get("cost_regret")),
                    "greedy_normalized_regret": _float(metrics.get("greedy_normalized_regret")),
                    "cost_ratio": cost_ratio,
                    "fallback_displacement_cost": _float(metrics.get("fallback_displacement_cost")),
                    "messages_per_meeting": communication_efficiency,
                    "communication_efficiency": communication_efficiency,
                    "legacy_game_pooled_messages_per_meeting": _float(metrics.get("efficiency")),
                    "total_dms_sent": total_dms,
                    "total_cheap_talk_messages": total_cheap_talk,
                    "total_dm_chars": _float(metrics.get("total_dm_chars")),
                    "avg_dm_chars": _float(metrics.get("avg_dm_chars")),
                    "dm_chars_per_meeting": _float(metrics.get("dm_chars_per_meeting")),
                    "fairness": fairness_cost,
                    "fairness_cost": fairness_cost,
                    "legacy_fairness_ratio": _float(metrics.get("fairness")),
                    "per_agent_cost_range": _per_agent_range(trace),
                    "per_agent_cost_max": _per_agent_max(trace),
                    "costly_full_coordination": costly_full_coordination,
                    "uniform_vps_loss_mean": uniform_vps.get("vps_loss_mean", math.nan),
                    "uniform_vps_loss_per_weight": uniform_vps.get("vps_loss_per_weight", math.nan),
                    "cost_weighted_vps_loss_mean": cost_vps.get("vps_loss_mean", math.nan),
                    "cost_weighted_vps_loss_per_weight": cost_vps.get("vps_loss_per_weight", math.nan),
                    "vps_observation_count": uniform_vps.get("observation_count", math.nan),
                    "privacy_leak_count": _float(metrics.get("privacy_leak_count")),
                }
            )
    return rows


SUMMARY_METRICS = [
    "headline_score",
    "headline_score_weighted",
    "success_score",
    "cost_score",
    "privacy_score",
    "efficiency_score",
    "communication_score",
    "excess_cost_score",
    "normalized_communication_cost",
    "normalized_privacy_leakage",
    "normalized_excess_cost",
    "coordination_rate",
    "task_success",
    "coordination_failure_rate",
    "meetings_scheduled",
    "failed_meetings",
    "excess_cost",
    "coordination_cost",
    "legacy_full_task_excess_cost",
    "cost_regret",
    "greedy_normalized_regret",
    "cost_ratio",
    "messages_per_meeting",
    "communication_efficiency",
    "legacy_game_pooled_messages_per_meeting",
    "total_dms_sent",
    "total_cheap_talk_messages",
    "total_dm_chars",
    "avg_dm_chars",
    "dm_chars_per_meeting",
    "fairness",
    "fairness_cost",
    "legacy_fairness_ratio",
    "per_agent_cost_range",
    "per_agent_cost_max",
    "slot_conflict_rate",
    "fallback_displacement_cost",
    "costly_full_coordination",
    "uniform_vps_loss_mean",
    "cost_weighted_vps_loss_mean",
    "uniform_vps_loss_per_weight",
    "cost_weighted_vps_loss_per_weight",
    "privacy_leak_count",
]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["setting"], row["difficulty"])].append(row)

    out: list[dict[str, Any]] = []
    for (model, setting, difficulty), items in sorted(
        groups.items(), key=lambda item: (item[0][0], SETTING_ORDER[item[0][1]], DIFFICULTY_ORDER[item[0][2]])
    ):
        summary: dict[str, Any] = {
            "model": model,
            "setting": setting,
            "difficulty": difficulty,
            "trace_count": len(items),
        }
        for metric in SUMMARY_METRICS:
            values = [_float(item.get(metric)) for item in items]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_median"] = _median(values)
            summary[f"{metric}_p90"] = _p90(values)
            summary[f"{metric}_ci95"] = _ci95(values)
        out.append(summary)
    return out


def rank_summaries(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(row["setting"], row["difficulty"])].append(row)
    rankings: list[dict[str, Any]] = []
    for (setting, difficulty), rows in sorted(groups.items(), key=lambda item: (SETTING_ORDER[item[0][0]], DIFFICULTY_ORDER[item[0][1]])):
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["coordination_rate_mean"],
                row["excess_cost_mean"],
                row["messages_per_meeting_mean"],
                row["fairness_mean"],
            ),
        )
        for idx, row in enumerate(ranked, start=1):
            rankings.append(
                {
                    "setting": setting,
                    "difficulty": difficulty,
                    "rank": idx,
                    "model": row["model"],
                    "coordination_rate_mean": row["coordination_rate_mean"],
                    "excess_cost_mean": row["excess_cost_mean"],
                    "messages_per_meeting_mean": row["messages_per_meeting_mean"],
                    "fairness_mean": row["fairness_mean"],
                    "uniform_vps_loss_mean": row["uniform_vps_loss_mean_mean"],
                }
            )
    return rankings


def rank_stability(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["setting"], row["difficulty"], row["model"]): int(row["rank"]) for row in rankings}
    models = sorted({row["model"] for row in rankings})
    out: list[dict[str, Any]] = []
    for setting in SETTING_ORDER:
        for model in models:
            easy = by_key.get((setting, "easy", model))
            medium = by_key.get((setting, "medium", model))
            hard = by_key.get((setting, "hard", model))
            if easy is None and medium is None and hard is None:
                continue
            out.append(
                {
                    "setting": setting,
                    "model": model,
                    "easy_rank": easy or "",
                    "medium_rank": medium or "",
                    "hard_rank": hard or "",
                    "hard_minus_easy_rank": (hard - easy) if easy is not None and hard is not None else "",
                }
            )
    return out


def difficulty_interactions(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["setting"], row["difficulty"]): row for row in summary_rows}
    models = sorted({row["model"] for row in summary_rows})
    out: list[dict[str, Any]] = []
    metrics = [
        ("coordination_rate_mean", "coord_delta_varied_minus_uniform"),
        ("meetings_scheduled_mean", "meetings_delta_varied_minus_uniform"),
        ("excess_cost_mean", "excess_delta_varied_minus_uniform"),
        ("messages_per_meeting_mean", "communication_efficiency_delta_varied_minus_uniform"),
        ("fairness_mean", "fairness_delta_varied_minus_uniform"),
        ("uniform_vps_loss_mean_mean", "uniform_vps_delta_varied_minus_uniform"),
    ]
    for model in models:
        for difficulty in DIFFICULTY_ORDER:
            uniform = by_key.get((model, "uniform", difficulty))
            varied = by_key.get((model, "varied", difficulty))
            if not uniform or not varied:
                continue
            row: dict[str, Any] = {"model": model, "difficulty": difficulty}
            for metric, out_name in metrics:
                row[out_name] = varied[metric] - uniform[metric]
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = _fmt(value, 2)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Difficulty Bucket Metrics")
    lines.append("")
    lines.append("Grouped by model, setting (`uniform`/`varied`), and difficulty bucket (`easy`/`medium`/`hard`).")
    lines.append("Ranking sort: task success descending, coordination cost ascending, communication efficiency ascending, fairness cost ascending.")
    lines.append("")

    compact: list[dict[str, Any]] = []
    for row in summary_rows:
        compact.append(
            {
                "model": row["model"],
                "setting": row["setting"],
                "difficulty": row["difficulty"],
                "n": row["trace_count"],
                "coord_%": 100 * row["coordination_rate_mean"],
                "meetings": row["meetings_scheduled_mean"],
                "excess_mean": row["excess_cost_mean"],
                "excess_p90": row["excess_cost_p90"],
                "communication_efficiency": row["messages_per_meeting_mean"],
                "fairness_cost": row["fairness_mean"],
                "uniform_vps": row["uniform_vps_loss_mean_mean"],
            }
        )
    for setting in SETTING_ORDER:
        lines.append(f"## {setting.title()}")
        lines.append("")
        rows = [row for row in compact if row["setting"] == setting]
        rows.sort(key=lambda row: (DIFFICULTY_ORDER[row["difficulty"]], row["model"]))
        lines.append(_markdown_table(rows, ["model", "difficulty", "n", "coord_%", "meetings", "excess_mean", "excess_p90", "communication_efficiency", "fairness_cost", "uniform_vps"]))
        lines.append("")
        lines.append("### Rankings")
        rank_rows = [row for row in rankings if row["setting"] == setting]
        rank_rows.sort(key=lambda row: (DIFFICULTY_ORDER[row["difficulty"]], row["rank"]))
        lines.append(_markdown_table(rank_rows, ["difficulty", "rank", "model", "coordination_rate_mean", "excess_cost_mean", "messages_per_meeting_mean", "fairness_mean"]))
        lines.append("")

    lines.append("## Rank Stability")
    lines.append("")
    lines.append(_markdown_table(stability, ["setting", "model", "easy_rank", "medium_rank", "hard_rank", "hard_minus_easy_rank"]))
    lines.append("")
    lines.append("## Varied Minus Uniform Interaction")
    lines.append("")
    interaction_rows = []
    for row in interactions:
        interaction_rows.append(
            {
                "model": row["model"],
                "difficulty": row["difficulty"],
                "coord_delta": 100 * row["coord_delta_varied_minus_uniform"],
                "excess_delta": row["excess_delta_varied_minus_uniform"],
                "communication_delta": row["communication_efficiency_delta_varied_minus_uniform"],
                "fairness_delta": row["fairness_delta_varied_minus_uniform"],
                "vps_delta": row["uniform_vps_delta_varied_minus_uniform"],
            }
        )
    lines.append(_markdown_table(interaction_rows, ["model", "difficulty", "coord_delta", "excess_delta", "communication_delta", "fairness_delta", "vps_delta"]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-traces", default=str(_repo_root() / "shared-traces"))
    parser.add_argument("--out-dir", default=str(_calendar_root() / "analysis" / "outputs" / "difficulty_bucket_metrics"))
    args = parser.parse_args()

    shared_traces = Path(args.shared_traces)
    out_dir = Path(args.out_dir)
    rows = collect_trace_rows(shared_traces, MODEL_RUNS)
    summary_rows = summarize(rows)
    rankings = rank_summaries(summary_rows)
    stability = rank_stability(rankings)
    interactions = difficulty_interactions(summary_rows)

    write_csv(out_dir / "trace_metrics.csv", rows)
    write_csv(out_dir / "summary_by_model_setting_difficulty.csv", summary_rows)
    write_csv(out_dir / "rankings_by_setting_difficulty.csv", rankings)
    write_csv(out_dir / "rank_stability.csv", stability)
    write_csv(out_dir / "varied_minus_uniform_by_model_difficulty.csv", interactions)
    write_report(out_dir / "summary.md", summary_rows, rankings, stability, interactions)

    print(f"trace rows: {len(rows)}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
