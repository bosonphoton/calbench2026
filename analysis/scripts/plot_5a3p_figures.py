"""Generate 5-agent/3-participant calendar benchmark figures.

Examples:
    cd games/calendar
    uv run python analysis/scripts/plot_5a3p_figures.py \
      --coord-traces results/uniform_full_gemini3flash results/varied_5a3p_b041_gpt55_medium \
      --redteam-traces results/redteam_c006_uniform_5a3p_c020 \
      --out-dir analysis/figures/5a3p
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:  # pragma: no cover - environment/dependency guard
    raise SystemExit(
        "matplotlib is required for plotting. Install/sync calendar dependencies "
        "with `uv sync` from games/calendar, then rerun this script."
    ) from exc


SeriesKey = tuple[str, bool]

BASELINE_COLORS = {
    "DSM Private (baseline)": "#000000",
    "DSM Welfare (baseline)": "#7f7f7f",
    "IMAP (baseline)": "#d62728",
    "SD (baseline)": "#8c564b",
}

BASELINE_MARKERS = {
    "DSM Private (baseline)": "D",
    "DSM Welfare (baseline)": "s",
    "IMAP (baseline)": "x",
    "SD (baseline)": "^",
}


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    if root_candidate.exists() or not candidate.exists():
        return root_candidate
    return candidate


def _trace_paths(inputs: Iterable[str], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = _resolve(raw, root=root)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif any(char in raw for char in "*?[]"):
            paths.extend(sorted(root.glob(raw)))
        else:
            paths.append(path)
    return [
        path
        for path in paths
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
    ]


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_5a3p_trace(trace: dict[str, Any]) -> bool:
    config = trace.get("config") or {}
    if int(config.get("num_agents") or 0) != 5:
        return False
    num_participants = config.get("num_participants")
    return num_participants is None or int(num_participants) == 3


def _is_redteam_trace(path: Path, trace: dict[str, Any]) -> bool:
    config = trace.get("config") or {}
    haystack = " ".join(
        [
            str(path).casefold(),
            str(trace.get("experiment_name") or "").casefold(),
            str(config.get("experiment_name") or "").casefold(),
            str(config.get("task_path") or "").casefold(),
        ]
    )
    return "redteam" in haystack or "red_team" in haystack


def _is_baseline_trace(path: Path, trace: dict[str, Any]) -> bool:
    config = trace.get("config") or {}
    agent_types = {
        str(agent.get("type") or "").casefold()
        for agent in config.get("agents") or []
        if isinstance(agent, dict)
    }
    if agent_types & {"dsm", "imap", "sd"}:
        return True
    haystack = " ".join(
        [
            str(path).casefold(),
            str(trace.get("experiment_name") or "").casefold(),
            str(config.get("experiment_name") or "").casefold(),
            str(config.get("task_path") or "").casefold(),
        ]
    )
    baseline_markers = [
        "baseline",
        "dsm-private",
        "_dsm",
        "dsm_welfare",
        "dsm-welfare",
        "imap-redteam",
        "_imap",
        "/local-sd/",
        "_sd",
    ]
    return any(marker in haystack for marker in baseline_markers)


def _dataset_case(trace: dict[str, Any]) -> str:
    config = trace.get("config") or {}
    task_path = str(config.get("task_path") or "").casefold()
    if "varied" in task_path:
        return "varied"
    if "uniform" in task_path:
        return "uniform"
    experiment = str(trace.get("experiment_name") or config.get("experiment_name") or "").casefold()
    if "varied" in experiment:
        return "varied"
    if "uniform" in experiment:
        return "uniform"
    return "unknown"


def _model_label(trace: dict[str, Any]) -> str:
    config = trace.get("config") or {}
    models = [
        str(agent.get("model"))
        for agent in config.get("agents") or []
        if isinstance(agent, dict) and agent.get("model")
    ]
    unique = sorted(set(models))
    if len(unique) == 1:
        return _clean_model_label(unique[0])
    if unique:
        return "mixed"
    return str(trace.get("experiment_name") or config.get("experiment_name") or "unknown")


def _clean_model_label(model: str) -> str:
    if "/models/" in model:
        return model.rsplit("/models/", maxsplit=1)[-1]
    if "/" in model:
        return model.rsplit("/", maxsplit=1)[-1]
    return model


def _baseline_label(path: Path, trace: dict[str, Any]) -> str:
    config = trace.get("config") or {}
    haystack = " ".join(
        [
            str(path).casefold(),
            str(trace.get("experiment_name") or "").casefold(),
            str(config.get("experiment_name") or "").casefold(),
        ]
    )
    if "dsm-private" in haystack or "private_dsm" in haystack:
        return "DSM Private"
    if "dsm-welfare" in haystack or "dsm_welfare" in haystack or "paper_dsm_social" in haystack:
        return "DSM Welfare"
    if "imap" in haystack:
        return "IMAP"
    if "sd" in haystack:
        return "SD"

    agent_types = sorted(
        {
            str(agent.get("type") or "").casefold()
            for agent in config.get("agents") or []
            if isinstance(agent, dict) and agent.get("type")
        }
    )
    if len(agent_types) == 1 and agent_types[0] in {"dsm", "imap", "sd"}:
        return agent_types[0].upper()

    return str(trace.get("experiment_name") or config.get("experiment_name") or "Baseline")


def _series_label(path: Path, trace: dict[str, Any]) -> str:
    if _is_baseline_trace(path, trace):
        return f"{_baseline_label(path, trace)} (baseline)"
    return _model_label(trace)


def _line_style(is_baseline: bool) -> str:
    return "--" if is_baseline else "-"


def _series_sort_key(key: SeriesKey) -> tuple[int, int, str]:
    label, is_baseline = key
    # DSM Private overlaps IMAP in current runs; draw it last so it remains visible.
    baseline_order = {
        "DSM Welfare (baseline)": 0,
        "IMAP (baseline)": 1,
        "SD (baseline)": 2,
        "DSM Private (baseline)": 3,
    }
    return (1 if is_baseline else 0, baseline_order.get(label, 0), label)


def _series_color(key: SeriesKey, colors: dict[SeriesKey, Any]) -> Any:
    label, _is_baseline = key
    return BASELINE_COLORS.get(label, colors.get(key))


def _marker_style(label: str, is_baseline: bool) -> dict[str, Any]:
    if not is_baseline:
        return {"marker": "o"}
    style: dict[str, Any] = {
        "marker": BASELINE_MARKERS.get(label, "o"),
        "markersize": 6,
        "markeredgewidth": 1.3,
    }
    if label == "DSM Private (baseline)":
        style["markerfacecolor"] = "white"
    return style


def _mean_and_ci(values: list[int | float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    ci = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return mean, ci


def _binomial_rate_ci_percent(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    rate = successes / total
    ci = 1.96 * math.sqrt(rate * (1.0 - rate) / total)
    lower = max(0.0, rate - ci) * 100.0
    upper = min(1.0, rate + ci) * 100.0
    return rate * 100.0, lower, upper


def _annotate_baseline_overlap(ax: Any) -> None:
    ax.text(
        0.02,
        0.96,
        "DSM Private overlaps IMAP",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )


def _meeting_index(event: dict[str, Any]) -> int | None:
    data = event.get("data") or {}
    if data.get("meeting_id") is not None:
        return int(data["meeting_id"])
    if data.get("round") is not None:
        return int(data["round"]) + 1
    return None


MESSAGE_EVENT_TYPES = {"dm_sent", "groupchat_sent", "participant_groupchat_sent", "all_groupchat_sent"}


def _cheap_talk_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in trace.get("events", []) if event.get("type") in MESSAGE_EVENT_TYPES]


def _dm_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in trace.get("events", []) if event.get("type") == "dm_sent"]


def _messages_per_meeting(trace: dict[str, Any]) -> float | None:
    values = [
        float(row["messages_per_scheduled_meeting"])
        for row in (trace.get("final_state") or {}).get("contribution_scores") or []
        if isinstance(row, dict) and row.get("messages_per_scheduled_meeting") is not None
    ]
    if values:
        return sum(values) / len(values)
    metrics = trace.get("metrics") or {}
    total_messages = metrics.get("total_cheap_talk_messages")
    if total_messages is None:
        total_messages = len(_cheap_talk_events(trace))
    meetings = metrics.get("meetings_scheduled")
    if meetings is None:
        meetings = len((trace.get("final_state") or {}).get("round_outcomes") or [])
    if not meetings:
        return None
    return float(total_messages) / float(meetings)


def _excess_cost(trace: dict[str, Any]) -> float | None:
    metrics = trace.get("metrics") or {}
    values = metrics.get("per_agent_scheduled_only_excess_burden") or []
    clean = [float(value) for value in values if value is not None]
    if clean:
        return sum(clean) / len(clean)
    scheduled_total = metrics.get("total_scheduled_only_excess_burden")
    if scheduled_total is not None:
        num_agents = float((trace.get("config") or {}).get("num_agents") or len(clean) or 1)
        return float(scheduled_total) / max(num_agents, 1.0)
    realized = metrics.get("realized_cost")
    optimal = metrics.get("scheduled_only_optimal_cost")
    if optimal is None:
        optimal = metrics.get("optimal_cost")
    if realized is None or optimal is None:
        return None
    return float(realized) - float(optimal)


def plot_average_dms_by_meeting(traces: list[tuple[Path, dict[str, Any]]], out_path: Path) -> None:
    counts: dict[str, dict[SeriesKey, dict[int, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path, trace in traces:
        case = _dataset_case(trace)
        if case not in {"uniform", "varied"}:
            continue
        key = (_series_label(path, trace), _is_baseline_trace(path, trace))
        per_meeting = {idx: 0 for idx in range(1, 6)}
        for event in _cheap_talk_events(trace):
            idx = _meeting_index(event)
            if idx in per_meeting:
                per_meeting[idx] += 1
        for idx, count in per_meeting.items():
            counts[case][key][idx].append(count)

    series = sorted({key for by_series in counts.values() for key in by_series}, key=_series_sort_key)
    colors = dict(zip(series, plt.get_cmap("tab20").colors, strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, case in zip(axes, ["uniform", "varied"], strict=True):
        for key in series:
            if key not in counts[case]:
                continue
            label, is_baseline = key
            xs = sorted(counts[case][key])
            stats = [_mean_and_ci(counts[case][key][idx]) for idx in xs]
            ys = [mean for mean, _ci in stats]
            yerr = [ci for _mean, ci in stats]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                linewidth=2,
                linestyle=_line_style(is_baseline),
                label=label,
                color=_series_color(key, colors),
                capsize=3,
                elinewidth=1.1,
                zorder=4 if label == "DSM Private (baseline)" else 3 if is_baseline else 2,
                **_marker_style(label, is_baseline),
            )
        ax.set_title(case.title())
        ax.set_xlabel("Meeting index")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, axis="y", alpha=0.3)
        _annotate_baseline_overlap(ax)
    axes[0].set_ylabel("Average cheap-talk messages sent")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        next_handles, next_labels = ax.get_legend_handles_labels()
        handles.extend(next_handles)
        labels.extend(next_labels)
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(4, max(1, len(by_label))), frameon=False)
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_communication_efficiency(traces: list[tuple[Path, dict[str, Any]]], out_path: Path) -> None:
    points: dict[str, dict[SeriesKey, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for path, trace in traces:
        if _is_baseline_trace(path, trace):
            continue
        case = _dataset_case(trace)
        if case not in {"uniform", "varied"}:
            continue
        messages_per_meeting = _messages_per_meeting(trace)
        excess_cost = _excess_cost(trace)
        if messages_per_meeting is None or excess_cost is None:
            continue
        key = (_series_label(path, trace), _is_baseline_trace(path, trace))
        points[case][key].append((messages_per_meeting, excess_cost))

    series = sorted({key for by_series in points.values() for key in by_series}, key=_series_sort_key)
    colors = dict(zip(series, plt.get_cmap("tab20").colors, strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    for ax, case in zip(axes, ["uniform", "varied"], strict=True):
        plotted_y: list[float] = []
        for key in series:
            values = points[case].get(key)
            if not values:
                continue
            label, is_baseline = key
            xs = [value[0] for value in values]
            ys = [value[1] for value in values]
            mean_x, _ci_x = _mean_and_ci(xs)
            mean_y, _ci_y = _mean_and_ci(ys)
            plotted_y.append(mean_y)
            marker_style = _marker_style(label, is_baseline)
            scatter_kwargs = {
                "label": label,
                "c": [_series_color(key, colors)],
                "s": 130 if is_baseline else 110,
                "linewidths": 1.7 if is_baseline else 1.1,
                "alpha": 0.9,
                "zorder": 4 if label == "DSM Private (baseline)" else 3 if is_baseline else 2,
                "marker": marker_style.get("marker", "o"),
            }
            if marker_style.get("marker") != "x":
                scatter_kwargs["edgecolors"] = _series_color(key, colors)
            ax.scatter(
                mean_x,
                mean_y,
                **scatter_kwargs,
            )
            if label == "DSM Private (baseline)":
                ax.scatter(
                    mean_x,
                    mean_y,
                    c=["white"],
                    s=130,
                    linewidths=1.7,
                    edgecolors=_series_color(key, colors),
                    zorder=5,
                    marker=marker_style.get("marker", "D"),
                )
        ax.set_title(case.title())
        ax.set_xlabel("Messages per meeting")
        ax.grid(True, axis="y", alpha=0.3)
        if plotted_y:
            lower = min(0.0, min(plotted_y))
            upper = max(plotted_y)
            padding = max(0.5, (upper - lower) * 0.12)
            ax.set_ylim(lower - padding, upper + padding)
        if case == "varied":
            positive_y = [value for value in plotted_y if value > 0]
            if positive_y:
                ax.set_yscale("log")
                ax.set_ylim(min(positive_y) * 0.8, max(positive_y) * 1.25)
    axes[0].set_ylabel("Excess cost")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        next_handles, next_labels = ax.get_legend_handles_labels()
        handles.extend(next_handles)
        legend_labels.extend(next_labels)
    by_label = dict(zip(legend_labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(4, max(1, len(by_label))), frameon=False)
    fig.subplots_adjust(bottom=0.26, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _speaker_orders(trace: dict[str, Any]) -> dict[int, list[int]]:
    orders: dict[int, list[int]] = {}
    for event in trace.get("events", []):
        if event.get("type") != "round_start":
            continue
        data = event.get("data") or {}
        if data.get("round") is None:
            continue
        order = data.get("speaker_order") or []
        if len(order) >= 3:
            orders[int(data["round"])] = [int(agent_id) for agent_id in order[:3]]
    return orders


def plot_average_dms_by_speaker_position(traces: list[tuple[Path, dict[str, Any]]], out_path: Path) -> None:
    counts: dict[str, dict[SeriesKey, dict[int, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path, trace in traces:
        case = _dataset_case(trace)
        if case not in {"uniform", "varied"}:
            continue
        key = (_series_label(path, trace), _is_baseline_trace(path, trace))
        orders = _speaker_orders(trace)
        per_round_position_counts: dict[tuple[int, int], int] = defaultdict(int)
        for event in _cheap_talk_events(trace):
            data = event.get("data") or {}
            round_idx = data.get("round")
            sender = data.get("agent_id", data.get("from_agent"))
            if round_idx is None or sender is None:
                continue
            order = orders.get(int(round_idx), [])
            if int(sender) not in order:
                continue
            position = order.index(int(sender)) + 1
            if position <= 3:
                per_round_position_counts[(int(round_idx), position)] += 1
        for round_idx, order in orders.items():
            for position in range(1, min(len(order), 3) + 1):
                counts[case][key][position].append(per_round_position_counts[(round_idx, position)])

    labels = {1: "First", 2: "Second", 3: "Third"}
    series = sorted({key for by_series in counts.values() for key in by_series}, key=_series_sort_key)
    colors = dict(zip(series, plt.get_cmap("tab20").colors, strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, case in zip(axes, ["uniform", "varied"], strict=True):
        for key in series:
            if key not in counts[case]:
                continue
            label, is_baseline = key
            xs = [1, 2, 3]
            stats = [_mean_and_ci(counts[case][key][position]) for position in xs]
            ys = [mean for mean, _ci in stats]
            yerr = [ci for _mean, ci in stats]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                linewidth=2,
                linestyle=_line_style(is_baseline),
                label=label,
                color=_series_color(key, colors),
                capsize=3,
                elinewidth=1.1,
                zorder=4 if label == "DSM Private (baseline)" else 3 if is_baseline else 2,
                **_marker_style(label, is_baseline),
            )
        ax.set_title(case.title())
        ax.set_xlabel("Speaker position")
        ax.set_xticks([1, 2, 3], [labels[i] for i in [1, 2, 3]])
        ax.grid(True, axis="y", alpha=0.3)
        _annotate_baseline_overlap(ax)
    axes[0].set_ylabel("Average cheap-talk messages sent")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        next_handles, next_labels = ax.get_legend_handles_labels()
        handles.extend(next_handles)
        legend_labels.extend(next_labels)
    by_label = dict(zip(legend_labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(4, max(1, len(by_label))), frameon=False)
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_privacy_module(root: Path):
    script_path = root / "analysis" / "scripts" / "rq1_privacy_leakage_prevalence.py"
    spec = importlib.util.spec_from_file_location("rq1_privacy_leakage_prevalence", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plot_privacy_leakage_ratio(
    traces: list[tuple[Path, dict[str, Any]]],
    *,
    root: Path,
    out_path: Path,
) -> None:
    rq1 = _load_privacy_module(root)
    leakage_terms, public_terms = rq1.load_match_terms(
        root / rq1.DEFAULT_ERRAND_BANK,
        root / rq1.DEFAULT_MEETING_BANK,
    )
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"dms": 0, "leaks": 0})
    for path, trace in traces:
        model = _model_label(trace)
        rows = rq1._message_rows_for_trace(path, leakage_terms=leakage_terms, public_terms=public_terms)
        regular_rows = [row for row in rows if not row["from_is_adversarial"]]
        totals[model]["dms"] += len(regular_rows)
        totals[model]["leaks"] += sum(1 for row in regular_rows if row["privacy_leakage"])

    models = sorted(totals)
    ratio_stats = [
        _binomial_rate_ci_percent(totals[model]["leaks"], totals[model]["dms"])
        for model in models
    ]
    ratios = [rate for rate, _lower, _upper in ratio_stats]
    yerr = [
        [rate - lower for rate, lower, _upper in ratio_stats],
        [upper - rate for rate, _lower, upper in ratio_stats],
    ]
    fig_width = max(10.0, 2.4 * len(models))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    bars = ax.bar(
        models,
        ratios,
        color="#1f77b4",
        edgecolor="none",
        width=0.62,
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1.1},
    )
    ax.set_title("Private Information Leakage Induced by Adversarial Agents", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Private Information Leakage Rate (%)")
    ax.set_ylim(0, max(ratios + [0.25]) * 1.45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.2f}%"))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, model in zip(bars, models, strict=True):
        leaks = totals[model]["leaks"]
        dms = totals[model]["dms"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.025,
            f"{leaks}/{dms}\n(privacy leaked DMs / total)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    fig.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    root = _calendar_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coord-traces",
        nargs="+",
        default=[],
        help="Trace files/directories for non-red-team uniform/varied coordination figures.",
    )
    parser.add_argument(
        "--redteam-traces",
        nargs="+",
        default=[],
        help="Red-team trace files/directories for privacy leakage figure.",
    )
    parser.add_argument("--out-dir", default="analysis/figures/5a3p")
    parser.add_argument("--include-non-5a3p", action="store_true")
    parser.add_argument(
        "--include-redteam-in-coord",
        action="store_true",
        help="Include red-team traces in Figures 1 and 2 instead of filtering them out.",
    )
    parser.add_argument(
        "--include-non-redteam-privacy",
        action="store_true",
        help="Include non-red-team traces in Figure 3 instead of filtering them out.",
    )
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir, root=root)
    coord_paths = _trace_paths(args.coord_traces, root=root)
    redteam_paths = _trace_paths(args.redteam_traces, root=root)
    coord_traces = [(path, _load_trace(path)) for path in coord_paths]
    redteam_traces = [(path, _load_trace(path)) for path in redteam_paths]
    if not args.include_non_5a3p:
        coord_traces = [(path, trace) for path, trace in coord_traces if _is_5a3p_trace(trace)]
        redteam_traces = [(path, trace) for path, trace in redteam_traces if _is_5a3p_trace(trace)]
    if not args.include_redteam_in_coord:
        coord_traces = [(path, trace) for path, trace in coord_traces if not _is_redteam_trace(path, trace)]
    if not args.include_non_redteam_privacy:
        redteam_traces = [(path, trace) for path, trace in redteam_traces if _is_redteam_trace(path, trace)]
    redteam_traces = [(path, trace) for path, trace in redteam_traces if not _is_baseline_trace(path, trace)]

    if coord_traces:
        plot_average_dms_by_meeting(coord_traces, out_dir / "figure1_avg_dms_by_meeting.png")
        plot_average_dms_by_speaker_position(coord_traces, out_dir / "figure2_avg_dms_by_speaker_position.png")
        plot_communication_efficiency(coord_traces, out_dir / "figure4_communication_efficiency.png")
    if redteam_traces:
        plot_privacy_leakage_ratio(redteam_traces, root=root, out_path=out_dir / "figure3_privacy_leakage_ratio.png")

    print(f"coordination traces: {len(coord_traces)}")
    print(f"red-team traces: {len(redteam_traces)}")
    print(f"wrote figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
