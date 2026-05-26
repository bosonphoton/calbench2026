"""Benchmark-lite 2D scatter: excess cost vs VPS privacy.

Produces two panels (uniform_full and varied_full) with local protocol baselines
and benchmarked model runs as points at (mean excess cost, mean VPS loss).
By default, the benchmark points are hydrated from the shared CalBench
leaderboard and filtered to exclude redteam/readteam and blocked-condition runs.

Usage::

    cd games/calendar
    uv run python analysis/scripts/benchmark_lite_protocol_scatter.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LEADERBOARD_URL = "https://calbench-traces.s3.amazonaws.com/calendar-traces/_leaderboard/leaderboard.json"
SETTING_FROM_LEADERBOARD = {
    "uniform": "uniform_full",
    "varied": "varied_full",
}
VPS_METRIC_FIELDS = {
    "uniform": "uniform_vps_loss_mean",
    "cost": "vps_loss_mean",
}


def _paper_excess_cost(trace: dict) -> float | None:
    metrics = trace.get("metrics") or {}
    values = metrics.get("per_agent_scheduled_only_excess_burden") or []
    clean = [float(value) for value in values if value is not None]
    if clean:
        return float(sum(clean) / len(clean))
    scheduled_total = metrics.get("total_scheduled_only_excess_burden")
    if scheduled_total is not None:
        num_agents = float((trace.get("config") or {}).get("num_agents") or 1)
        return float(scheduled_total) / max(num_agents, 1.0)
    realized = metrics.get("realized_cost")
    optimal = metrics.get("scheduled_only_optimal_cost", metrics.get("optimal_cost"))
    if realized is None or optimal is None:
        return None
    return float(realized) - float(optimal)


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_benchmark_lite(root: Path) -> dict[tuple[str, str], dict]:
    """Return {(client, task_id): row} from benchmark_lite per_game.csv."""
    path = root / "analysis/outputs/benchmark_lite_5a3p_baselines/per_game.csv"
    out: dict[tuple[str, str], dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["client"], row["task_id"])] = row
    return out


# Map (client_label, setting) -> VPS output dir name
VPS_DIRS: dict[tuple[str, str], str] = {
    ("IMAP", "uniform_full"): "rq5_vps_uniform_full_imap",
    ("IMAP", "varied_full"): "rq5_vps_varied_full_imap",
    ("DSM-privacy", "uniform_full"): "rq5_vps_uniform_full_private_dsm",
    ("DSM-privacy", "varied_full"): "rq5_vps_varied_full_private_dsm",
    ("DSM-welfare", "uniform_full"): "rq5_vps_uniform_full_paper_dsm_social",
    ("DSM-welfare", "varied_full"): "rq5_vps_varied_full_paper_dsm_social",
    ("SD", "uniform_full"): "rq5_vps_benchmark_lite_uniform_sd",
    ("SD", "varied_full"): "rq5_vps_benchmark_lite_varied_sd",
}


def _load_vps_for_tasks(
    vps_dir: Path, target_task_ids: set[str]
) -> dict[str, float]:
    """Return {task_id: vps_loss_mean} for benchmark-lite tasks in a VPS output dir."""
    game_summary = vps_dir / "game_summary.csv"
    if not game_summary.exists():
        return {}

    result: dict[str, float] = {}
    with open(game_summary) as f:
        for row in csv.DictReader(f):
            trace_path = Path(row["trace_path"])
            if not trace_path.exists():
                continue
            try:
                trace = json.loads(trace_path.read_text())
            except Exception:
                continue
            task_id = trace.get("config", {}).get("task_id", "")
            if task_id in target_task_ids:
                result[task_id] = float(row["vps_loss_mean"])
    return result


def _task_id_from_trace(trace: dict, trace_path: Path) -> str:
    config = trace.get("config") or {}
    task_id = config.get("task_id")
    if task_id:
        return str(task_id)
    experiment_run_id = str(config.get("experiment_run_id") or trace_path.parent.name)
    if "." in experiment_run_id:
        task_id = experiment_run_id.split(".", 1)[1]
    else:
        task_id = experiment_run_id
    return re.sub(r"\.\d+$", "", task_id)


def _load_model_rows_for_tasks(
    vps_dir: Path,
    *,
    client: str,
    setting: str,
    target_task_ids: set[str],
) -> list[dict]:
    game_summary = vps_dir / "game_summary.csv"
    if not game_summary.exists():
        return []

    rows: list[dict] = []
    with open(game_summary) as f:
        for row in csv.DictReader(f):
            trace_path = Path(row["trace_path"])
            if not trace_path.exists():
                continue
            try:
                trace = json.loads(trace_path.read_text())
            except Exception:
                continue
            task_id = _task_id_from_trace(trace, trace_path)
            if task_id not in target_task_ids:
                continue
            excess_cost = _paper_excess_cost(trace)
            if excess_cost is None:
                continue
            rows.append({
                "client": client,
                "series": "model",
                "setting": setting,
                "task_id": task_id,
                "excess_cost": excess_cost,
                "vps_loss_mean": float(row["vps_loss_mean"]),
            })
    return rows


MODEL_LABELS: dict[str, str] = {
    "gemini31pro": "Gemini 3.1 Pro",
    "gemini3flash": "Gemini 3 Flash",
    "gpt54_mini": "GPT-5.4 Mini",
    "claude_sonnet46": "Claude Sonnet 4.6",
    "deepseek_v4_pro": "DeepSeek V4 Pro",
    "qwen36_plus": "Qwen3.6 Plus",
}

PROTOCOL_VPS_KEYS = {
    "dsm",
    "imap",
    "paper_dsm_social",
    "private_dsm",
}


def _discover_model_vps_dirs(root: Path) -> dict[tuple[str, str], str]:
    outputs = root / "analysis/outputs"
    discovered: dict[tuple[str, str], str] = {}
    for path in sorted(outputs.glob("rq5_vps_*")):
        match = re.fullmatch(r"rq5_vps_(uniform_full|varied_full)_(.+)", path.name)
        if not match:
            continue
        setting, key = match.groups()
        if key in PROTOCOL_VPS_KEYS or key.endswith("_uniform_weights") or "expanded_language" in key:
            continue
        label = MODEL_LABELS.get(key, key.replace("_", " ").title())
        discovered[(label, setting)] = path.name
    return discovered


def collect_data(root: Path) -> dict[str, list[dict]]:
    """Return {setting: [row_dicts]} with excess_cost and vps for each run/task."""
    bl = _load_benchmark_lite(root)

    bench_tasks: dict[str, set[str]] = {"uniform_full": set(), "varied_full": set()}
    for (client, task_id), row in bl.items():
        bench_tasks[row["setting"]].add(task_id)

    rows_by_setting: dict[str, list[dict]] = {"uniform_full": [], "varied_full": []}

    for (client, setting), dir_name in VPS_DIRS.items():
        vps_dir = root / "analysis/outputs" / dir_name
        task_ids = bench_tasks[setting]
        vps_map = _load_vps_for_tasks(vps_dir, task_ids)

        for task_id in task_ids:
            bl_row = bl.get((client, task_id))
            if bl_row is None:
                continue
            vps_loss = vps_map.get(task_id)
            if vps_loss is None:
                continue
            rows_by_setting[setting].append({
                "client": client,
                "series": "protocol",
                "setting": setting,
                "task_id": task_id,
                "excess_cost": float(bl_row["excess_cost"]),
                "vps_loss_mean": vps_loss,
            })

    for (client, setting), dir_name in _discover_model_vps_dirs(root).items():
        rows_by_setting[setting].extend(
            _load_model_rows_for_tasks(
                root / "analysis/outputs" / dir_name,
                client=client,
                setting=setting,
                target_task_ids=bench_tasks[setting],
            )
        )

    return rows_by_setting


def _read_json_source(source: str) -> dict:
    if re.match(r"https?://", source):
        with urlopen(source, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _excluded_leaderboard_run(run: dict) -> bool:
    display = run.get("display") or {}
    blocked_condition = str(run.get("blocked_condition") or "").lower()
    kind = str(display.get("kind") or "").lower()
    label = str(display.get("label") or "").lower()
    run_id = str(run.get("run_id") or "").lower()
    text = f"{kind} {label} {run_id}"
    if blocked_condition != "unblocked":
        return True
    return any(token in text for token in ("redteam", "readteam"))


def _leaderboard_series(kind: str) -> str:
    return "protocol" if kind == "baseline" else "model"


def collect_data_from_leaderboard(leaderboard: dict, *, vps_metric: str = "uniform") -> dict[str, list[dict]]:
    vps_field = VPS_METRIC_FIELDS[vps_metric]
    rows_by_setting: dict[str, list[dict]] = {"uniform_full": [], "varied_full": []}
    for run in leaderboard.get("runs") or []:
        if _excluded_leaderboard_run(run):
            continue
        display = run.get("display") or {}
        kind = str(display.get("kind") or "")
        if kind not in {"baseline", "model"}:
            continue
        client = str(display.get("label") or run.get("run_id") or "unknown")
        for leaderboard_setting, setting in SETTING_FROM_LEADERBOARD.items():
            metrics = (run.get("by_setting") or {}).get(leaderboard_setting) or {}
            excess_cost = metrics.get("excess_cost_mean")
            vps_loss = metrics.get(vps_field)
            if vps_loss is None and vps_metric == "uniform":
                vps_loss = metrics.get("vps_loss_mean")
            if excess_cost is None or vps_loss is None:
                continue
            rows_by_setting[setting].append({
                "client": client,
                "series": _leaderboard_series(kind),
                "setting": setting,
                "task_id": "leaderboard_aggregate",
                "n": int(metrics.get("trace_count") or 0),
                "excess_cost": float(excess_cost),
                "excess_cost_median": (
                    float(metrics["excess_cost_median"])
                    if metrics.get("excess_cost_median") is not None
                    else None
                ),
                "vps_loss_mean": float(vps_loss),
            })
    return rows_by_setting


def collect_data_with_leaderboard(
    root: Path,
    *,
    leaderboard_source: str | None,
    cache_path: Path,
    vps_metric: str,
) -> tuple[dict[str, list[dict]], dict | None, str]:
    source = leaderboard_source or os.environ.get("CALBENCH_LEADERBOARD") or LEADERBOARD_URL
    try:
        leaderboard = _read_json_source(source)
    except (OSError, URLError, json.JSONDecodeError) as exc:
        if cache_path.exists():
            leaderboard = json.loads(cache_path.read_text(encoding="utf-8"))
            return collect_data_from_leaderboard(leaderboard, vps_metric=vps_metric), leaderboard, f"cached leaderboard ({cache_path})"
        print(f"Warning: could not load leaderboard from {source}: {exc}")
        print("Falling back to local analysis outputs.")
        return collect_data(root), None, "local analysis outputs"

    cache_path.write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return collect_data_from_leaderboard(leaderboard, vps_metric=vps_metric), leaderboard, source


PROTOCOL_STYLE: dict[str, dict] = {
    "SD":          {"color": "#9467bd", "marker": "D", "label": "SD", "series": "protocol"},
    "IMAP":        {"color": "#1f77b4", "marker": "o", "label": "IMAP", "series": "protocol"},
    "DSM-private": {"color": "#2ca02c", "marker": "s", "label": "DSM-private", "series": "protocol"},
    "DSM-privacy": {"color": "#2ca02c", "marker": "s", "label": "DSM-privacy", "series": "protocol"},
    "DSM-welfare": {"color": "#d62728", "marker": "^", "label": "DSM-welfare", "series": "protocol"},
}

MODEL_STYLE: dict[str, dict] = {
    "Gemini 3.1 Pro": {"color": "#111827", "marker": "X", "label": "Gemini 3.1 Pro", "series": "model"},
    "Gemini 3 Flash": {"color": "#6b7280", "marker": "P", "label": "Gemini 3 Flash", "series": "model"},
    "GPT-5.4 Mini": {"color": "#f59e0b", "marker": "*", "label": "GPT-5.4 Mini", "series": "model"},
    "Claude Sonnet 4.6": {"color": "#8b5cf6", "marker": "v", "label": "Claude Sonnet 4.6", "series": "model"},
    "DeepSeek V4 Pro": {"color": "#0891b2", "marker": "<", "label": "DeepSeek V4 Pro", "series": "model"},
    "Qwen3.6 Plus": {"color": "#db2777", "marker": ">", "label": "Qwen3.6 Plus", "series": "model"},
    "Llama 4 Maverick": {"color": "#65a30d", "marker": "h", "label": "Llama 4 Maverick", "series": "model"},
}

SETTING_LABELS = {
    "uniform_full": "Uniform costs",
    "varied_full": "Varied costs",
}

MODEL_SHORT_LABELS = {
    "Gemini 3.1 Pro": "Gemini Pro",
    "Gemini 3 Flash": "Gemini Flash",
    "GPT-5.4 Mini": "GPT-5.4 Mini",
    "Claude Sonnet 4.6": "Claude",
    "DeepSeek V4 Pro": "DeepSeek",
    "Qwen3.6 Plus": "Qwen",
    "Llama 4 Maverick": "Llama",
}


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _stderr(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return (var / len(vals)) ** 0.5


def plot_setting(
    rows: list[dict],
    setting: str,
    ax: plt.Axes,
    *,
    y_label: str = "Uniform VPS privacy loss (mean per game)",
) -> None:
    styles = {**PROTOCOL_STYLE, **MODEL_STYLE}
    for client in sorted({r["client"] for r in rows if r["series"] == "model"}):
        styles.setdefault(client, {"color": "#374151", "marker": "X", "label": client, "series": "model"})

    for client, style in styles.items():
        subset = [r for r in rows if r["client"] == client]
        if not subset:
            continue
        xs = [r["excess_cost"] for r in subset]
        ys = [r["vps_loss_mean"] for r in subset]
        mx, my = _mean(xs), _mean(ys)
        ex, ey = _stderr(xs), _stderr(ys)
        is_model = style.get("series") == "model"
        ax.errorbar(
            mx, my,
            xerr=ex, yerr=ey,
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor="white" if is_model else style["color"],
            markeredgecolor=style["color"] if is_model else "#111827",
            markeredgewidth=2.2 if is_model else 0.9,
            markersize=11 if is_model else 10,
            capsize=4,
            linewidth=1.5,
            label=style["label"],
        )

    if setting == "varied_full":
        positive_xs = [r["excess_cost"] for r in rows if r["excess_cost"] > 0]
        if positive_xs:
            ax.set_xscale("log")
            ax.set_xlim(left=min(positive_xs) * 0.5)
    else:
        ax.set_xlim(left=0)

    ax.set_xlabel("Coordination cost (scheduled-only excess)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(SETTING_LABELS[setting], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def _set_zoomed_limits(rows: list[dict], setting: str, ax: plt.Axes, x_field: str = "excess_cost") -> None:
    xs = [r[x_field] for r in rows if r.get(x_field) is not None]
    ys = [r["vps_loss_mean"] for r in rows]
    if not xs or not ys:
        return

    if setting == "varied_full":
        if x_field == "excess_cost_median":
            ax.set_xlim(-25, max(xs) * 1.25)
        else:
            ax.set_xscale("log")
            ax.set_xlim(min(xs) * 0.75, max(xs) * 1.25)
    else:
        x_pad = max((max(xs) - min(xs)) * 0.22, 0.12)
        ax.set_xlim(0.8, max(xs) + x_pad)

    y_pad = max((max(ys) - min(ys)) * 0.22, 0.06)
    y_min = 0.55 if setting == "uniform_full" else max(min(ys) - y_pad, 0)
    ax.set_ylim(y_min, max(ys) + y_pad)


def _expand_ylim_to_include(ax: plt.Axes, value: float | None, *, pad_fraction: float = 0.08) -> None:
    if value is None or np.isnan(value):
        return
    bottom, top = ax.get_ylim()
    if bottom <= value <= top:
        return
    bottom = min(bottom, value)
    top = max(top, value)
    pad = max((top - bottom) * pad_fraction, 0.02)
    ax.set_ylim(bottom - pad, top + pad)


def plot_model_setting(
    rows: list[dict],
    setting: str,
    ax: plt.Axes,
    *,
    x_field: str = "excess_cost",
    x_label: str = "Coordination cost (scheduled-only excess)",
    y_label: str = "Uniform VPS privacy loss (mean per game)",
) -> None:
    model_rows = [r for r in rows if r["series"] == "model" and r.get(x_field) is not None]
    imap_rows = [r for r in rows if r["client"] == "IMAP"]
    dsm_private_rows = [r for r in rows if r["client"] in {"DSM-private", "DSM-privacy"}]
    imap_excess = _mean([r[x_field] for r in imap_rows if r.get(x_field) is not None])
    dsm_private_vps = _mean([r["vps_loss_mean"] for r in dsm_private_rows])
    styles = MODEL_STYLE.copy()
    for client in sorted({r["client"] for r in model_rows}):
        styles.setdefault(client, {"color": "#374151", "marker": "X", "label": client, "series": "model"})

    for client, style in styles.items():
        subset = [r for r in model_rows if r["client"] == client]
        if not subset:
            continue
        xs = [r[x_field] for r in subset]
        ys = [r["vps_loss_mean"] for r in subset]
        mx, my = _mean(xs), _mean(ys)
        ax.scatter(
            [mx], [my],
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=2.6,
            s=130,
            label="_nolegend_",
            zorder=3,
        )
        ax.annotate(
            MODEL_SHORT_LABELS.get(client, client),
            xy=(mx, my),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
            color=style["color"],
            weight="bold",
        )

    _set_zoomed_limits(model_rows, setting, ax, x_field=x_field)
    if not np.isnan(imap_excess):
        ax.axvline(
            imap_excess,
            color=PROTOCOL_STYLE["IMAP"]["color"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            label="IMAP median excess cost" if x_field == "excess_cost_median" else "IMAP excess cost",
            zorder=1,
        )
    if not np.isnan(dsm_private_vps):
        ax.axhline(
            dsm_private_vps,
            color=PROTOCOL_STYLE["DSM-private"]["color"],
            linestyle=":",
            linewidth=1.8,
            alpha=0.85,
            label="DSM-private VPS",
            zorder=1,
        )
        _expand_ylim_to_include(ax, dsm_private_vps)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(SETTING_LABELS[setting], fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaderboard",
        help=(
            "Leaderboard JSON path or URL. Defaults to CALBENCH_LEADERBOARD or "
            f"{LEADERBOARD_URL}."
        ),
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use the legacy local analysis-output discovery instead of the leaderboard.",
    )
    parser.add_argument(
        "--median-excess-models-only",
        action="store_true",
        help="Write only new models-only plots using median excess cost on the x-axis.",
    )
    parser.add_argument(
        "--vps-metric",
        choices=sorted(VPS_METRIC_FIELDS),
        default="uniform",
        help="Leaderboard VPS metric to plot. Defaults to uniform VPS.",
    )
    args = parser.parse_args()

    root = _calendar_root()
    out_dir = root / "analysis/outputs/benchmark_lite_protocol_scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.local_only:
        data = collect_data(root)
        leaderboard = None
        data_source = "local analysis outputs"
    else:
        data, leaderboard, data_source = collect_data_with_leaderboard(
            root,
            leaderboard_source=args.leaderboard,
            cache_path=out_dir / "leaderboard.json",
            vps_metric=args.vps_metric,
        )
    output_suffix = "_uniform_vps" if args.vps_metric == "uniform" else ""
    y_label = (
        "Uniform VPS privacy loss (mean per game)"
        if args.vps_metric == "uniform"
        else "Cost-weighted VPS privacy loss (mean per game)"
    )

    if args.median_excess_models_only:
        model_fig, model_axes = plt.subplots(1, 2, figsize=(14, 6))
        model_fig.suptitle(
            "Cost Efficiency vs Privacy",
            fontsize=14, fontweight="bold",
        )
        for ax, setting in zip(model_axes, ["uniform_full", "varied_full"]):
            plot_model_setting(
                data[setting],
                setting,
                ax,
                x_field="excess_cost_median",
                x_label="Median coordination cost (scheduled-only excess)",
                y_label=y_label,
            )
        plt.tight_layout()
        median_pdf = out_dir / f"protocol_scatter_models_only_median_excess{output_suffix}.pdf"
        model_fig.savefig(median_pdf, bbox_inches="tight")
        print(f"Saved: {median_pdf}")
        median_png = out_dir / f"protocol_scatter_models_only_median_excess{output_suffix}.png"
        model_fig.savefig(median_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {median_png}")

        summary_rows = []
        for setting in ("uniform_full", "varied_full"):
            for row in [r for r in data[setting] if r["series"] == "model" and r.get("excess_cost_median") is not None]:
                summary_rows.append({
                    "setting": setting,
                    "client": row["client"],
                    "series": row["series"],
                    "n": row.get("n", 1),
                    "excess_cost_median": round(row["excess_cost_median"], 3),
                    "vps_loss_mean": round(row["vps_loss_mean"], 3),
                })
        with open(out_dir / f"summary_models_only_median_excess{output_suffix}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Data source: {data_source}")
        if leaderboard:
            print(f"Leaderboard generated_at: {leaderboard.get('generated_at')}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Cost Efficiency vs Privacy",
        fontsize=14, fontweight="bold",
    )

    for ax, setting in zip(axes, ["uniform_full", "varied_full"]):
        plot_setting(data[setting], setting, ax, y_label=y_label)

    plt.tight_layout()
    out_path = out_dir / f"protocol_scatter{output_suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_path_png = out_dir / f"protocol_scatter{output_suffix}.png"
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path_png}")

    model_fig, model_axes = plt.subplots(1, 2, figsize=(14, 6))
    model_fig.suptitle(
        "Cost Efficiency vs Privacy",
        fontsize=14, fontweight="bold",
    )
    for ax, setting in zip(model_axes, ["uniform_full", "varied_full"]):
        plot_model_setting(data[setting], setting, ax, y_label=y_label)
    plt.tight_layout()
    model_pdf = out_dir / f"protocol_scatter_models_only{output_suffix}.pdf"
    model_fig.savefig(model_pdf, bbox_inches="tight")
    print(f"Saved: {model_pdf}")
    model_png = out_dir / f"protocol_scatter_models_only{output_suffix}.png"
    model_fig.savefig(model_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {model_png}")

    print(f"Data source: {data_source}")
    if leaderboard:
        print(f"Leaderboard generated_at: {leaderboard.get('generated_at')}")

    # Also dump summary stats
    summary_rows = []
    for setting in ("uniform_full", "varied_full"):
        rows = data[setting]
        ordered_clients = [*PROTOCOL_STYLE, *MODEL_STYLE]
        ordered_clients.extend(sorted({r["client"] for r in rows if r["client"] not in ordered_clients}))
        clients = [client for client in ordered_clients if any(r["client"] == client for r in rows)]
        for client in clients:
            subset = [r for r in rows if r["client"] == client]
            xs = [r["excess_cost"] for r in subset]
            ys = [r["vps_loss_mean"] for r in subset]
            summary_rows.append({
                "setting": setting,
                "client": client,
                "series": subset[0]["series"],
                "n": int(sum(r.get("n", 1) for r in subset)),
                "excess_cost_mean": round(_mean(xs), 3),
                "excess_cost_stderr": round(_stderr(xs), 3),
                "vps_loss_mean": round(_mean(ys), 3),
                "vps_loss_stderr": round(_stderr(ys), 3),
            })

    with open(out_dir / f"summary{output_suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary:")
    print(f"{'Setting':<15} {'Client':<18} {'Series':<9} {'N':>4}  {'ExcessCost':>12}  {'VPS':>10}")
    print("-" * 82)
    for r in summary_rows:
        print(
            f"{r['setting']:<15} {r['client']:<18} {r['series']:<9} {r['n']:>4}  "
            f"{r['excess_cost_mean']:>8.2f}±{r['excess_cost_stderr']:.2f}  "
            f"{r['vps_loss_mean']:>7.3f}±{r['vps_loss_stderr']:.3f}"
        )


if __name__ == "__main__":
    main()
