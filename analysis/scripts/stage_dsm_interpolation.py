"""Stage DSM privacy/welfare interpolation runs and plot their tradeoff.

This script creates temporary benchmark-lite experiments that use ``paper_dsm``
agents with knobs linearly interpolated between the DSM-welfare and DSM-private
presets. It can run those local scripted experiments, score uniform VPS, and
write a staging chart without touching the polished protocol scatter figures.

Usage::

    cd games/calendar
    uv run python analysis/scripts/stage_dsm_interpolation.py --run --plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

from canonical_uniform_excess_vps_scatter import _agent_point_cost


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = CALENDAR_ROOT / "analysis/outputs/dsm_interpolation_stage"
EXPERIMENT_DIR = OUT_DIR / "experiments"
RESULTS_DIR = OUT_DIR / "results"
VPS_DIR = OUT_DIR / "vps"
MODEL_SUMMARY_CSV = (
    CALENDAR_ROOT
    / "analysis/outputs/benchmark_lite_protocol_scatter/summary_uniform_vps.csv"
)

BASE_EXPERIMENTS = {
    "uniform_full": CALENDAR_ROOT / "experiments/benchmark_lite_uniform_gemini31pro.yaml",
    "varied_full": CALENDAR_ROOT / "experiments/benchmark_lite_varied_gemini31pro.yaml",
}

ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 1.0]


def _interp(welfare: float, private: float, alpha: float) -> float:
    return welfare + alpha * (private - welfare)


def _params(alpha: float) -> dict[str, Any]:
    """Interpolate from welfare (0.0) to private (1.0)."""
    return {
        "dsm_lmin": 1,
        "dsm_lmax": max(2, round(_interp(12, 2, alpha))),
        "dsm_beta": round(_interp(0.0, 0.25, alpha), 4),
        "dsm_theta": round(_interp(0.0, 10.0, alpha), 4),
        "dsm_social_welfare_weight": round(_interp(10.0, 0.25, alpha), 4),
        "dsm_privacy_unit_cost": round(_interp(0.0, 1.0, alpha), 4),
        "dsm_initial_budget": round(_interp(1_000_000, 100, alpha)),
        "dsm_cascade_depth": 2 if alpha < 0.5 else 1,
    }


def _label(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def _agent_specs(num_agents: int) -> list[dict[str, str]]:
    return [{"type": "paper_dsm"} for _ in range(num_agents)]


def write_experiments() -> list[Path]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for setting, base_path in BASE_EXPERIMENTS.items():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for alpha in ALPHAS:
            spec = dict(base)
            name = f"dsm_interpolation_{_label(alpha)}_{setting}"
            defaults = dict(spec.get("defaults") or {})
            defaults.update({
                "enable_fallback": False,
                "dm_cap": 1_000_000,
                **_params(alpha),
            })
            if defaults.get("num_agents"):
                defaults["agents"] = _agent_specs(int(defaults["num_agents"]))
            batches = []
            for batch in spec.get("batches") or []:
                batch = dict(batch)
                config = dict(batch.get("config") or {})
                num_agents = int(config.get("num_agents", defaults.get("num_agents", 0) or 0))
                config["agents"] = _agent_specs(num_agents)
                batch["config"] = config
                batches.append(batch)
            spec.update({
                "name": name,
                "description": (
                    f"Staged DSM interpolation alpha={alpha:.2f}; "
                    "0=welfare preset, 1=private-like preset."
                ),
                "defaults": defaults,
                "batches": batches,
            })
            path = EXPERIMENT_DIR / f"{name}.yaml"
            path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            paths.append(path)
    return paths


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CALENDAR_ROOT, check=True)


def run_experiments(max_parallelism: int) -> None:
    for path in write_experiments():
        _run([
            "uv", "run", "python", "run.py", str(path),
            "--max-parallelism", str(max_parallelism),
            "--results-dir", str(RESULTS_DIR),
            "--resume",
        ])


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.glob("*.json")
        if not path.name.endswith(".metadata.json") and path.name != "_run_manifest.jsonl"
    )


def score_vps() -> None:
    VPS_DIR.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted(RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else []):
        if not run_dir.is_dir():
            continue
        traces = _trace_paths(run_dir)
        if not traces:
            continue
        out_dir = VPS_DIR / run_dir.name
        if (out_dir / "game_summary.csv").exists():
            continue
        _run([
            "uv", "run", "python", "analysis/scripts/rq5_vps_privacy_metric.py",
            "--weight-mode", "uniform",
            "--out-dir", str(out_dir),
            str(run_dir),
        ])


def _mean_excess_cost(run_dir: Path) -> float:
    values: list[float] = []
    for path in _trace_paths(run_dir):
        trace = json.loads(path.read_text(encoding="utf-8"))
        num_agents = int((trace.get("config") or {}).get("num_agents") or 0)
        for agent_id in range(num_agents):
            values.append(_agent_point_cost(trace, agent_id, "scheduled-only"))
    return fmean(values)


def _mean_uniform_vps(run_name: str) -> float:
    path = VPS_DIR / run_name / "game_summary.csv"
    values: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["vps_loss_mean"]))
    return fmean(values)


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else []):
        if not run_dir.is_dir() or not _trace_paths(run_dir):
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 5 or parts[:2] != ["dsm", "interpolation"]:
            continue
        alpha = int(parts[2][1:]) / 100
        setting = "_".join(parts[3:])
        rows.append({
            "setting": setting,
            "alpha": alpha,
            "label": f"DSM interp {alpha:.2f}",
            "excess_cost_mean": _mean_excess_cost(run_dir),
            "uniform_vps_loss_mean": _mean_uniform_vps(run_dir.name),
            "n": len(_trace_paths(run_dir)),
            **_params(alpha),
        })
    return sorted(rows, key=lambda r: (r["setting"], r["alpha"]))


def write_summary(rows: list[dict[str, Any]]) -> Path:
    path = OUT_DIR / "dsm_interpolation_summary.csv"
    if not rows:
        raise RuntimeError("no interpolation rows found")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    fig.suptitle("DSM Cost Efficiency vs Privacy", fontsize=14, fontweight="bold")
    colors = {"uniform_full": "#2563eb", "varied_full": "#dc2626"}
    titles = {"uniform_full": "Uniform costs", "varied_full": "Varied costs"}
    for ax, setting in zip(axes, ["uniform_full", "varied_full"]):
        subset = [row for row in rows if row["setting"] == setting]
        xs = [row["excess_cost_mean"] for row in subset]
        ys = [row["uniform_vps_loss_mean"] for row in subset]
        ax.plot(xs, ys, color=colors[setting], linewidth=1.8, alpha=0.8)
        ax.scatter(
            xs, ys,
            c=[row["alpha"] for row in subset],
            cmap="viridis_r",
            s=90,
            edgecolors="#111827",
            linewidths=0.8,
            zorder=3,
        )

        grouped: dict[tuple[float, float], list[float]] = {}
        for row in subset:
            key = (round(row["excess_cost_mean"], 6), round(row["uniform_vps_loss_mean"], 6))
            grouped.setdefault(key, []).append(row["alpha"])
        for (x, y), alphas in grouped.items():
            label = ",".join(f"{alpha:.2f}" for alpha in alphas)
            ax.annotate(
                label,
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
            )
        ax.set_title(titles[setting], fontsize=12, fontweight="bold")
        ax.set_xlabel("Coordination cost (scheduled-only excess)")
        ax.set_ylabel("Uniform VPS privacy loss (mean per game)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.text(
            0.02, 0.02,
            "alpha: 0=welfare, 1=private-like",
            transform=ax.transAxes,
            fontsize=9,
            color="#374151",
        )
    png = OUT_DIR / "dsm_interpolation_stage_scatter.png"
    pdf = OUT_DIR / "dsm_interpolation_stage_scatter.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    return png, pdf


MODEL_STYLE: dict[str, dict[str, str]] = {
    "Gemini 3.1 Pro": {"color": "#111827", "marker": "X", "short": "Gemini Pro"},
    "Gemini 3 Flash": {"color": "#6b7280", "marker": "P", "short": "Gemini Flash"},
    "GPT-5.4 Mini": {"color": "#f59e0b", "marker": "*", "short": "GPT-5.4 Mini"},
    "Claude Sonnet 4.6": {"color": "#8b5cf6", "marker": "v", "short": "Claude"},
    "DeepSeek V4 Pro": {"color": "#0891b2", "marker": "<", "short": "DeepSeek"},
    "Qwen3.6 Plus": {"color": "#db2777", "marker": ">", "short": "Qwen"},
    "Llama 4 Maverick": {"color": "#65a30d", "marker": "h", "short": "Llama"},
}

BASELINE_STYLE: dict[str, dict[str, str]] = {
    "IMAP": {"color": "#1f77b4", "marker": "o", "short": "IMAP"},
    "SD": {"color": "#9467bd", "marker": "D", "short": "SD"},
}

MODEL_LABEL_OFFSETS = {
    ("uniform_full", "Gemini 3.1 Pro"): (-24, -13),
    ("uniform_full", "Gemini 3 Flash"): (8, -16),
    ("uniform_full", "GPT-5.4 Mini"): (8, -2),
    ("uniform_full", "Claude Sonnet 4.6"): (8, 12),
    ("uniform_full", "DeepSeek V4 Pro"): (8, 10),
    ("uniform_full", "Qwen3.6 Plus"): (-18, 8),
    ("uniform_full", "Llama 4 Maverick"): (8, 6),
    ("varied_full", "Gemini 3.1 Pro"): (8, 8),
    ("varied_full", "Gemini 3 Flash"): (8, -11),
    ("varied_full", "GPT-5.4 Mini"): (8, -3),
    ("varied_full", "Claude Sonnet 4.6"): (8, 8),
    ("varied_full", "DeepSeek V4 Pro"): (8, 8),
    ("varied_full", "Qwen3.6 Plus"): (8, 6),
    ("varied_full", "Llama 4 Maverick"): (8, 7),
}

BASELINE_LABEL_OFFSETS = {
    ("uniform_full", "IMAP"): (8, -18),
    ("uniform_full", "SD"): (8, 5),
    ("varied_full", "IMAP"): (8, 7),
    ("varied_full", "SD"): (8, -13),
}


def load_overlay_summary(path: Path = MODEL_SUMMARY_CSV) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(
            f"model summary not found: {path}. Run benchmark_lite_protocol_scatter.py first."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("series") != "model" and row.get("client") not in BASELINE_STYLE:
                continue
            rows.append({
                "setting": row["setting"],
                "client": row["client"],
                "series": row["series"],
                "excess_cost_mean": float(row["excess_cost_mean"]),
                "uniform_vps_loss_mean": float(row["vps_loss_mean"]),
            })
    return rows


def plot_with_models(
    dsm_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), constrained_layout=True)
    fig.suptitle("DSM + LLM Models: Cost Efficiency vs Privacy", fontsize=14, fontweight="bold")
    colors = {"uniform_full": "#2563eb", "varied_full": "#dc2626"}
    titles = {"uniform_full": "Uniform costs", "varied_full": "Varied costs"}
    shared_ys = [
        *(row["uniform_vps_loss_mean"] for row in dsm_rows),
        *(row["uniform_vps_loss_mean"] for row in overlay_rows),
    ]
    shared_y_pad = max((max(shared_ys) - min(shared_ys)) * 0.12, 0.08)
    shared_ylim = (max(0, min(shared_ys) - shared_y_pad), max(shared_ys) + shared_y_pad)
    for ax, setting in zip(axes, ["uniform_full", "varied_full"]):
        subset = [row for row in dsm_rows if row["setting"] == setting]
        model_subset = [
            row for row in overlay_rows
            if row["setting"] == setting and row["series"] == "model"
        ]
        baseline_subset = [
            row for row in overlay_rows
            if row["setting"] == setting and row["client"] in BASELINE_STYLE
        ]
        xs = [row["excess_cost_mean"] for row in subset]
        ys = [row["uniform_vps_loss_mean"] for row in subset]
        ax.plot(
            xs,
            ys,
            color=colors[setting],
            linewidth=1.8,
            alpha=0.55,
            label="DSM interpolation",
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            c=[row["alpha"] for row in subset],
            cmap="viridis_r",
            s=72,
            edgecolors="#111827",
            linewidths=0.7,
            zorder=2,
        )
        grouped: dict[tuple[float, float], list[float]] = {}
        for row in subset:
            key = (round(row["excess_cost_mean"], 6), round(row["uniform_vps_loss_mean"], 6))
            grouped.setdefault(key, []).append(row["alpha"])
        for (x, y), alphas in grouped.items():
            if min(alphas) == 0.0:
                label = "DSM-welfare\n" + ",".join(f"{alpha:.2f}" for alpha in alphas)
                offset = (6, 5)
                weight = "bold"
            elif max(alphas) == 1.0:
                label = "DSM-private\n" + ",".join(f"{alpha:.2f}" for alpha in alphas)
                offset = (-54, 5) if setting == "varied_full" else (8, 3)
                weight = "bold"
            else:
                label = ",".join(f"{alpha:.2f}" for alpha in alphas)
                offset = (5, 4)
                weight = "normal"
            ax.annotate(
                label,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                weight=weight,
                color="#374151",
            )

        for row in baseline_subset:
            style = BASELINE_STYLE[row["client"]]
            x = row["excess_cost_mean"]
            y = row["uniform_vps_loss_mean"]
            ax.scatter(
                [x],
                [y],
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="#111827",
                linewidths=0.9,
                s=115,
                zorder=4,
            )
            ax.annotate(
                style["short"],
                (x, y),
                xytext=BASELINE_LABEL_OFFSETS.get((setting, row["client"]), (7, 6)),
                textcoords="offset points",
                fontsize=8.5,
                weight="bold",
                color=style["color"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
                zorder=6,
            )

        for row in model_subset:
            style = MODEL_STYLE.get(
                row["client"],
                {"color": "#374151", "marker": "X", "short": row["client"]},
            )
            x = row["excess_cost_mean"]
            y = row["uniform_vps_loss_mean"]
            ax.scatter(
                [x],
                [y],
                marker=style["marker"],
                facecolors="white",
                edgecolors=style["color"],
                linewidths=2.5,
                s=150,
                zorder=5,
            )

        ax.set_title(titles[setting], fontsize=12, fontweight="bold")
        ax.set_xlabel("Coordination cost (scheduled-only excess)")
        ax.set_ylabel("Uniform VPS privacy loss (mean per game)")
        ax.grid(True, alpha=0.3)
        if setting == "varied_full":
            ax.set_xscale("log")
            all_xs = [
                *(row["excess_cost_mean"] for row in subset),
                *(row["excess_cost_mean"] for row in model_subset),
                *(row["excess_cost_mean"] for row in baseline_subset),
            ]
            positive_xs = [x for x in all_xs if x > 0]
            ax.set_xlim(min(positive_xs) * 0.75, max(positive_xs) * 1.2)
        else:
            all_xs = [
                *(row["excess_cost_mean"] for row in subset),
                *(row["excess_cost_mean"] for row in model_subset),
                *(row["excess_cost_mean"] for row in baseline_subset),
            ]
            ax.set_xlim(0.55, max(all_xs) + 0.45)
        ax.set_ylim(*shared_ylim)
    present_models = [
        client for client in MODEL_STYLE
        if any(row["client"] == client for row in overlay_rows)
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_STYLE[client]["marker"],
            color="none",
            markerfacecolor="white",
            markeredgecolor=MODEL_STYLE[client]["color"],
            markeredgewidth=2.2,
            markersize=9,
            label=MODEL_STYLE[client]["short"],
        )
        for client in present_models
    ]
    fig.legend(
        handles=model_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        title="LLM models",
        title_fontsize=10,
    )
    png = OUT_DIR / "dsm_interpolation_with_models_scatter.png"
    pdf = OUT_DIR / "dsm_interpolation_with_models_scatter.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run staged DSM experiments before plotting.")
    parser.add_argument("--score-vps", action="store_true", help="Compute uniform VPS for staged traces.")
    parser.add_argument("--plot", action="store_true", help="Write summary CSV and chart.")
    parser.add_argument(
        "--plot-model-overlay",
        action="store_true",
        help="Write a separate DSM interpolation chart with LLM model points overlaid.",
    )
    parser.add_argument(
        "--model-summary",
        type=Path,
        default=MODEL_SUMMARY_CSV,
        help="Protocol scatter summary CSV with model aggregates.",
    )
    parser.add_argument("--max-parallelism", type=int, default=8)
    args = parser.parse_args()

    write_experiments()
    if args.run:
        run_experiments(args.max_parallelism)
    if args.score_vps or args.plot or args.plot_model_overlay:
        score_vps()
    if args.plot or args.plot_model_overlay:
        rows = collect_rows()
        summary = write_summary(rows)
        print(summary)
    if args.plot:
        png, pdf = plot(rows)
        print(png)
        print(pdf)
    if args.plot_model_overlay:
        overlay_png, overlay_pdf = plot_with_models(rows, load_overlay_summary(args.model_summary))
        print(overlay_png)
        print(overlay_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
