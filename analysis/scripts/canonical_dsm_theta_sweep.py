"""Stage and plot a canonical Paper-DSM theta sweep.

The sweep holds the welfare-oriented Paper-DSM preset fixed and varies only
``dsm_theta``.  This makes the resulting curve easier to interpret than the
older interpolation script, which changed several DSM knobs at once.

Usage:

    cd games/calendar
    uv run python analysis/scripts/canonical_dsm_theta_sweep.py --run --plot
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
import yaml

from canonical_uniform_excess_vps_scatter import _agent_point_cost


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = CALENDAR_ROOT / "analysis/outputs/canonical_dsm_theta_sweep"
EXPERIMENT_DIR = OUT_DIR / "experiments"
RESULTS_DIR = OUT_DIR / "results"
VPS_DIR = OUT_DIR / "vps"

BASE_EXPERIMENTS = {
    "uniform": CALENDAR_ROOT / "experiments/calbench90_uniform_mixed5_reflection.yaml",
    "varied": CALENDAR_ROOT / "experiments/calbench90_varied_mixed5_reflection.yaml",
}

THETAS = [0.0, 0.10, 0.15, 0.25, 0.50, 1.00]
THETA_SET = {round(theta, 8) for theta in THETAS}


def _theta_label(theta: float) -> str:
    return f"theta{theta:.2f}".replace(".", "p")


def _agent_specs(num_agents: int) -> list[dict[str, str]]:
    return [{"type": "paper_dsm"} for _ in range(num_agents)]


def _dsm_params(theta: float) -> dict[str, Any]:
    return {
        "dsm_lmin": 1,
        "dsm_lmax": 12,
        "dsm_beta": 0.0,
        "dsm_theta": theta,
        "dsm_social_welfare_weight": 10.0,
        "dsm_privacy_unit_cost": 1.0,
        "dsm_initial_budget": 1_000_000,
        "dsm_cascade_depth": 2,
    }


def write_experiments() -> list[Path]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for setting, base_path in BASE_EXPERIMENTS.items():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for theta in THETAS:
            spec = dict(base)
            defaults = dict(spec.get("defaults") or {})
            defaults.update({
                "enable_fallback": False,
                "enable_reflection": False,
                "dm_cap": 1_000_000,
                **_dsm_params(theta),
            })
            if defaults.get("num_agents"):
                defaults["agents"] = _agent_specs(int(defaults["num_agents"]))

            batches = []
            for batch in spec.get("batches") or []:
                batch = dict(batch)
                config = dict(batch.get("config") or {})
                num_agents = int(config.get("num_agents", defaults.get("num_agents", 0) or 0))
                if num_agents:
                    config["agents"] = _agent_specs(num_agents)
                batch["config"] = config
                batches.append(batch)

            name = f"canonical_dsm_{_theta_label(theta)}_{setting}"
            spec.update({
                "name": name,
                "description": (
                    f"Canonical Paper-DSM theta sweep, {setting} tasks, "
                    f"dsm_theta={theta:.2f}; all other welfare-style DSM knobs fixed."
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
            "uv",
            "run",
            "python",
            "run.py",
            str(path),
            "--max-parallelism",
            str(max_parallelism),
            "--results-dir",
            str(RESULTS_DIR),
            "--resume",
        ])


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.glob("**/*.json")
        if not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
        and "_reports" not in path.parts
        and "_index" not in path.parts
    )


def score_vps() -> None:
    VPS_DIR.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted(RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else []):
        if not run_dir.is_dir() or not _trace_paths(run_dir):
            continue
        parsed = _parse_run_name(run_dir.name)
        if parsed is None or round(parsed[0], 8) not in THETA_SET:
            continue
        out_dir = VPS_DIR / run_dir.name
        if (out_dir / "game_summary.csv").exists():
            continue
        _run([
            "uv",
            "run",
            "python",
            "analysis/scripts/rq5_vps_privacy_metric.py",
            "--weight-mode",
            "uniform",
            "--out-dir",
            str(out_dir),
            str(run_dir),
        ])


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _run_metrics(run_dir: Path) -> dict[str, float | int | None]:
    legacy_full_task_excess_costs: list[float] = []
    scheduled_only_agent_costs: list[float] = []
    task_success_values: list[float] = []
    scheduled: list[float] = []
    for path in _trace_paths(run_dir):
        trace = json.loads(path.read_text(encoding="utf-8"))
        metrics = trace.get("metrics") or {}
        realized = metrics.get("realized_cost")
        optimal = metrics.get("optimal_cost")
        if realized is not None and optimal is not None:
            legacy_full_task_excess_costs.append(float(realized) - float(optimal))
        if metrics.get("meetings_scheduled") is not None:
            scheduled.append(float(metrics["meetings_scheduled"]))
        num_agents = int((trace.get("config") or {}).get("num_agents") or 0)
        contribution_by_agent = {
            int(row["agent_id"]): row
            for row in (trace.get("final_state") or {}).get("contribution_scores") or []
            if isinstance(row, dict) and row.get("agent_id") is not None
        }
        for agent_id in range(num_agents):
            contribution = contribution_by_agent.get(agent_id, {})
            task_success = contribution.get("task_success", contribution.get("coordination_rate"))
            if task_success is not None:
                task_success_values.append(float(task_success))
            scheduled_only_agent_costs.append(_agent_point_cost(trace, agent_id, "scheduled-only"))
    return {
        "n": len(_trace_paths(run_dir)),
        "legacy_full_task_excess_cost_mean": _mean(legacy_full_task_excess_costs),
        "raw_game_excess_cost_mean": _mean(legacy_full_task_excess_costs),
        "adjusted_excess_cost_mean": _mean(scheduled_only_agent_costs),
        "scheduled_only_excess_cost_mean": _mean(scheduled_only_agent_costs),
        "coordination_rate_mean": _mean(task_success_values),
        "task_success_mean": _mean(task_success_values),
        "meetings_scheduled_mean": _mean(scheduled),
    }


def _num_meetings_by_game(run_dir: Path) -> dict[str, int]:
    meetings_by_game: dict[str, int] = {}
    for path in _trace_paths(run_dir):
        trace = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(trace.get("game_id") or path.stem)
        config = trace.get("config") or {}
        meetings_by_game[game_id] = int(config.get("num_meetings") or 1)
    return meetings_by_game


def _mean_excess_vps(run_name: str, run_dir: Path) -> float | None:
    path = VPS_DIR / run_name / "game_target_summary.csv"
    if not path.exists():
        return None
    meetings_by_game = _num_meetings_by_game(run_dir)
    values: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("weight_mode") or "uniform") != "uniform":
                continue
            value = row.get("excess_vps_loss_total")
            if value not in (None, ""):
                game_id = str(row.get("game_id") or "")
                values.append(float(value) / max(1, meetings_by_game.get(game_id, 1)))
    return _mean(values)


def _parse_run_name(name: str) -> tuple[float, str] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[:2] != ["canonical", "dsm"]:
        return None
    theta_text = parts[2].removeprefix("theta").replace("p", ".")
    try:
        theta = float(theta_text)
    except ValueError:
        return None
    return theta, parts[-1]


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else []):
        parsed = _parse_run_name(run_dir.name)
        if parsed is None or not run_dir.is_dir() or not _trace_paths(run_dir):
            continue
        theta, setting = parsed
        if round(theta, 8) not in THETA_SET:
            continue
        rows.append({
            "setting": setting,
            "theta": theta,
            "label": f"DSM theta={theta:.2f}",
            **_run_metrics(run_dir),
            "excess_vps_mean": _mean_excess_vps(run_dir.name, run_dir),
            **_dsm_params(theta),
        })
    return sorted(rows, key=lambda row: (row["setting"], row["theta"]))


def write_summary(rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise RuntimeError("no theta-sweep rows found; run with --run first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    fig.suptitle("Paper-DSM Theta Sweep", fontsize=14, fontweight="bold")
    colors = {"uniform": "#2563eb", "varied": "#dc2626"}
    titles = {"uniform": "Uniform costs", "varied": "Varied costs"}
    for ax, setting in zip(axes, ["uniform", "varied"]):
        subset = [row for row in rows if row["setting"] == setting]
        xs = [row["excess_vps_mean"] for row in subset]
        ys = [row["adjusted_excess_cost_mean"] for row in subset]
        ax.plot(xs, ys, color=colors[setting], linewidth=1.8, alpha=0.65, zorder=1)
        sc = ax.scatter(
            xs,
            ys,
            c=[row["theta"] for row in subset],
            cmap="viridis_r",
            s=95,
            edgecolors="#111827",
            linewidths=0.8,
            zorder=2,
        )
        grouped: dict[tuple[float, float], list[float]] = {}
        for row in subset:
            key = (
                round(float(row["excess_vps_mean"]), 6),
                round(float(row["adjusted_excess_cost_mean"]), 6),
            )
            grouped.setdefault(key, []).append(float(row["theta"]))
        for (x, y), thetas in grouped.items():
            label = "/".join(f"{theta:.2f}" for theta in thetas)
            ax.annotate(
                label,
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8.5,
                color="#374151",
            )
        ax.set_title(titles[setting], fontsize=12, fontweight="bold")
        ax.set_xlabel("Excess VPS privacy loss (mean per target agent-meeting)")
        ax.set_ylabel("Coordination-adjusted excess cost (mean per participant-meeting)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.colorbar(sc, ax=axes, shrink=0.85, label="dsm_theta")
    png = OUT_DIR / "theta_sweep_scatter.png"
    pdf = OUT_DIR / "theta_sweep_scatter.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run generated Paper-DSM experiments.")
    parser.add_argument("--score-vps", action="store_true", help="Score protocol-semantic VPS.")
    parser.add_argument("--plot", action="store_true", help="Write summary and theta-sweep plot.")
    parser.add_argument("--max-parallelism", type=int, default=8)
    args = parser.parse_args()

    experiments = write_experiments()
    for path in experiments:
        print(path)
    if args.run:
        run_experiments(args.max_parallelism)
    if args.score_vps or args.plot:
        score_vps()
    if args.plot:
        rows = collect_rows()
        summary = write_summary(rows)
        png, pdf = plot(rows)
        print(summary)
        print(png)
        print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
