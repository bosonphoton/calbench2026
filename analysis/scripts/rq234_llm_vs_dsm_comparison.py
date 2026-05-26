"""RQ2/RQ3/RQ4 comparison: Gemini-3.1-Pro vs DSM baseline.

RQ2 (efficiency)  — per-agent cheap-talk messages per scheduled meeting
RQ3 (optimality)  — scheduled-only excess coordination cost
RQ4 (fairness)    — per-agent absolute relative excess burden

Usage::

    cd games/calendar
    uv run python analysis/scripts/rq234_llm_vs_dsm_comparison.py \\
        --llm  ../../shared-traces \\
        --dsm  results/varied_full_dsm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Allow running from repo root or games/calendar
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calendar_game.dataset import CalendarGameDataset


def load(llm_root: Path, dsm_root: Path):
    llm_ds = CalendarGameDataset.from_dir(llm_root)
    dsm_ds = CalendarGameDataset.from_dir(dsm_root)

    lg = llm_ds.to_game_df(); lg["client"] = "Gemini-3.1-Pro"
    dg = dsm_ds.to_game_df(); dg["client"] = "DSM"

    la = llm_ds.to_agent_df(); la["client"] = "Gemini-3.1-Pro"
    da = dsm_ds.to_agent_df(); da["client"] = "DSM"

    games  = pd.concat([lg, dg], ignore_index=True)
    agents = pd.concat([la, da], ignore_index=True)
    return games, agents


def report(games: pd.DataFrame, agents: pd.DataFrame) -> str:
    lines = []

    def h(title): lines.append(f"\n=== {title} ===")
    def show(df): lines.append(df.to_string())

    h("Game counts")
    show(games.groupby("client").size().rename("n"))

    h("RQ2: Communication efficiency (per-agent cheap-talk per scheduled meeting)")
    show(agents.groupby("client")["communication_efficiency"]
         .agg(["mean", "median", "std"]).round(2))

    h("RQ3: Task success")
    show(agents.groupby("client")["task_success"]
         .agg(["mean", "median"]).round(3))

    h("RQ3: Coordination cost (scheduled-only excess)")
    show(agents.groupby("client")["coordination_cost"]
         .agg(["mean", "median", "std"]).round(2))

    h("RQ3: Legacy cost ratio (realized / full-task optimal) — games where optimal > 0")
    cr = games[games["optimal_cost"] > 0]
    show(cr.groupby("client")["cost_ratio"]
         .agg(["mean", "median", "std"]).round(3))

    h("RQ3: Coordination cost by num_agents")
    show(agents.groupby(["client", "num_agents"])["coordination_cost"]
         .mean().round(2).unstack("num_agents"))

    h("RQ4: Legacy realized-cost Gini coefficient (0=equal, 1=all on one agent)")
    show(games.groupby("client")["cost_gini"]
         .agg(["mean", "median", "std"]).round(3))

    h("RQ4: Fairness cost")
    show(agents.groupby("client")["fairness_cost"]
         .agg(["mean", "median", "std"]).round(3))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="../../shared-traces",
                        help="Root dir for LLM traces (default: ../../shared-traces)")
    parser.add_argument("--dsm", default="results/varied_full_dsm",
                        help="Root dir for DSM traces (default: results/varied_full_dsm)")
    args = parser.parse_args()

    games, agents = load(Path(args.llm), Path(args.dsm))
    print(report(games, agents))


if __name__ == "__main__":
    main()
