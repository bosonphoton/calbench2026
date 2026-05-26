"""CLI sweep for the calendar scheduling benchmark."""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Literal

from calendar_game.scenario import generate_scenario
from calendar_game.solver import solve_greedy, solve_optimal

Strategy = Literal["optimal", "greedy"]


def score(optimal_cost: int | None, realized_cost: int | None) -> dict[str, int | bool | None]:
    if optimal_cost is None or realized_cost is None:
        return {"correct": False, "gap": None}
    return {"correct": True, "gap": realized_cost - optimal_cost}


def run_sweep(
    *,
    densities: list[float],
    pref_levels: list[int],
    runs: int = 50,
    strategy: Strategy = "greedy",
    seed: int = 0,
    num_agents: int = 2,
    num_slots: int = 16,
    num_meetings: int = 1,
) -> list[dict]:
    rows: list[dict] = []
    for density in densities:
        for pref_level in pref_levels:
            optimal_costs: list[int] = []
            greedy_costs: list[int] = []
            realized_costs: list[int] = []
            gaps: list[int] = []
            failures = 0
            for i in range(runs):
                scenario = generate_scenario(
                    seed + i,
                    num_agents,
                    num_slots,
                    density,
                    pref_level,
                    num_meetings,
                )
                optimal = solve_optimal(scenario["calendars"], scenario["meetings"], scenario["num_slots"])
                greedy = solve_greedy(scenario["calendars"], scenario["meetings"], scenario["num_slots"])
                realized = optimal if strategy == "optimal" else greedy
                if optimal["cost"] is None or greedy["cost"] is None or realized["cost"] is None:
                    failures += 1
                    continue
                optimal_costs.append(optimal["cost"])
                greedy_costs.append(greedy["cost"])
                realized_costs.append(realized["cost"])
                gaps.append(realized["cost"] - optimal["cost"])
            rows.append({
                "density": density,
                "pref_level": pref_level,
                "runs": runs,
                "failures": failures,
                "avg_optimal": mean(optimal_costs) if optimal_costs else None,
                "avg_greedy": mean(greedy_costs) if greedy_costs else None,
                "avg_realized": mean(realized_costs) if realized_costs else None,
                "avg_gap": mean(gaps) if gaps else None,
            })
    return rows


def _print_table(rows: list[dict]) -> None:
    headers = ["density", "pref_level", "runs", "failures", "avg_optimal", "avg_greedy", "avg_realized", "avg_gap"]
    print("  ".join(f"{h:>12}" for h in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:12.2f}")
            else:
                values.append(f"{str(value):>12}")
        print("  ".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the calendar benchmark sweep.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--strategy", choices=["optimal", "greedy"], default="greedy")
    parser.add_argument("--densities", default="0.3,0.5,0.8")
    parser.add_argument("--pref-levels", default="1,2,5")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-meetings", type=int, default=1)
    args = parser.parse_args()

    rows = run_sweep(
        densities=[float(x) for x in args.densities.split(",")],
        pref_levels=[int(x) for x in args.pref_levels.split(",")],
        runs=args.runs,
        strategy=args.strategy,
        num_agents=args.num_agents,
        num_meetings=args.num_meetings,
    )
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
