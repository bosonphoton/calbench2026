#!/usr/bin/env python3
"""Rebuild calendar OpenSkill ratings from deterministic trace logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from a2a_engine.ratings import OpenSkillRater
from calendar_game.ratings import (
    CALENDAR_RATING_VARIANT_METRICS,
    CALENDAR_RATING_METRICS,
    extract_calendar_rating_event,
    load_calendar_trace,
    load_target_vps_from_game_target_csv,
    load_target_vps_from_pair_csv,
)


def _trace_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                child
                for child in path.rglob("*.json")
                if not child.name.endswith(".metadata.json")
            )
        elif path.exists():
            paths.append(path)
        else:
            paths.extend(Path().glob(raw))
    return sorted(set(paths))


def _write_leaderboard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Trace JSON files, directories, or globs.")
    parser.add_argument("--vps-pair-csv", help="Optional pair_round_vps.csv with target/observer VPS rows.")
    parser.add_argument("--vps-game-target-csv", help="Optional game_target_summary.csv with per-target excess VPS rows.")
    parser.add_argument("--reflection-vps-csv", help="Deprecated alias for --vps-game-target-csv.")
    parser.add_argument("--vps-weight-mode", default="uniform", choices=["cost", "uniform", ""])
    parser.add_argument("--rating-variant", default="default", choices=["default", "score_margin_v1"])
    parser.add_argument(
        "--vps-value-column",
        default=None,
        help="Value column for --vps-game-target-csv. Defaults to excess_calibrated_vps_loss_total.",
    )
    parser.add_argument("--out", default="analysis/outputs/openskill_ratings/rating_snapshot.json")
    parser.add_argument("--leaderboard-csv", default="analysis/outputs/openskill_ratings/leaderboard.csv")
    args = parser.parse_args()

    game_target_csv = args.vps_game_target_csv or args.reflection_vps_csv
    if args.vps_pair_csv and game_target_csv:
        parser.error("Use only one of --vps-pair-csv or --vps-game-target-csv.")
    has_vps_report = bool(args.vps_pair_csv or game_target_csv)
    if game_target_csv:
        value_column = args.vps_value_column
        if value_column is None:
            value_column = "excess_calibrated_vps_loss_total"
        vps_by_game = load_target_vps_from_game_target_csv(
            game_target_csv,
            value_column=value_column,
            weight_mode=args.vps_weight_mode or None,
        )
    elif args.vps_pair_csv:
        vps_by_game = load_target_vps_from_pair_csv(args.vps_pair_csv, weight_mode=args.vps_weight_mode or None)
    else:
        vps_by_game = {}
    events = []
    skipped = 0
    for path in _trace_paths(args.traces):
        trace = load_calendar_trace(path)
        event = extract_calendar_rating_event(
            trace,
            source_path=str(path),
            excess_vps_by_agent=(
                vps_by_game.get(str(trace.get("game_id") or path.stem), {})
                if has_vps_report
                else None
            ),
            rating_variant=args.rating_variant,
        )
        if event is None:
            skipped += 1
            continue
        events.append(event)

    metrics = CALENDAR_RATING_VARIANT_METRICS.get(args.rating_variant, CALENDAR_RATING_METRICS)
    snapshot = OpenSkillRater(metrics).rate_events(events)
    snapshot.metadata.update({
        "trace_count": len(events),
        "skipped_trace_count": skipped,
        "vps_pair_csv": args.vps_pair_csv,
        "vps_game_target_csv": game_target_csv,
        "reflection_vps_csv": args.reflection_vps_csv,
        "vps_value_column": args.vps_value_column,
        "vps_weight_mode": args.vps_weight_mode,
        "rating_variant": args.rating_variant,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_leaderboard(Path(args.leaderboard_csv), snapshot.leaderboard())

    print(f"rating_events={len(events)} skipped={skipped}")
    print(f"snapshot={out_path}")
    print(f"leaderboard={args.leaderboard_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
