#!/usr/bin/env python3
"""Compute reflection-based target-attributed VPS privacy loss.

This consumes ``reflection_slot_calibration.csv`` from
``reflection_logprob_calibration.py`` and turns agents' self-reported belief
movement into privacy-loss rows. Loss is reported in slot-equivalent units: a
maximal ``+3`` or ``-3`` belief movement about one slot counts as one slot of
belief movement. A belief movement only counts as calibrated privacy loss when
it moves toward the target's actual calendar state:

* actual busy + positive occupied delta -> loss
* actual free + negative occupied delta -> loss
* wrong-direction movement -> 0 calibrated loss

The uncalibrated movement is also reported as ``raw_vps_loss``. Per-game target
summaries include ``excess_calibrated_vps_loss_total``, which subtracts the
unavoidable five-slot communication floor for a five-round calendar game.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DELTA_MAX = 3.0
DEFAULT_GAME_VPS_FLOOR = 5.0


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _value(row: dict[str, str], source: str) -> float | None:
    if source == "sampled":
        delta = _float(row.get("sampled_delta"))
    elif source == "floor":
        delta = _float(row.get("floor_expected_delta"))
    elif source == "upper":
        delta = _float(row.get("floor_upper_abs_delta"))
    else:
        raise ValueError(f"Unknown source: {source}")
    return delta


def _slot_rows(rows: list[dict[str, str]], *, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        actual_busy = _int(row.get("actual_busy"))
        delta = _value(row, source)
        target_agent = _int(row.get("target_agent_id"))
        observer_agent = _int(row.get("agent_id"))
        slot = _int(row.get("slot"))
        if actual_busy is None or delta is None or target_agent is None or observer_agent is None or slot is None:
            continue
        truth_sign = 1.0 if actual_busy else -1.0
        raw_loss = abs(delta) / DELTA_MAX
        calibrated_loss = max(0.0, truth_sign * delta) / DELTA_MAX
        out.append({
            "trace_path": row.get("trace_path"),
            "game_id": row.get("game_id"),
            "round": row.get("round"),
            "observer_agent": observer_agent,
            "target_agent": target_agent,
            "slot": slot,
            "actual_busy": actual_busy,
            "belief_delta_occupied": delta,
            "raw_vps_loss": raw_loss,
            "calibrated_vps_loss": calibrated_loss,
            "direction_correct": bool(delta and truth_sign * delta > 0),
            "direction_wrong": bool(delta and truth_sign * delta < 0),
            "source": source,
        })
    return out


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...], *, vps_floor: float | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        rec = grouped.setdefault(key, {
            **dict(zip(keys, key)),
            "slot_count": 0,
            "nonzero_count": 0,
            "direction_correct_count": 0,
            "direction_wrong_count": 0,
            "raw_vps_loss_total": 0.0,
            "calibrated_vps_loss_total": 0.0,
        })
        rec["slot_count"] += 1
        if row["belief_delta_occupied"] != 0:
            rec["nonzero_count"] += 1
        if row["direction_correct"]:
            rec["direction_correct_count"] += 1
        if row["direction_wrong"]:
            rec["direction_wrong_count"] += 1
        rec["raw_vps_loss_total"] += float(row["raw_vps_loss"])
        rec["calibrated_vps_loss_total"] += float(row["calibrated_vps_loss"])

    out = []
    for rec in grouped.values():
        nonzero = int(rec["nonzero_count"])
        slots = int(rec["slot_count"])
        rec["raw_vps_loss_mean"] = rec["raw_vps_loss_total"] / slots if slots else 0.0
        rec["calibrated_vps_loss_mean"] = rec["calibrated_vps_loss_total"] / slots if slots else 0.0
        if vps_floor is not None:
            rec["vps_floor"] = vps_floor
            rec["excess_raw_vps_loss_total"] = max(0.0, rec["raw_vps_loss_total"] - vps_floor)
            rec["excess_calibrated_vps_loss_total"] = max(0.0, rec["calibrated_vps_loss_total"] - vps_floor)
        rec["nonzero_precision"] = rec["direction_correct_count"] / nonzero if nonzero else None
        out.append(rec)
    return sorted(out, key=lambda rec: tuple(str(rec.get(name)) for name in keys))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reflection_slot_csv", help="reflection_slot_calibration.csv")
    parser.add_argument("--out-dir", default="analysis/outputs/reflection_vps_metric")
    parser.add_argument("--source", choices=["sampled", "floor", "upper"], default="sampled")
    parser.add_argument(
        "--game-vps-floor",
        type=float,
        default=DEFAULT_GAME_VPS_FLOOR,
        help="Unavoidable slot-equivalent VPS floor subtracted from every game-target total.",
    )
    args = parser.parse_args()

    input_path = Path(args.reflection_slot_csv)
    out_dir = Path(args.out_dir)
    rows = _slot_rows(_read_rows(input_path), source=args.source)
    by_game_target = _group(rows, ("game_id", "target_agent"), vps_floor=args.game_vps_floor)
    by_target = _group(rows, ("target_agent",))
    by_observer = _group(rows, ("observer_agent",))

    _write_csv(out_dir / "reflection_slot_vps.csv", rows)
    _write_csv(out_dir / "game_target_summary.csv", by_game_target)
    _write_csv(out_dir / "target_summary.csv", by_target)
    _write_csv(out_dir / "observer_summary.csv", by_observer)

    summary = {
        "source": args.source,
        "game_vps_floor": args.game_vps_floor,
        "slot_count": len(rows),
        "game_target_count": len(by_game_target),
        "raw_vps_loss_total": sum(float(row["raw_vps_loss"]) for row in rows),
        "calibrated_vps_loss_total": sum(float(row["calibrated_vps_loss"]) for row in rows),
        "excess_calibrated_vps_loss_total": sum(
            float(row.get("excess_calibrated_vps_loss_total") or 0.0)
            for row in by_game_target
        ),
        "excess_raw_vps_loss_total": sum(
            float(row.get("excess_raw_vps_loss_total") or 0.0)
            for row in by_game_target
        ),
        "nonzero_count": sum(1 for row in rows if row["belief_delta_occupied"] != 0),
        "direction_correct_count": sum(1 for row in rows if row["direction_correct"]),
        "direction_wrong_count": sum(1 for row in rows if row["direction_wrong"]),
    }
    summary["nonzero_precision"] = (
        summary["direction_correct_count"] / summary["nonzero_count"]
        if summary["nonzero_count"]
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} reflection VPS slot rows to {out_dir}")
    print(
        "excess_calibrated={excess_calibrated_vps_loss_total:.6f}, "
        "calibrated={calibrated_vps_loss_total:.6f}, raw={raw_vps_loss_total:.6f}, "
        "precision={nonzero_precision}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
