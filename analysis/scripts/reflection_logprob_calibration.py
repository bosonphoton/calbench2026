"""Post-hoc calibration for calendar reflection logprobs.

The calendar game records end-of-game reflection prompts that ask each agent
how much more or less willing they are to say another agent is occupied at
each slot. This script replays those reflection outputs from saved traces and
computes several offline calibrations:

* sampled_delta: direct score from the emitted integer in [-3, 3]
* floor_expected_delta: softmax over {-3..3}, filling missing alternatives with
  the minimum reported top-logprob for that slot token
* floor_upper_abs_delta: directional upper bound when missing alternatives are
  allowed up to that same top-logprob floor
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


DELTA_VALUES = (-3, -2, -1, 0, 1, 2, 3)
DELTA_MAX = 3


@dataclass
class SlotLogprobs:
    values: dict[int, float]
    floor: float | None


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    if root_candidate.exists():
        return root_candidate
    return Path.cwd() / candidate


def _trace_paths(inputs: Iterable[str], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = _resolve(raw, root=root)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif any(char in raw for char in "*?[]"):
            paths.extend(sorted(Path().glob(raw)))
        else:
            paths.append(path)
    return [
        path for path in paths
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
        and "_reports" not in path.parts
        and "_index" not in path.parts
    ]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_deltas(text: str | None, num_slots: int) -> list[int | None]:
    deltas: list[int | None] = [None for _ in range(num_slots)]
    if not text:
        return deltas
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("deltas") or parsed.get("states")
    if isinstance(parsed, list):
        for index, value in enumerate(parsed[:num_slots]):
            try:
                delta = int(value)
            except (TypeError, ValueError):
                continue
            deltas[index] = max(-DELTA_MAX, min(DELTA_MAX, delta))
        return deltas
    for index, value in enumerate(re.findall(r"(?<!\d)-?[0-3](?!\d)", text)[:num_slots]):
        deltas[index] = int(value)
    return deltas


def _flatten_content(raw: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            content = obj.get("content")
            if isinstance(content, list):
                for entry in content:
                    if isinstance(entry, dict) and "token" in entry:
                        items.append(entry)
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(raw)
    return items


def _token_digit(token: Any) -> int | None:
    text = str(token).strip()
    return int(text) if text in {"0", "1", "2", "3"} else None


def _extract_slot_logprobs(raw: Any, deltas: list[int | None], num_slots: int) -> list[SlotLogprobs]:
    """Best-effort extraction for the numeric JSON decision tokens.

    Negative JSON numbers are often tokenized as a sign-bearing comma token
    followed by a digit token. We use the emitted delta's sign to interpret
    digit alternatives in that context.
    """
    slots: list[SlotLogprobs] = []
    for entry in _flatten_content(raw):
        digit = _token_digit(entry.get("token"))
        if digit is None:
            continue
        slot_index = len(slots)
        if slot_index >= num_slots:
            break
        emitted_delta = deltas[slot_index]
        sign = -1 if emitted_delta is not None and emitted_delta < 0 else 1
        values: dict[int, float] = {}
        floors: list[float] = []

        if isinstance(entry.get("logprob"), (int, float)):
            emitted_digit = abs(emitted_delta) if emitted_delta is not None else digit
            emitted_value = 0 if emitted_digit == 0 else sign * int(emitted_digit)
            values[emitted_value] = float(entry["logprob"])
            floors.append(float(entry["logprob"]))

        top = entry.get("top_logprobs")
        if isinstance(top, list):
            for top_entry in top:
                if not isinstance(top_entry, dict):
                    continue
                if not isinstance(top_entry.get("logprob"), (int, float)):
                    continue
                top_lp = float(top_entry["logprob"])
                floors.append(top_lp)
                top_digit = _token_digit(top_entry.get("token"))
                if top_digit is None:
                    continue
                candidate = 0 if top_digit == 0 else sign * top_digit
                values[candidate] = top_lp

        slots.append(SlotLogprobs(values=values, floor=min(floors) if floors else None))

    while len(slots) < num_slots:
        slots.append(SlotLogprobs(values={}, floor=None))
    return slots


def _softmax(logprobs: dict[int, float]) -> dict[int, float]:
    if not logprobs:
        return {}
    baseline = max(logprobs.values())
    weights = {key: math.exp(value - baseline) for key, value in logprobs.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else {}


def _expected_delta(logprobs: dict[int, float]) -> float | None:
    probs = _softmax(logprobs)
    if not probs:
        return None
    return sum(delta * prob for delta, prob in probs.items())


def _floor_filled_expected(slot: SlotLogprobs) -> float | None:
    if slot.floor is None and not slot.values:
        return None
    floor = slot.floor if slot.floor is not None else min(slot.values.values())
    filled = {delta: slot.values.get(delta, floor) for delta in DELTA_VALUES}
    return _expected_delta(filled)


def _floor_upper_abs_expected(slot: SlotLogprobs) -> float | None:
    if slot.floor is None and not slot.values:
        return None
    floor = slot.floor if slot.floor is not None else min(slot.values.values())

    def directional(sign: int) -> float | None:
        candidates: dict[int, float] = {}
        for delta, logprob in slot.values.items():
            if delta == 0 or delta * sign > 0:
                candidates[delta] = logprob
        for delta in DELTA_VALUES:
            if delta * sign > 0 and delta not in candidates:
                candidates[delta] = floor
        return _expected_delta(candidates)

    positive = directional(1)
    negative = directional(-1)
    candidates = [value for value in (positive, negative) if value is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda value: abs(value))


def _prob_busy_from_delta(delta: float | None) -> float | None:
    if delta is None:
        return None
    return min(1.0, max(0.0, 0.5 + delta / (2 * DELTA_MAX)))


def _slot_busy(calendars: Any, target_agent_id: int, slot: int) -> int | None:
    try:
        return 1 if calendars[target_agent_id][slot] is not None else 0
    except (TypeError, IndexError):
        return None


def _reflection_rows(trace: dict[str, Any], trace_path: Path) -> list[dict[str, Any]]:
    calendars = (trace.get("final_state") or {}).get("calendars") or []
    rows: list[dict[str, Any]] = []
    for event in trace.get("events") or []:
        if event.get("type") != "reflection_end":
            continue
        data = event.get("data") or {}
        try:
            agent_id = int(data["agent_id"])
            target_agent_id = int(data["target_agent_id"])
            num_slots = int(data.get("num_slots") or len(data.get("estimates") or []))
        except (KeyError, TypeError, ValueError):
            continue
        deltas = _parse_deltas(data.get("text"), num_slots)
        slot_logprobs = _extract_slot_logprobs(data.get("raw_api_response"), deltas, num_slots)
        for slot in range(num_slots):
            sampled_delta = deltas[slot]
            floor_delta = _floor_filled_expected(slot_logprobs[slot])
            upper_delta = _floor_upper_abs_expected(slot_logprobs[slot])
            rows.append({
                "trace_path": str(trace_path),
                "game_id": trace.get("game_id") or trace_path.stem,
                "agent_id": agent_id,
                "target_agent_id": target_agent_id,
                "slot": slot,
                "actual_busy": _slot_busy(calendars, target_agent_id, slot),
                "sampled_delta": sampled_delta,
                "sampled_probability_busy": _prob_busy_from_delta(sampled_delta),
                "sampled_vps_loss": abs(sampled_delta) / (2 * DELTA_MAX) if sampled_delta is not None else None,
                "floor_expected_delta": floor_delta,
                "floor_probability_busy": _prob_busy_from_delta(floor_delta),
                "floor_vps_loss": abs(floor_delta) / (2 * DELTA_MAX) if floor_delta is not None else None,
                "floor_upper_abs_delta": upper_delta,
                "floor_upper_probability_busy": _prob_busy_from_delta(upper_delta),
                "floor_upper_vps_loss": abs(upper_delta) / (2 * DELTA_MAX) if upper_delta is not None else None,
                "observed_logprob_values": json.dumps(slot_logprobs[slot].values, sort_keys=True),
                "top_logprob_floor": slot_logprobs[slot].floor,
            })
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def nums(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    sampled = nums("sampled_vps_loss")
    floor = nums("floor_vps_loss")
    upper = nums("floor_upper_vps_loss")
    nonzero_sampled = [row for row in rows if row.get("sampled_delta") not in (None, 0)]
    return {
        "reflection_estimate_count": len(rows),
        "sampled_vps_loss_total": sum(sampled),
        "sampled_vps_loss_mean": _mean(sampled),
        "floor_vps_loss_total": sum(floor),
        "floor_vps_loss_mean": _mean(floor),
        "floor_upper_vps_loss_total": sum(upper),
        "floor_upper_vps_loss_mean": _mean(upper),
        "nonzero_sampled_delta_count": len(nonzero_sampled),
        "logprob_slot_count": len(floor),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trace JSON files, directories, or glob patterns.")
    parser.add_argument("--out-dir", default="analysis/outputs/reflection_logprob_calibration")
    args = parser.parse_args()

    root = _calendar_root()
    out_dir = _resolve(args.out_dir, root=root)
    rows: list[dict[str, Any]] = []
    for trace_path in _trace_paths(args.inputs, root=root):
        trace = _load_json(trace_path)
        if isinstance(trace, dict):
            rows.extend(_reflection_rows(trace, trace_path))

    summary = _summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "reflection_slot_calibration.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} reflection slot rows to {out_dir}")
    print(
        "sampled={sampled_vps_loss_total:.6f}, floor={floor_vps_loss_total:.6f}, "
        "upper={floor_upper_vps_loss_total:.6f}".format(**summary)
    )


if __name__ == "__main__":
    main()
