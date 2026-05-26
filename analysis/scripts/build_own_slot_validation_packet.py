"""Build a human annotation packet for own-slot disclosure regex validation.

The packet samples messages from the deterministic own-slot disclosure audit and
leaves human label columns blank. After annotation, the same CSV can be used to
compute precision/recall for the appendix table.

Usage:
    cd games/calendar
    uv run python analysis/scripts/build_own_slot_validation_packet.py
"""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
from pathlib import Path


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = CALENDAR_ROOT / "analysis" / "outputs" / "own_slot_disclosure" / "own_slot_message_rows.csv"
DEFAULT_OUT_DIR = CALENDAR_ROOT / "analysis" / "outputs" / "own_slot_disclosure"

METRICS = [
    (
        "own_slot_proposal_message",
        "Own-slot proposal",
        "Does this message propose, suggest, or endorse a slot as workable for the speaker?",
    ),
    (
        "own_slot_availability_message",
        "Own-slot availability",
        "Does this message state that a slot is free, available, or workable for the speaker?",
    ),
    (
        "own_cost_or_constraint_message",
        "Own cost/constraint disclosure",
        "Does this message disclose the speaker's own cost, burden, blocked status, or rescheduling constraint for a slot?",
    ),
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool_text(value: str) -> str:
    return "yes" if str(value).lower() == "true" else "no"


def build_sample(rows: list[dict[str, str]], *, seed: int, per_metric_label: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    sample_id = 1
    nonempty = [row for row in rows if row.get("content", "").strip()]
    for metric_name, metric_label, annotation_question in METRICS:
        positives = [row for row in nonempty if row.get(metric_name) == "True"]
        negatives = [row for row in nonempty if row.get(metric_name) == "False"]
        for regex_label, pool in [("yes", positives), ("no", negatives)]:
            for row in rng.sample(pool, min(per_metric_label, len(pool))):
                out.append(
                    {
                        "sample_id": f"R{sample_id:03d}",
                        "metric": metric_name,
                        "metric_label": metric_label,
                        "annotation_question": annotation_question,
                        "regex_label": regex_label,
                        "human_label_yes_no": "",
                        "human_notes": "",
                        "model": row.get("model", ""),
                        "task_id": row.get("task_id", ""),
                        "game_id": row.get("game_id", ""),
                        "round": row.get("round", ""),
                        "agent_id": row.get("agent_id", ""),
                        "channel": row.get("channel", ""),
                        "slot_mentions": row.get("slot_mentions", ""),
                        "slots": row.get("slots", ""),
                        "content": row.get("content", ""),
                        "trace_path": row.get("trace_path", ""),
                    }
                )
                sample_id += 1
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], *, seed: int, per_metric_label: int) -> None:
    lines = [
        "# Own-Slot Regex Validation Packet",
        "",
        f"Sampling: seed `{seed}`, {per_metric_label} regex-positive and {per_metric_label} regex-negative messages per metric.",
        "",
        "For each row, set `human_label_yes_no` in the CSV to `yes` if the message satisfies the annotation question, otherwise `no`. Add notes only when useful.",
        "",
    ]
    by_metric: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_metric.setdefault(row["metric_label"], []).append(row)

    for metric_label, metric_rows in by_metric.items():
        lines.extend([f"## {metric_label}", "", metric_rows[0]["annotation_question"], ""])
        for row in metric_rows:
            content = textwrap.indent(row["content"].strip(), "> ")
            lines.extend(
                [
                    f"### {row['sample_id']} | Regex: {row['regex_label']} | Model: {row['model']} | Task: {row['task_id']} | Round: {row['round']} | Agent: {row['agent_id']}",
                    "",
                    "- [ ] Human label: yes",
                    "- [ ] Human label: no",
                    "- Notes:",
                    "",
                    content,
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-metric-label", type=int, default=5)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input))
    sample = build_sample(rows, seed=args.seed, per_metric_label=args.per_metric_label)
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "own_slot_regex_validation_sample.csv"
    md_path = out_dir / "own_slot_regex_validation_checklist.md"
    write_csv(csv_path, sample)
    write_markdown(md_path, sample, seed=args.seed, per_metric_label=args.per_metric_label)
    print(f"Wrote {len(sample)} validation rows")
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
