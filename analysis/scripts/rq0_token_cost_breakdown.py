"""Report LLM token cost breakdowns for calendar traces.

This separates total prompt tokens into fresh input tokens and cached prompt
hits. Reasoning tokens are reported separately and also included in output
tokens because Gemini accounts for them in total token usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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
        path for path in paths
        if path.is_file()
        and path.suffix == ".json"
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
    ]


def _empty() -> dict[str, int]:
    return {
        "llm_calls": 0,
        "calls_with_token_usage": 0,
        "prompt_tokens": 0,
        "fresh_prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def token_breakdown(trace: dict[str, Any]) -> dict[str, int]:
    summary = _empty()
    for event in trace.get("events", []):
        raw = event.get("data", {}).get("raw_api_response") or {}
        if not isinstance(raw, dict) or "duration_s" not in raw:
            continue
        summary["llm_calls"] += 1
        if raw.get("prompt_tokens") is None and raw.get("completion_tokens") is None:
            continue
        summary["calls_with_token_usage"] += 1
        prompt = int(raw.get("prompt_tokens") or 0)
        cached = int(raw.get("cached_prompt_tokens") or 0)
        completion = int(raw.get("completion_tokens") or 0)
        reasoning = int(raw.get("reasoning_tokens") or 0)
        total = int(raw.get("total_tokens") or 0)
        summary["prompt_tokens"] += prompt
        summary["cached_prompt_tokens"] += cached
        summary["fresh_prompt_tokens"] += max(prompt - cached, 0)
        summary["completion_tokens"] += completion
        summary["reasoning_tokens"] += reasoning
        summary["output_tokens"] += completion + reasoning
        summary["total_tokens"] += total
    return summary


def _row(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    config = trace.get("config", {})
    metrics = trace.get("metrics", {})
    breakdown = token_breakdown(trace)
    return {
        "trace_path": str(path),
        "experiment_name": config.get("experiment_name") or config.get("name") or path.parent.name,
        "game_id": trace.get("game_id") or path.stem,
        "model_names": ",".join(
            str(agent.get("model") or agent.get("type"))
            for agent in config.get("agents", [])
            if isinstance(agent, dict)
        ),
        "realized_cost": metrics.get("realized_cost"),
        "optimal_cost": metrics.get("optimal_cost"),
        **breakdown,
    }


def main() -> int:
    root = _calendar_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Trace JSON files, directories, or globs.")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows instead of TSV.")
    args = parser.parse_args()

    rows = [_row(path) for path in _trace_paths(args.traces, root=root)]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    columns = [
        "experiment_name",
        "game_id",
        "llm_calls",
        "calls_with_token_usage",
        "fresh_prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "output_tokens",
        "total_tokens",
        "realized_cost",
        "optimal_cost",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column) or 0) for column in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
