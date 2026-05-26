"""Index calendar trace outcomes into SQLite for fast experiment lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


DEFAULT_DB = Path("analysis/outcomes/calendar_outcomes.sqlite3")


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
            paths.extend(sorted(Path().glob(raw)))
        else:
            paths.append(path)
    return [
        path for path in paths
        if path.is_file()
        and path.suffix == ".json"
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
    ]


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trace_outcomes (
            trace_path TEXT PRIMARY KEY,
            game_id TEXT,
            experiment_name TEXT,
            started_at TEXT,
            ended_at TEXT,
            scenario_seed INTEGER,
            num_agents INTEGER,
            num_slots INTEGER,
            density REAL,
            pref_level INTEGER,
            num_meetings INTEGER,
            num_participants INTEGER,
            model_names TEXT,
            coordination_rate REAL,
            meetings_scheduled INTEGER,
            total_dms_sent INTEGER,
            total_dm_chars INTEGER,
            avg_dm_chars REAL,
            max_dm_chars INTEGER,
            dm_chars_per_meeting REAL,
            slot_conflict_rate REAL,
            realized_cost REAL,
            optimal_cost REAL,
            cost_gap REAL,
            is_optimal INTEGER,
            fallback_displacement_cost REAL,
            efficiency REAL,
            fairness REAL,
            llm_calls INTEGER,
            prompt_tokens INTEGER,
            fresh_prompt_tokens INTEGER,
            cached_prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            metrics_json TEXT,
            config_json TEXT,
            indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_trace_outcomes_experiment
            ON trace_outcomes(experiment_name);
        CREATE INDEX IF NOT EXISTS idx_trace_outcomes_gap
            ON trace_outcomes(cost_gap);
        CREATE INDEX IF NOT EXISTS idx_trace_outcomes_seed
            ON trace_outcomes(scenario_seed);
        """
    )
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(trace_outcomes)").fetchall()
    }
    for column, spec in {
        "llm_calls": "INTEGER",
        "prompt_tokens": "INTEGER",
        "fresh_prompt_tokens": "INTEGER",
        "cached_prompt_tokens": "INTEGER",
        "completion_tokens": "INTEGER",
        "reasoning_tokens": "INTEGER",
        "output_tokens": "INTEGER",
        "total_tokens": "INTEGER",
    }.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE trace_outcomes ADD COLUMN {column} {spec}")


def _experiment_name(trace_path: Path, trace: dict[str, Any]) -> str:
    config = trace.get("config", {})
    for key in ("experiment_name", "name"):
        value = config.get(key)
        if value:
            return str(value)
    parent = trace_path.parent.name
    return parent if parent != "results" else ""


def _model_names(config: dict[str, Any]) -> str:
    agents = config.get("agents") or []
    names = []
    for agent in agents:
        if isinstance(agent, dict):
            model = agent.get("model") or agent.get("type")
            if model:
                names.append(str(model))
    return ",".join(names)


def _token_breakdown(events: list[dict[str, Any]]) -> dict[str, int | None]:
    calls = 0
    prompt = 0
    cached = 0
    completion = 0
    reasoning = 0
    total = 0
    known_any = False
    for event in events:
        raw = event.get("data", {}).get("raw_api_response") or {}
        if not isinstance(raw, dict) or "duration_s" not in raw:
            continue
        calls += 1
        for key, bucket in (
            ("prompt_tokens", "prompt"),
            ("cached_prompt_tokens", "cached"),
            ("completion_tokens", "completion"),
            ("reasoning_tokens", "reasoning"),
            ("total_tokens", "total"),
        ):
            value = raw.get(key)
            if value is None:
                continue
            known_any = True
            if bucket == "prompt":
                prompt += int(value)
            elif bucket == "cached":
                cached += int(value)
            elif bucket == "completion":
                completion += int(value)
            elif bucket == "reasoning":
                reasoning += int(value)
            elif bucket == "total":
                total += int(value)
    if not known_any:
        return {
            "llm_calls": calls,
            "prompt_tokens": None,
            "fresh_prompt_tokens": None,
            "cached_prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
    return {
        "llm_calls": calls,
        "prompt_tokens": prompt,
        "fresh_prompt_tokens": max(prompt - cached, 0),
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "output_tokens": completion + reasoning,
        "total_tokens": total,
    }


def _row_for_trace(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    config = trace.get("config", {})
    metrics = trace.get("metrics", {})
    events = trace.get("events", [])
    game_start = next((event.get("data", {}) for event in events if event.get("type") == "game_start"), {})
    round_starts = [event for event in events if event.get("type") == "round_start"]
    participant_sizes = [
        len(event.get("data", {}).get("meeting", {}).get("participants", []))
        for event in round_starts
        if event.get("data", {}).get("meeting", {}).get("participants")
    ]
    realized_cost = metrics.get("realized_cost")
    optimal_cost = metrics.get("optimal_cost")
    cost_gap = None
    is_optimal = None
    if realized_cost is not None and optimal_cost is not None:
        cost_gap = float(realized_cost) - float(optimal_cost)
        is_optimal = int(cost_gap == 0)
    token_breakdown = _token_breakdown(events)
    dm_events = [event for event in events if event.get("type") == "dm_sent"]
    dm_lengths = [
        int(event.get("data", {}).get("content_chars", len(str(event.get("data", {}).get("content", "")))))
        for event in dm_events
    ]
    total_dm_chars = metrics.get("total_dm_chars", sum(dm_lengths))
    avg_dm_chars = metrics.get("avg_dm_chars", (sum(dm_lengths) / len(dm_lengths) if dm_lengths else 0.0))
    max_dm_chars = metrics.get("max_dm_chars", (max(dm_lengths) if dm_lengths else 0))
    dm_chars_per_meeting = metrics.get(
        "dm_chars_per_meeting",
        (sum(dm_lengths) / metrics.get("meetings_scheduled") if metrics.get("meetings_scheduled") else None),
    )
    return {
        "trace_path": str(trace_path),
        "game_id": trace.get("game_id") or trace_path.stem,
        "experiment_name": _experiment_name(trace_path, trace),
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "scenario_seed": config.get("seed") or config.get("scenario_seed") or game_start.get("scenario_seed"),
        "num_agents": config.get("num_agents") or game_start.get("num_agents"),
        "num_slots": config.get("num_slots") or game_start.get("num_slots"),
        "density": config.get("density"),
        "pref_level": config.get("pref_level"),
        "num_meetings": config.get("num_meetings") or len(round_starts) or None,
        "num_participants": config.get("num_participants") or (participant_sizes[0] if len(set(participant_sizes)) == 1 and participant_sizes else None),
        "model_names": _model_names(config),
        "coordination_rate": metrics.get("coordination_rate"),
        "meetings_scheduled": metrics.get("meetings_scheduled"),
        "total_dms_sent": metrics.get("total_dms_sent"),
        "total_dm_chars": total_dm_chars,
        "avg_dm_chars": avg_dm_chars,
        "max_dm_chars": max_dm_chars,
        "dm_chars_per_meeting": dm_chars_per_meeting,
        "slot_conflict_rate": metrics.get("slot_conflict_rate"),
        "realized_cost": realized_cost,
        "optimal_cost": optimal_cost,
        "cost_gap": cost_gap,
        "is_optimal": is_optimal,
        "fallback_displacement_cost": metrics.get("fallback_displacement_cost"),
        "efficiency": metrics.get("efficiency"),
        "fairness": metrics.get("fairness"),
        **token_breakdown,
        "metrics_json": json.dumps(metrics, sort_keys=True),
        "config_json": json.dumps(config, sort_keys=True),
    }


def index_traces(db_path: Path, trace_paths: list[Path]) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _init_db(conn)
        rows = [_row_for_trace(path) for path in trace_paths]
        if not rows:
            return 0
        columns = list(rows[0].keys())
        placeholders = ", ".join(":" + column for column in columns)
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column != "trace_path"
        )
        conn.executemany(
            f"""
            INSERT INTO trace_outcomes ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(trace_path) DO UPDATE SET {updates}, indexed_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def main() -> int:
    root = _calendar_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Trace JSON files, directories, or globs.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    trace_paths = _trace_paths(args.traces, root=root)
    db_path = _resolve(args.db, root=root)
    count = index_traces(db_path, trace_paths)
    print(f"indexed {count} trace(s) into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
