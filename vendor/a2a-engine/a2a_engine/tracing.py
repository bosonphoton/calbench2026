"""Local JSON tracing helpers — no Firestore, no network."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from a2a_engine.schemas import GameEvent, GameTraceBase


class EventLog:
    """Thread-safe in-memory event accumulator."""

    def __init__(self) -> None:
        self._events: list[GameEvent] = []
        self._lock = threading.Lock()

    def append(self, type: str, data: dict[str, Any] | None = None, **extra) -> GameEvent:
        ev = GameEvent(type=type, data=data or {}, **extra)
        with self._lock:
            self._events.append(ev)
        return ev

    def all(self) -> list[GameEvent]:
        with self._lock:
            return list(self._events)


def write_trace(trace: GameTraceBase, results_dir: str | Path, experiment_name: str | None = None) -> Path:
    """Write a GameTrace to ``<results_dir>/<experiment_name>/<game_id>.json``."""
    base = Path(results_dir)
    if experiment_name:
        base = base / experiment_name
    base.mkdir(parents=True, exist_ok=True)
    if trace.ended_at is None:
        trace.ended_at = datetime.utcnow()
    out_path = base / f"{trace.game_id}.json"
    out_path.write_text(trace.model_dump_json(indent=2))
    return out_path


def read_trace(path: str | Path) -> GameTraceBase:
    return GameTraceBase.model_validate_json(Path(path).read_text())
