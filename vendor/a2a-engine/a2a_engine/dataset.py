"""Game-agnostic Dataset interface over GameTraceBase JSON dumps.

`GameDataset` mirrors the structure of `NegotiationDataset` (from the
a2a-llm-judge analysis layer) but knows nothing about a specific game. Use
``GameDataset.from_dir(path)`` to load every trace under a results directory
and then ``to_games_df`` / ``to_messages_df`` / ``to_events_df`` for analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from pydantic import BaseModel, ConfigDict

from a2a_engine.schemas import AgentInfo, GameTraceBase


class GameMessage(BaseModel):
    """One natural-language message extracted from a trace's events."""

    model_config = ConfigDict(extra="allow")

    turn: int
    speaker: str
    text: str
    timestamp: datetime


class GameRecord(BaseModel):
    """Thin pydantic wrapper around a GameTraceBase with computed views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace: GameTraceBase

    # ---- pass-through ids ----
    @property
    def game_id(self) -> str:
        return self.trace.game_id

    @property
    def game_name(self) -> str:
        return self.trace.config.game_name

    @property
    def experiment_name(self) -> str | None:
        return self.trace.config.experiment_name

    @property
    def experiment_run_id(self) -> str | None:
        return self.trace.config.experiment_run_id

    @property
    def duration_seconds(self) -> float | None:
        if self.trace.ended_at is None:
            return None
        return (self.trace.ended_at - self.trace.started_at).total_seconds()

    @property
    def agents(self) -> list[AgentInfo]:
        return list(self.trace.config.agents)

    @property
    def messages(self) -> list[GameMessage]:
        out: list[GameMessage] = []
        turn = 0
        for ev in self.trace.events:
            data = ev.data or {}
            speaker = data.get("speaker")
            text = data.get("text")
            if ev.type == "message" or (speaker and text is not None):
                if speaker is None or text is None:
                    continue
                out.append(GameMessage(turn=turn, speaker=str(speaker), text=str(text), timestamp=ev.timestamp))
                turn += 1
        return out

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.trace.metrics)

    @property
    def final_state(self) -> dict[str, Any]:
        return dict(self.trace.final_state)


class GameDataset:
    """A collection of GameRecord with DataFrame builders."""

    def __init__(self, records: list[GameRecord]) -> None:
        self.records = list(records)

    # ---- constructors ----
    @classmethod
    def from_dir(cls, path: str | Path) -> "GameDataset":
        root = Path(path)
        records: list[GameRecord] = []
        for jp in sorted(root.rglob("*.json")):
            try:
                trace = GameTraceBase.model_validate_json(jp.read_text())
            except Exception:
                continue
            records.append(GameRecord(trace=trace))
        return cls(records)

    @classmethod
    def from_traces(cls, traces: list[GameTraceBase]) -> "GameDataset":
        return cls([GameRecord(trace=t) for t in traces])

    # ---- container ----
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[GameRecord]:
        return iter(self.records)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return GameDataset(self.records[idx])
        return self.records[idx]

    # ---- filtering ----
    def filter(self, predicate: Callable[[GameRecord], bool]) -> "GameDataset":
        return GameDataset([r for r in self.records if predicate(r)])

    def filter_by(self, **kwargs: Any) -> "GameDataset":
        def ok(r: GameRecord) -> bool:
            for k, v in kwargs.items():
                if getattr(r, k, None) != v:
                    return False
            return True
        return self.filter(ok)

    # ---- DataFrames ----
    def to_games_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in self.records:
            row: dict[str, Any] = {
                "game_id": r.game_id,
                "game_name": r.game_name,
                "experiment_name": r.experiment_name,
                "experiment_run_id": r.experiment_run_id,
                "num_agents": len(r.agents),
                "num_events": len(r.trace.events),
                "num_messages": len(r.messages),
                "duration_s": r.duration_seconds,
            }
            for k, v in r.metrics.items():
                row[f"metrics_{k}"] = v
            for k, v in r.final_state.items():
                row[f"final_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def to_messages_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in self.records:
            for m in r.messages:
                rows.append({
                    "game_id": r.game_id,
                    "turn": m.turn,
                    "speaker": m.speaker,
                    "text": m.text,
                    "char_count": len(m.text),
                    "timestamp": m.timestamp,
                })
        return pd.DataFrame(rows)

    def to_events_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in self.records:
            for i, ev in enumerate(r.trace.events):
                rows.append({
                    "game_id": r.game_id,
                    "event_idx": i,
                    "type": ev.type,
                    "timestamp": ev.timestamp,
                    "data": json.dumps(ev.data, default=str),
                })
        return pd.DataFrame(rows)
