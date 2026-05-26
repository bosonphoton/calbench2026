"""Serializable schemas for game-agnostic player ratings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MetricSpec(BaseModel):
    """One independently-rated metric in a game."""

    name: str
    higher_is_better: bool = True
    weight: float = 1.0
    tie_tolerance: float = 0.0


class RatingParticipant(BaseModel):
    """A concrete seat in a rating event, mapped to a stable player identity."""

    participant_id: str
    player_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RatingEvent(BaseModel):
    """A single free-for-all rating event extracted from a game trace."""

    game_id: str
    game_name: str
    participants: list[RatingParticipant]
    metric_scores: dict[str, dict[str, float]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillRating(BaseModel):
    """Serializable OpenSkill mu/sigma pair."""

    mu: float
    sigma: float


class PlayerRatingState(BaseModel):
    """All rating state for one stable player identity."""

    player_id: str
    games_played: int = 0
    version: int = 0
    skills: dict[str, SkillRating] = Field(default_factory=dict)

    def mmr(self, metrics: list[MetricSpec]) -> float:
        total_weight = sum(max(0.0, metric.weight) for metric in metrics)
        if total_weight <= 0:
            total_weight = float(len(metrics) or 1)
        score = 0.0
        for metric in metrics:
            skill = self.skills.get(metric.name)
            if skill is None:
                continue
            score += (max(0.0, metric.weight) / total_weight) * skill.mu
        return score


class RatingSnapshot(BaseModel):
    """A deterministic rating snapshot derived from processed rating events."""

    schema_version: int = 1
    model: str = "openskill.plackett_luce"
    metrics: list[MetricSpec]
    players: dict[str, PlayerRatingState] = Field(default_factory=dict)
    processed_game_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def leaderboard(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for player in self.players.values():
            row: dict[str, Any] = {
                "player_id": player.player_id,
                "games_played": player.games_played,
                "mmr": player.mmr(self.metrics),
            }
            for metric in self.metrics:
                skill = player.skills.get(metric.name)
                row[f"{metric.name}_mu"] = skill.mu if skill else None
                row[f"{metric.name}_sigma"] = skill.sigma if skill else None
            rows.append(row)
        return sorted(rows, key=lambda row: row["mmr"], reverse=True)
