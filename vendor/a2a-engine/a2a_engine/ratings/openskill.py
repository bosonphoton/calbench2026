"""OpenSkill-backed free-for-all rating updates."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from a2a_engine.ratings.schemas import (
    MetricSpec,
    PlayerRatingState,
    RatingEvent,
    RatingSnapshot,
    SkillRating,
)


def _require_openskill():
    try:
        from openskill.models import PlackettLuce
    except ImportError as exc:
        raise ImportError(
            "OpenSkill ratings require the 'openskill' package. "
            "Install a2a-engine with its declared dependencies."
        ) from exc
    return PlackettLuce


class OpenSkillRater:
    """Stateful OpenSkill updater for extracted rating events.

    Events are seat-level free-for-all matches. Multiple seats may map to the
    same stable player_id, which is common while player identity is only a model
    name. In that case, rating deltas from those seats are averaged back into the
    single player state for each metric.
    """

    def __init__(
        self,
        metrics: Iterable[MetricSpec],
        *,
        players: dict[str, PlayerRatingState] | None = None,
        processed_game_ids: Iterable[str] | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.metric_by_name = {metric.name: metric for metric in self.metrics}
        self.model = _require_openskill()()
        self.players: dict[str, PlayerRatingState] = dict(players or {})
        self.processed_game_ids: list[str] = list(processed_game_ids or [])

    def _default_skill(self) -> SkillRating:
        rating = self.model.rating()
        return SkillRating(mu=float(rating.mu), sigma=float(rating.sigma))

    def _ensure_player(self, player_id: str) -> PlayerRatingState:
        player = self.players.get(player_id)
        if player is None:
            player = PlayerRatingState(player_id=player_id)
            self.players[player_id] = player
        for metric in self.metrics:
            player.skills.setdefault(metric.name, self._default_skill())
        return player

    def _rating_for(self, player_id: str, metric_name: str):
        skill = self._ensure_player(player_id).skills[metric_name]
        return self.model.rating(mu=skill.mu, sigma=skill.sigma)

    @staticmethod
    def _ranks(scores: list[float], *, higher_is_better: bool, tie_tolerance: float) -> list[int]:
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda item: item[1], reverse=higher_is_better)
        ranks_by_idx: dict[int, int] = {}
        group_score: float | None = None
        current_rank = 0
        for sorted_idx, (original_idx, score) in enumerate(indexed):
            if group_score is None or abs(score - group_score) > tie_tolerance:
                current_rank = sorted_idx
                group_score = score
            ranks_by_idx[original_idx] = current_rank
        return [ranks_by_idx[idx] for idx in range(len(scores))]

    def rate_event(self, event: RatingEvent, *, skip_processed: bool = True) -> bool:
        if skip_processed and event.game_id in self.processed_game_ids:
            return False
        participant_by_id = {p.participant_id: p for p in event.participants}
        participant_counts: dict[str, int] = {}
        for participant in event.participants:
            self._ensure_player(participant.player_id)
            participant_counts[participant.player_id] = participant_counts.get(participant.player_id, 0) + 1

        for metric in self.metrics:
            raw_scores = event.metric_scores.get(metric.name, {})
            participant_ids = [
                participant.participant_id
                for participant in event.participants
                if participant.participant_id in raw_scores
                and math.isfinite(float(raw_scores[participant.participant_id]))
            ]
            if len(participant_ids) < 2:
                continue
            scores = [float(raw_scores[participant_id]) for participant_id in participant_ids]
            metric_tolerances = event.metadata.get("metric_tie_tolerances") or {}
            tie_tolerance = metric_tolerances.get(metric.name, metric.tie_tolerance)
            ranks = self._ranks(
                scores,
                higher_is_better=metric.higher_is_better,
                tie_tolerance=max(0.0, float(tie_tolerance)),
            )
            teams = [
                [self._rating_for(participant_by_id[participant_id].player_id, metric.name)]
                for participant_id in participant_ids
            ]
            updated_teams = self.model.rate(teams, ranks=ranks)

            deltas: dict[str, list[tuple[float, float]]] = {}
            for participant_id, updated_team in zip(participant_ids, updated_teams, strict=True):
                player_id = participant_by_id[participant_id].player_id
                before = self.players[player_id].skills[metric.name]
                after = updated_team[0]
                deltas.setdefault(player_id, []).append((float(after.mu) - before.mu, float(after.sigma) - before.sigma))

            for player_id, player_deltas in deltas.items():
                before = self.players[player_id].skills[metric.name]
                mu_delta = sum(delta[0] for delta in player_deltas) / len(player_deltas)
                sigma_delta = sum(delta[1] for delta in player_deltas) / len(player_deltas)
                self.players[player_id].skills[metric.name] = SkillRating(
                    mu=before.mu + mu_delta,
                    sigma=max(0.0, before.sigma + sigma_delta),
                )

        for player_id, count in participant_counts.items():
            self.players[player_id].games_played += count
        self.processed_game_ids.append(event.game_id)
        return True

    def rate_events(self, events: Iterable[RatingEvent]) -> RatingSnapshot:
        for event in sorted(events, key=lambda item: (item.timestamp, item.game_id)):
            self.rate_event(event)
        return self.snapshot()

    def snapshot(self, **metadata: object) -> RatingSnapshot:
        return RatingSnapshot(
            metrics=self.metrics,
            players=self.players,
            processed_game_ids=self.processed_game_ids,
            generated_at=datetime.now(timezone.utc),
            metadata=dict(metadata),
        )

    @classmethod
    def from_snapshot(cls, snapshot: RatingSnapshot) -> "OpenSkillRater":
        return cls(
            snapshot.metrics,
            players=snapshot.players,
            processed_game_ids=snapshot.processed_game_ids,
        )
