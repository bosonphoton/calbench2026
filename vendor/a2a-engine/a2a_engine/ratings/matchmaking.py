"""Matchmaking helpers for live rating-backed leaderboards."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from a2a_engine.ratings.schemas import MetricSpec, PlayerRatingState


@dataclass(frozen=True)
class MatchmakingPlan:
    """A concrete free-for-all player selection."""

    target_player_id: str
    opponent_player_ids: tuple[str, ...]
    phase: str

    @property
    def player_ids(self) -> tuple[str, ...]:
        return (self.target_player_id, *self.opponent_player_ids)


class Matchmaker:
    """Select opponents for live post-benchmark rating phases."""

    def __init__(
        self,
        metrics: Sequence[MetricSpec],
        *,
        warmup_games: int = 90,
        num_opponents: int = 4,
        search_pool_size: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.warmup_games = warmup_games
        self.num_opponents = num_opponents
        self.search_pool_size = search_pool_size
        self.rng = rng or random.Random()

    def _sample(self, player_ids: Sequence[str], count: int) -> tuple[str, ...]:
        if len(player_ids) < count:
            raise ValueError(f"Need {count} opponents, found {len(player_ids)}.")
        return tuple(self.rng.sample(list(player_ids), count))

    def plan_match(
        self,
        target: PlayerRatingState,
        pool: Sequence[PlayerRatingState],
    ) -> MatchmakingPlan:
        """Return a target-plus-opponents plan for an on-the-fly match.

        New players must first complete the fixed benchmark suite represented by
        ``warmup_games`` rated seats. This method only schedules extra live
        arena games after that benchmark barrier has been crossed.
        """

        if target.games_played < self.warmup_games:
            raise ValueError(
                f"{target.player_id!r} needs the fixed benchmark warmup first "
                f"({target.games_played}/{self.warmup_games} rated seats)."
            )

        excluded = {target.player_id}
        eligible = [
            player
            for player in pool
            if player.player_id not in excluded and player.games_played >= self.warmup_games
        ]
        target_mmr = target.mmr(self.metrics)
        eligible.sort(key=lambda player: abs(player.mmr(self.metrics) - target_mmr))
        window = self.search_pool_size or max(self.num_opponents * 3, 15)
        candidates = eligible[:window]
        return MatchmakingPlan(
            target_player_id=target.player_id,
            opponent_player_ids=self._sample([player.player_id for player in candidates], self.num_opponents),
            phase="live",
        )
