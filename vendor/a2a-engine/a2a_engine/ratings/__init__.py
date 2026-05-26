"""Game-agnostic rating utilities."""

from a2a_engine.ratings.openskill import OpenSkillRater
from a2a_engine.ratings.schemas import (
    MetricSpec,
    PlayerRatingState,
    RatingEvent,
    RatingParticipant,
    RatingSnapshot,
    SkillRating,
)
from a2a_engine.ratings.matchmaking import Matchmaker, MatchmakingPlan
from a2a_engine.ratings.store import (
    ConcurrentRatingUpdate,
    DynamoDBRatingStore,
    InMemoryRatingStore,
    RatingEventAlreadyProcessed,
)

__all__ = [
    "ConcurrentRatingUpdate",
    "DynamoDBRatingStore",
    "InMemoryRatingStore",
    "Matchmaker",
    "MatchmakingPlan",
    "MetricSpec",
    "OpenSkillRater",
    "PlayerRatingState",
    "RatingEventAlreadyProcessed",
    "RatingEvent",
    "RatingParticipant",
    "RatingSnapshot",
    "SkillRating",
]
