"""Live rating stores for concurrent leaderboard updates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

from a2a_engine.ratings.openskill import OpenSkillRater
from a2a_engine.ratings.schemas import MetricSpec, PlayerRatingState, RatingEvent


class RatingEventAlreadyProcessed(RuntimeError):
    """Raised when an event idempotency key is already present."""


class ConcurrentRatingUpdate(RuntimeError):
    """Raised when optimistic rating-state writes lose a race."""


class RatingStore(Protocol):
    """Storage contract used by live workers."""

    def get_players(self, player_ids: Sequence[str]) -> dict[str, PlayerRatingState]:
        ...

    def apply_event(self, event: RatingEvent) -> bool:
        ...

    def leaderboard(self) -> list[dict[str, Any]]:
        ...


def _participants_player_ids(event: RatingEvent) -> list[str]:
    return sorted({participant.player_id for participant in event.participants})


def _updated_players_for_event(
    metrics: Sequence[MetricSpec],
    event: RatingEvent,
    players: dict[str, PlayerRatingState],
) -> dict[str, PlayerRatingState]:
    rater = OpenSkillRater(metrics, players=deepcopy(players))
    rater.rate_event(event, skip_processed=False)
    return {player_id: rater.players[player_id] for player_id in _participants_player_ids(event)}


class InMemoryRatingStore:
    """Thread-light in-memory store with the same idempotency semantics."""

    def __init__(self, metrics: Sequence[MetricSpec]) -> None:
        self.metrics = list(metrics)
        self.players: dict[str, PlayerRatingState] = {}
        self.processed_game_ids: set[str] = set()

    def get_players(self, player_ids: Sequence[str]) -> dict[str, PlayerRatingState]:
        return {
            player_id: deepcopy(self.players.get(player_id) or PlayerRatingState(player_id=player_id))
            for player_id in player_ids
        }

    def apply_event(self, event: RatingEvent) -> bool:
        if event.game_id in self.processed_game_ids:
            return False
        player_ids = _participants_player_ids(event)
        current = self.get_players(player_ids)
        updated = _updated_players_for_event(self.metrics, event, current)
        for player_id, state in updated.items():
            state.version = current[player_id].version + 1
            self.players[player_id] = state
        self.processed_game_ids.add(event.game_id)
        return True

    def leaderboard(self) -> list[dict[str, Any]]:
        return OpenSkillRater(self.metrics, players=deepcopy(self.players)).snapshot().leaderboard()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dynamo(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _to_dynamo(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_dynamo(item) for item in value]
    return value


def _from_dynamo(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {key: _from_dynamo(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_dynamo(item) for item in value]
    return value


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


class DynamoDBRatingStore:
    """DynamoDB-backed store for multi-worker live OpenSkill updates.

    Table shape:
    - partition key: ``pk`` string
    - sort key: ``sk`` string

    Player rows are stored under ``pk=rating#<leaderboard_id>`` and
    ``sk=player#<player_id>``. Processed event rows use ``sk=event#<game_id>``.
    Each event update is a DynamoDB transaction: create the event idempotency row
    and conditionally replace only the participating player rows by version.
    """

    def __init__(
        self,
        table_name: str,
        metrics: Sequence[MetricSpec],
        *,
        leaderboard_id: str = "default",
        dynamodb_resource: Any | None = None,
        dynamodb_client: Any | None = None,
        region_name: str | None = None,
        max_retries: int = 5,
    ) -> None:
        if dynamodb_client is None and dynamodb_resource is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError("DynamoDBRatingStore requires boto3.") from exc
            dynamodb_client = boto3.client("dynamodb", region_name=region_name)
        elif dynamodb_client is None:
            dynamodb_client = dynamodb_resource.meta.client
        self.table_name = table_name
        self.client = dynamodb_client
        try:
            from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
        except ImportError as exc:
            raise ImportError("DynamoDBRatingStore requires boto3.") from exc
        self.serializer = TypeSerializer()
        self.deserializer = TypeDeserializer()
        self.metrics = list(metrics)
        self.leaderboard_id = leaderboard_id
        self.pk = f"rating#{leaderboard_id}"
        self.max_retries = max_retries

    @staticmethod
    def player_sk(player_id: str) -> str:
        return f"player#{player_id}"

    @staticmethod
    def event_sk(game_id: str) -> str:
        return f"event#{game_id}"

    def _serialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self.serializer.serialize(value)
            for key, value in _to_dynamo(_strip_none(item)).items()
        }

    def _deserialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: _from_dynamo(self.deserializer.deserialize(value))
            for key, value in item.items()
        }

    def get_players(self, player_ids: Sequence[str]) -> dict[str, PlayerRatingState]:
        if not player_ids:
            return {}
        keys = [
            self._serialize_item({"pk": self.pk, "sk": self.player_sk(player_id)})
            for player_id in sorted(set(player_ids))
        ]
        response = self.client.batch_get_item(RequestItems={self.table_name: {"Keys": keys}})
        items = response.get("Responses", {}).get(self.table_name, [])
        found: dict[str, PlayerRatingState] = {}
        for raw_item in items:
            item = self._deserialize_item(raw_item)
            state_data = _from_dynamo(item.get("state") or {})
            player = PlayerRatingState.model_validate(state_data)
            found[player.player_id] = player
        return {
            player_id: found.get(player_id) or PlayerRatingState(player_id=player_id)
            for player_id in player_ids
        }

    def _event_exists(self, game_id: str) -> bool:
        response = self.client.get_item(
            TableName=self.table_name,
            Key=self._serialize_item({"pk": self.pk, "sk": self.event_sk(game_id)}),
            ProjectionExpression="pk",
        )
        return "Item" in response

    def _put_event_txn(self, event: RatingEvent) -> dict[str, Any]:
        item = {
            "pk": self.pk,
            "sk": self.event_sk(event.game_id),
            "item_type": "event",
            "game_id": event.game_id,
            "game_name": event.game_name,
            "timestamp": event.timestamp.isoformat(),
            "source_path": event.source_path,
            "participant_player_ids": _participants_player_ids(event),
            "event": event.model_dump(mode="json"),
            "metadata": event.metadata,
            "created_at": _utc_now(),
        }
        return {
            "Put": {
                "TableName": self.table_name,
                "Item": self._serialize_item(item),
                "ConditionExpression": "attribute_not_exists(pk)",
            }
        }

    def _put_player_txn(self, state: PlayerRatingState, expected_version: int) -> dict[str, Any]:
        state.version = expected_version + 1
        item = {
            "pk": self.pk,
            "sk": self.player_sk(state.player_id),
            "item_type": "player",
            "player_id": state.player_id,
            "version": state.version,
            "games_played": state.games_played,
            "state": state.model_dump(mode="json"),
            "updated_at": _utc_now(),
        }
        return {
            "Put": {
                "TableName": self.table_name,
                "Item": self._serialize_item(item),
                "ConditionExpression": "attribute_not_exists(pk) OR #version = :expected_version",
                "ExpressionAttributeNames": {"#version": "version"},
                "ExpressionAttributeValues": {
                    ":expected_version": self.serializer.serialize(expected_version),
                },
            }
        }

    def apply_event(self, event: RatingEvent) -> bool:
        player_ids = _participants_player_ids(event)
        for _ in range(self.max_retries):
            current = self.get_players(player_ids)
            expected_versions = {player_id: current[player_id].version for player_id in player_ids}
            updated = _updated_players_for_event(self.metrics, event, current)
            transact_items = [self._put_event_txn(event)]
            transact_items.extend(
                self._put_player_txn(updated[player_id], expected_versions[player_id])
                for player_id in player_ids
            )
            try:
                self.client.transact_write_items(TransactItems=transact_items)
                return True
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if code != "TransactionCanceledException":
                    raise
                if self._event_exists(event.game_id):
                    return False
        raise ConcurrentRatingUpdate(f"Could not apply rating event {event.game_id!r} after retries.")

    def leaderboard(self) -> list[dict[str, Any]]:
        rows: list[PlayerRatingState] = []
        start_key: dict[str, Any] | None = None
        while True:
            kwargs: dict[str, Any] = {
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {":pk": self.serializer.serialize(self.pk)},
            }
            if start_key:
                kwargs["ExclusiveStartKey"] = start_key
            response = self.client.query(**kwargs)
            for raw_item in response.get("Items", []):
                item = self._deserialize_item(raw_item)
                if item.get("item_type") != "player":
                    continue
                rows.append(PlayerRatingState.model_validate(_from_dynamo(item["state"])))
            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                break
        return OpenSkillRater(self.metrics, players={row.player_id: row for row in rows}).snapshot().leaderboard()
