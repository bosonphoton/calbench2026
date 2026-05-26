import random

from a2a_engine.ratings import (
    InMemoryRatingStore,
    Matchmaker,
    MetricSpec,
    OpenSkillRater,
    PlayerRatingState,
    RatingEvent,
    RatingParticipant,
)


def test_openskill_rater_updates_independent_metrics():
    metrics = [
        MetricSpec(name="coordination", higher_is_better=True),
        MetricSpec(name="cost", higher_is_better=False),
    ]
    event = RatingEvent(
        game_id="g1",
        game_name="test",
        participants=[
            RatingParticipant(participant_id="seat0", player_id="model-a"),
            RatingParticipant(participant_id="seat1", player_id="model-b"),
        ],
        metric_scores={
            "coordination": {"seat0": 1.0, "seat1": 0.0},
            "cost": {"seat0": 5.0, "seat1": 1.0},
        },
    )

    snapshot = OpenSkillRater(metrics).rate_events([event])

    assert snapshot.players["model-a"].skills["coordination"].mu > snapshot.players["model-b"].skills["coordination"].mu
    assert snapshot.players["model-b"].skills["cost"].mu > snapshot.players["model-a"].skills["cost"].mu


def test_duplicate_player_seats_are_aggregated():
    metrics = [MetricSpec(name="coordination", higher_is_better=True)]
    event = RatingEvent(
        game_id="g1",
        game_name="test",
        participants=[
            RatingParticipant(participant_id="seat0", player_id="model-a"),
            RatingParticipant(participant_id="seat1", player_id="model-a"),
            RatingParticipant(participant_id="seat2", player_id="model-b"),
        ],
        metric_scores={"coordination": {"seat0": 1.0, "seat1": 0.5, "seat2": 0.0}},
    )

    snapshot = OpenSkillRater(metrics).rate_events([event])

    assert set(snapshot.players) == {"model-a", "model-b"}
    assert snapshot.players["model-a"].games_played == 2
    assert snapshot.players["model-b"].games_played == 1


def test_in_memory_rating_store_applies_event_once():
    metrics = [MetricSpec(name="coordination", higher_is_better=True)]
    event = RatingEvent(
        game_id="g1",
        game_name="test",
        participants=[
            RatingParticipant(participant_id="seat0", player_id="model-a"),
            RatingParticipant(participant_id="seat1", player_id="model-b"),
        ],
        metric_scores={"coordination": {"seat0": 1.0, "seat1": 0.0}},
    )
    store = InMemoryRatingStore(metrics)

    assert store.apply_event(event) is True
    assert store.apply_event(event) is False

    players = store.get_players(["model-a", "model-b"])
    assert players["model-a"].games_played == 1
    assert players["model-a"].version == 1
    assert players["model-a"].skills["coordination"].mu > players["model-b"].skills["coordination"].mu


def test_ranks_use_margin_from_group_anchor():
    ranks = OpenSkillRater._ranks([0.0, 1.0, 2.0], higher_is_better=False, tie_tolerance=1.0)

    assert ranks == [0, 0, 2]


def test_event_metadata_can_override_metric_tie_tolerance():
    metrics = [MetricSpec(name="cost", higher_is_better=False, tie_tolerance=0.0)]
    event = RatingEvent(
        game_id="g1",
        game_name="test",
        participants=[
            RatingParticipant(participant_id="seat0", player_id="model-a"),
            RatingParticipant(participant_id="seat1", player_id="model-b"),
        ],
        metric_scores={"cost": {"seat0": 0.0, "seat1": 1.0}},
        metadata={"metric_tie_tolerances": {"cost": 1.0}},
    )

    snapshot = OpenSkillRater(metrics).rate_events([event])

    assert snapshot.players["model-a"].skills["cost"].mu == snapshot.players["model-b"].skills["cost"].mu


def test_matchmaker_requires_fixed_benchmark_warmup_first():
    metrics = [MetricSpec(name="coordination", higher_is_better=True)]
    matchmaker = Matchmaker(
        metrics,
        warmup_games=90,
        rng=random.Random(7),
    )

    try:
        matchmaker.plan_match(PlayerRatingState(player_id="new-model", games_played=12), [])
    except ValueError as exc:
        assert "fixed benchmark warmup" in str(exc)
    else:
        raise AssertionError("Expected benchmark warmup barrier to reject live matchmaking.")


def test_matchmaker_samples_nearest_live_neighbors():
    metrics = [MetricSpec(name="coordination", higher_is_better=True)]
    target = PlayerRatingState(
        player_id="target",
        games_played=90,
        skills={"coordination": OpenSkillRater(metrics)._default_skill()},
    )
    target.skills["coordination"].mu = 30.0
    pool = []
    for idx, mu in enumerate([10.0, 28.0, 29.0, 31.0, 32.0, 45.0]):
        state = PlayerRatingState(
            player_id=f"p{idx}",
            games_played=90,
            skills={"coordination": OpenSkillRater(metrics)._default_skill()},
        )
        state.skills["coordination"].mu = mu
        pool.append(state)
    matchmaker = Matchmaker(metrics, warmup_games=90, num_opponents=2, search_pool_size=3, rng=random.Random(1))

    plan = matchmaker.plan_match(target, pool)

    assert plan.phase == "live"
    assert set(plan.opponent_player_ids).issubset({"p1", "p2", "p3"})
