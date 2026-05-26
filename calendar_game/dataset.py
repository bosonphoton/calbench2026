"""Structured abstraction over calendar game traces.

Mirrors the NegotiationDataset interface from a2a-llm-judge.

Usage::

    from calendar_game.dataset import CalendarGameDataset

    ds = CalendarGameDataset.from_dir("shared-traces")
    game_df  = ds.to_game_df()   # one row per game
    round_df = ds.to_round_df()  # one row per (game, round)
    agent_df = ds.to_agent_df()  # one row per (game, round, agent)
    msg_df   = ds.to_message_df() # one row per cheap-talk message sent

Research questions surfaced:

    Task success       — agent_df["task_success"]
    Coordination cost  — agent_df["coordination_cost"]
    Privacy cost       — external uniform/excess VPS summaries
    Communication      — agent_df["communication_efficiency"]
    Fairness           — agent_df["fairness_cost"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level event wrapper
# ---------------------------------------------------------------------------

class CalendarEvent:
    """A single event from the trace event log."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.type: str = raw["type"]
        self.timestamp: str = raw.get("timestamp", "")
        data = raw.get("data", {})
        self.round: int = data.get("round", -1)
        self.turn: int = data.get("turn", -1)
        self.phase: str = data.get("phase", "")
        self.agent_id: Optional[int] = data.get("agent_id")
        self.data: dict[str, Any] = data

    # Convenience accessors for common event types
    @property
    def tool_calls(self) -> list[dict]:
        return self.data.get("tool_calls", [])

    @property
    def dm_content(self) -> Optional[str]:
        return (
            self.data.get("content")
            if self.type in {"dm_sent", "groupchat_sent", "participant_groupchat_sent", "all_groupchat_sent"}
            else None
        )

    @property
    def dm_content_chars(self) -> int:
        if self.type not in {"dm_sent", "groupchat_sent", "participant_groupchat_sent", "all_groupchat_sent"}:
            return 0
        if "content_chars" in self.data:
            return int(self.data.get("content_chars") or 0)
        return len(self.dm_content or "")

    @property
    def from_agent(self) -> Optional[int]:
        return self.data.get("from_agent")

    @property
    def to_agent(self) -> Optional[int]:
        return self.data.get("to_agent")

    @property
    def latency_ms(self) -> Optional[float]:
        return self.data.get("latency_ms")

    @property
    def usage(self) -> Optional[dict]:
        resp = self.data.get("raw_api_response", {})
        if resp:
            return {
                "prompt_tokens": resp.get("prompt_tokens"),
                "completion_tokens": resp.get("completion_tokens"),
                "total_tokens": resp.get("total_tokens"),
                "cached_prompt_tokens": resp.get("cached_prompt_tokens"),
            }
        return self.data.get("usage")


# ---------------------------------------------------------------------------
# Round wrapper
# ---------------------------------------------------------------------------

class CalendarRound:
    """Data and metrics for a single scheduling round (one meeting)."""

    def __init__(self, game: "CalendarGame", round_idx: int):
        self.game = game
        self.round_idx = round_idx  # 0-based
        self._outcome: dict[str, Any] = (
            game.raw["final_state"]["round_outcomes"][round_idx]
            if round_idx < len(game.raw["final_state"]["round_outcomes"])
            else {}
        )

    # --- Identity ---

    @property
    def meeting_id(self) -> int:
        return self._outcome.get("meeting_id", self.round_idx + 1)

    @property
    def round_number(self) -> int:
        return self.round_idx

    # --- Coordination ---

    @property
    def coordinated(self) -> bool:
        return self._outcome.get("coordinated", False)

    @property
    def slot_conflicts(self) -> dict:
        return self._outcome.get("slot_conflicts", {})

    @property
    def consistency_violations(self) -> list:
        return self._outcome.get("consistency_violated_meeting_ids", [])

    @property
    def has_slot_conflict(self) -> bool:
        return bool(self.slot_conflicts)

    # --- Events for this round ---

    @property
    def events(self) -> list[CalendarEvent]:
        return [e for e in self.game.events if e.round == self.round_idx]

    @property
    def dms(self) -> list[CalendarEvent]:
        return [
            e for e in self.events
            if e.type in {"dm_sent", "groupchat_sent", "participant_groupchat_sent", "all_groupchat_sent"}
        ]

    @property
    def num_dms(self) -> int:
        return len(self.dms)

    @property
    def num_cheap_talk_turns(self) -> int:
        turns = {e.turn for e in self.events if e.phase == "CHEAP_TALK" and e.type == "turn_end"}
        return len(turns)

    # --- Per-round cost from events (reschedule actions in this round) ---

    def _reschedule_costs(self) -> dict[int, float]:
        """Sum of displacement costs per agent for reschedule actions in this round."""
        costs: dict[int, float] = {}
        for e in self.events:
            if e.type != "turn_end" or e.phase not in ("DECISION", "VOLUNTARY"):
                continue
            if e.agent_id is None:
                continue
            for tc in e.tool_calls:
                if tc.get("type") == "reschedule":
                    # Cost is not in the tool call itself; we rely on game-level accounting
                    costs[e.agent_id] = costs.get(e.agent_id, 0.0)
        return costs


# ---------------------------------------------------------------------------
# Game wrapper
# ---------------------------------------------------------------------------

class CalendarGame:
    """A full calendar game trace."""

    def __init__(self, raw: dict[str, Any], path: Optional[Path] = None):
        self.raw = raw
        self.path = path
        self.game_id: str = raw.get("game_id", "")
        self._events: Optional[list[CalendarEvent]] = None

    # --- Config accessors ---

    @property
    def config(self) -> dict[str, Any]:
        return self.raw.get("config", {})

    @property
    def experiment_name(self) -> str:
        return self.config.get("experiment_name", "")

    @property
    def experiment_run_id(self) -> str:
        return self.config.get("experiment_run_id", "")

    @property
    def seed(self) -> Optional[int]:
        return self.config.get("seed")

    @property
    def num_agents(self) -> int:
        return self.config.get("num_agents", len(self.config.get("agents", [])))

    @property
    def agents_config(self) -> list[dict]:
        return self.config.get("agents", [])

    @property
    def models(self) -> list[str]:
        return [a.get("model") or a.get("type") or "unknown" for a in self.agents_config]

    @property
    def agent_types(self) -> list[str]:
        return [a.get("type") or "unknown" for a in self.agents_config]

    @property
    def model_label(self) -> str:
        unique = list(dict.fromkeys(self.models))
        return unique[0] if len(unique) == 1 else " vs ".join(unique)

    @property
    def team_model_counts(self) -> dict[str, int]:
        return self.metrics.get(
            "team_model_counts",
            self.raw.get("final_state", {}).get("team_model_counts", {}),
        )

    @property
    def is_heterogeneous_team(self) -> bool:
        return bool(
            self.metrics.get(
                "is_heterogeneous_team",
                len(set(self.models)) > 1 or len(set(self.agent_types)) > 1,
            )
        )

    # --- Scenario metadata (from game_start event) ---

    @property
    def _game_start_data(self) -> dict[str, Any]:
        for e in self.events:
            if e.type == "game_start":
                return e.data
        return {}

    @property
    def num_slots(self) -> int:
        return self._game_start_data.get("num_slots", 0)

    @property
    def optimal_cost(self) -> float:
        return self._game_start_data.get("optimal_cost", float("nan"))

    @property
    def greedy_cost(self) -> float:
        return self._game_start_data.get("greedy_cost", float("nan"))

    # --- Metrics (pre-computed by game engine) ---

    @property
    def metrics(self) -> dict[str, Any]:
        return self.raw.get("metrics", {})

    @property
    def coordination_rate(self) -> float:
        return self.metrics.get("coordination_rate", float("nan"))

    @property
    def meetings_scheduled(self) -> int:
        return self.metrics.get("meetings_scheduled", 0)

    @property
    def total_dms(self) -> int:
        return self.metrics.get("total_dms_sent", 0)

    @property
    def total_groupchat_messages(self) -> int:
        return int(
            self.metrics.get(
                "total_groupchat_messages_sent",
                int(self.metrics.get("total_participant_groupchat_messages_sent", 0) or 0)
                + int(self.metrics.get("total_all_groupchat_messages_sent", 0) or 0),
            )
            or 0
        )

    @property
    def total_participant_groupchat_messages(self) -> int:
        return int(self.metrics.get("total_participant_groupchat_messages_sent", 0) or 0)

    @property
    def total_all_groupchat_messages(self) -> int:
        return int(self.metrics.get("total_all_groupchat_messages_sent", 0) or 0)

    @property
    def total_cheap_talk_messages(self) -> int:
        if "total_cheap_talk_messages" in self.metrics:
            return int(self.metrics.get("total_cheap_talk_messages") or 0)
        return self.total_dms + self.total_groupchat_messages

    @property
    def total_dm_chars(self) -> int:
        if "total_dm_chars" in self.metrics:
            return int(self.metrics.get("total_dm_chars") or 0)
        return sum(e.dm_content_chars for e in self.events if e.type == "dm_sent")

    @property
    def avg_dm_chars(self) -> float:
        if "avg_dm_chars" in self.metrics:
            return float(self.metrics.get("avg_dm_chars") or 0.0)
        return self.total_dm_chars / self.total_dms if self.total_dms else 0.0

    @property
    def dm_chars_per_meeting(self) -> float:
        if "dm_chars_per_meeting" in self.metrics:
            return float(self.metrics.get("dm_chars_per_meeting") or 0.0)
        if not self.meetings_scheduled:
            return float("nan")
        return self.total_dm_chars / self.meetings_scheduled

    @property
    def realized_cost(self) -> float:
        return self.metrics.get("realized_cost", float("nan"))

    @property
    def fallback_displacement_cost(self) -> float:
        return self.metrics.get("fallback_displacement_cost", float("nan"))

    @property
    def efficiency_metric(self) -> float:
        """Engine-computed cheap-talk messages per coordinated meeting."""
        return self.metrics.get("efficiency", float("nan"))

    @property
    def fairness_metric(self) -> float:
        return self.metrics.get("fairness", float("nan"))

    # --- Derived RQ metrics ---

    @property
    def msgs_per_meeting(self) -> float:
        """Legacy game-pooled cheap-talk messages per successfully scheduled meeting."""
        if not self.meetings_scheduled:
            return float("nan")
        return self.total_cheap_talk_messages / self.meetings_scheduled

    @property
    def excess_cost(self) -> float:
        """Paper coordination cost summed over agents for scheduled meetings."""
        value = self.metrics.get("scheduled_only_excess_cost")
        if value is None:
            value = self.metrics.get("total_scheduled_only_excess_burden")
        if value is not None:
            return float(value)
        scheduled_optimal = self.metrics.get("scheduled_only_optimal_cost")
        if scheduled_optimal is None:
            scheduled_optimal = self.optimal_cost
        return self.realized_cost - scheduled_optimal

    @property
    def legacy_full_task_excess_cost(self) -> float:
        """Legacy full-task realized_cost - optimal_cost diagnostic."""
        return self.realized_cost - self.optimal_cost

    @property
    def cost_ratio(self) -> float:
        """RQ3: realized / optimal — >1 means over-spending; nan when optimal=0."""
        if self.optimal_cost == 0:
            return float("nan")
        return self.realized_cost / self.optimal_cost

    # --- Per-agent costs ---

    @property
    def per_agent_cost(self) -> list[float]:
        return self.raw.get("final_state", {}).get("per_agent_cost", [])

    @property
    def per_agent_fallback_cost(self) -> list[float]:
        return self.raw.get("final_state", {}).get("per_agent_fallback_cost", [])

    @property
    def oracle_per_agent_cost(self) -> list[float]:
        return self.raw.get("final_state", {}).get("oracle_per_agent_cost", [])

    @property
    def per_agent_excess_burden(self) -> list[float]:
        return self.raw.get("final_state", {}).get("per_agent_excess_burden", [])

    @property
    def contribution_scores(self) -> list[dict[str, Any]]:
        return self.raw.get("final_state", {}).get("contribution_scores", [])

    @property
    def representation_elo_by_agent(self) -> list[float]:
        return self.raw.get("final_state", {}).get("representation_elo_by_agent", [])

    @property
    def initial_calendar_density_by_agent(self) -> list[float]:
        return self.raw.get("final_state", {}).get("initial_calendar_density_by_agent", [])

    @property
    def total_agent_cost(self) -> float:
        return sum(self.per_agent_cost)

    @property
    def cost_gini(self) -> float:
        """RQ4: Gini coefficient of per-agent realized cost (0=equal, 1=all on one)."""
        costs = [float(c) for c in self.per_agent_cost]
        if not costs or sum(costs) == 0:
            return float("nan")
        n = len(costs)
        costs_sorted = sorted(costs)
        cumsum = np.cumsum(costs_sorted)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    # --- Events and rounds ---

    @property
    def events(self) -> list[CalendarEvent]:
        if self._events is None:
            self._events = [CalendarEvent(e) for e in self.raw.get("events", [])]
        return self._events

    @property
    def rounds(self) -> list[CalendarRound]:
        n = len(self.raw.get("final_state", {}).get("round_outcomes", []))
        return [CalendarRound(self, i) for i in range(n)]

    # --- Metadata ---

    @property
    def stopped(self) -> bool:
        return self.raw.get("stopped", False)

    def __repr__(self) -> str:
        return (
            f"CalendarGame(id={self.game_id[:8]}, model={self.model_label}, "
            f"meetings={self.meetings_scheduled}, cost={self.realized_cost})"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalendarGameDataset:
    """A collection of CalendarGame traces with DataFrame export methods."""

    def __init__(self, games: list[CalendarGame]):
        self.games = games

    @classmethod
    def from_dir(cls, root: str | Path) -> "CalendarGameDataset":
        """Load all *.json trace files (excluding *.metadata.json) under root."""
        root = Path(root)
        paths = sorted(
            p for p in root.rglob("*.json")
            if not p.name.endswith(".metadata.json")
            and p.name != "_run_manifest.jsonl"
        )
        games = []
        for p in paths:
            try:
                raw = json.loads(p.read_text())
                games.append(CalendarGame(raw, path=p))
            except Exception as exc:
                log.warning("Skipping %s: %s", p, exc)
        log.info("Loaded %d games from %s", len(games), root)
        return cls(games)

    @classmethod
    def from_traces(cls, traces: list[dict[str, Any]]) -> "CalendarGameDataset":
        return cls([CalendarGame(t) for t in traces])

    # ------------------------------------------------------------------
    # DataFrame builders
    # ------------------------------------------------------------------

    def to_game_df(self) -> pd.DataFrame:
        """One row per game.

        Primary metric columns use the paper definitions. Legacy game-pooled
        fields are retained with ``legacy_`` prefixes for diagnostics.
        """
        rows = []
        for g in self.games:
            rows.append({
                "game_id": g.game_id,
                "experiment_name": g.experiment_name,
                "experiment_run_id": g.experiment_run_id,
                "seed": g.seed,
                "model_label": g.model_label,
                "team_model_counts": g.team_model_counts,
                "is_heterogeneous_team": g.is_heterogeneous_team,
                "agent_types": g.agent_types,
                "num_agents": g.num_agents,
                "num_participants": g.config.get("num_participants"),
                "density": g.config.get("density"),
                "dataset": (
                    "uniform"
                    if "uniform" in str(g.config.get("task_path", "")).lower()
                    else "varied"
                    if "varied" in str(g.config.get("task_path", "")).lower()
                    else None
                ),
                "difficulty": g.metrics.get("difficulty"),
                "difficulty_score": g.metrics.get("difficulty_score"),
                "difficulty_scorer": g.metrics.get("difficulty_scorer"),
                "nosy_agent_count": g.metrics.get("nosy_agent_count"),
                "communication_protocol": g.metrics.get("communication_protocol"),
                "num_slots": g.num_slots,
                "stopped": g.stopped,
                # Scheduling outcomes
                "meetings_scheduled": g.meetings_scheduled,
                "coordination_rate": g.coordination_rate,
                "success_score": g.metrics.get("success_score"),
                "headline_score": g.metrics.get("headline_score"),
                "headline_score_weighted": g.metrics.get("headline_score_weighted"),
                "cost_score": g.metrics.get("cost_score"),
                "privacy_score": g.metrics.get("privacy_score"),
                "privacy_score_available": g.metrics.get("privacy_score_available"),
                "efficiency_score": g.metrics.get("efficiency_score"),
                "communication_score": g.metrics.get("communication_score"),
                "excess_cost_score": g.metrics.get("excess_cost_score"),
                "normalized_communication_cost": g.metrics.get("normalized_communication_cost"),
                "normalized_privacy_leakage": g.metrics.get("normalized_privacy_leakage"),
                "normalized_excess_cost": g.metrics.get("normalized_excess_cost"),
                # RQ2: efficiency
                "total_dms": g.total_dms,
                "total_groupchat_messages": g.total_groupchat_messages,
                "total_participant_groupchat_messages": g.total_participant_groupchat_messages,
                "total_all_groupchat_messages": g.total_all_groupchat_messages,
                "total_cheap_talk_messages": g.total_cheap_talk_messages,
                "legacy_msgs_per_meeting": g.msgs_per_meeting,
                "total_dm_chars": g.total_dm_chars,
                "avg_dm_chars": g.avg_dm_chars,
                "dm_chars_per_meeting": g.dm_chars_per_meeting,
                # RQ3: optimality
                "optimal_cost": g.optimal_cost,
                "greedy_cost": g.greedy_cost,
                "scheduled_only_optimal_cost": g.metrics.get("scheduled_only_optimal_cost"),
                "scheduled_only_greedy_cost": g.metrics.get("scheduled_only_greedy_cost"),
                "scheduled_only_oracle_cost_replayed": g.metrics.get("scheduled_only_oracle_cost_replayed"),
                "scheduled_meeting_ids": g.metrics.get("scheduled_meeting_ids"),
                "unscheduled_meeting_ids": g.metrics.get("unscheduled_meeting_ids"),
                "realized_cost": g.realized_cost,
                "fallback_displacement_cost": g.fallback_displacement_cost,
                "excess_cost": g.excess_cost,
                "scheduled_only_excess_cost": g.excess_cost,
                "legacy_full_task_excess_cost": g.legacy_full_task_excess_cost,
                "cost_regret": g.metrics.get("cost_regret"),
                "cost_score_scope": g.metrics.get("cost_score_scope"),
                "cost_score_optimal_cost": g.metrics.get("cost_score_optimal_cost"),
                "cost_score_greedy_cost": g.metrics.get("cost_score_greedy_cost"),
                "greedy_normalized_regret": g.metrics.get("greedy_normalized_regret"),
                "oracle_normalized_regret": g.metrics.get("oracle_normalized_regret"),
                "cost_ratio": g.cost_ratio,
                "privacy_leak_count": g.metrics.get("privacy_leak_count"),
                "high_sensitivity_leak_count": g.metrics.get("high_sensitivity_leak_count"),
                "third_party_leak_count": g.metrics.get("third_party_leak_count"),
                # RQ4: fairness
                "cost_gini": g.cost_gini,
                "legacy_fairness_ratio": g.fairness_metric,
                "fairness_cost": g.metrics.get("fairness_cost"),
                "mean_contribution_score": g.metrics.get("mean_contribution_score"),
                "model_contribution_summary": g.metrics.get("model_contribution_summary"),
            })
        return pd.DataFrame(rows)

    def to_round_df(self) -> pd.DataFrame:
        """One row per (game, round)."""
        rows = []
        for g in self.games:
            game_meta = self._game_meta(g)
            for r in g.rounds:
                rows.append({
                    **game_meta,
                    "round_number": r.round_number,
                    "meeting_id": r.meeting_id,
                    "coordinated": r.coordinated,
                    "has_slot_conflict": r.has_slot_conflict,
                    "num_dms": r.num_dms,
                    "num_cheap_talk_turns": r.num_cheap_talk_turns,
                })
        return pd.DataFrame(rows)

    def to_agent_df(self) -> pd.DataFrame:
        """One row per (game, agent).

        Key columns for RQ4:
            coordination_cost      — scheduled-only excess burden for the agent
            communication_efficiency — sent cheap-talk messages per scheduled meeting
            fairness_cost          — absolute relative scheduled-only excess burden
        """
        rows = []
        for g in self.games:
            game_meta = self._game_meta(g)
            total = g.total_agent_cost
            for i, (cost, fallback) in enumerate(
                zip(g.per_agent_cost, g.per_agent_fallback_cost)
            ):
                model = g.models[i] if i < len(g.models) else "unknown"
                atype = g.agent_types[i] if i < len(g.agent_types) else "unknown"
                contribution = (
                    g.contribution_scores[i]
                    if i < len(g.contribution_scores)
                    else {}
                )
                rows.append({
                    **game_meta,
                    "agent_id": i,
                    "model": model,
                    "agent_type": atype,
                    "cost": float(cost),
                    "fallback_cost": float(fallback),
                    "oracle_cost": contribution.get(
                        "oracle_cost",
                        g.oracle_per_agent_cost[i]
                        if i < len(g.oracle_per_agent_cost)
                        else float("nan"),
                    ),
                    "excess_burden": contribution.get(
                        "excess_burden",
                        g.metrics.get("per_agent_scheduled_only_excess_burden", [])[i]
                        if i < len(g.metrics.get("per_agent_scheduled_only_excess_burden", []))
                        else g.per_agent_excess_burden[i]
                        if i < len(g.per_agent_excess_burden)
                        else float("nan"),
                    ),
                    "legacy_full_task_excess_burden": contribution.get(
                        "legacy_full_task_excess_burden",
                        g.metrics.get("legacy_per_agent_full_task_excess_burden", [])[i]
                        if i < len(g.metrics.get("legacy_per_agent_full_task_excess_burden", []))
                        else float("nan"),
                    ),
                    "scheduled_only_oracle_cost": contribution.get(
                        "scheduled_only_oracle_cost",
                        g.metrics.get("per_agent_scheduled_only_oracle_cost", [])[i]
                        if i < len(g.metrics.get("per_agent_scheduled_only_oracle_cost", []))
                        else float("nan"),
                    ),
                    "scheduled_only_excess_burden": contribution.get(
                        "scheduled_only_excess_burden",
                        g.metrics.get("per_agent_scheduled_only_excess_burden", [])[i]
                        if i < len(g.metrics.get("per_agent_scheduled_only_excess_burden", []))
                        else float("nan"),
                    ),
                    "coordination_cost": contribution.get(
                        "coordination_cost",
                        contribution.get(
                            "scheduled_only_excess_burden",
                            g.metrics.get("per_agent_scheduled_only_excess_burden", [])[i]
                            if i < len(g.metrics.get("per_agent_scheduled_only_excess_burden", []))
                            else float("nan"),
                        ),
                    ),
                    "excess_cost": contribution.get(
                        "excess_cost",
                        contribution.get(
                            "scheduled_only_excess_burden",
                            g.metrics.get("per_agent_scheduled_only_excess_burden", [])[i]
                            if i < len(g.metrics.get("per_agent_scheduled_only_excess_burden", []))
                            else float("nan"),
                        ),
                    ),
                    "fairness_cost": contribution.get(
                        "fairness_cost",
                        g.metrics.get("per_agent_fairness_cost", [])[i]
                        if i < len(g.metrics.get("per_agent_fairness_cost", []))
                        else float("nan"),
                    ),
                    "communication_score": contribution.get(
                        "communication_score",
                        g.metrics.get("per_agent_communication_score", [])[i]
                        if i < len(g.metrics.get("per_agent_communication_score", []))
                        else float("nan"),
                    ),
                    "normalized_excess_cost": contribution.get(
                        "normalized_excess_cost",
                        g.metrics.get("per_agent_normalized_excess_cost", [])[i]
                        if i < len(g.metrics.get("per_agent_normalized_excess_cost", []))
                        else float("nan"),
                    ),
                    "excess_cost_score": contribution.get(
                        "excess_cost_score",
                        g.metrics.get("per_agent_excess_cost_score", [])[i]
                        if i < len(g.metrics.get("per_agent_excess_cost_score", []))
                        else float("nan"),
                    ),
                    "privacy_score": (
                        g.metrics.get("per_agent_privacy_score", [])[i]
                        if i < len(g.metrics.get("per_agent_privacy_score", []))
                        else float("nan")
                    ),
                    "cost_share": float(cost) / total if total > 0 else float("nan"),
                    "calendar_density": contribution.get(
                        "calendar_density",
                        g.initial_calendar_density_by_agent[i]
                        if i < len(g.initial_calendar_density_by_agent)
                        else float("nan"),
                    ),
                    "representation_elo": contribution.get(
                        "representation_elo",
                        g.representation_elo_by_agent[i]
                        if i < len(g.representation_elo_by_agent)
                        else float("nan"),
                    ),
                    "participant_rounds": contribution.get("participant_rounds"),
                    "coordinated_participant_rounds": contribution.get("coordinated_participant_rounds"),
                    "task_success": contribution.get("task_success"),
                    "messages_sent": contribution.get("messages_sent"),
                    "messages_received": contribution.get("messages_received"),
                    "messages_per_scheduled_meeting": contribution.get("messages_per_scheduled_meeting"),
                    "communication_efficiency": contribution.get(
                        "communication_efficiency",
                        contribution.get("messages_per_scheduled_meeting"),
                    ),
                    "contribution_score": contribution.get("contribution_score"),
                    "density_adjusted_contribution_score": contribution.get(
                        "density_adjusted_contribution_score"
                    ),
                })
        return pd.DataFrame(rows)

    def to_message_df(self) -> pd.DataFrame:
        """One row per cheap-talk message sent — useful for communication pattern analysis."""
        rows = []
        for g in self.games:
            game_meta = self._game_meta(g)
            for e in g.events:
                if e.type not in {"dm_sent", "groupchat_sent", "participant_groupchat_sent", "all_groupchat_sent"}:
                    continue
                rows.append({
                    **game_meta,
                    "round_number": e.round,
                    "turn": e.turn,
                    "channel": e.data.get("channel", "dm" if e.type == "dm_sent" else e.type.removesuffix("_sent")),
                    "from_agent": e.from_agent,
                    "to_agent": e.to_agent,
                    "to_agents": e.data.get("to_agents"),
                    "content": e.dm_content,
                    "content_chars": e.dm_content_chars,
                    "meeting_id": e.data.get("meeting_id"),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        experiment_name: Optional[str] = None,
        model: Optional[str] = None,
        agent_type: Optional[str] = None,
        min_meetings: Optional[int] = None,
        exclude_stopped: bool = False,
    ) -> "CalendarGameDataset":
        games = self.games
        if experiment_name is not None:
            games = [g for g in games if g.experiment_name == experiment_name]
        if model is not None:
            games = [g for g in games if any(model in m for m in g.models)]
        if agent_type is not None:
            games = [g for g in games if any(agent_type in t for t in g.agent_types)]
        if min_meetings is not None:
            games = [g for g in games if g.meetings_scheduled >= min_meetings]
        if exclude_stopped:
            games = [g for g in games if not g.stopped]
        return CalendarGameDataset(games)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _game_meta(self, g: CalendarGame) -> dict[str, Any]:
        return {
            "game_id": g.game_id,
            "experiment_name": g.experiment_name,
            "experiment_run_id": g.experiment_run_id,
            "seed": g.seed,
            "model_label": g.model_label,
            "team_model_counts": g.team_model_counts,
            "is_heterogeneous_team": g.is_heterogeneous_team,
            "num_agents": g.num_agents,
            "num_participants": g.config.get("num_participants"),
            "density": g.config.get("density"),
            "dataset": (
                "uniform"
                if "uniform" in str(g.config.get("task_path", "")).lower()
                else "varied"
                if "varied" in str(g.config.get("task_path", "")).lower()
                else None
            ),
            "difficulty": g.metrics.get("difficulty"),
            "nosy_agent_count": g.metrics.get("nosy_agent_count"),
            "communication_protocol": g.metrics.get("communication_protocol"),
        }

    def __len__(self) -> int:
        return len(self.games)

    def __repr__(self) -> str:
        return f"CalendarGameDataset({len(self.games)} games)"
