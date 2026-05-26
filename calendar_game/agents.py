"""Agents for the calendar scheduling game."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:
    from calendar_game.calendar import Calendar
except ImportError:
    Calendar = Any  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    num_agents: int
    num_slots: int
    agent_id: int               # this specific agent's identity
    all_agent_ids: list[int]    # all agent ids in the game
    dm_cap: int = 1_000_000     # legacy name: max delivered cheap-talk messaging actions per agent per round
    decision_retries: int = 3   # max retries allowed in DECISION phase
    dsm_num_proposals: int = 4
    dsm_cascade_depth: int = 1
    dsm_displacement_targets: int = 4
    dsm_exhaustive_search: bool = True
    dsm_stop_on_perfect: bool = True
    dsm_prior_meetings: list[dict] = field(default_factory=list)
    dsm_lmin: int = 1
    dsm_lmax: int | None = None
    dsm_beta: float = 1.0
    dsm_theta: float = 0.0
    dsm_social_welfare_weight: float = 1.0
    dsm_privacy_unit_cost: float = 1.0
    dsm_initial_budget: int = 100
    sd_model: dict[int, float] = field(default_factory=dict)
    communication_protocol: str = "dm"


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None


@dataclass
class TurnResult:
    tool_calls: list[dict]      # parsed tool calls; empty list = pass
    text: str | None            # raw model text output
    thinking: str | None        # reasoning/thinking trace if available
    usage: TokenUsage | None    # token counts
    latency_ms: float | None    # wall time ms for the client call
    raw: dict | None            # full raw API response, unmodified


@dataclass
class DecideResult(TurnResult):
    retry_count: int = 0        # how many retries were used (0 = first attempt succeeded)


@dataclass
class ReflectionResult:
    target_agent_id: int
    estimates: list[dict]
    text: str | None
    usage: TokenUsage | None
    latency_ms: float | None
    raw: dict | None


# ---------------------------------------------------------------------------
# BaseClient (abstract interface)
# ---------------------------------------------------------------------------

class BaseClient(ABC):
    @abstractmethod
    def register(self, agent_id: int, game_config: GameConfig) -> None:
        """Called once at game start. LLM clients build system prompt here."""

    @abstractmethod
    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        """Called at start of each round the agent participates in."""

    @abstractmethod
    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        """
        Called each CHEAP_TALK turn. messages is the drained inbox in arrival order.
        Each message: {"from": int, "meeting_id": int, "content": str}
        Return TurnResult with tool_calls being a list of dm tool dicts, or [] to pass.
        """

    @abstractmethod
    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        """
        Called once per DECISION phase. calendar_render is the frozen snapshot.
        Return DecideResult with tool_calls being schedule/reschedule dicts.
        """

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        """
        Called when a decision batch fails validation. Default raises NotImplementedError.
        attempt is 1-indexed (first retry = 1).
        """
        raise NotImplementedError(
            f"Client does not support decision retries (attempt {attempt}/{max_attempts}): {conflict}"
        )

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        """
        Called for non-participants who received DMs during CHEAP_TALK.
        May return reschedule actions to honor coordination commitments.
        Default: no-op (pass).
        """
        return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def observe_calendar(self, calendar_render: str) -> None:
        """Optional hook for scripted clients that keep parsed local-calendar state."""
        return None

    def observe_penalty(self, incurred_penalty: int) -> None:
        """Optional hook for clients that surface accumulated private penalty to the agent."""
        return None

    def reflect_calendar_belief(
        self,
        target_agent_id: int,
        num_slots: int,
        round_num: int | None = None,
    ) -> ReflectionResult | None:
        """
        Optional measurement hook. LLM clients can report belief movement about
        another agent's occupied/free state for each calendar slot.
        """
        return None


# ---------------------------------------------------------------------------
# Agent (game-engine-facing wrapper)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, client: BaseClient) -> None:
        self.agent_id: int = -1
        self.client: BaseClient = client
        self.inbox_queue: deque = deque()
        self.calendar: Calendar = None  # type: ignore

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        """Store agent_id, delegate to client.register()."""
        self.agent_id = agent_id
        self.client.register(agent_id, game_config)

    def start_round(self, meeting: dict, round_num: int, incurred_penalty: int = 0) -> None:
        """Delegate to client.start_round() passing meeting, calendar.render(), round_num."""
        self.client.observe_penalty(incurred_penalty)
        self.client.start_round(meeting, self.calendar.render(), round_num)

    def turn(self, turn_index: int | None = None, max_turns_per_round: int | None = None) -> TurnResult:
        """
        Drain inbox_queue into a list (in order), clear queue,
        call client.turn(drained_messages), return result.
        """
        self.client.observe_calendar(self.calendar.render())
        drained_messages = list(self.inbox_queue)
        self.inbox_queue.clear()
        return self.client.turn(drained_messages, turn_index, max_turns_per_round)

    def decide(self, meeting: dict) -> DecideResult:
        """
        Call client.decide(meeting, calendar.render(snapshot)), return result.
        Note: pass calendar.snapshot().render() so client sees frozen state.
        """
        return self.client.decide(meeting, self.calendar.snapshot().render())

    def voluntary_decide(self, meeting: dict) -> DecideResult:
        """Call client.voluntary_decide() for non-participant reschedule phase."""
        return self.client.voluntary_decide(meeting, self.calendar.render())


# ---------------------------------------------------------------------------
# CapturingClient (test double)
# ---------------------------------------------------------------------------

@dataclass
class CapturingClient(BaseClient):
    """
    Records every call without hitting any API.
    Returns configurable canned responses.
    """
    calls: list[dict] = field(default_factory=list)
    canned_turn_result: TurnResult = field(
        default_factory=lambda: TurnResult(
            tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None
        )
    )
    canned_decide_result: DecideResult = field(
        default_factory=lambda: DecideResult(
            tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None
        )
    )

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.calls.append({"method": "register", "args": {"agent_id": agent_id, "game_config": game_config}})

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self.calls.append({
            "method": "start_round",
            "args": {"meeting": meeting, "calendar_render": calendar_render, "round_num": round_num},
        })

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        self.calls.append({
            "method": "turn",
            "args": {
                "messages": messages,
                "turn_index": turn_index,
                "max_turns_per_round": max_turns_per_round,
            },
        })
        return self.canned_turn_result

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        self.calls.append({"method": "decide", "args": {"meeting": meeting, "calendar_render": calendar_render}})
        return self.canned_decide_result

    canned_voluntary_result: DecideResult = field(
        default_factory=lambda: DecideResult(
            tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None
        )
    )

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        self.calls.append({"method": "voluntary_decide", "args": {"meeting": meeting, "calendar_render": calendar_render}})
        return self.canned_voluntary_result
