"""Deterministic scripted client for dry-run / testing."""

from __future__ import annotations

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult


def _parse_free_slots(calendar_render: str) -> list[int]:
    """Extract free slot indices from a calendar render string."""
    free = []
    for line in calendar_render.splitlines():
        if "[FREE]" in line:
            try:
                slot_idx = int(line.split("Slot")[1].split(":")[0].strip())
                free.append(slot_idx)
            except (IndexError, ValueError):
                pass
    return free


class ScriptedClient(BaseClient):
    """Deterministic scripted client for dry-run / testing."""

    def __init__(self) -> None:
        self.agent_id: int = -1
        self.game_config: GameConfig | None = None
        self.meeting: dict | None = None
        self.calendar_render: str = ""
        self._turned: bool = False

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self.game_config = game_config

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self.meeting = meeting
        self.calendar_render = calendar_render
        self._turned = False

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        if not self._turned:
            self._turned = True
            free_slots = _parse_free_slots(self.calendar_render)
            tool_calls = []
            if self.meeting is not None:
                protocol = (
                    self.game_config.communication_protocol
                    if self.game_config is not None
                    else "dm"
                )
                if "participant_groupchat" in protocol:
                    tool_calls.append({
                        "type": "participant_groupchat",
                        "meeting_id": self.meeting["id"],
                        "content": f"I'm free at slots: {free_slots}",
                    })
                elif "all_groupchat" in protocol or protocol == "groupchat":
                    tool_calls.append({
                        "type": "all_groupchat",
                        "meeting_id": self.meeting["id"],
                        "content": f"I'm free at slots: {free_slots}",
                    })
                else:
                    for other in self.meeting.get("participants", []):
                        if other != self.agent_id:
                            tool_calls.append({
                                "type": "dm",
                                "to": other,
                                "meeting_id": self.meeting["id"],
                                "content": f"I'm free at slots: {free_slots}",
                            })
            return TurnResult(
                tool_calls=tool_calls,
                text=None,
                thinking=None,
                usage=None,
                latency_ms=None,
                raw=None,
            )
        return TurnResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        free_slots = _parse_free_slots(calendar_render)
        if free_slots:
            return DecideResult(
                tool_calls=[{"type": "schedule", "meeting_id": meeting["id"], "slot": free_slots[0]}],
                text=None,
                thinking=None,
                usage=None,
                latency_ms=None,
                raw=None,
            )
        return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        if self.meeting is None:
            return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)
        free_slots = _parse_free_slots(self.calendar_render)
        idx = attempt
        if idx < len(free_slots):
            return DecideResult(
                tool_calls=[{"type": "schedule", "meeting_id": self.meeting["id"], "slot": free_slots[idx]}],
                text=None,
                thinking=None,
                usage=None,
                latency_ms=None,
                raw=None,
            )
        return DecideResult(tool_calls=[], text=None, thinking=None, usage=None, latency_ms=None, raw=None)
