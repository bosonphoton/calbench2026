"""LLM-backed client wrapping a2a_engine provider clients."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from collections.abc import Callable

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # pragma: no cover
    _repair_json = None

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TokenUsage, TurnResult
from calendar_game.agents import ReflectionResult
from calendar_game.prompts import (
    build_decision_message,
    build_reflection_message,
    build_retry_message,
    build_round_start_message,
    build_system_prompt,
    build_turn_message,
    build_voluntary_reschedule_message,
)
from calendar_game.privacy import hydrate_calendar_render_for_llm, hydrate_meeting_for_llm

SystemPromptBuilder = Callable[[dict], str]


def _parse_response(text: str) -> tuple[list[dict], str | None]:
    """Parse model output into (tool_calls, thinking).

    Accepts either the new {"thinking": "...", "actions": [...]} object format
    or the legacy bare-list format. Falls back through fence-stripping, json_repair,
    and regex extraction.
    """
    if not text:
        return [], None

    def _extract(parsed: object) -> tuple[list[dict], str | None] | None:
        if isinstance(parsed, dict) and "actions" in parsed:
            actions = parsed["actions"]
            if isinstance(actions, list):
                return [a for a in actions if isinstance(a, dict)], parsed.get("thinking") or None
        if isinstance(parsed, list):
            return [a for a in parsed if isinstance(a, dict)], None
        return None

    stripped = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()
    for candidate in (text.strip(), stripped):
        try:
            result = _extract(json.loads(candidate))
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass
    if _repair_json is not None:
        try:
            result = _extract(_repair_json(stripped, return_objects=True))
            if result is not None:
                return result
        except Exception:
            pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, stripped, re.DOTALL)
        if m and _repair_json is not None:
            try:
                result = _extract(_repair_json(m.group(), return_objects=True))
                if result is not None:
                    return result
            except Exception:
                pass
    return [], None


def _make_usage(result: dict) -> TokenUsage | None:
    pt = result.get("prompt_tokens")
    ct = result.get("completion_tokens")
    tt = result.get("total_tokens")
    if pt is None and ct is None:
        return None
    return TokenUsage(
        prompt_tokens=pt or 0,
        completion_tokens=ct or 0,
        total_tokens=tt or (pt or 0) + (ct or 0),
        reasoning_tokens=result.get("reasoning_tokens"),
        cached_prompt_tokens=result.get("cached_prompt_tokens"),
    )


_REFLECTION_DELTA_MAX = 3
_REFLECTION_MIN_MAX_TOKENS = 1024
_REFLECTION_TOKENS_PER_SLOT = 128


def _reflection_max_tokens(num_slots: int) -> int:
    return max(_REFLECTION_MIN_MAX_TOKENS, num_slots * _REFLECTION_TOKENS_PER_SLOT)


def _parse_reflection_deltas(text: str | None, num_slots: int) -> list[int | None]:
    deltas: list[int | None] = [None for _ in range(num_slots)]
    if text is None:
        return deltas
    stripped = text.strip()
    if not stripped:
        return deltas
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        if _repair_json is not None:
            try:
                parsed = _repair_json(stripped, return_objects=True)
            except Exception:
                parsed = None
        else:
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("states"), list):
        parsed = parsed["states"]
    if isinstance(parsed, dict) and isinstance(parsed.get("deltas"), list):
        parsed = parsed["deltas"]
    if isinstance(parsed, list):
        for index, value in enumerate(parsed[:num_slots]):
            try:
                delta = int(value)
            except (TypeError, ValueError):
                continue
            deltas[index] = max(-_REFLECTION_DELTA_MAX, min(_REFLECTION_DELTA_MAX, delta))
        return deltas
    matches = re.findall(r"(?<!\d)-?[0-3](?!\d)", stripped)
    for index, value in enumerate(matches[:num_slots]):
        deltas[index] = int(value)
    return deltas


def _norm_binary_token(token: object) -> str | None:
    text = str(token).strip()
    if text in {"0", "1"}:
        return text
    return None


def _extract_binary_logprobs_by_slot(raw: object, num_slots: int) -> list[dict[str, float | None]]:
    """Best-effort extraction for 0/1 token logprobs in generated slot order."""
    found: list[dict[str, float | None]] = []

    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            top = obj.get("top_logprobs")
            if isinstance(top, list):
                local = {"0": None, "1": None, "_top_logprob_floor": None}
                token_key = _norm_binary_token(obj.get("token"))
                if token_key is not None and isinstance(obj.get("logprob"), (int, float)):
                    local[token_key] = float(obj["logprob"])
                for entry in top:
                    if not isinstance(entry, dict):
                        continue
                    if isinstance(entry.get("logprob"), (int, float)):
                        floor = local["_top_logprob_floor"]
                        entry_logprob = float(entry["logprob"])
                        local["_top_logprob_floor"] = (
                            entry_logprob if floor is None else min(float(floor), entry_logprob)
                        )
                    key = _norm_binary_token(entry.get("token"))
                    if key is not None and isinstance(entry.get("logprob"), (int, float)):
                        local[key] = float(entry["logprob"])
                if local["0"] is not None or local["1"] is not None:
                    found.append(local)
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(raw)
    found = found[:num_slots]
    while len(found) < num_slots:
        found.append({"0": None, "1": None, "_top_logprob_floor": None})
    return found


def _softmax_binary(lp0: float | None, lp1: float | None) -> tuple[float | None, float | None]:
    if lp0 is None or lp1 is None:
        return None, None
    baseline = max(lp0, lp1)
    p0 = math.exp(lp0 - baseline)
    p1 = math.exp(lp1 - baseline)
    total = p0 + p1
    if total <= 0:
        return None, None
    return p0 / total, p1 / total


def _missing_alternative_logprob(known_logprob: float, floor_logprob: float | None) -> float:
    known_prob = math.exp(min(0.0, known_logprob))
    residual_prob = max(1.0 - known_prob, 1e-12)
    if floor_logprob is not None:
        residual_prob = min(residual_prob, math.exp(min(0.0, floor_logprob)))
    return math.log(max(residual_prob, 1e-12))


def _is_logprob_unsupported_error(result: dict) -> bool:
    error = str(result.get("_error") or "").lower()
    return bool(error) and "logprob" in error and "not supported" in error


def _llm_supports_logprobs(llm_client: object) -> bool:
    """Return whether the configured provider/model should receive logprob params."""
    explicit = getattr(llm_client, "supports_logprobs", None)
    extra = getattr(llm_client, "extra", None)
    if explicit is None and isinstance(extra, dict):
        explicit = extra.get("supports_logprobs")
    if explicit is not None:
        return bool(explicit)

    api_format = str(getattr(llm_client, "api_format", "") or "").lower()
    model = str(getattr(llm_client, "model", "") or "").lower()

    if api_format in {"anthropic", "vertexai_anthropic", "vertexai_openai"}:
        return False
    if api_format in {"gemini", "vertexai"} or "gemini" in model:
        return False
    if api_format == "openai":
        return True
    return False


class LLMClient(BaseClient):
    """LLM client wrapping an underlying provider client (OpenAI, Anthropic, etc.)."""

    def __init__(self, llm_client: object, system_prompt_builder: SystemPromptBuilder = build_system_prompt) -> None:
        self._llm = llm_client
        self._system_prompt_builder = system_prompt_builder
        self.agent_id: int = -1
        self._system_prompt: str = ""
        self._history: list[dict] = []

    def _call(self, user_message: str, *, record_history: bool = True, **kwargs) -> dict:
        user_turn = {"role": "user", "content": user_message}
        if record_history:
            self._history.append(user_turn)
            messages = [{"role": "system", "content": self._system_prompt}] + self._history
        else:
            messages = [{"role": "system", "content": self._system_prompt}] + self._history + [user_turn]
        try:
            result = self._llm.streaming_with_retry(messages, **kwargs)
            if result is None:
                raise RuntimeError("LLM call exhausted retries")
        except Exception as exc:
            logging.getLogger("calendar_game.llm").error(
                "LLM call failed (agent %d): %s: %s", self.agent_id, type(exc).__name__, exc
            )
            if record_history:
                self._history.pop()
            return {"text": None, "prompt_tokens": None, "completion_tokens": None,
                    "total_tokens": None, "duration_s": None, "finish_reason": "error",
                    "_error": str(exc)}
        assistant_text = result.get("text") or ""
        if record_history:
            self._history.append({"role": "assistant", "content": assistant_text})
        return result

    def _make_turn_result(self, result: dict) -> TurnResult:
        text = result.get("text") or None
        tool_calls, model_thinking = _parse_response(text or "")
        return TurnResult(
            tool_calls=tool_calls,
            text=text,
            thinking=model_thinking or result.get("reasoning") or None,
            usage=_make_usage(result),
            latency_ms=(result.get("duration_s") or 0) * 1000 or None,
            raw=result.get("_raw_response") or result,
        )

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self._system_prompt = self._system_prompt_builder(dataclasses.asdict(game_config))
        self._history = []
        self._round_meeting: dict | None = None
        self._round_calendar: str = ""
        self._round_num: int = 0
        self._incurred_penalty: int = 0
        self._first_turn: bool = True
        self._communication_protocol = game_config.communication_protocol

    def observe_penalty(self, incurred_penalty: int) -> None:
        self._incurred_penalty = incurred_penalty

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self._round_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{round_num}",
        )
        self._round_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{round_num}",
        )
        self._round_num = round_num
        self._first_turn = True

    def observe_calendar(self, calendar_render: str) -> None:
        if self._round_meeting is None:
            self._round_calendar = hydrate_calendar_render_for_llm(
                calendar_render,
                stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
            )

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        if self._first_turn:
            self._first_turn = False
            if self._round_meeting is None:
                user_msg = (
                    f"=== YOUR CALENDAR ===\n{self._round_calendar}\n\n"
                    f"{build_turn_message(messages, turn_index, max_turns_per_round, self._communication_protocol)}"
                )
            else:
                user_msg = build_round_start_message(
                    self._round_meeting,
                    self._round_calendar,
                    self._round_num,
                    incurred_penalty=self._incurred_penalty,
                    turn_index=turn_index,
                    max_turns_per_round=max_turns_per_round,
                    communication_protocol=self._communication_protocol,
                )
                if messages:
                    user_msg += "\n\n" + build_turn_message(
                        messages,
                        turn_index,
                        max_turns_per_round,
                        self._communication_protocol,
                    )
        else:
            user_msg = build_turn_message(
                messages,
                turn_index,
                max_turns_per_round,
                self._communication_protocol,
            )
        result = self._call(user_msg)
        return self._make_turn_result(result)

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        hydrated_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        hydrated_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        msg = build_decision_message(hydrated_meeting, hydrated_calendar)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=0,
        )

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        msg = build_retry_message(attempt, max_attempts, conflict)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=attempt,
        )

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        hydrated_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        hydrated_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        msg = build_voluntary_reschedule_message(hydrated_meeting, hydrated_calendar)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=0,
        )

    def reflect_calendar_belief(
        self,
        target_agent_id: int,
        num_slots: int,
        round_num: int | None = None,
    ) -> ReflectionResult | None:
        msg = build_reflection_message(target_agent_id, num_slots, round_num=round_num)
        max_tokens = _reflection_max_tokens(num_slots)
        call_kwargs = {
            "record_history": False,
            "max_tokens": max_tokens,
            "temperature": 0,
            "thinking_config": {"thinking_budget": 0},
            "response_mime_type": "application/json",
        }
        if _llm_supports_logprobs(self._llm):
            call_kwargs["logprobs"] = True
            call_kwargs["top_logprobs"] = 5
        result = self._call(msg, **call_kwargs)
        calibration_error = result.get("_error") if _is_logprob_unsupported_error(result) else None
        if calibration_error:
            result = self._call(
                msg,
                record_history=False,
                max_tokens=max_tokens,
                temperature=0,
                thinking_config={"thinking_budget": 0},
                response_mime_type="application/json",
            )
            result["_calibration_error"] = calibration_error
        raw = result.get("_raw_response") or result
        if calibration_error and isinstance(raw, dict):
            raw["_calibration_error"] = calibration_error
        deltas = _parse_reflection_deltas(result.get("text"), num_slots)
        slot_logprobs = _extract_binary_logprobs_by_slot(raw, num_slots)
        estimates: list[dict] = []
        for slot, delta in enumerate(deltas):
            logprobs = slot_logprobs[slot]
            probability_busy = None
            probability_free = None
            confidence = None
            estimate_state = None
            if delta is not None:
                probability_busy = min(1.0, max(0.0, 0.5 + (delta / (2 * _REFLECTION_DELTA_MAX))))
                probability_free = 1.0 - probability_busy
                confidence = abs(delta) / _REFLECTION_DELTA_MAX
                if delta > 0:
                    estimate_state = 1
                elif delta < 0:
                    estimate_state = 0
            estimates.append({
                "target_agent_id": target_agent_id,
                "slot": slot,
                "belief_delta_occupied": delta,
                "estimate_state": estimate_state,
                "probability_free": probability_free,
                "probability_busy": probability_busy,
                "confidence": confidence,
                "logprobs": logprobs,
            })
        return ReflectionResult(
            target_agent_id=target_agent_id,
            estimates=estimates,
            text=result.get("text") or None,
            usage=_make_usage(result),
            latency_ms=(result.get("duration_s") or 0) * 1000 or None,
            raw=raw,
        )
