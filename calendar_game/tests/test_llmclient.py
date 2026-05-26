"""Tests for the LLMClient class in calendar_game.game."""
from __future__ import annotations

import pytest

from calendar_game.agents import GameConfig, TurnResult, DecideResult, TokenUsage
from calendar_game.clients import LLMClient


# ---------------------------------------------------------------------------
# MockLLM stub
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self, text="[]"):
        self.text = text
        self.calls = []  # record each call's messages list
        self.kwargs = []

    def streaming(self, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return {
            "text": self.text,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "duration_s": 0.1,
            "finish_reason": "stop",
        }

    def streaming_with_retry(self, messages, **kwargs):
        return self.streaming(messages, **kwargs)


class SequenceMockLLM(MockLLM):
    def __init__(self, texts):
        super().__init__(text="")
        self.texts = list(texts)

    def streaming(self, messages, **kwargs):
        index = len(self.calls)
        self.text = self.texts[index] if index < len(self.texts) else self.texts[-1]
        return super().streaming(messages, **kwargs)


class LogprobMockLLM(MockLLM):
    def __init__(self, text="[0,1,-2,3]"):
        super().__init__(text=text)
        self.api_format = "openai"
        self.model = "gpt-5.4-mini"

    def streaming(self, messages, **kwargs):
        result = super().streaming(messages, **kwargs)
        result["logprobs"] = [{
            "content": [{
                "token": "0",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "0", "logprob": -0.2},
                    {"token": "1", "logprob": -1.7},
                ],
            }, {
                "token": "1",
                "logprob": -0.4,
                "top_logprobs": [
                    {"token": "0", "logprob": -1.4},
                    {"token": "1", "logprob": -0.4},
                ],
            }, {
                "token": "0",
                "logprob": -0.3,
                "top_logprobs": [
                    {"token": "0", "logprob": -0.3},
                    {"token": "1", "logprob": -2.3},
                ],
            }, {
                "token": "1",
                "logprob": -0.7,
                "top_logprobs": [
                    {"token": "0", "logprob": -0.9},
                    {"token": "1", "logprob": -0.7},
                ],
            }]
        }]
        result["_raw_response"] = dict(result)
        return result


class LogprobUnsupportedThenTextMockLLM(MockLLM):
    def __init__(self, fallback_text="1"):
        super().__init__(text="")
        self.api_format = "openai"
        self.model = "gpt-5.4-mini"
        self.fallback_text = fallback_text

    def streaming(self, messages, **kwargs):
        if kwargs.get("logprobs"):
            self.calls.append(messages)
            self.kwargs.append(kwargs)
            return {
                "text": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "duration_s": None,
                "finish_reason": "error",
                "_error": "Logprobs is not supported for this model",
            }
        self.text = self.fallback_text
        return super().streaming(messages, **kwargs)


class MissingAlternativeLogprobMockLLM(MockLLM):
    def __init__(self):
        super().__init__(text="[0]")
        self.api_format = "openai"
        self.model = "gpt-5.4-mini"

    def streaming(self, messages, **kwargs):
        result = super().streaming(messages, **kwargs)
        result["logprobs"] = [{
            "content": [{
                "token": "0",
                "logprob": -0.01,
                "top_logprobs": [
                    {"token": "0", "logprob": -0.01},
                    {"token": "2", "logprob": -5.0},
                    {"token": ",", "logprob": -6.0},
                ],
            }]
        }]
        result["_raw_response"] = dict(result)
        return result


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def make_game_config(agent_id=0):
    return GameConfig(
        num_agents=2,
        num_slots=8,
        agent_id=agent_id,
        all_agent_ids=[0, 1],
        dm_cap=10,
        decision_retries=3,
    )


def make_meeting(meeting_id=1):
    return {"id": meeting_id, "participants": [0, 1], "duration": 1}


def make_llm_client(text="[]"):
    mock = MockLLM(text)
    client = LLMClient(mock)
    return client, mock


def assert_payloads_are_append_only(calls, assistant_texts):
    """Each provider payload should preserve the previous payload as a prefix."""
    for index in range(1, len(calls)):
        previous = calls[index - 1]
        current = calls[index]
        assert current[:len(previous)] == previous
        assert current[len(previous)] == {
            "role": "assistant",
            "content": assistant_texts[index - 1],
        }
        assert current[len(previous) + 1]["role"] == "user"
        assert len(current) == len(previous) + 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_register_sets_system_prompt():
    """After register(), turn() sends messages where first has role=system and contains agent_id."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])
    assert len(mock.calls) == 1
    messages = mock.calls[0]
    assert messages[0]["role"] == "system"
    # System prompt should mention agent id in some form
    assert "0" in messages[0]["content"]


def test_first_turn_includes_round_start():
    """First turn() after start_round() sends a user message with round number and meeting id."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting(meeting_id=5)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=3)
    client.turn([])
    messages = mock.calls[0]
    # Skip the system message
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= 1
    user_content = user_messages[0]["content"]
    # Should contain round number and meeting id
    assert "3" in user_content or "round" in user_content.lower()
    assert "5" in user_content


def test_first_turn_includes_observed_penalty():
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    client.observe_penalty(37)
    client.start_round(make_meeting(meeting_id=5), "Slot 0: [FREE]", round_num=3)
    client.turn([])
    user_messages = [m for m in mock.calls[0] if m["role"] == "user"]
    assert "YOUR PENALTY SO FAR" in user_messages[0]["content"]
    assert "37 total penalty points" in user_messages[0]["content"]


def test_first_turn_with_inbox_appends_inbox():
    """First turn() with inbox messages sends a user message containing both round-start and inbox."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting(meeting_id=1)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    inbox = [{"from": 1, "meeting_id": 1, "content": "hello"}]
    client.turn(inbox)
    messages = mock.calls[0]
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= 1
    user_content = user_messages[0]["content"]
    # Should contain both round start context (meeting id) and inbox content
    assert "1" in user_content
    assert "hello" in user_content


def test_external_first_turn_without_start_round_uses_inbox_and_calendar():
    """Non-participants can receive cheap-talk DMs before start_round() is called."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    client.observe_calendar("Slot 0: Meeting M100 (cost=1)")
    inbox = [{"from": 1, "meeting_id": 1, "content": "can you move M100?"}]
    client.turn(inbox)
    messages = mock.calls[0]
    user_content = [m for m in messages if m["role"] == "user"][0]["content"]
    assert "Slot 0: Meeting M100" in user_content
    assert "can you move M100?" in user_content


def test_subsequent_turn_sends_only_inbox():
    """After first turn, subsequent turn() sends only inbox content (no round-start block again)."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting(meeting_id=7)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=2)
    # First turn
    client.turn([])
    first_user_msg = [m for m in mock.calls[0] if m["role"] == "user"][0]["content"]

    # Second turn with inbox
    inbox = [{"from": 1, "meeting_id": 7, "content": "hi"}]
    client.turn(inbox)
    second_messages = mock.calls[1]
    # The new user message is the last user message in second_messages
    second_user_msgs = [m for m in second_messages if m["role"] == "user"]
    # The last user message should be the second-turn one (just inbox)
    new_user_msg = second_user_msgs[-1]["content"]
    assert "hi" in new_user_msg
    # Should not re-send the round-start block (which was in first_user_msg)
    # The round-start block is only in the first user message, not repeated
    assert len(new_user_msg) < len(first_user_msg) or "hi" in new_user_msg


def test_history_grows_across_turns():
    """Each turn() adds 2 messages (user + assistant); after 3 turns LLM gets system + 6 messages."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])
    client.turn([])
    client.turn([])
    # Third call's messages: system + 2 prior (user+asst) + current user = 6
    # The 3rd assistant reply is appended AFTER the call, so it's not in the sent messages.
    last_messages = mock.calls[2]
    assert last_messages[0]["role"] == "system"
    assert len(last_messages) == 6  # 1 system + 2*(turn1 user+asst) + turn3 user


def test_history_accumulates_across_rounds():
    """History persists across rounds — second round's turn() includes prior context."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    # First round: two turns build up history
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])
    client.turn([])
    # Second round does NOT reset history
    client.start_round(meeting, "Slot 1: [FREE]", round_num=1)
    client.turn([])
    last_messages = mock.calls[2]
    # Should be system + prior turns + new user message (more than 2)
    assert last_messages[0]["role"] == "system"
    assert len(last_messages) > 2


def test_provider_payload_context_is_append_only_across_calls():
    """The exact LLM API payload should grow by assistant+user turns only."""
    assistant_texts = [
        '{"thinking": "round 1 hello", "actions": []}',
        '{"thinking": "round 1 followup", "actions": []}',
        '{"thinking": "round 1 decide", "actions": []}',
        '{"thinking": "round 1 retry", "actions": []}',
        '{"thinking": "round 2 hello", "actions": []}',
    ]
    mock = SequenceMockLLM(assistant_texts)
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))

    meeting = make_meeting(meeting_id=1)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])
    client.turn([{"from": 1, "meeting_id": 1, "content": "slot 3?"}])
    client.decide(meeting, "Slot 0: [FREE]")
    client.retry_decide(1, 3, "forced retry for context audit")

    second_meeting = make_meeting(meeting_id=2)
    client.start_round(second_meeting, "Slot 1: [FREE]", round_num=1)
    client.turn([])

    assert len(mock.calls) == len(assistant_texts)
    assert_payloads_are_append_only(mock.calls, assistant_texts)
    for payload in mock.calls:
        assert payload[0]["role"] == "system"
        assert payload.count(payload[0]) == 1


def test_turn_result_populated():
    """turn() returns TurnResult with text set, non-null usage with correct tokens, ~100ms latency."""
    client, mock = make_llm_client(text='[]')
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.turn([])
    assert isinstance(result, TurnResult)
    assert result.text is not None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.latency_ms is not None
    assert abs(result.latency_ms - 100.0) < 1.0
    assert result.raw is not None


def test_turn_parses_valid_json_tool_calls():
    """When MockLLM returns valid JSON list, turn() returns tool_calls with one entry."""
    text = '[{"type": "dm", "to": 1, "meeting_id": 1, "content": "hi"}]'
    client, mock = make_llm_client(text=text)
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.turn([])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["type"] == "dm"


def test_turn_parses_json_with_surrounding_text():
    """When response has surrounding text, tool_calls is still extracted correctly."""
    text = 'Here are my tool calls: [{"type": "dm", "to": 1, "meeting_id": 1, "content": "hi"}] done.'
    client, mock = make_llm_client(text=text)
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.turn([])
    assert len(result.tool_calls) == 1


def test_turn_returns_empty_on_invalid_json():
    """When MockLLM returns non-JSON text, tool_calls is []."""
    client, mock = make_llm_client(text='not json')
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.turn([])
    assert result.tool_calls == []


def test_turn_filters_non_object_actions():
    """Malformed list entries from the model are ignored before the game loop sees them."""
    text = '{"thinking": "mixed output", "actions": [null, "oops", {"type": "dm", "to": 1, "content": "hi"}]}'
    client, mock = make_llm_client(text=text)
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.turn([])
    assert result.tool_calls == [{"type": "dm", "to": 1, "content": "hi"}]


def test_decide_sends_decision_message():
    """decide() sends a user message containing 'DECISION' and the meeting id; returns DecideResult with retry_count=0."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting(meeting_id=3)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    result = client.decide(meeting, "Slot 0: [FREE]")
    assert isinstance(result, DecideResult)
    assert result.retry_count == 0
    # Check the message sent includes DECISION and meeting id
    messages = mock.calls[0]
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= 1
    last_user_content = user_messages[-1]["content"]
    assert "DECISION" in last_user_content or "decision" in last_user_content.lower()
    assert "3" in last_user_content


def test_retry_decide_appends_retry_message():
    """retry_decide() appends a user message with '[RETRY 1/3]' and the conflict reason."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting(meeting_id=1)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.decide(meeting, "Slot 0: [FREE]")
    result = client.retry_decide(1, 3, "conflict reason")
    assert isinstance(result, DecideResult)
    assert result.retry_count == 1
    # The retry call's messages should include [RETRY 1/3] and conflict reason
    retry_messages = mock.calls[1]
    user_messages = [m for m in retry_messages if m["role"] == "user"]
    last_user_content = user_messages[-1]["content"]
    assert "1" in last_user_content and "3" in last_user_content
    assert "conflict reason" in last_user_content


def test_decide_history_continues_from_cheap_talk():
    """After start_round + turn, decide() sends a message list including prior turn's user+assistant."""
    client, mock = make_llm_client()
    config = make_game_config(agent_id=0)
    client.register(0, config)
    meeting = make_meeting()
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])  # first cheap-talk turn
    client.decide(meeting, "Slot 0: [FREE]")
    # decide() call is mock.calls[1]
    decide_messages = mock.calls[1]
    # Should have: system + user(turn1) + assistant(turn1) + user(decide)
    assert len(decide_messages) == 4
    assert decide_messages[0]["role"] == "system"
    assert decide_messages[1]["role"] == "user"
    assert decide_messages[2]["role"] == "assistant"
    assert decide_messages[3]["role"] == "user"


def test_context_checkpoint_B():
    """First turn() after start_round sends a user message with round number,
    meeting id, calendar_render verbatim, and CHEAP_TALK signal."""
    cal_render = "Slot 0: [FREE]\nSlot 1: Errand #1 (cost=2)"
    client, mock = make_llm_client()
    client.register(0, make_game_config())
    meeting = make_meeting(meeting_id=1)
    client.start_round(meeting, cal_render, round_num=2)
    client.turn([])

    user_msgs = [m for m in mock.calls[0] if m["role"] == "user"]
    content = user_msgs[-1]["content"]

    assert "2" in content
    assert str(meeting["id"]) in content
    assert cal_render in content
    assert "CHEAP_TALK" in content


def test_context_checkpoint_C():
    """Second turn() sends inbox content but NOT the calendar_render or ROUND block from start_round."""
    cal_render = "Slot 0: [FREE]\nSlot 1: Errand #1 (cost=2)"
    client, mock = make_llm_client()
    client.register(0, make_game_config())
    client.start_round(make_meeting(), cal_render, round_num=1)
    client.turn([])

    client.turn([{"from": 1, "meeting_id": 1, "content": "use slot 5"}])

    user_msgs = [m for m in mock.calls[1] if m["role"] == "user"]
    content = user_msgs[-1]["content"]

    assert "use slot 5" in content
    assert "Agent 1" in content
    assert cal_render not in content
    assert "ROUND" not in content


def test_context_checkpoint_D():
    """decide() sends DECISION signal, meeting id, calendar snapshot; no inbox section."""
    cal_snapshot = "Slot 0: [FREE]\nSlot 1: Errand #1 (cost=2)"
    client, mock = make_llm_client()
    client.register(0, make_game_config())
    meeting = make_meeting(meeting_id=1)
    client.start_round(meeting, "Slot 0: [FREE]", round_num=0)
    client.turn([])
    client.decide(meeting, cal_snapshot)

    user_msgs = [m for m in mock.calls[1] if m["role"] == "user"]
    content = user_msgs[-1]["content"]

    assert "DECISION" in content
    assert str(meeting["id"]) in content
    assert cal_snapshot in content
    assert "inbox" not in content.lower()


def test_reflection_uses_batched_logprob_call_without_mutating_history():
    mock = LogprobMockLLM()
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))
    client.start_round(make_meeting(), "Slot 0: [FREE]", round_num=0)
    client.turn([])

    before_history_len = len(client._history)
    result = client.reflect_calendar_belief(target_agent_id=1, num_slots=4)

    assert result is not None
    assert [estimate["belief_delta_occupied"] for estimate in result.estimates] == [0, 1, -2, 3]
    assert [estimate["estimate_state"] for estimate in result.estimates] == [None, 1, 0, 1]
    assert result.estimates[0]["probability_free"] == 0.5
    assert result.estimates[0]["probability_busy"] == 0.5
    assert result.estimates[1]["probability_busy"] > result.estimates[1]["probability_free"]
    assert result.estimates[2]["probability_free"] > result.estimates[2]["probability_busy"]
    assert result.estimates[0]["confidence"] == 0
    assert result.estimates[1]["confidence"] == 1 / 3
    assert result.estimates[3]["confidence"] == 1
    assert result.estimates[0]["logprobs"] == {"0": -0.2, "1": -1.7, "_top_logprob_floor": -1.7}
    assert mock.kwargs[-1]["max_tokens"] == 1024
    assert mock.kwargs[-1]["logprobs"] is True
    assert mock.kwargs[-1]["top_logprobs"] == 5
    assert len(client._history) == before_history_len
    assert "Agent 1" in mock.calls[-1][-1]["content"]
    assert "exactly 4 integers" in mock.calls[-1][-1]["content"]


def test_reflection_uses_large_output_budget_for_full_slot_array():
    mock = MockLLM(text="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))

    result = client.reflect_calendar_belief(target_agent_id=1, num_slots=16)

    assert result is not None
    assert mock.kwargs[-1]["max_tokens"] == 2048
    assert len(result.estimates) == 16


def test_reflection_prompt_does_not_bleed_into_next_round_context():
    mock = SequenceMockLLM([
        '{"thinking":"round zero","actions":[]}',
        '[0,1,-2,3]',
        '{"thinking":"round one","actions":[]}',
    ])
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))
    client.start_round(make_meeting(meeting_id=1), "Slot 0: [FREE]", round_num=0)
    client.turn([])
    client.reflect_calendar_belief(target_agent_id=1, num_slots=4, round_num=0)

    client.start_round(make_meeting(meeting_id=2), "Slot 0: [FREE]\nSlot 1: [FREE]", round_num=1)
    client.turn([])

    next_turn_messages = mock.calls[-1]
    next_turn_text = "\n\n".join(message["content"] for message in next_turn_messages)
    assert "END-OF-ROUND" not in next_turn_text
    assert "belief changed" not in next_turn_text
    assert "[0,1,-2,3]" not in next_turn_text


def test_reflection_fallback_defaults_to_half_confidence_without_logprobs():
    mock = LogprobUnsupportedThenTextMockLLM(fallback_text="1")
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))

    result = client.reflect_calendar_belief(target_agent_id=1, num_slots=4)

    assert result is not None
    assert result.estimates[0]["belief_delta_occupied"] == 1
    assert result.estimates[0]["estimate_state"] == 1
    assert result.estimates[0]["probability_free"] == pytest.approx(1 / 3)
    assert result.estimates[0]["probability_busy"] == pytest.approx(2 / 3)
    assert result.estimates[0]["confidence"] == pytest.approx(1 / 3)
    assert result.estimates[0]["logprobs"] == {"0": None, "1": None, "_top_logprob_floor": None}
    assert len(mock.calls) == 2
    assert mock.kwargs[0]["logprobs"] is True
    assert "Logprobs is not supported" in result.raw["_calibration_error"]


def test_reflection_skips_logprobs_for_known_unsupported_provider():
    mock = MockLLM(text="[0,1,-1,3]")
    mock.api_format = "vertexai"
    mock.model = "gemini-3.1-pro-preview"
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))

    result = client.reflect_calendar_belief(target_agent_id=1, num_slots=4)

    assert result is not None
    assert [estimate["belief_delta_occupied"] for estimate in result.estimates] == [0, 1, -1, 3]
    assert len(mock.calls) == 1
    assert "logprobs" not in mock.kwargs[0]
    assert "top_logprobs" not in mock.kwargs[0]
    assert result.estimates[1]["logprobs"] == {"0": None, "1": None, "_top_logprob_floor": None}


def test_reflection_missing_binary_alternative_uses_top_logprob_floor():
    mock = MissingAlternativeLogprobMockLLM()
    client = LLMClient(mock)
    client.register(0, make_game_config(agent_id=0))

    result = client.reflect_calendar_belief(target_agent_id=1, num_slots=1)

    estimate = result.estimates[0]
    assert estimate["belief_delta_occupied"] == 0
    assert estimate["estimate_state"] is None
    assert estimate["probability_free"] == 0.5
    assert estimate["probability_busy"] == 0.5
    assert estimate["confidence"] == 0
    assert estimate["logprobs"]["0"] == -0.01
    assert estimate["logprobs"]["1"] is None
    assert estimate["logprobs"]["_top_logprob_floor"] == -6.0
