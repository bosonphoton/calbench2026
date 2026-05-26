"""Generic agent interface.

A game provides ``observation`` and a ``tools`` dict of callables (e.g.
``send_p2p``, ``broadcast``, ``finalize``). The agent returns an action dict
describing what it did. Action schema is game-defined.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from a2a_engine._context import current_conversation_id
from a2a_engine.llm.api import LLMClient
from a2a_engine.tracing_otel import get_tracer


class AgentInterface(ABC):
    """Minimal agent contract. All concrete agents implement ``act``."""

    #: Human-readable agent name; used as the `gen_ai.agent.name` span attr.
    name: str = ""

    @abstractmethod
    async def act(self, observation: dict[str, Any], tools: dict[str, Callable]) -> dict[str, Any]:
        """Return an action dict given an observation and available tools."""
        raise NotImplementedError


class LLMAgent(AgentInterface):
    """Base class for LLM-backed agents.

    Subclasses override ``build_messages`` (and optionally ``parse_response``)
    to map game observations to chat messages and back. Keeps the LLM-call
    plumbing in one place so per-game agents stay tiny.
    """

    def __init__(self, client: LLMClient, system_prompt: str = "", name: str | None = None) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.name = name or self.__class__.__name__

    def build_messages(
        self,
        observation: dict[str, Any],
        tools: dict[str, Callable],
    ) -> list[dict]:
        """Build chat messages for the LLM. Override per-game."""
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": str(observation)})
        return msgs

    def parse_response(
        self,
        text: str,
        observation: dict[str, Any],
        tools: dict[str, Callable],
    ) -> dict[str, Any]:
        """Parse the LLM's text into an action dict. Override per-game."""
        return {"text": text}

    async def act(
        self,
        observation: dict[str, Any],
        tools: dict[str, Callable],
    ) -> dict[str, Any]:
        tracer = get_tracer()
        with tracer.start_as_current_span(f"invoke_agent {self.name}") as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", self.name)
            agent_id = getattr(self, "id", None)
            if agent_id:
                span.set_attribute("gen_ai.agent.id", str(agent_id))
            conv_id = current_conversation_id.get()
            if conv_id:
                span.set_attribute("gen_ai.conversation.id", conv_id)
            messages = self.build_messages(observation, tools)
            text = self.client.oneshot(messages)
            return self.parse_response(text, observation, tools)
