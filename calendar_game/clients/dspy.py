"""DSPy-optimized LLM client for calendar games.

This client is intentionally separate from LLMClient so prompt optimization can
iterate without changing the baseline prompt used by ordinary LLM agents.
"""

from __future__ import annotations

from calendar_game.clients.llm import LLMClient
from calendar_game.prompts import make_dspy_system_prompt_builder


class DSPyClient(LLMClient):
    """LLM-backed client using the DSPy-optimization prompt entrypoint."""

    def __init__(
        self,
        llm_client: object,
        prompt_variant: str | None = None,
        prompt_variant_dir: str | None = None,
    ) -> None:
        super().__init__(
            llm_client,
            system_prompt_builder=make_dspy_system_prompt_builder(prompt_variant, prompt_variant_dir),
        )
