"""Factory for LLM clients with provider auto-detection from model name."""

import os

from a2a_engine.llm.api import LLMClient
from a2a_engine.llm.providers import LLM_PROVIDERS


def detect_provider(model: str) -> str | None:
    """Detect provider key from a model name. Returns None if unknown."""
    m = model.lower()
    if m == "gemini-3-flash-preview" or m.startswith("publishers/google/models/gemini-3"):
        return "gemini_vertexai"
    if m == "claude-sonnet-4-6" or ("claude" in m and "@" in m):
        return "claude_vertexai"
    if m.startswith("meta/") and "maas" in m:
        return "vertexai_openai"
    if "/" in model:
        return "openrouter"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m or m.startswith(("o1-", "o1_", "o3-", "o3_", "o4-", "o4_")):
        return "openai"
    if "gemini" in m:
        return "gemini"
    if "llama" in m or "mistral" in m or "qwen" in m:
        return "ollama"
    return None


def get_api_key_for_provider(provider: str) -> str:
    """Read the API key for a provider from environment."""
    if provider in {"gemini_vertexai", "claude_vertexai", "vertexai_openai"}:
        return ""
    env_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    return os.environ.get(env_keys.get(provider, ""), "")


def make_llm_client(cfg: dict) -> LLMClient:
    """Build an ``LLMClient`` from a config dict.

    Auto-fills api_base, api_format, and api_key from LLM_PROVIDERS / env when
    not explicitly set. Required: ``model``.
    """
    model = cfg.get("model", "gpt-4o-mini")
    api_base = cfg.get("api_base")
    api_key = cfg.get("api_key")
    api_format = cfg.get("api_format")

    provider = detect_provider(model)
    if provider and provider in LLM_PROVIDERS:
        p = LLM_PROVIDERS[provider]
        api_base = api_base or p["api_base"]
        api_format = api_format or p["api_format"]
        if not api_key:
            api_key = get_api_key_for_provider(provider)

    return LLMClient(
        api_format=api_format or "openai",
        api_base=api_base or "",
        api_key=api_key or "",
        model=model,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens"),
        vertex_adc_file=cfg.get("vertex_adc_file") or cfg.get("adc_file"),
        gcp_project=cfg.get("gcp_project"),
        gcp_location=cfg.get("gcp_location"),
        extra=cfg.get("extra", {}),
    )
