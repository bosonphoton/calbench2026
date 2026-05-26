"""LLM provider presets and HTTP retry/backoff settings.

Game-agnostic: reuse from any benchmark built on a2a-engine.
"""

import os

# Each preset: api_base, default model, api_format, key_placeholder.
# api_format: "openai" for /chat/completions, "anthropic" for /v1/messages,
# "vertexai" for Gemini via Google ADC, "vertexai_anthropic" for Claude via
# Google ADC, and "vertexai_openai" for Vertex MaaS OpenAI-compatible models.
LLM_PROVIDERS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_format": "openai",
        "key_placeholder": "OPENAI_API_KEY",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "api_base": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-5-20250929",
        "api_format": "anthropic",
        "key_placeholder": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "api_format": "openai",
        "key_placeholder": "GOOGLE_API_KEY",
    },
    "gemini_vertexai": {
        "label": "Google Gemini (Vertex AI / ADC)",
        "api_base": "",
        "model": "publishers/google/models/gemini-3-flash-preview",
        "api_format": "vertexai",
        "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "gcp_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        "key_placeholder": "",
    },
    "claude_vertexai": {
        "label": "Anthropic Claude (Vertex AI / ADC)",
        "api_base": "",
        "model": "claude-haiku-4-5@20251001",
        "api_format": "vertexai_anthropic",
        "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "gcp_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        "key_placeholder": "",
    },
    "vertexai_openai": {
        "label": "Vertex AI OpenAI-compatible MaaS / ADC",
        "api_base": "",
        "model": "meta/llama-4-maverick-17b-128e-instruct-maas",
        "api_format": "vertexai_openai",
        "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "gcp_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east5"),
        "key_placeholder": "",
    },
    "ollama": {
        "label": "Ollama (local)",
        "api_base": "http://localhost:11434/v1",
        "model": "llama3",
        "api_format": "openai",
        "key_placeholder": "ollama",
    },
    "openrouter": {
        "label": "OpenRouter",
        "api_base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "api_format": "openai",
        "key_placeholder": "OPENROUTER_API_KEY",
    },
    "custom": {
        "label": "Custom Endpoint",
        "api_base": "",
        "model": "",
        "api_format": "openai",
        "key_placeholder": "",
    },
}

# Minimum seconds between consecutive LLM API requests (per agent)
API_REQUEST_COOLDOWN = 1.0

# Retry settings for transient API errors (429, 5xx)
API_MAX_RETRIES = 10
API_BACKOFF_BASE = 2.0  # seconds; exponential
API_BACKOFF_MAX = 120.0  # cap per-retry wait time
