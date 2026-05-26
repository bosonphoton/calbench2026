"""
Model-specific API configuration.

Maps model identifiers to their required API parameters and quirks.
Update this file when new models are released or API requirements change.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for a specific model's API requirements."""

    # Parameter name for token limit
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    # Default max output tokens for this model
    default_max_tokens: int = 4096

    # Whether the model supports temperature parameter
    supports_temperature: bool = True

    # Provider-specific notes
    notes: str = ""


# Explicit model registry
# Add new models here as they're released
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # OpenAI GPT-4 family (standard API)
    "gpt-4": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "gpt-4-turbo": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "gpt-4o": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "gpt-4o-mini": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),

    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),

    # OpenAI GPT-5 family (new API format, no custom temperature, reasoning models)
    "gpt-5-mini": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),
    "gpt-5-mini-2025-08-07": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),
    "gpt-5.4": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),

    # OpenAI o-series (reasoning models)
    "o1-preview": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),
    "o1-mini": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),
    "o3-mini": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),

    # OpenAI GPT-5.4 mini
    "gpt-5.4-mini": ModelConfig(
        max_tokens_param="max_completion_tokens",
        default_max_tokens=16000,
        supports_temperature=False,
    ),

    # Google Gemini family (via OpenAI-compatible endpoint)
    "gemini-2.0-flash": ModelConfig(default_max_tokens=16384),
    "gemini-2.5-flash": ModelConfig(default_max_tokens=16384),
    "gemini-2.5-pro": ModelConfig(default_max_tokens=16384),
    "gemini-3-flash-preview": ModelConfig(default_max_tokens=16384),

    # Anthropic Claude 3 family
    "claude-3-opus-20240229": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "claude-3-sonnet-20240229": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "claude-3-haiku-20240307": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "claude-3-5-sonnet-20240620": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "claude-3-5-sonnet-20241022": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),
    "claude-3-5-haiku-20241022": ModelConfig(max_tokens_param="max_tokens", supports_temperature=True),

    # Anthropic Claude 4 family
    "claude-sonnet-4-5-20250929": ModelConfig(max_tokens_param="max_tokens", default_max_tokens=8192, supports_temperature=True),
    "claude-sonnet-4-6": ModelConfig(max_tokens_param="max_tokens", default_max_tokens=16000, supports_temperature=True),
    "claude-haiku-4-5-20251001": ModelConfig(max_tokens_param="max_tokens", default_max_tokens=8192, supports_temperature=True),
    "claude-haiku-4-5@20251001": ModelConfig(max_tokens_param="max_tokens", default_max_tokens=8192, supports_temperature=True),
}

# Fallback defaults for unknown models
DEFAULT_CONFIG = ModelConfig(
    max_tokens_param="max_tokens",
    supports_temperature=True,
    notes="Using default config - model not in registry"
)


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model.

    Args:
        model_name: Full model identifier (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")

    Returns:
        ModelConfig with the model's API requirements.
        Returns DEFAULT_CONFIG if model is not in registry.

    Examples:
        >>> config = get_model_config("gpt-4o-mini")
        >>> config.max_tokens_param
        'max_tokens'
        >>> config.supports_temperature
        True

        >>> config = get_model_config("o1-preview")
        >>> config.max_tokens_param
        'max_completion_tokens'
        >>> config.supports_temperature
        False

        >>> config = get_model_config("meta-llama/llama-3.3-70b-instruct")
        >>> config.max_tokens_param
        'max_tokens'
        >>> config.default_max_tokens
        8192
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Explicit OpenRouter model overrides
    openrouter_overrides = {
        "qwen/qwen3.6-plus-preview:free": ModelConfig(
            default_max_tokens=65536,
            notes="Qwen 3.6 Plus Preview (free tier)",
        ),
    }
    if model_name in openrouter_overrides:
        return openrouter_overrides[model_name]

    # OpenRouter models (detected by slash in model ID)
    if "/" in model_name:
        return ModelConfig(
            max_tokens_param="max_tokens",
            default_max_tokens=16000,  # Standard for all OpenRouter models
            supports_temperature=True,
            notes=f"Auto-detected OpenRouter model: {model_name}",
        )

    # Prefix-based fallback for model families
    if model_name.startswith("gemini"):
        return ModelConfig(
            default_max_tokens=8192,
            notes=f"Auto-detected from prefix: {model_name}",
        )
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return ModelConfig(
            max_tokens_param="max_completion_tokens",
            default_max_tokens=16000,
            supports_temperature=False,
            notes=f"Auto-detected from prefix: {model_name}",
        )

    return DEFAULT_CONFIG
