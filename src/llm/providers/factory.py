"""Factory for creating LLM providers."""

import logging
from typing import Any, Optional

from .base import LLMProvider, ProviderCapabilities, ProviderType

logger = logging.getLogger(__name__)


def create_llm_provider(
    provider: ProviderType | str,
    api_key: Optional[str] = None,
    default_model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    This factory function creates the appropriate provider implementation
    based on the specified provider type.

    Args:
        provider: The provider type (ProviderType enum or string)
        api_key: Optional API key (falls back to environment variables)
        default_model: Optional default model name
        **kwargs: Additional provider-specific arguments

    Returns:
        An LLMProvider instance

    Raises:
        ValueError: If the provider type is unknown

    Example:
        # Create Anthropic provider
        provider = create_llm_provider(
            ProviderType.ANTHROPIC,
            default_model="claude-sonnet-4-20250514",
        )

        # Create Gemini provider
        provider = create_llm_provider(
            "gemini",  # String also works
            default_model="gemini-1.5-pro",
        )

        # Use the provider
        async with provider:
            response = await provider.complete("Hello!")
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: {[p.value for p in ProviderType]}"
            )

    if provider == ProviderType.ANTHROPIC:
        from .anthropic import AnthropicProvider

        # Set default model for Anthropic
        model = default_model or "claude-sonnet-4-20250514"

        return AnthropicProvider(
            api_key=api_key,
            default_model=model,
            **kwargs,
        )

    elif provider == ProviderType.GEMINI:
        from .gemini import GeminiProvider

        # Set default model for Gemini (Gemini 3 Pro is the latest and most intelligent)
        model = default_model or "gemini-3-pro-preview"

        return GeminiProvider(
            api_key=api_key,
            default_model=model,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: {[p.value for p in ProviderType]}"
        )


def get_provider_capabilities(provider: ProviderType | str) -> ProviderCapabilities:
    """
    Get the capabilities of a provider without creating an instance.

    Args:
        provider: The provider type

    Returns:
        ProviderCapabilities for the specified provider
    """
    if isinstance(provider, str):
        provider = ProviderType(provider.lower())

    if provider == ProviderType.ANTHROPIC:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_prompt=True,
            supports_thinking=True,
            supports_vision=True,
            supports_function_calling=True,
            max_context_window=200_000,
            supports_caching=True,
            supports_batching=True,
        )

    elif provider == ProviderType.GEMINI:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_prompt=True,
            supports_thinking=False,
            supports_vision=True,
            supports_function_calling=True,
            max_context_window=1_000_000,
            supports_caching=True,
            supports_batching=False,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_model(provider: ProviderType | str) -> str:
    """
    Get the default model for a provider.

    Args:
        provider: The provider type

    Returns:
        Default model name for the provider
    """
    if isinstance(provider, str):
        provider = ProviderType(provider.lower())

    defaults = {
        ProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
        ProviderType.GEMINI: "gemini-3-pro-preview",
    }

    if provider not in defaults:
        raise ValueError(f"Unknown provider: {provider}")

    return defaults[provider]


def list_available_models(provider: ProviderType | str) -> list[str]:
    """
    List available models for a provider.

    Args:
        provider: The provider type

    Returns:
        List of available model names
    """
    if isinstance(provider, str):
        provider = ProviderType(provider.lower())

    models = {
        ProviderType.ANTHROPIC: [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        ProviderType.GEMINI: [
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp",
            "gemini-2.0-pro-exp",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ],
    }

    if provider not in models:
        raise ValueError(f"Unknown provider: {provider}")

    return models[provider]
