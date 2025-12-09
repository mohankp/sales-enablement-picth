"""Base LLM provider interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional


class ProviderType(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class ProviderCapabilities:
    """Capabilities supported by a provider."""

    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_thinking: bool = False  # Extended thinking (Anthropic-specific)
    supports_vision: bool = False
    supports_function_calling: bool = False
    max_context_window: int = 200_000

    # Provider-specific features
    supports_caching: bool = False  # Anthropic prompt caching
    supports_batching: bool = False


@dataclass
class TokenUsage:
    """Track token usage for a request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM call."""

    content: str
    model: str
    usage: TokenUsage
    stop_reason: Optional[str] = None
    thinking: Optional[str] = None  # Extended thinking content (Anthropic)
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None
    provider: Optional[ProviderType] = None
    cost_usd: float = 0.0  # Cost in USD for this response


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    text: str
    is_final: bool = False
    usage: Optional[TokenUsage] = None
    thinking: Optional[str] = None


@dataclass
class CostTracker:
    """Track cumulative costs across requests."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0
    _costs_by_model: dict[str, float] = field(default_factory=dict)

    def add(self, response: LLMResponse, cost_usd: float) -> None:
        """Add a response to the tracker."""
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_cost_usd += cost_usd
        self.request_count += 1

        model_key = response.model
        self._costs_by_model[model_key] = (
            self._costs_by_model.get(model_key, 0.0) + cost_usd
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of tracked costs."""
        return {
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "costs_by_model": {k: round(v, 4) for k, v in self._costs_by_model.items()},
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    across different LLM backends (Anthropic, Google Gemini, etc.).

    Usage:
        provider = create_llm_provider(ProviderType.ANTHROPIC, config)
        async with provider:
            response = await provider.complete("Hello, world!")
    """

    def __init__(self):
        self.cost_tracker = CostTracker()
        self._started = False

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        ...

    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    @abstractmethod
    async def start(self) -> None:
        """Initialize the provider (create client connections, etc.)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Clean up provider resources."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            prompt: The user prompt (ignored if messages provided)
            system: Optional system prompt
            messages: Optional list of message dicts (overrides prompt)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with content, usage, and metadata
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion response from the LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            messages: Optional list of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Provider-specific arguments

        Yields:
            StreamChunk objects with incremental text
        """
        ...

    @abstractmethod
    def calculate_cost(self, usage: TokenUsage, model: str) -> float:
        """
        Calculate cost in USD for token usage.

        Args:
            usage: Token usage from a response
            model: Model name used

        Returns:
            Cost in USD
        """
        ...

    def get_cost_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked costs."""
        return self.cost_tracker.get_summary()

    def reset_cost_tracker(self) -> None:
        """Reset the cost tracker."""
        self.cost_tracker = CostTracker()
