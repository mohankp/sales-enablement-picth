"""Anthropic (Claude) LLM provider implementation."""

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Optional

import anthropic
from anthropic import APIError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import (
    CostTracker,
    LLMProvider,
    LLMResponse,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
    TokenUsage,
)

logger = logging.getLogger(__name__)


# Anthropic model pricing per million tokens (as of 2025)
ANTHROPIC_PRICING = {
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}

# Default pricing for unknown models
DEFAULT_ANTHROPIC_PRICING = {"input": 3.00, "output": 15.00}


class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) LLM provider.

    Supports all Claude models with features like:
    - Extended thinking
    - Prompt caching
    - Streaming responses
    - System prompts

    Usage:
        provider = AnthropicProvider(
            api_key="...",
            default_model="claude-sonnet-4-20250514",
        )
        async with provider:
            response = await provider.complete("Hello!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
        api_base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = 50,
        track_costs: bool = True,
        log_requests: bool = False,
        log_responses: bool = False,
    ):
        super().__init__()
        self._api_key = api_key
        self.default_model = default_model
        self.api_base_url = api_base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self.track_costs = track_costs
        self.log_requests = log_requests
        self.log_responses = log_responses

        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_prompt=True,
            supports_thinking=True,  # Extended thinking
            supports_vision=True,
            supports_function_calling=True,
            max_context_window=200_000,
            supports_caching=True,  # Prompt caching
            supports_batching=True,
        )

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self._api_key:
            return self._api_key

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        return api_key

    async def start(self) -> None:
        """Initialize the async client."""
        api_key = self._get_api_key()
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.api_base_url,
            timeout=self.timeout_seconds,
        )
        self._started = True
        logger.info("Anthropic provider initialized")

    async def stop(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
        self._started = False
        logger.info("Anthropic provider closed")

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "Provider not initialized. Use 'async with provider' or call start()."
            )
        return self._client

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.requests_per_minute:
            min_interval = 60.0 / self.requests_per_minute
            async with self._rate_limit_lock:
                elapsed = time.time() - self._last_request_time
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                self._last_request_time = time.time()

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Build messages array for the API call."""
        if messages:
            return system, messages
        return system, [{"role": "user", "content": prompt}]

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        model: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 10000,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to Claude.

        Args:
            prompt: The user prompt (ignored if messages provided)
            system: Optional system prompt
            messages: Optional list of message dicts (overrides prompt)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling (Anthropic-specific)
            stop_sequences: Sequences that stop generation
            model: Model to use (defaults to default_model)
            enable_thinking: Enable extended thinking (Anthropic-specific)
            thinking_budget_tokens: Token budget for thinking
            **kwargs: Additional arguments passed to the API

        Returns:
            LLMResponse with content, usage, and metadata
        """
        client = self._ensure_client()
        model = model or self.default_model

        system_prompt, message_list = self._build_messages(prompt, system, messages)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": message_list,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if temperature is not None:
            request_params["temperature"] = temperature

        if top_p is not None:
            request_params["top_p"] = top_p

        if top_k is not None:
            request_params["top_k"] = top_k

        if stop_sequences:
            request_params["stop_sequences"] = stop_sequences

        # Handle extended thinking
        if enable_thinking:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
            # Temperature must be 1 for thinking
            request_params["temperature"] = 1.0

        request_params.update(kwargs)

        # Apply rate limiting
        await self._apply_rate_limit()

        # Execute with retry logic
        start_time = time.time()

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential(multiplier=1.0, max=60.0, exp_base=2.0),
                retry=retry_if_exception_type((RateLimitError, APIError)),
                reraise=True,
            ):
                with attempt:
                    if self.log_requests:
                        logger.debug(f"Anthropic Request: {request_params}")

                    response = await client.messages.create(**request_params)
        except RetryError as e:
            raise e.last_attempt.result()

        latency_ms = (time.time() - start_time) * 1000

        # Extract response content
        content = ""
        thinking = None

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                thinking = block.thinking

        # Build usage tracking
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )

        # Calculate cost
        cost_usd = self.calculate_cost(usage, model)

        llm_response = LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason,
            thinking=thinking,
            latency_ms=latency_ms,
            raw_response=response if self.log_responses else None,
            provider=ProviderType.ANTHROPIC,
            cost_usd=cost_usd,
        )

        # Track costs
        if self.track_costs:
            self.cost_tracker.add(llm_response, cost_usd)

        if self.log_responses:
            logger.debug(f"Anthropic Response: {content[:200]}...")

        return llm_response

    async def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion response from Claude.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            messages: Optional list of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            model: Model to use
            **kwargs: Additional arguments

        Yields:
            StreamChunk objects with incremental text
        """
        client = self._ensure_client()
        model = model or self.default_model

        system_prompt, message_list = self._build_messages(prompt, system, messages)

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": message_list,
            "stream": True,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if temperature is not None:
            request_params["temperature"] = temperature

        request_params.update(kwargs)

        await self._apply_rate_limit()

        start_time = time.time()
        full_content = ""
        usage = TokenUsage()

        async with client.messages.stream(**request_params) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text = event.delta.text
                            full_content += text
                            yield StreamChunk(text=text, is_final=False)

                    elif event.type == "message_delta":
                        if hasattr(event, "usage"):
                            usage.output_tokens = getattr(event.usage, "output_tokens", 0)

                    elif event.type == "message_start":
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            usage.input_tokens = event.message.usage.input_tokens

        # Yield final chunk with usage
        latency_ms = (time.time() - start_time) * 1000
        yield StreamChunk(text="", is_final=True, usage=usage)

        # Track costs for streamed response
        if self.track_costs:
            cost_usd = self.calculate_cost(usage, model)
            response = LLMResponse(
                content=full_content,
                model=model,
                usage=usage,
                latency_ms=latency_ms,
                provider=ProviderType.ANTHROPIC,
                cost_usd=cost_usd,
            )
            self.cost_tracker.add(response, cost_usd)

    def calculate_cost(self, usage: TokenUsage, model: str) -> float:
        """Calculate cost in USD for token usage."""
        pricing = ANTHROPIC_PRICING.get(model, DEFAULT_ANTHROPIC_PRICING)
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


# For backwards compatibility, expose AnthropicClient as an alias
AnthropicClient = AnthropicProvider
