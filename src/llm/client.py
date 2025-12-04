"""Anthropic API client wrapper with async support, retry logic, and streaming."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import anthropic
from anthropic import APIError, APIStatusError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLMConfig, ModelName, ModelSettings

logger = logging.getLogger(__name__)


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

    def calculate_cost(self, model: ModelName) -> float:
        """Calculate cost in USD for this usage."""
        from .config import MODEL_PRICING

        pricing = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    usage: TokenUsage
    stop_reason: Optional[str] = None
    thinking: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None

    @property
    def cost_usd(self) -> float:
        """Calculate cost for this response."""
        try:
            model_name = ModelName(self.model)
        except ValueError:
            model_name = ModelName.SONNET
        return self.usage.calculate_cost(model_name)


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

    def add(self, response: LLMResponse) -> None:
        """Add a response to the tracker."""
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_cost_usd += response.cost_usd
        self.request_count += 1

        model_key = response.model
        self._costs_by_model[model_key] = (
            self._costs_by_model.get(model_key, 0.0) + response.cost_usd
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


class AnthropicClient:
    """
    Async wrapper around the Anthropic API with retry logic, streaming, and cost tracking.

    Usage:
        config = LLMConfig()
        async with AnthropicClient(config) as client:
            response = await client.complete("Hello, world!")
            print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._sync_client: Optional[anthropic.Anthropic] = None
        self.cost_tracker = CostTracker()
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> "AnthropicClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize the async client."""
        api_key = self._get_api_key()
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.config.api_base_url,
            timeout=self.config.timeout_seconds,
        )
        logger.info("Anthropic async client initialized")

    async def stop(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Anthropic client closed")

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.config.api_key:
            return self.config.api_key.get_secret_value()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key in LLMConfig."
            )
        return api_key

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AnthropicClient()' or call start()."
            )
        return self._client

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.config.requests_per_minute:
            min_interval = 60.0 / self.config.requests_per_minute
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
        settings: Optional[ModelSettings] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to Claude.

        Args:
            prompt: The user prompt (ignored if messages provided)
            system: Optional system prompt
            messages: Optional list of message dicts (overrides prompt)
            settings: Model settings (uses defaults if not provided)
            **kwargs: Additional arguments passed to the API

        Returns:
            LLMResponse with content, usage, and metadata
        """
        client = self._ensure_client()
        settings = settings or self.config.get_model_settings()

        system_prompt, message_list = self._build_messages(prompt, system, messages)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": settings.model.value,
            "max_tokens": settings.max_tokens,
            "messages": message_list,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if settings.temperature is not None:
            request_params["temperature"] = settings.temperature

        if settings.top_p is not None:
            request_params["top_p"] = settings.top_p

        if settings.top_k is not None:
            request_params["top_k"] = settings.top_k

        if settings.stop_sequences:
            request_params["stop_sequences"] = settings.stop_sequences

        # Handle extended thinking
        if settings.enable_thinking:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget_tokens,
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
                stop=stop_after_attempt(self.config.retry.max_retries + 1),
                wait=wait_exponential(
                    multiplier=self.config.retry.initial_delay_seconds,
                    max=self.config.retry.max_delay_seconds,
                    exp_base=self.config.retry.exponential_base,
                ),
                retry=retry_if_exception_type((RateLimitError, APIError)),
                reraise=True,
            ):
                with attempt:
                    if self.config.log_requests:
                        logger.debug(f"LLM Request: {request_params}")

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

        llm_response = LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason,
            thinking=thinking,
            latency_ms=latency_ms,
            raw_response=response if self.config.log_responses else None,
        )

        # Track costs
        if self.config.track_costs:
            self.cost_tracker.add(llm_response)

            # Check cost alert threshold
            if (
                self.config.cost_alert_threshold_usd
                and self.cost_tracker.total_cost_usd > self.config.cost_alert_threshold_usd
            ):
                logger.warning(
                    f"Cost threshold exceeded: ${self.cost_tracker.total_cost_usd:.4f} "
                    f"(threshold: ${self.config.cost_alert_threshold_usd:.2f})"
                )

        if self.config.log_responses:
            logger.debug(f"LLM Response: {content[:200]}...")

        return llm_response

    async def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        settings: Optional[ModelSettings] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion response from Claude.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            messages: Optional list of message dicts
            settings: Model settings
            **kwargs: Additional arguments

        Yields:
            StreamChunk objects with incremental text
        """
        client = self._ensure_client()
        settings = settings or self.config.get_model_settings()

        system_prompt, message_list = self._build_messages(prompt, system, messages)

        request_params: dict[str, Any] = {
            "model": settings.model.value,
            "max_tokens": settings.max_tokens,
            "messages": message_list,
            "stream": True,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if settings.temperature is not None:
            request_params["temperature"] = settings.temperature

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
        if self.config.track_costs:
            response = LLMResponse(
                content=full_content,
                model=settings.model.value,
                usage=usage,
                latency_ms=latency_ms,
            )
            self.cost_tracker.add(response)

    async def complete_with_structured_output(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        *,
        system: Optional[str] = None,
        settings: Optional[ModelSettings] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Request a completion with structured JSON output.

        Args:
            prompt: The user prompt
            output_schema: JSON schema for the expected output
            system: Optional system prompt
            settings: Model settings
            **kwargs: Additional arguments

        Returns:
            LLMResponse with JSON content
        """
        import json

        schema_str = json.dumps(output_schema, indent=2)

        structured_system = system or ""
        structured_system += f"""

You must respond with valid JSON that matches this schema:
```json
{schema_str}
```

Respond ONLY with the JSON object, no additional text or markdown formatting."""

        return await self.complete(
            prompt,
            system=structured_system.strip(),
            settings=settings,
            **kwargs,
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked costs."""
        return self.cost_tracker.get_summary()

    def reset_cost_tracker(self) -> None:
        """Reset the cost tracker."""
        self.cost_tracker = CostTracker()


# Convenience function for one-off calls
async def quick_complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: ModelName = ModelName.SONNET,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """
    Quick one-off completion without managing client lifecycle.

    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        The response content string
    """
    config = LLMConfig()
    settings = ModelSettings(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    async with AnthropicClient(config) as client:
        response = await client.complete(prompt, system=system, settings=settings)
        return response.content
