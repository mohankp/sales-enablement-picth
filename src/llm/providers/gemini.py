"""Google Gemini LLM provider implementation."""

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Optional

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


# Gemini model pricing per million tokens (as of 2025)
GEMINI_PRICING = {
    # Gemini 3 (most intelligent)
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.00},
    # Gemini 2.5
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    # Gemini 2.0
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
    "gemini-2.0-pro-exp": {"input": 0.0, "output": 0.0},  # Free during preview
    # Latest aliases
    "gemini-flash-latest": {"input": 0.15, "output": 0.60},
    "gemini-pro-latest": {"input": 1.25, "output": 10.00},
}

# Default pricing for unknown models
DEFAULT_GEMINI_PRICING = {"input": 0.15, "output": 0.60}


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider.

    Supports Gemini models including:
    - Gemini 2.0 Flash (fast, efficient)
    - Gemini 1.5 Pro (high capability)
    - Gemini 1.5 Flash (balanced)

    Note: Gemini handles system prompts differently - they are prepended
    to the user message as context.

    Usage:
        provider = GeminiProvider(
            api_key="...",
            default_model="gemini-1.5-pro",
        )
        async with provider:
            response = await provider.complete("Hello!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-3-pro-preview",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = 60,
        track_costs: bool = True,
        log_requests: bool = False,
        log_responses: bool = False,
    ):
        super().__init__()
        self._api_key = api_key
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self.track_costs = track_costs
        self.log_requests = log_requests
        self.log_responses = log_responses

        self._client: Optional[Any] = None  # GenerativeModel
        self._genai_module: Optional[Any] = None
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GEMINI

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_prompt=True,  # Via system_instruction
            supports_thinking=False,  # No extended thinking
            supports_vision=True,
            supports_function_calling=True,
            max_context_window=1_000_000,  # Gemini 1.5 supports up to 1M tokens
            supports_caching=True,  # Context caching available
            supports_batching=False,
        )

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self._api_key:
            return self._api_key

        # Check multiple possible env var names
        for env_var in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key

        raise ValueError(
            "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
            "environment variable or pass api_key parameter."
        )

    async def start(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            self._genai_module = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        api_key = self._get_api_key()
        self._genai_module.configure(api_key=api_key)
        self._started = True
        logger.info("Gemini provider initialized")

    async def stop(self) -> None:
        """Clean up resources."""
        self._client = None
        self._started = False
        logger.info("Gemini provider closed")

    def _get_model(self, model: Optional[str] = None) -> Any:
        """Get or create a GenerativeModel instance."""
        if self._genai_module is None:
            raise RuntimeError(
                "Provider not initialized. Use 'async with provider' or call start()."
            )

        model_name = model or self.default_model

        # Ensure model name doesn't have "models/" prefix (SDK adds it automatically)
        if model_name.startswith("models/"):
            model_name = model_name[7:]

        return self._genai_module.GenerativeModel(model_name)

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.requests_per_minute:
            min_interval = 60.0 / self.requests_per_minute
            async with self._rate_limit_lock:
                elapsed = time.time() - self._last_request_time
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                self._last_request_time = time.time()

    def _build_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Build prompt for Gemini.

        Gemini uses system_instruction parameter for system prompts,
        or they can be prepended to the conversation.

        Returns:
            Tuple of (user_prompt, system_instruction)
        """
        if messages:
            # Convert messages to a single prompt
            # Gemini's chat API is different, so we concatenate for simple completion
            parts = []
            system_instruction = system
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            return "\n\n".join(parts), system_instruction

        return prompt, system

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
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to Gemini.

        Args:
            prompt: The user prompt (ignored if messages provided)
            system: Optional system prompt (used as system_instruction)
            messages: Optional list of message dicts (overrides prompt)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling
            stop_sequences: Sequences that stop generation
            model: Model to use (defaults to default_model)
            **kwargs: Additional arguments

        Returns:
            LLMResponse with content, usage, and metadata
        """
        model_name = model or self.default_model
        user_prompt, system_instruction = self._build_prompt(prompt, system, messages)

        # Create generation config
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if top_k is not None:
            generation_config["top_k"] = top_k

        if stop_sequences:
            generation_config["stop_sequences"] = stop_sequences

        # Apply rate limiting
        await self._apply_rate_limit()

        start_time = time.time()

        # Get model with optional system instruction
        genai_model = self._get_model(model_name)

        if self.log_requests:
            logger.debug(f"Gemini Request: model={model_name}, prompt={user_prompt[:200]}...")

        # Execute with retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Use system instruction if provided
                if system_instruction:
                    # Create a new model instance with system instruction
                    genai_model = self._genai_module.GenerativeModel(
                        model_name,
                        system_instruction=system_instruction,
                    )

                response = await asyncio.to_thread(
                    genai_model.generate_content,
                    user_prompt,
                    generation_config=generation_config,
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Gemini request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error

        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        content = ""
        try:
            content = response.text
        except ValueError:
            # Response may be blocked or empty
            if response.prompt_feedback:
                logger.warning(f"Gemini response blocked: {response.prompt_feedback}")
            content = ""

        # Extract token usage
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage.input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage.output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        # Calculate cost
        cost_usd = self.calculate_cost(usage, model_name)

        # Determine stop reason
        stop_reason = None
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                stop_reason = str(candidate.finish_reason)

        llm_response = LLMResponse(
            content=content,
            model=model_name,
            usage=usage,
            stop_reason=stop_reason,
            thinking=None,  # Gemini doesn't support extended thinking
            latency_ms=latency_ms,
            raw_response=response if self.log_responses else None,
            provider=ProviderType.GEMINI,
            cost_usd=cost_usd,
        )

        # Track costs
        if self.track_costs:
            self.cost_tracker.add(llm_response, cost_usd)

        if self.log_responses:
            logger.debug(f"Gemini Response: {content[:200]}...")

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
        Stream a completion response from Gemini.

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
        model_name = model or self.default_model
        user_prompt, system_instruction = self._build_prompt(prompt, system, messages)

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        await self._apply_rate_limit()

        start_time = time.time()
        full_content = ""

        # Get model with optional system instruction
        if system_instruction:
            genai_model = self._genai_module.GenerativeModel(
                model_name,
                system_instruction=system_instruction,
            )
        else:
            genai_model = self._get_model(model_name)

        # Gemini streaming is synchronous, so we run in thread
        def stream_sync():
            return genai_model.generate_content(
                user_prompt,
                generation_config=generation_config,
                stream=True,
            )

        response_stream = await asyncio.to_thread(stream_sync)

        # Process stream
        usage = TokenUsage()
        for chunk in response_stream:
            try:
                if chunk.text:
                    text = chunk.text
                    full_content += text
                    yield StreamChunk(text=text, is_final=False)
            except ValueError:
                # Chunk may be empty or blocked
                pass

            # Update usage if available
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage.input_tokens = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                usage.output_tokens = getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0

        # Yield final chunk
        latency_ms = (time.time() - start_time) * 1000
        yield StreamChunk(text="", is_final=True, usage=usage)

        # Track costs
        if self.track_costs:
            cost_usd = self.calculate_cost(usage, model_name)
            response = LLMResponse(
                content=full_content,
                model=model_name,
                usage=usage,
                latency_ms=latency_ms,
                provider=ProviderType.GEMINI,
                cost_usd=cost_usd,
            )
            self.cost_tracker.add(response, cost_usd)

    def calculate_cost(self, usage: TokenUsage, model: str) -> float:
        """Calculate cost in USD for token usage."""
        pricing = GEMINI_PRICING.get(model, DEFAULT_GEMINI_PRICING)
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
