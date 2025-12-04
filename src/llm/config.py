"""Configuration models for LLM integration."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr


class ModelName(str, Enum):
    """Available Claude model names."""

    # Claude 4 models
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"

    # Claude 3.5 models
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU_3_5 = "claude-3-5-haiku-20241022"

    # Aliases for convenience
    OPUS = "claude-opus-4-20250514"
    SONNET = "claude-sonnet-4-20250514"
    HAIKU = "claude-3-5-haiku-20241022"


class ModelCapability(str, Enum):
    """Model capabilities for task routing."""

    ANALYSIS = "analysis"  # Deep content analysis
    EXTRACTION = "extraction"  # Entity/data extraction
    GENERATION = "generation"  # Content generation
    SUMMARIZATION = "summarization"  # Summarizing content
    CLASSIFICATION = "classification"  # Content classification
    REASONING = "reasoning"  # Complex reasoning tasks


# Model pricing per million tokens (as of 2025)
MODEL_PRICING = {
    ModelName.CLAUDE_OPUS_4: {"input": 15.00, "output": 75.00},
    ModelName.CLAUDE_SONNET_4: {"input": 3.00, "output": 15.00},
    ModelName.CLAUDE_SONNET_3_5: {"input": 3.00, "output": 15.00},
    ModelName.CLAUDE_HAIKU_3_5: {"input": 0.80, "output": 4.00},
}

# Model context windows
MODEL_CONTEXT_WINDOWS = {
    ModelName.CLAUDE_OPUS_4: 200_000,
    ModelName.CLAUDE_SONNET_4: 200_000,
    ModelName.CLAUDE_SONNET_3_5: 200_000,
    ModelName.CLAUDE_HAIKU_3_5: 200_000,
}


class ModelSettings(BaseModel):
    """Settings for a specific model invocation."""

    model: ModelName = ModelName.SONNET
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    stop_sequences: list[str] = Field(default_factory=list)

    # Extended thinking (for complex reasoning)
    enable_thinking: bool = False
    thinking_budget_tokens: int = Field(default=10000, ge=1000, le=100000)

    @property
    def context_window(self) -> int:
        """Get context window size for the model."""
        return MODEL_CONTEXT_WINDOWS.get(self.model, 200_000)

    @property
    def input_cost_per_million(self) -> float:
        """Get input token cost per million."""
        return MODEL_PRICING.get(self.model, {}).get("input", 3.00)

    @property
    def output_cost_per_million(self) -> float:
        """Get output token cost per million."""
        return MODEL_PRICING.get(self.model, {}).get("output", 15.00)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_delay_seconds: float = Field(default=60.0, ge=1.0)
    exponential_base: float = Field(default=2.0, ge=1.5, le=3.0)
    retry_on_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    retry_on_timeout: bool = True


class LLMConfig(BaseModel):
    """Main configuration for LLM integration."""

    # API Configuration
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key. If not set, reads from ANTHROPIC_API_KEY env var",
    )
    api_base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (for proxies)",
    )
    timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    # Default model settings
    default_model: ModelName = ModelName.SONNET
    default_max_tokens: int = 4096
    default_temperature: float = 0.7

    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Rate limiting
    requests_per_minute: Optional[int] = Field(default=50, ge=1)
    tokens_per_minute: Optional[int] = Field(default=100_000, ge=1000)

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(default=3600, ge=60)

    # Cost tracking
    track_costs: bool = True
    cost_alert_threshold_usd: Optional[float] = Field(default=10.0, ge=0.0)

    # Logging
    log_requests: bool = False
    log_responses: bool = False

    def get_model_settings(
        self,
        model: Optional[ModelName] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ModelSettings:
        """Create ModelSettings with defaults from config."""
        return ModelSettings(
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            **kwargs,
        )


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    # Template settings
    template_dir: Optional[str] = Field(
        default=None,
        description="Directory containing custom prompt templates",
    )
    enable_versioning: bool = True

    # Variable handling
    strict_variables: bool = Field(
        default=True,
        description="Raise error if template variables are missing",
    )
    default_variables: dict[str, Any] = Field(default_factory=dict)

    # System prompt settings
    include_system_context: bool = True
    system_context_template: Optional[str] = None

    # Output format hints
    prefer_json_output: bool = True
    json_schema_validation: bool = True


class TaskModelMapping(BaseModel):
    """Maps task types to preferred models."""

    analysis: ModelName = ModelName.SONNET
    extraction: ModelName = ModelName.SONNET
    generation: ModelName = ModelName.SONNET
    summarization: ModelName = ModelName.HAIKU
    classification: ModelName = ModelName.HAIKU
    reasoning: ModelName = ModelName.OPUS

    def get_model_for_task(self, capability: ModelCapability) -> ModelName:
        """Get the preferred model for a task capability."""
        mapping = {
            ModelCapability.ANALYSIS: self.analysis,
            ModelCapability.EXTRACTION: self.extraction,
            ModelCapability.GENERATION: self.generation,
            ModelCapability.SUMMARIZATION: self.summarization,
            ModelCapability.CLASSIFICATION: self.classification,
            ModelCapability.REASONING: self.reasoning,
        }
        return mapping.get(capability, ModelName.SONNET)
