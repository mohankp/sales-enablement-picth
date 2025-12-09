"""Configuration models for LLM integration."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr


class ProviderType(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


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


class GeminiModelName(str, Enum):
    """Available Gemini model names."""

    # Gemini 3 (most intelligent, latest)
    GEMINI_3_PRO = "gemini-3-pro-preview"

    # Gemini 2.5
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # Gemini 2.0
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_2_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_PRO_EXP = "gemini-2.0-pro-exp"

    # Latest aliases
    GEMINI_FLASH_LATEST = "gemini-flash-latest"
    GEMINI_PRO_LATEST = "gemini-pro-latest"

    # Aliases for convenience
    PRO = "gemini-3-pro-preview"  # Updated to Gemini 3
    FLASH = "gemini-2.5-flash"
    FAST = "gemini-2.5-flash"
    LITE = "gemini-2.5-flash-lite"


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
    # Anthropic Claude models
    ModelName.CLAUDE_OPUS_4: {"input": 15.00, "output": 75.00},
    ModelName.CLAUDE_SONNET_4: {"input": 3.00, "output": 15.00},
    ModelName.CLAUDE_SONNET_3_5: {"input": 3.00, "output": 15.00},
    ModelName.CLAUDE_HAIKU_3_5: {"input": 0.80, "output": 4.00},
}

# Gemini model pricing per million tokens (as of 2025)
GEMINI_PRICING = {
    # Gemini 3
    GeminiModelName.GEMINI_3_PRO: {"input": 1.25, "output": 10.00},  # Preview pricing
    # Gemini 2.5
    GeminiModelName.GEMINI_2_5_FLASH: {"input": 0.15, "output": 0.60},
    GeminiModelName.GEMINI_2_5_FLASH_LITE: {"input": 0.075, "output": 0.30},
    GeminiModelName.GEMINI_2_5_PRO: {"input": 1.25, "output": 5.00},
    # Gemini 2.0
    GeminiModelName.GEMINI_2_FLASH: {"input": 0.10, "output": 0.40},
    GeminiModelName.GEMINI_2_FLASH_LITE: {"input": 0.075, "output": 0.30},
    GeminiModelName.GEMINI_2_FLASH_EXP: {"input": 0.0, "output": 0.0},  # Free during preview
    GeminiModelName.GEMINI_2_PRO_EXP: {"input": 0.0, "output": 0.0},  # Free during preview
    # Aliases
    GeminiModelName.GEMINI_FLASH_LATEST: {"input": 0.15, "output": 0.60},
    GeminiModelName.GEMINI_PRO_LATEST: {"input": 1.25, "output": 10.00},  # Points to Gemini 3
}

# Model context windows
MODEL_CONTEXT_WINDOWS = {
    # Anthropic
    ModelName.CLAUDE_OPUS_4: 200_000,
    ModelName.CLAUDE_SONNET_4: 200_000,
    ModelName.CLAUDE_SONNET_3_5: 200_000,
    ModelName.CLAUDE_HAIKU_3_5: 200_000,
}

# Gemini context windows
GEMINI_CONTEXT_WINDOWS = {
    # Gemini 3
    GeminiModelName.GEMINI_3_PRO: 1_048_576,  # 1M+ tokens
    # Gemini 2.5
    GeminiModelName.GEMINI_2_5_FLASH: 1_000_000,
    GeminiModelName.GEMINI_2_5_FLASH_LITE: 1_000_000,
    GeminiModelName.GEMINI_2_5_PRO: 1_000_000,
    # Gemini 2.0
    GeminiModelName.GEMINI_2_FLASH: 1_000_000,
    GeminiModelName.GEMINI_2_FLASH_LITE: 1_000_000,
    GeminiModelName.GEMINI_2_FLASH_EXP: 1_000_000,
    GeminiModelName.GEMINI_2_PRO_EXP: 1_000_000,
    # Aliases
    GeminiModelName.GEMINI_FLASH_LATEST: 1_000_000,
    GeminiModelName.GEMINI_PRO_LATEST: 1_048_576,
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

    # Provider selection
    provider: ProviderType = Field(
        default=ProviderType.ANTHROPIC,
        description="LLM provider to use (anthropic or gemini)",
    )

    # API Configuration
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key. If not set, reads from environment variable",
    )
    api_base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (for proxies)",
    )
    timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    # Default model settings (for Anthropic)
    default_model: ModelName = ModelName.SONNET
    default_max_tokens: int = 4096
    default_temperature: float = 0.7

    # Gemini-specific default model
    gemini_default_model: GeminiModelName = GeminiModelName.GEMINI_3_PRO

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

    def get_default_model_name(self) -> str:
        """Get the default model name for the configured provider."""
        if self.provider == ProviderType.GEMINI:
            return self.gemini_default_model.value
        return self.default_model.value

    def get_api_key_value(self) -> Optional[str]:
        """Get the API key value as a plain string."""
        if self.api_key:
            return self.api_key.get_secret_value()
        return None


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
