"""Tests for LLM configuration models."""

import pytest

from src.llm.config import (
    LLMConfig,
    ModelCapability,
    ModelName,
    ModelSettings,
    PromptConfig,
    RetryConfig,
    TaskModelMapping,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
)


class TestModelName:
    """Tests for ModelName enum."""

    def test_model_values(self):
        """Test model name values."""
        assert ModelName.SONNET.value == "claude-sonnet-4-20250514"
        assert ModelName.OPUS.value == "claude-opus-4-20250514"
        assert ModelName.HAIKU.value == "claude-3-5-haiku-20241022"

    def test_aliases(self):
        """Test that aliases resolve to correct values."""
        assert ModelName.OPUS == ModelName.CLAUDE_OPUS_4
        assert ModelName.SONNET == ModelName.CLAUDE_SONNET_4


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_defaults(self):
        """Test default settings."""
        settings = ModelSettings()
        assert settings.model == ModelName.SONNET
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
        assert settings.top_p == 1.0
        assert settings.enable_thinking is False

    def test_custom_settings(self):
        """Test custom settings."""
        settings = ModelSettings(
            model=ModelName.OPUS,
            max_tokens=8192,
            temperature=0.3,
            enable_thinking=True,
        )
        assert settings.model == ModelName.OPUS
        assert settings.max_tokens == 8192
        assert settings.temperature == 0.3
        assert settings.enable_thinking is True

    def test_context_window_property(self):
        """Test context window property."""
        settings = ModelSettings(model=ModelName.SONNET)
        assert settings.context_window == 200_000

    def test_cost_properties(self):
        """Test cost calculation properties."""
        settings = ModelSettings(model=ModelName.SONNET)
        assert settings.input_cost_per_million == 3.00
        assert settings.output_cost_per_million == 15.00

    def test_validation_bounds(self):
        """Test validation of bounds."""
        with pytest.raises(ValueError):
            ModelSettings(max_tokens=0)  # Must be >= 1

        with pytest.raises(ValueError):
            ModelSettings(temperature=1.5)  # Must be <= 1.0


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0
        assert config.exponential_base == 2.0
        assert 429 in config.retry_on_status_codes
        assert config.retry_on_timeout is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=2.0,
            retry_on_timeout=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay_seconds == 2.0
        assert config.retry_on_timeout is False


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.api_key is None
        assert config.default_model == ModelName.SONNET
        assert config.default_max_tokens == 4096
        assert config.timeout_seconds == 120.0
        assert config.track_costs is True

    def test_get_model_settings(self):
        """Test creating model settings from config."""
        config = LLMConfig(
            default_model=ModelName.OPUS,
            default_max_tokens=8192,
            default_temperature=0.5,
        )
        settings = config.get_model_settings()
        assert settings.model == ModelName.OPUS
        assert settings.max_tokens == 8192
        assert settings.temperature == 0.5

    def test_get_model_settings_override(self):
        """Test overriding settings."""
        config = LLMConfig(default_model=ModelName.SONNET)
        settings = config.get_model_settings(
            model=ModelName.HAIKU,
            temperature=0.9,
        )
        assert settings.model == ModelName.HAIKU
        assert settings.temperature == 0.9


class TestPromptConfig:
    """Tests for PromptConfig."""

    def test_defaults(self):
        """Test default prompt configuration."""
        config = PromptConfig()
        assert config.strict_variables is True
        assert config.enable_versioning is True
        assert config.prefer_json_output is True

    def test_default_variables(self):
        """Test default variables."""
        config = PromptConfig(
            default_variables={"tone": "professional", "length": "brief"}
        )
        assert config.default_variables["tone"] == "professional"


class TestTaskModelMapping:
    """Tests for TaskModelMapping."""

    def test_defaults(self):
        """Test default task model mapping."""
        mapping = TaskModelMapping()
        assert mapping.analysis == ModelName.SONNET
        assert mapping.summarization == ModelName.HAIKU
        assert mapping.reasoning == ModelName.OPUS

    def test_get_model_for_task(self):
        """Test getting model for capability."""
        mapping = TaskModelMapping()
        assert mapping.get_model_for_task(ModelCapability.REASONING) == ModelName.OPUS
        assert mapping.get_model_for_task(ModelCapability.SUMMARIZATION) == ModelName.HAIKU


class TestModelPricing:
    """Tests for model pricing constants."""

    def test_pricing_exists(self):
        """Test that pricing exists for all models."""
        for model in [ModelName.OPUS, ModelName.SONNET, ModelName.HAIKU]:
            assert model in MODEL_PRICING
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]

    def test_context_windows_exist(self):
        """Test that context windows exist for all models."""
        for model in [ModelName.OPUS, ModelName.SONNET, ModelName.HAIKU]:
            assert model in MODEL_CONTEXT_WINDOWS
            assert MODEL_CONTEXT_WINDOWS[model] >= 100_000
