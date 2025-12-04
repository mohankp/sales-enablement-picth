"""Tests for the Anthropic client wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.client import (
    AnthropicClient,
    CostTracker,
    LLMResponse,
    TokenUsage,
    StreamChunk,
)
from src.llm.config import LLMConfig, ModelName, ModelSettings


class TestTokenUsage:
    """Tests for TokenUsage."""

    def test_total_tokens(self):
        """Test total token calculation."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_calculate_cost_sonnet(self):
        """Test cost calculation for Sonnet."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000)
        cost = usage.calculate_cost(ModelName.SONNET)
        # Input: $3/M * 1M = $3
        # Output: $15/M * 0.1M = $1.5
        assert cost == pytest.approx(4.5, rel=0.01)

    def test_calculate_cost_haiku(self):
        """Test cost calculation for Haiku."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = usage.calculate_cost(ModelName.HAIKU)
        # Input: $0.80/M * 1M = $0.80
        # Output: $4/M * 1M = $4
        assert cost == pytest.approx(4.8, rel=0.01)


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_properties(self):
        """Test response properties."""
        response = LLMResponse(
            content="Hello, world!",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            latency_ms=150.0,
        )
        assert response.content == "Hello, world!"
        assert response.stop_reason == "end_turn"
        assert response.latency_ms == 150.0

    def test_cost_usd_property(self):
        """Test cost calculation property."""
        response = LLMResponse(
            content="Test",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        assert response.cost_usd >= 0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_add_response(self):
        """Test adding response to tracker."""
        tracker = CostTracker()
        response = LLMResponse(
            content="Test",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )
        tracker.add(response)

        assert tracker.request_count == 1
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost_usd > 0

    def test_multiple_responses(self):
        """Test tracking multiple responses."""
        tracker = CostTracker()

        for _ in range(3):
            response = LLMResponse(
                content="Test",
                model="claude-sonnet-4-20250514",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
            tracker.add(response)

        assert tracker.request_count == 3
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150

    def test_get_summary(self):
        """Test getting summary."""
        tracker = CostTracker()
        response = LLMResponse(
            content="Test",
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )
        tracker.add(response)

        summary = tracker.get_summary()
        assert "total_requests" in summary
        assert "total_cost_usd" in summary
        assert "costs_by_model" in summary
        assert summary["total_requests"] == 1


class TestAnthropicClientInit:
    """Tests for AnthropicClient initialization."""

    def test_default_config(self):
        """Test client with default config."""
        client = AnthropicClient()
        assert client.config is not None
        assert client.config.default_model == ModelName.SONNET

    def test_custom_config(self):
        """Test client with custom config."""
        config = LLMConfig(
            default_model=ModelName.OPUS,
            default_max_tokens=8192,
        )
        client = AnthropicClient(config)
        assert client.config.default_model == ModelName.OPUS
        assert client.config.default_max_tokens == 8192

    def test_client_not_started(self):
        """Test that client raises error when not started."""
        client = AnthropicClient()
        with pytest.raises(RuntimeError, match="not initialized"):
            client._ensure_client()


class TestAnthropicClientMethods:
    """Tests for AnthropicClient methods (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked Anthropic API."""
        config = LLMConfig()
        client = AnthropicClient(config)

        # Create mock async client
        mock_anthropic = AsyncMock()
        client._client = mock_anthropic

        return client, mock_anthropic

    def test_build_messages_from_prompt(self):
        """Test building messages from prompt."""
        client = AnthropicClient()
        system, messages = client._build_messages("Hello", system="Be helpful")

        assert system == "Be helpful"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_build_messages_from_list(self):
        """Test building messages from message list."""
        client = AnthropicClient()
        msg_list = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        system, messages = client._build_messages("ignored", messages=msg_list)

        assert len(messages) == 3
        assert messages[2]["content"] == "How are you?"

    @pytest.mark.asyncio
    async def test_complete_builds_correct_params(self, mock_client):
        """Test that complete builds correct API parameters."""
        client, mock_anthropic = mock_client

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response text")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        settings = ModelSettings(
            model=ModelName.SONNET,
            max_tokens=1024,
            temperature=0.5,
        )

        await client.complete(
            "Test prompt",
            system="Be helpful",
            settings=settings,
        )

        # Verify the call was made with correct params
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs

        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_complete_tracks_costs(self, mock_client):
        """Test that complete tracks costs."""
        client, mock_anthropic = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_response.stop_reason = "end_turn"
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        response = await client.complete("Test")

        assert client.cost_tracker.request_count == 1
        assert client.cost_tracker.total_input_tokens == 100
        assert client.cost_tracker.total_output_tokens == 50

    def test_get_cost_summary(self):
        """Test getting cost summary."""
        client = AnthropicClient()
        summary = client.get_cost_summary()
        assert summary["total_requests"] == 0
        assert summary["total_cost_usd"] == 0

    def test_reset_cost_tracker(self):
        """Test resetting cost tracker."""
        client = AnthropicClient()
        client.cost_tracker.request_count = 5
        client.cost_tracker.total_cost_usd = 10.0

        client.reset_cost_tracker()

        assert client.cost_tracker.request_count == 0
        assert client.cost_tracker.total_cost_usd == 0


class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_chunk_properties(self):
        """Test chunk properties."""
        chunk = StreamChunk(text="Hello", is_final=False)
        assert chunk.text == "Hello"
        assert chunk.is_final is False
        assert chunk.usage is None

    def test_final_chunk(self):
        """Test final chunk with usage."""
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        chunk = StreamChunk(text="", is_final=True, usage=usage)
        assert chunk.is_final is True
        assert chunk.usage.total_tokens == 15


class TestApiKeyHandling:
    """Tests for API key handling."""

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error."""
        config = LLMConfig(api_key=None)
        client = AnthropicClient(config)

        # Clear environment variable for test
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                client._get_api_key()

    def test_api_key_from_env(self):
        """Test getting API key from environment."""
        config = LLMConfig(api_key=None)
        client = AnthropicClient(config)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            key = client._get_api_key()
            assert key == "test-key"

    def test_api_key_from_config(self):
        """Test getting API key from config."""
        from pydantic import SecretStr

        config = LLMConfig(api_key=SecretStr("config-key"))
        client = AnthropicClient(config)

        key = client._get_api_key()
        assert key == "config-key"
