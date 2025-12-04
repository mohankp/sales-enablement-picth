"""Tests for the content processor module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.processing.processor import (
    ContentProcessor,
    ProcessingConfig,
    ProcessingResult,
)
from src.models.content import (
    ContentBlock,
    ContentType,
    ExtractedContent,
    PageContent,
)
from src.models.processed import (
    AudienceType,
    FeatureCategory,
    PricingModel,
    ProcessedContent,
)
from src.llm import LLMResponse, TokenUsage, ModelName


class TestProcessingConfig:
    """Tests for ProcessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()

        assert config.default_model == ModelName.SONNET
        assert config.enable_features is True
        assert config.enable_benefits is True
        assert config.enable_use_cases is True
        assert config.enable_competitive is True
        assert config.enable_audience is True
        assert config.enable_pricing is True
        assert config.enable_technical is True
        assert config.max_concurrent_requests == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            default_model=ModelName.HAIKU,
            enable_features=True,
            enable_benefits=False,
            max_concurrent_requests=5,
        )

        assert config.default_model == ModelName.HAIKU
        assert config.enable_benefits is False
        assert config.max_concurrent_requests == 5


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = ProcessingResult(
            success=True,
            data={"features": []},
            tokens_used=100,
            cost_usd=0.01,
            latency_ms=500.0,
        )

        assert result.success is True
        assert result.data == {"features": []}
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = ProcessingResult(
            success=False,
            error="API call failed",
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "API call failed"


class TestContentProcessor:
    """Tests for ContentProcessor class."""

    @pytest.fixture
    def sample_content(self):
        """Create sample extracted content for testing."""
        blocks = [
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Amazing Product",
                heading_level=1,
            ),
            ContentBlock(
                content_type=ContentType.PARAGRAPH,
                text="Our product helps teams collaborate better with AI-powered features.",
            ),
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Features",
                heading_level=2,
            ),
            ContentBlock(
                content_type=ContentType.LIST,
                text="Features",
                list_items=[
                    "Real-time collaboration",
                    "AI-powered suggestions",
                    "Enterprise security",
                ],
            ),
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Pricing",
                heading_level=2,
            ),
            ContentBlock(
                content_type=ContentType.PARAGRAPH,
                text="Free tier available. Pro plan at $10/month. Enterprise contact us.",
            ),
        ]

        page = PageContent(
            url="https://example.com/product",
            title="Amazing Product - AI Collaboration",
            meta_description="The best AI collaboration tool for teams",
            content_blocks=blocks,
            word_count=50,
        )

        return ExtractedContent(
            product_name="Amazing Product",
            product_url="https://example.com",
            pages=[page],
            extraction_id="test_extraction_123",
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return LLMResponse(
            content='{"features": [{"name": "Real-time Collaboration", "description": "Work together in real-time", "category": "collaboration", "benefits": ["Faster teamwork"], "use_cases": ["Remote teams"], "is_flagship": true, "is_unique": false}]}',
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=1000, output_tokens=200),
            latency_ms=500.0,
        )

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        config = ProcessingConfig()
        processor = ContentProcessor(config)

        assert processor.config == config
        assert processor._client is None  # Not started yet

    @pytest.mark.asyncio
    async def test_processor_context_manager(self):
        """Test processor as async context manager."""
        config = ProcessingConfig()

        with patch.object(ContentProcessor, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(ContentProcessor, 'stop', new_callable=AsyncMock) as mock_stop:
                async with ContentProcessor(config) as processor:
                    pass

                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_raises_when_not_started(self):
        """Test that _ensure_client raises error when not started."""
        processor = ContentProcessor()

        with pytest.raises(RuntimeError, match="Processor not initialized"):
            processor._ensure_client()

    def test_combine_chunks_for_analysis(self, sample_content):
        """Test combining chunks into analysis string."""
        processor = ContentProcessor()
        chunks = processor._chunker.chunk_extracted_content(sample_content)

        combined = processor._combine_chunks_for_analysis(chunks)

        assert isinstance(combined, str)
        assert len(combined) > 0
        assert "Amazing Product" in combined

    @pytest.mark.asyncio
    async def test_detect_product_name(self, sample_content):
        """Test product name detection."""
        processor = ContentProcessor()

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=LLMResponse(
            content="Amazing Product",
            model="claude-3-5-haiku-20241022",
            usage=TokenUsage(input_tokens=100, output_tokens=5),
        ))

        processor._client = mock_client

        content_text = "Amazing Product is a collaboration tool..."
        name = await processor._detect_product_name(content_text)

        assert name == "Amazing Product"

    @pytest.mark.asyncio
    async def test_extract_features_success(self, mock_llm_response):
        """Test successful feature extraction."""
        processor = ContentProcessor()

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_llm_response)
        processor._client = mock_client

        result = await processor._extract_features(
            "Product content here",
            "Test Product",
        )

        assert result.success is True
        assert result.data is not None
        assert len(result.data.features) == 1
        assert result.data.features[0].name == "Real-time Collaboration"
        assert result.data.features[0].is_flagship is True

    @pytest.mark.asyncio
    async def test_extract_features_handles_error(self):
        """Test feature extraction handles errors gracefully."""
        processor = ContentProcessor()

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=Exception("API Error"))
        processor._client = mock_client

        result = await processor._extract_features(
            "Product content here",
            "Test Product",
        )

        assert result.success is False
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_extract_benefits(self):
        """Test benefit extraction."""
        processor = ContentProcessor()

        mock_response = LLMResponse(
            content='{"benefits": [{"headline": "Save Time", "description": "Reduce manual work by 50%", "target_audience": ["business", "technical"], "supporting_features": ["Automation"], "proof_points": ["Customer X saved 10 hours/week"], "business_impact": "Cost reduction"}]}',
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=500, output_tokens=100),
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        processor._client = mock_client

        result = await processor._extract_benefits("Content", "Product")

        assert result.success is True
        assert len(result.data.benefits) == 1
        assert result.data.benefits[0].headline == "Save Time"
        assert AudienceType.BUSINESS in result.data.benefits[0].target_audience

    @pytest.mark.asyncio
    async def test_extract_pricing(self):
        """Test pricing extraction."""
        processor = ContentProcessor()

        mock_response = LLMResponse(
            content='{"pricing_model": "tiered", "has_free_tier": true, "has_free_trial": true, "trial_duration": "14 days", "tiers": [{"name": "Free", "price": "$0", "features_included": ["Basic features"], "is_popular": false}], "currency": "USD"}',
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=300, output_tokens=80),
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        processor._client = mock_client

        result = await processor._extract_pricing("Content", "Product")

        assert result.success is True
        assert result.data.pricing_model == PricingModel.TIERED
        assert result.data.has_free_tier is True
        assert result.data.has_free_trial is True
        assert len(result.data.tiers) == 1

    @pytest.mark.asyncio
    async def test_generate_summary(self):
        """Test summary generation."""
        processor = ContentProcessor()

        mock_response = LLMResponse(
            content='{"executive_summary": "Amazing Product is an AI collaboration tool.", "detailed_summary": "This product helps teams work together more effectively using AI.", "comprehensive_summary": "# Overview\\nAmazing Product...", "key_points": ["AI-powered", "Real-time collaboration", "Enterprise ready"], "product_category": "Collaboration Software", "tagline": "Work smarter together", "value_proposition": "Boost team productivity with AI"}',
            model="claude-3-5-haiku-20241022",
            usage=TokenUsage(input_tokens=800, output_tokens=150),
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        processor._client = mock_client

        result = await processor._generate_summary("Content", "Amazing Product")

        assert result.success is True
        assert "AI collaboration tool" in result.data.executive_summary
        assert len(result.data.key_points) == 3
        assert result.data.product_category == "Collaboration Software"

    @pytest.mark.asyncio
    async def test_process_single_aspect(self, sample_content):
        """Test processing a single aspect."""
        processor = ContentProcessor()

        mock_response = LLMResponse(
            content='{"features": []}',
            model="claude-sonnet-4-20250514",
            usage=TokenUsage(input_tokens=500, output_tokens=50),
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        processor._client = mock_client

        result = await processor.process_single_aspect(
            sample_content,
            "features",
            product_name="Test Product",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_single_aspect_invalid(self, sample_content):
        """Test processing invalid aspect."""
        processor = ContentProcessor()
        processor._client = MagicMock()

        result = await processor.process_single_aspect(
            sample_content,
            "invalid_aspect",
        )

        assert result.success is False
        assert "Unknown aspect" in result.error


class TestProcessedContentModel:
    """Tests for ProcessedContent model."""

    def test_to_pitch_context(self):
        """Test conversion to pitch context dictionary."""
        from src.models.processed import (
            ContentSummary,
            FeatureSet,
            ProductFeature,
            BenefitSet,
            CustomerBenefit,
        )

        processed = ProcessedContent(
            product_name="Test Product",
            product_url="https://example.com",
            summary=ContentSummary(
                executive_summary="A great product",
                detailed_summary="More details here",
                comprehensive_summary="Full summary",
                key_points=["Point 1", "Point 2"],
                tagline="Best in class",
                value_proposition="Save time and money",
            ),
            features=FeatureSet(
                features=[
                    ProductFeature(
                        name="Feature 1",
                        description="Desc 1",
                        is_flagship=True,
                    )
                ],
            ),
            benefits=BenefitSet(
                benefits=[
                    CustomerBenefit(
                        headline="Benefit 1",
                        description="Desc",
                    )
                ],
            ),
        )

        # Compute stats first
        processed.compute_stats()

        context = processed.to_pitch_context()

        assert context["product_name"] == "Test Product"
        assert context["tagline"] == "Best in class"
        assert "Feature 1" in context["flagship_features"]
        assert len(context["all_features"]) == 1

    def test_compute_stats(self):
        """Test statistics computation."""
        from src.models.processed import (
            ContentSummary,
            FeatureSet,
            ProductFeature,
            BenefitSet,
            UseCaseSet,
        )

        processed = ProcessedContent(
            product_name="Test",
            product_url="https://test.com",
            summary=ContentSummary(
                executive_summary="",
                detailed_summary="",
                comprehensive_summary="",
            ),
            features=FeatureSet(
                features=[
                    ProductFeature(
                        name="F1",
                        description="D1",
                        category=FeatureCategory.CORE,
                        is_flagship=True,
                    ),
                    ProductFeature(
                        name="F2",
                        description="D2",
                        category=FeatureCategory.SECURITY,
                    ),
                ]
            ),
        )

        processed.compute_stats()

        assert "F1" in processed.features.flagship_features
        assert processed.features.feature_count_by_category["core"] == 1
        assert processed.features.feature_count_by_category["security"] == 1
