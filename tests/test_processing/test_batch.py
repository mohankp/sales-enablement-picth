"""Tests for the batch processing module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.processing.batch import (
    BatchConfig,
    BatchItemResult,
    BatchResult,
    BatchProcessor,
)
from src.models.content import (
    ContentBlock,
    ContentType,
    ExtractedContent,
    PageContent,
)
from src.models.processed import (
    ContentSummary,
    FeatureSet,
    ProcessedContent,
)


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()

        assert config.max_concurrent_items == 3
        assert config.continue_on_error is True
        assert config.max_retries == 2
        assert config.retry_delay_seconds == 5.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchConfig(
            max_concurrent_items=5,
            max_retries=3,
            continue_on_error=False,
        )

        assert config.max_concurrent_items == 5
        assert config.max_retries == 3
        assert config.continue_on_error is False


class TestBatchItemResult:
    """Tests for BatchItemResult."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = BatchItemResult(
            item_id="item1",
            success=True,
            result=MagicMock(),
            processing_time_ms=1500.0,
        )

        assert result.success is True
        assert result.error is None
        assert result.retries == 0

    def test_failed_result(self):
        """Test creating a failed result."""
        result = BatchItemResult(
            item_id="item1",
            success=False,
            error="API Error",
            retries=2,
        )

        assert result.success is False
        assert result.error == "API Error"
        assert result.retries == 2


class TestBatchResult:
    """Tests for BatchResult."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = BatchResult(
            total_items=10,
            successful=8,
            failed=2,
        )

        assert result.success_rate == 0.8

    def test_success_rate_empty(self):
        """Test success rate with no items."""
        result = BatchResult(
            total_items=0,
            successful=0,
            failed=0,
        )

        assert result.success_rate == 0.0

    def test_get_failed_items(self):
        """Test getting failed items."""
        result = BatchResult(
            total_items=3,
            successful=2,
            failed=1,
            results=[
                BatchItemResult(item_id="1", success=True),
                BatchItemResult(item_id="2", success=False, error="Error"),
                BatchItemResult(item_id="3", success=True),
            ],
        )

        failed = result.get_failed_items()
        assert len(failed) == 1
        assert failed[0].item_id == "2"

    def test_get_successful_items(self):
        """Test getting successful items."""
        result = BatchResult(
            total_items=3,
            successful=2,
            failed=1,
            results=[
                BatchItemResult(item_id="1", success=True),
                BatchItemResult(item_id="2", success=False, error="Error"),
                BatchItemResult(item_id="3", success=True),
            ],
        )

        successful = result.get_successful_items()
        assert len(successful) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BatchResult(
            total_items=5,
            successful=4,
            failed=1,
            total_time_ms=5000.0,
            total_tokens=1000,
            total_cost_usd=0.05,
        )

        data = result.to_dict()

        assert data["total_items"] == 5
        assert data["successful"] == 4
        assert data["success_rate"] == "80.0%"
        assert data["total_cost_usd"] == 0.05


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.fixture
    def sample_content(self):
        """Create sample extracted content."""
        page = PageContent(
            url="https://example.com/product",
            title="Test Product",
            content_blocks=[
                ContentBlock(
                    content_type=ContentType.PARAGRAPH,
                    text="Test content for processing.",
                ),
            ],
            word_count=10,
        )

        return ExtractedContent(
            product_name="Test Product",
            product_url="https://example.com",
            pages=[page],
            extraction_id="test123",
        )

    @pytest.fixture
    def mock_processed_content(self):
        """Create mock processed content."""
        return ProcessedContent(
            product_name="Test Product",
            product_url="https://example.com",
            summary=ContentSummary(
                executive_summary="Test summary",
                detailed_summary="Detailed test summary",
                comprehensive_summary="Comprehensive summary",
            ),
            total_llm_tokens_used=500,
            total_llm_cost_usd=0.01,
        )

    def test_batch_processor_initialization(self):
        """Test batch processor initializes correctly."""
        config = BatchConfig(max_concurrent_items=5)
        processor = BatchProcessor(config)

        assert processor.config.max_concurrent_items == 5
        assert processor._processor is None  # Not started yet

    @pytest.mark.asyncio
    async def test_batch_processor_context_manager(self):
        """Test batch processor as async context manager."""
        config = BatchConfig()

        with patch.object(BatchProcessor, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(BatchProcessor, 'stop', new_callable=AsyncMock) as mock_stop:
                async with BatchProcessor(config) as processor:
                    pass

                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_success(self, sample_content, mock_processed_content):
        """Test successful batch processing."""
        config = BatchConfig()
        processor = BatchProcessor(config)

        # Mock the internal processor
        mock_content_processor = MagicMock()
        mock_content_processor.process = AsyncMock(return_value=mock_processed_content)
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        items = [
            ("item1", sample_content),
            ("item2", sample_content),
        ]

        result = await processor.process_batch(items)

        assert result.total_items == 2
        assert result.successful == 2
        assert result.failed == 0
        assert result.total_tokens == 1000  # 500 * 2
        assert result.total_cost_usd == 0.02  # 0.01 * 2

    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self, sample_content, mock_processed_content):
        """Test batch processing with some failures."""
        config = BatchConfig(max_retries=0)
        processor = BatchProcessor(config)

        # Mock processor that fails on second call
        call_count = 0

        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Processing failed")
            return mock_processed_content

        mock_content_processor = MagicMock()
        mock_content_processor.process = mock_process
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        items = [
            ("item1", sample_content),
            ("item2", sample_content),
            ("item3", sample_content),
        ]

        result = await processor.process_batch(items)

        assert result.total_items == 3
        assert result.successful == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_process_with_retry(self, sample_content, mock_processed_content):
        """Test retry logic on failure."""
        config = BatchConfig(max_retries=2, retry_delay_seconds=0.01)
        processor = BatchProcessor(config)

        # Mock processor that fails twice then succeeds
        call_count = 0

        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return mock_processed_content

        mock_content_processor = MagicMock()
        mock_content_processor.process = mock_process
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        result = await processor._process_with_retry("item1", sample_content)

        assert result.success is True
        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_process_with_retry_exhausted(self, sample_content):
        """Test when retries are exhausted."""
        config = BatchConfig(max_retries=2, retry_delay_seconds=0.01)
        processor = BatchProcessor(config)

        # Mock processor that always fails
        mock_content_processor = MagicMock()
        mock_content_processor.process = AsyncMock(side_effect=Exception("Always fails"))
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        result = await processor._process_with_retry("item1", sample_content)

        assert result.success is False
        assert result.error == "Always fails"
        assert result.retries == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_progress_callback(self, sample_content, mock_processed_content):
        """Test progress callback is called."""
        progress_calls = []

        def progress_callback(completed, total, current_item):
            progress_calls.append((completed, total, current_item))

        config = BatchConfig()
        processor = BatchProcessor(config, progress_callback=progress_callback)

        mock_content_processor = MagicMock()
        mock_content_processor.process = AsyncMock(return_value=mock_processed_content)
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        items = [("item1", sample_content)]

        await processor.process_batch(items)

        # Should have progress calls
        assert len(progress_calls) >= 1

    @pytest.mark.asyncio
    async def test_product_name_override(self, sample_content, mock_processed_content):
        """Test product name override is passed."""
        config = BatchConfig()
        processor = BatchProcessor(config)

        mock_content_processor = MagicMock()
        mock_content_processor.process = AsyncMock(return_value=mock_processed_content)
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        items = [("item1", sample_content)]
        product_names = {"item1": "Custom Product Name"}

        await processor.process_batch(items, product_names=product_names)

        # Verify product name was passed
        mock_content_processor.process.assert_called_once()
        call_kwargs = mock_content_processor.process.call_args[1]
        assert call_kwargs.get("product_name") == "Custom Product Name"


class TestBatchProcessorStream:
    """Tests for streaming batch processing."""

    @pytest.fixture
    def sample_content(self):
        """Create sample content."""
        return ExtractedContent(
            product_url="https://example.com",
            pages=[],
        )

    @pytest.fixture
    def mock_processed_content(self):
        """Create mock result."""
        return ProcessedContent(
            product_name="Test",
            product_url="https://example.com",
            summary=ContentSummary(
                executive_summary="",
                detailed_summary="",
                comprehensive_summary="",
            ),
        )

    @pytest.mark.asyncio
    async def test_process_stream_yields_results(
        self, sample_content, mock_processed_content
    ):
        """Test streaming yields results as they complete."""
        config = BatchConfig()
        processor = BatchProcessor(config)

        mock_content_processor = MagicMock()
        mock_content_processor.process = AsyncMock(return_value=mock_processed_content)
        processor._processor = mock_content_processor
        processor._semaphore = MagicMock()
        processor._semaphore.__aenter__ = AsyncMock()
        processor._semaphore.__aexit__ = AsyncMock()

        items = [
            ("item1", sample_content),
            ("item2", sample_content),
        ]

        results = []
        async for result in processor.process_stream(items):
            results.append(result)

        assert len(results) == 2
        assert all(r.success for r in results)
