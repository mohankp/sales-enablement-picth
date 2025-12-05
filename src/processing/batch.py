"""Batch processing for handling multiple extractions efficiently."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from pydantic import BaseModel, Field

from src.models.content import ExtractedContent
from src.models.processed import ProcessedContent
from .processor import ContentProcessor, ProcessingConfig, ProcessingResult

logger = logging.getLogger(__name__)


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    # Concurrency
    max_concurrent_items: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum items to process concurrently",
    )

    # Error handling
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing remaining items on error",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries per item",
    )
    retry_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="Delay between retries",
    )

    # Progress
    enable_progress_callback: bool = True

    # Processing config to use for each item
    processing_config: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
    )


@dataclass
class BatchItemResult:
    """Result for a single item in a batch."""

    item_id: str
    success: bool
    result: Optional[ProcessedContent] = None
    error: Optional[str] = None
    retries: int = 0
    processing_time_ms: float = 0.0


@dataclass
class BatchResult:
    """Result of a batch processing operation."""

    total_items: int
    successful: int
    failed: int
    results: list[BatchItemResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful / self.total_items if self.total_items > 0 else 0.0

    def get_failed_items(self) -> list[BatchItemResult]:
        """Get list of failed items."""
        return [r for r in self.results if not r.success]

    def get_successful_items(self) -> list[BatchItemResult]:
        """Get list of successful items."""
        return [r for r in self.results if r.success]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_items": self.total_items,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": f"{self.success_rate * 100:.1f}%",
            "total_time_ms": self.total_time_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [
                {
                    "item_id": r.item_id,
                    "success": r.success,
                    "error": r.error,
                    "retries": r.retries,
                    "processing_time_ms": r.processing_time_ms,
                }
                for r in self.results
            ],
        }


# Type for progress callback
ProgressCallback = Callable[[int, int, Optional[str]], None]


class BatchProcessor:
    """
    Process multiple extractions efficiently in batch.

    Handles:
    - Concurrent processing with rate limiting
    - Error recovery and retries
    - Progress tracking
    - Aggregated results

    Usage:
        config = BatchConfig(max_concurrent_items=3)
        processor = BatchProcessor(config)

        results = await processor.process_batch(extractions)
        print(f"Processed {results.successful}/{results.total_items}")
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.config = config or BatchConfig()
        self.progress_callback = progress_callback
        self._processor: Optional[ContentProcessor] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self) -> "BatchProcessor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize the batch processor."""
        self._processor = ContentProcessor(self.config.processing_config)
        await self._processor.start()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_items)
        logger.info(
            f"BatchProcessor started with concurrency={self.config.max_concurrent_items}"
        )

    async def stop(self) -> None:
        """Shutdown the batch processor."""
        if self._processor:
            await self._processor.stop()
            self._processor = None
        logger.info("BatchProcessor stopped")

    def _report_progress(
        self,
        completed: int,
        total: int,
        current_item: Optional[str] = None,
    ) -> None:
        """Report progress via callback if configured."""
        if self.config.enable_progress_callback and self.progress_callback:
            self.progress_callback(completed, total, current_item)

    async def process_batch(
        self,
        items: list[tuple[str, ExtractedContent]],
        product_names: Optional[dict[str, str]] = None,
    ) -> BatchResult:
        """
        Process a batch of extracted content items.

        Args:
            items: List of (item_id, extracted_content) tuples
            product_names: Optional mapping of item_id to product name overrides

        Returns:
            BatchResult with all results and statistics
        """
        if not self._processor:
            raise RuntimeError("BatchProcessor not started. Use 'async with' or call start().")

        start_time = time.time()
        product_names = product_names or {}

        result = BatchResult(
            total_items=len(items),
            successful=0,
            failed=0,
        )

        logger.info(f"Starting batch processing of {len(items)} items")

        # Process items with concurrency control
        completed = 0

        async def process_item(item_id: str, content: ExtractedContent) -> BatchItemResult:
            nonlocal completed

            async with self._semaphore:
                self._report_progress(completed, len(items), item_id)

                item_result = await self._process_with_retry(
                    item_id,
                    content,
                    product_names.get(item_id),
                )

                completed += 1
                self._report_progress(completed, len(items), None)

                return item_result

        # Run all items concurrently (limited by semaphore)
        tasks = [
            process_item(item_id, content)
            for item_id, content in items
        ]
        item_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for i, item_result in enumerate(item_results):
            if isinstance(item_result, Exception):
                # Task raised an exception
                item_id = items[i][0]
                result.results.append(
                    BatchItemResult(
                        item_id=item_id,
                        success=False,
                        error=str(item_result),
                    )
                )
                result.failed += 1
            else:
                result.results.append(item_result)
                if item_result.success:
                    result.successful += 1
                    if item_result.result:
                        result.total_tokens += item_result.result.total_llm_tokens_used
                        result.total_cost_usd += item_result.result.total_llm_cost_usd
                else:
                    result.failed += 1

        result.total_time_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.now(timezone.utc)

        logger.info(
            f"Batch complete: {result.successful}/{result.total_items} successful "
            f"({result.total_time_ms:.0f}ms, ${result.total_cost_usd:.4f})"
        )

        return result

    async def _process_with_retry(
        self,
        item_id: str,
        content: ExtractedContent,
        product_name: Optional[str] = None,
    ) -> BatchItemResult:
        """Process a single item with retry logic."""
        start_time = time.time()
        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            try:
                result = await self._processor.process(content, product_name=product_name)

                return BatchItemResult(
                    item_id=item_id,
                    success=True,
                    result=result,
                    retries=retries,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            except Exception as e:
                last_error = str(e)
                retries += 1
                logger.warning(
                    f"Item {item_id} failed (attempt {retries}): {e}"
                )

                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        return BatchItemResult(
            item_id=item_id,
            success=False,
            error=last_error,
            retries=retries,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    async def process_stream(
        self,
        items: list[tuple[str, ExtractedContent]],
        product_names: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[BatchItemResult]:
        """
        Process items and yield results as they complete.

        Useful for real-time progress updates or early termination.

        Args:
            items: List of (item_id, extracted_content) tuples
            product_names: Optional mapping of item_id to product name overrides

        Yields:
            BatchItemResult for each completed item
        """
        if not self._processor:
            raise RuntimeError("BatchProcessor not started.")

        product_names = product_names or {}

        # Create task queue
        queue: asyncio.Queue[Optional[BatchItemResult]] = asyncio.Queue()

        async def process_item(item_id: str, content: ExtractedContent) -> None:
            async with self._semaphore:
                result = await self._process_with_retry(
                    item_id,
                    content,
                    product_names.get(item_id),
                )
                await queue.put(result)

        # Start all tasks
        tasks = [
            asyncio.create_task(process_item(item_id, content))
            for item_id, content in items
        ]

        # Yield results as they complete
        for _ in range(len(items)):
            result = await queue.get()
            if result:
                yield result

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def process_from_files(
        self,
        file_paths: list[Path],
    ) -> BatchResult:
        """
        Process extractions from JSON files.

        Args:
            file_paths: Paths to extraction JSON files

        Returns:
            BatchResult with all results
        """
        import json

        items: list[tuple[str, ExtractedContent]] = []

        for path in file_paths:
            try:
                with open(path) as f:
                    data = json.load(f)
                content = ExtractedContent.model_validate(data)
                item_id = path.stem  # Use filename as ID
                items.append((item_id, content))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        if not items:
            return BatchResult(
                total_items=0,
                successful=0,
                failed=0,
            )

        return await self.process_batch(items)


class BatchQueue:
    """
    Queue-based batch processor for continuous processing.

    Accepts items dynamically and processes them in batches.
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        batch_size: int = 10,
        flush_interval_seconds: float = 30.0,
    ):
        self.config = config or BatchConfig()
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: asyncio.Queue[tuple[str, ExtractedContent]] = asyncio.Queue()
        self._results: list[BatchItemResult] = []
        self._processor: Optional[BatchProcessor] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the batch queue processor."""
        self._processor = BatchProcessor(self.config)
        await self._processor.start()
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("BatchQueue started")

    async def stop(self) -> None:
        """Stop the batch queue processor and flush remaining items."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._processor:
            await self._processor.stop()
        logger.info("BatchQueue stopped")

    async def add(self, item_id: str, content: ExtractedContent) -> None:
        """Add an item to the processing queue."""
        await self._queue.put((item_id, content))

    def get_results(self) -> list[BatchItemResult]:
        """Get all completed results."""
        return self._results.copy()

    async def _process_loop(self) -> None:
        """Main processing loop."""
        batch: list[tuple[str, ExtractedContent]] = []
        last_flush = time.time()

        while self._running:
            try:
                # Wait for items with timeout
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size
                    or (batch and time.time() - last_flush >= self.flush_interval)
                )

                if should_flush and batch:
                    result = await self._processor.process_batch(batch)
                    self._results.extend(result.results)
                    batch = []
                    last_flush = time.time()

            except asyncio.CancelledError:
                # Flush remaining items on shutdown
                if batch:
                    result = await self._processor.process_batch(batch)
                    self._results.extend(result.results)
                raise
