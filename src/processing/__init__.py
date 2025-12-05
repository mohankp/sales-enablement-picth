"""
Content Processing Module for Sales Enablement Pitch Generator.

This module provides LLM-powered analysis of extracted website content:
- Feature extraction and categorization
- Benefit identification and mapping
- Use case analysis
- Competitive differentiation
- Audience segmentation
- Pricing information parsing
- Technical specifications extraction

Usage:
    from src.processing import ContentProcessor, ProcessingConfig

    config = ProcessingConfig()
    async with ContentProcessor(config) as processor:
        result = await processor.process(extracted_content)
        print(f"Found {len(result.features.features)} features")

Caching:
    # Results are automatically cached based on content hash
    # Cache can be configured via ProcessingConfig:
    config = ProcessingConfig(
        enable_cache=True,
        cache_dir="./cache",
        cache_ttl_seconds=86400,
    )

    # Get cache stats
    stats = processor.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']}")
"""

from .processor import ContentProcessor, ProcessingConfig, ProcessingResult
from .chunker import ContentChunker, ChunkingStrategy
from .cache import ProcessingCache, CacheStats, CacheEntry
from .batch import (
    BatchProcessor,
    BatchConfig,
    BatchResult,
    BatchItemResult,
    BatchQueue,
)

__all__ = [
    # Core processor
    "ContentProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    # Chunking
    "ContentChunker",
    "ChunkingStrategy",
    # Caching
    "ProcessingCache",
    "CacheStats",
    "CacheEntry",
    # Batch processing
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "BatchItemResult",
    "BatchQueue",
]
