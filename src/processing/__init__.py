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
"""

from .processor import ContentProcessor, ProcessingConfig, ProcessingResult
from .chunker import ContentChunker, ChunkingStrategy

__all__ = [
    "ContentProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ContentChunker",
    "ChunkingStrategy",
]
