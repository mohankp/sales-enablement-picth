"""Data models for the sales pitch generator."""

from src.models.content import (
    ContentBlock,
    ContentType,
    ExtractedContent,
    ImageAsset,
    MediaAsset,
    MediaType,
    PageContent,
    SiteStructure,
    TableData,
    VideoAsset,
)
from src.models.config import (
    BrowserConfig,
    ExtractionConfig,
    SiteConfig,
    SPAConfig,
    WaitStrategy,
)

__all__ = [
    # Content models
    "ContentBlock",
    "ContentType",
    "ExtractedContent",
    "ImageAsset",
    "MediaAsset",
    "MediaType",
    "PageContent",
    "SiteStructure",
    "TableData",
    "VideoAsset",
    # Config models
    "BrowserConfig",
    "ExtractionConfig",
    "SiteConfig",
    "SPAConfig",
    "WaitStrategy",
]
