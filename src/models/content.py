"""Content-related data models for extracted website content."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class ContentType(str, Enum):
    """Type of content extracted from a webpage."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    NAVIGATION = "navigation"
    FEATURE = "feature"
    BENEFIT = "benefit"
    TESTIMONIAL = "testimonial"
    PRICING = "pricing"
    CTA = "cta"  # Call to action
    UNKNOWN = "unknown"


class MediaType(str, Enum):
    """Type of media asset."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"


class MediaAsset(BaseModel):
    """Base class for media assets extracted from pages."""

    url: str
    alt_text: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    local_path: Optional[str] = None  # Path after download
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    source_selector: Optional[str] = None  # CSS selector for traceability


class ImageAsset(MediaAsset):
    """Image asset with additional image-specific metadata."""

    media_type: MediaType = MediaType.IMAGE
    is_logo: bool = False
    is_icon: bool = False
    is_screenshot: bool = False
    is_diagram: bool = False
    dominant_colors: list[str] = Field(default_factory=list)


class VideoAsset(MediaAsset):
    """Video asset with additional video-specific metadata."""

    media_type: MediaType = MediaType.VIDEO
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None
    embed_url: Optional[str] = None  # For YouTube, Vimeo, etc.
    platform: Optional[str] = None  # youtube, vimeo, wistia, etc.


class TableData(BaseModel):
    """Structured table data extracted from a webpage."""

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    source_selector: Optional[str] = None


class ContentBlock(BaseModel):
    """
    A block of content extracted from a webpage.

    This represents a semantic unit of content with its type,
    text content, and associated metadata.
    """

    content_type: ContentType
    text: str
    html: Optional[str] = None
    heading_level: Optional[int] = None  # For HEADING type (1-6)
    list_items: list[str] = Field(default_factory=list)  # For LIST type
    table_data: Optional[TableData] = None  # For TABLE type
    media_assets: list[MediaAsset] = Field(default_factory=list)
    links: list[dict[str, str]] = Field(default_factory=list)  # [{url, text}]
    source_selector: Optional[str] = None  # CSS selector
    source_xpath: Optional[str] = None  # XPath for precise location
    confidence_score: float = 1.0  # How confident we are in extraction
    parent_section: Optional[str] = None  # Parent heading/section
    order: int = 0  # Order on the page

    model_config = {"use_enum_values": True}


class PageContent(BaseModel):
    """
    Complete extracted content from a single page.

    Contains all content blocks, media assets, and metadata
    from a webpage after extraction.
    """

    url: str
    title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: list[str] = Field(default_factory=list)
    canonical_url: Optional[str] = None

    # Content
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    images: list[ImageAsset] = Field(default_factory=list)
    videos: list[VideoAsset] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)

    # Navigation & Structure
    navigation_links: list[dict[str, str]] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)
    external_links: list[str] = Field(default_factory=list)

    # Metadata
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_duration_ms: Optional[int] = None
    content_hash: Optional[str] = None  # For change detection
    raw_html_hash: Optional[str] = None
    word_count: int = 0
    is_spa: bool = False
    required_javascript: bool = False

    # Quality indicators
    extraction_warnings: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)

    def get_full_text(self) -> str:
        """Get all text content concatenated."""
        texts = []
        for block in self.content_blocks:
            if block.text:
                texts.append(block.text)
            if block.list_items:
                texts.extend(block.list_items)
        return "\n\n".join(texts)

    def get_headings(self) -> list[tuple[int, str]]:
        """Get all headings with their levels."""
        return [
            (block.heading_level or 1, block.text)
            for block in self.content_blocks
            if block.content_type == ContentType.HEADING
        ]


class SiteStructure(BaseModel):
    """
    Analyzed structure of a website.

    Used to plan extraction strategy based on site characteristics.
    """

    base_url: str
    is_spa: bool = False
    spa_framework: Optional[str] = None  # react, vue, angular, etc.
    has_dynamic_content: bool = False
    requires_javascript: bool = False

    # Navigation structure
    main_navigation_selector: Optional[str] = None
    tab_selectors: list[str] = Field(default_factory=list)
    accordion_selectors: list[str] = Field(default_factory=list)
    modal_triggers: list[str] = Field(default_factory=list)

    # Content areas
    main_content_selector: Optional[str] = None
    article_selector: Optional[str] = None
    sidebar_selectors: list[str] = Field(default_factory=list)
    footer_selector: Optional[str] = None

    # Pages to crawl
    discovered_pages: list[str] = Field(default_factory=list)
    documentation_pages: list[str] = Field(default_factory=list)
    feature_pages: list[str] = Field(default_factory=list)
    pricing_pages: list[str] = Field(default_factory=list)

    # Technical details
    uses_lazy_loading: bool = False
    uses_infinite_scroll: bool = False
    has_cookie_banner: bool = False
    has_auth_wall: bool = False
    requires_interaction: bool = False  # Content requires clicks to reveal

    # Analysis metadata
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_confidence: float = 0.0


class ExtractedContent(BaseModel):
    """
    Aggregated content from multiple pages of a product website.

    This is the complete extraction result that feeds into
    the content processing pipeline.
    """

    product_name: Optional[str] = None
    product_url: str
    pages: list[PageContent] = Field(default_factory=list)
    site_structure: Optional[SiteStructure] = None

    # Aggregated content
    all_images: list[ImageAsset] = Field(default_factory=list)
    all_videos: list[VideoAsset] = Field(default_factory=list)
    all_tables: list[TableData] = Field(default_factory=list)

    # Extraction metadata
    extraction_id: str = Field(default_factory=lambda: "")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_pages_extracted: int = 0
    total_word_count: int = 0
    content_hash: Optional[str] = None  # Combined hash for change detection

    # Quality metrics
    extraction_success_rate: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def compute_totals(self) -> None:
        """Compute aggregate statistics from pages."""
        self.total_pages_extracted = len(self.pages)
        self.total_word_count = sum(p.word_count for p in self.pages)
        self.all_images = [img for p in self.pages for img in p.images]
        self.all_videos = [vid for p in self.pages for vid in p.videos]
        self.all_tables = [tbl for p in self.pages for tbl in p.tables]
