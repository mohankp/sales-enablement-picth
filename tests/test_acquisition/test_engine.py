"""Tests for the content acquisition engine."""

import pytest

from src.models.config import BrowserConfig, ExtractionConfig, SiteConfig, SPAConfig
from src.models.content import ContentBlock, ContentType, PageContent


class TestSiteConfig:
    """Tests for SiteConfig."""

    def test_from_url(self):
        """Test creating config from URL."""
        config = SiteConfig.from_url("https://example.com")
        assert config.url == "https://example.com"
        assert config.browser is not None
        assert config.spa is not None
        assert config.extraction is not None

    def test_for_spa(self):
        """Test creating SPA-optimized config."""
        config = SiteConfig.for_spa(
            "https://example.com",
            content_selector=".main-content",
        )
        assert config.url == "https://example.com"
        assert config.spa.content_ready_selector == ".main-content"
        assert config.spa.wait_for_hydration is True


class TestBrowserConfig:
    """Tests for BrowserConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = BrowserConfig()
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.viewport_width == 1920
        assert config.stealth_mode is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BrowserConfig(
            browser_type="firefox",
            headless=False,
            viewport_width=1280,
        )
        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.viewport_width == 1280


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_defaults(self):
        """Test default extraction configuration."""
        config = ExtractionConfig()
        assert config.download_images is True
        assert config.max_pages == 50
        assert config.min_content_length == 100

    def test_exclude_patterns(self):
        """Test exclude patterns."""
        config = ExtractionConfig()
        assert r"/blog/" in config.exclude_patterns
        assert r"/careers/" in config.exclude_patterns


class TestContentBlock:
    """Tests for ContentBlock model."""

    def test_heading_block(self):
        """Test creating a heading block."""
        block = ContentBlock(
            content_type=ContentType.HEADING,
            text="Product Features",
            heading_level=2,
        )
        assert block.content_type == ContentType.HEADING
        assert block.text == "Product Features"
        assert block.heading_level == 2

    def test_list_block(self):
        """Test creating a list block."""
        block = ContentBlock(
            content_type=ContentType.LIST,
            text="Features",
            list_items=["Feature 1", "Feature 2", "Feature 3"],
        )
        assert block.content_type == ContentType.LIST
        assert len(block.list_items) == 3


class TestPageContent:
    """Tests for PageContent model."""

    def test_empty_page(self):
        """Test creating an empty page."""
        page = PageContent(url="https://example.com")
        assert page.url == "https://example.com"
        assert page.word_count == 0
        assert len(page.content_blocks) == 0

    def test_get_full_text(self):
        """Test getting full text from page."""
        page = PageContent(
            url="https://example.com",
            content_blocks=[
                ContentBlock(content_type=ContentType.HEADING, text="Title"),
                ContentBlock(content_type=ContentType.PARAGRAPH, text="First paragraph."),
                ContentBlock(content_type=ContentType.PARAGRAPH, text="Second paragraph."),
            ],
        )
        full_text = page.get_full_text()
        assert "Title" in full_text
        assert "First paragraph." in full_text
        assert "Second paragraph." in full_text

    def test_get_headings(self):
        """Test extracting headings from page."""
        page = PageContent(
            url="https://example.com",
            content_blocks=[
                ContentBlock(content_type=ContentType.HEADING, text="Main Title", heading_level=1),
                ContentBlock(content_type=ContentType.PARAGRAPH, text="Some text."),
                ContentBlock(content_type=ContentType.HEADING, text="Section", heading_level=2),
            ],
        )
        headings = page.get_headings()
        assert len(headings) == 2
        assert headings[0] == (1, "Main Title")
        assert headings[1] == (2, "Section")


class TestContentFingerprinting:
    """Tests for content fingerprinting."""

    def test_hash_text(self):
        """Test text hashing."""
        from src.acquisition.fingerprint import ContentFingerprinter

        hash1 = ContentFingerprinter.hash_text("Hello World")
        hash2 = ContentFingerprinter.hash_text("hello world")  # Same after normalization
        hash3 = ContentFingerprinter.hash_text("Different text")

        assert hash1 == hash2  # Normalization should make these equal
        assert hash1 != hash3

    def test_hash_content_block(self):
        """Test content block hashing."""
        from src.acquisition.fingerprint import ContentFingerprinter

        block1 = ContentBlock(content_type=ContentType.PARAGRAPH, text="Hello World")
        block2 = ContentBlock(content_type=ContentType.PARAGRAPH, text="Hello World")
        block3 = ContentBlock(content_type=ContentType.HEADING, text="Hello World")

        hash1 = ContentFingerprinter.hash_content_block(block1)
        hash2 = ContentFingerprinter.hash_content_block(block2)
        hash3 = ContentFingerprinter.hash_content_block(block3)

        assert hash1 == hash2  # Same content
        assert hash1 != hash3  # Different type

    def test_fingerprint_page(self):
        """Test page fingerprinting."""
        from src.acquisition.fingerprint import ContentFingerprinter

        page = PageContent(
            url="https://example.com",
            title="Test Page",
            content_blocks=[
                ContentBlock(content_type=ContentType.PARAGRAPH, text="Content here."),
            ],
        )

        hash1 = ContentFingerprinter.fingerprint_page(page)
        assert page.content_hash == hash1

        # Same page should produce same hash
        page2 = PageContent(
            url="https://example.com",
            title="Test Page",
            content_blocks=[
                ContentBlock(content_type=ContentType.PARAGRAPH, text="Content here."),
            ],
        )
        hash2 = ContentFingerprinter.fingerprint_page(page2)
        assert hash1 == hash2


# Integration tests (require network access)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_browser_manager():
    """Test browser manager startup and shutdown."""
    from src.acquisition.browser import BrowserManager

    config = BrowserConfig(headless=True)
    manager = BrowserManager(config)

    await manager.start()
    assert manager._browser is not None

    async with manager.get_page() as page:
        await page.goto("about:blank")
        assert page is not None

    await manager.stop()
    assert manager._browser is None
