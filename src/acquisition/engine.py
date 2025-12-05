"""Main content acquisition engine orchestrating all extraction components."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import Page

from src.acquisition.browser import BrowserManager, PageInteractionHelper
from src.acquisition.extractors.images import ImageExtractor
from src.acquisition.extractors.tables import TableExtractor
from src.acquisition.extractors.text import TextExtractor
from src.acquisition.fingerprint import ContentFingerprinter, ExtractionStore
from src.acquisition.site_analyzer import SiteAnalyzer
from src.acquisition.spa_handler import SPAHandler
from src.models.config import SiteConfig
from src.models.content import ExtractedContent, PageContent, SiteStructure

logger = logging.getLogger(__name__)


class ContentAcquisitionEngine:
    """
    Main orchestrator for content extraction from product websites.

    This engine coordinates:
    1. Browser management (Playwright)
    2. Site analysis (structure detection)
    3. SPA handling (dynamic content)
    4. Content extraction (text, images, tables)
    5. Change detection (fingerprinting)

    Usage:
        config = SiteConfig.from_url("https://example.com/product")

        async with ContentAcquisitionEngine(config) as engine:
            content = await engine.extract()

        # Or for incremental updates:
        async with ContentAcquisitionEngine(config) as engine:
            content, changes = await engine.extract_with_diff(previous_extraction)
    """

    def __init__(
        self,
        config: SiteConfig,
        storage_path: Optional[str] = None,
    ):
        """Initialize the content acquisition engine.

        Args:
            config: Site configuration.
            storage_path: Optional path for storing extractions.
        """
        self.config = config
        self.storage = ExtractionStore(storage_path or "data/extractions")

        # Initialize components
        self.browser_manager = BrowserManager(config.browser)
        self.site_analyzer = SiteAnalyzer(config)
        self.spa_handler = SPAHandler(config.spa)
        self.text_extractor = TextExtractor(config.extraction)
        self.image_extractor = ImageExtractor(config.extraction)
        self.table_extractor = TableExtractor(config.extraction)

        # State
        self._site_structure: Optional[SiteStructure] = None
        self._extraction_result: Optional[ExtractedContent] = None

    async def __aenter__(self) -> "ContentAcquisitionEngine":
        """Async context manager entry."""
        await self.browser_manager.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.image_extractor.close()
        await self.browser_manager.stop()

    async def extract(
        self,
        product_id: Optional[str] = None,
        save: bool = True,
    ) -> ExtractedContent:
        """Extract all content from the configured site.

        This is the main extraction method that orchestrates the entire process.

        Args:
            product_id: Optional product identifier for storage.
            save: Whether to save the extraction to storage.

        Returns:
            ExtractedContent with all extracted data.
        """
        start_time = time.time()
        logger.info(f"Starting content extraction from: {self.config.url}")

        # Initialize result
        result = ExtractedContent(
            product_url=self.config.url,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Phase 1: Analyze site structure
            logger.info("Phase 1: Analyzing site structure...")
            async with self.browser_manager.get_page() as page:
                await self._navigate_with_handling(page, self.config.url)
                self._site_structure = await self.site_analyzer.analyze(
                    page, self.config.url
                )
                result.site_structure = self._site_structure

            # Phase 2: Get extraction recommendations
            recommendations = await self.site_analyzer.get_extraction_recommendations(
                self._site_structure
            )
            logger.info(
                f"Found {len(recommendations['priority_pages'])} priority pages to extract"
            )

            # Phase 3: Extract each page
            logger.info("Phase 3: Extracting page content...")
            pages_to_extract = [self.config.url] + recommendations["priority_pages"]
            pages_to_extract = list(dict.fromkeys(pages_to_extract))  # Remove duplicates

            # Limit pages based on config
            max_pages = self.config.extraction.max_pages
            if len(pages_to_extract) > max_pages:
                logger.info(f"Limiting extraction to {max_pages} pages")
                pages_to_extract = pages_to_extract[:max_pages]

            # Extract pages with rate limiting
            for i, url in enumerate(pages_to_extract):
                logger.info(f"Extracting page {i+1}/{len(pages_to_extract)}: {url}")

                try:
                    page_content = await self._extract_single_page(url)
                    if page_content:
                        result.pages.append(page_content)
                except Exception as e:
                    logger.error(f"Failed to extract {url}: {e}")
                    result.errors.append(f"{url}: {str(e)}")

                # Rate limiting
                if i < len(pages_to_extract) - 1:
                    delay = self.config.delay_between_pages_ms / 1000
                    await asyncio.sleep(delay)

            # Phase 4: Finalize
            logger.info("Phase 4: Finalizing extraction...")
            result.completed_at = datetime.now(timezone.utc)
            result.compute_totals()

            # Calculate success rate
            total_attempted = len(pages_to_extract)
            successful = len(result.pages)
            result.extraction_success_rate = successful / total_attempted if total_attempted > 0 else 0

            # Fingerprint the extraction
            ContentFingerprinter.fingerprint_extraction(result)

            # Try to extract product name
            if result.pages:
                main_page = result.pages[0]
                if main_page.title:
                    # Clean up title (remove common suffixes)
                    title = main_page.title
                    for suffix in [" | ", " - ", " – "]:
                        if suffix in title:
                            title = title.split(suffix)[0]
                    result.product_name = title.strip()

            # Save if requested
            if save and product_id:
                self.storage.save_extraction(result, product_id)

            duration = time.time() - start_time
            logger.info(
                f"Extraction complete: {successful}/{total_attempted} pages, "
                f"{result.total_word_count} words, {len(result.all_images)} images, "
                f"{duration:.1f}s"
            )

            self._extraction_result = result
            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            result.errors.append(str(e))
            result.completed_at = datetime.now(timezone.utc)
            raise

    async def extract_with_diff(
        self,
        previous: Optional[ExtractedContent] = None,
        product_id: Optional[str] = None,
    ) -> tuple[ExtractedContent, dict]:
        """Extract content and compare with previous extraction.

        Args:
            previous: Previous extraction to compare against.
            product_id: Product ID to load previous extraction if not provided.

        Returns:
            Tuple of (new extraction, diff results).
        """
        # Load previous if not provided
        if previous is None and product_id:
            previous = self.storage.get_latest_extraction(product_id)

        # Do the extraction
        current = await self.extract(product_id=product_id)

        # Compare if we have a previous extraction
        if previous:
            diff = ContentFingerprinter.compare_extractions(previous, current)
            logger.info(
                f"Change detection: {len(diff['added'])} added, "
                f"{len(diff['removed'])} removed, {len(diff['modified'])} modified"
            )
        else:
            diff = {
                "added": [p.url for p in current.pages],
                "removed": [],
                "modified": [],
                "unchanged": [],
                "has_changes": True,
            }

        return current, diff

    async def _extract_single_page(self, url: str) -> Optional[PageContent]:
        """Extract content from a single page.

        Args:
            url: URL to extract.

        Returns:
            PageContent or None if extraction failed.
        """
        start_time = time.time()

        async with self.browser_manager.get_page() as page:
            try:
                # Navigate with proper handling
                await self._navigate_with_handling(page, url)

                # Handle SPA content
                spa_result = await self.spa_handler.prepare_page_for_extraction(page)

                # Create page content object
                page_content = PageContent(
                    url=url,
                    is_spa=spa_result.get("framework") is not None,
                    required_javascript=spa_result.get("content_ready", False),
                )

                # Extract metadata
                metadata = await self.text_extractor.extract_metadata(page)
                page_content.title = metadata.get("title")
                page_content.meta_description = metadata.get("description")
                if metadata.get("keywords"):
                    page_content.meta_keywords = [
                        k.strip() for k in metadata["keywords"].split(",")
                    ]
                page_content.canonical_url = metadata.get("canonical")

                # Extract text content
                content_blocks = await self.text_extractor.extract(page)
                page_content.content_blocks = content_blocks

                # Calculate word count
                page_content.word_count = sum(
                    len(block.text.split()) for block in content_blocks
                )

                # Extract structured data
                structured_data = await self.text_extractor.extract_structured_data(page)
                if structured_data:
                    logger.debug(f"Found {len(structured_data)} structured data items")

                # Extract links
                links = await self.text_extractor.extract_links(page)
                page_content.internal_links = [l["url"] for l in links.get("internal", [])]
                page_content.external_links = [l["url"] for l in links.get("external", [])]

                # Extract images
                images = await self.image_extractor.extract(page)
                page_content.images = images

                # Extract tables
                tables = await self.table_extractor.extract(page)
                page_content.tables = tables

                # Link downloaded images to content blocks
                self._link_images_to_blocks(page_content, images)

                # Fingerprint the page
                ContentFingerprinter.fingerprint_page(page_content)

                # Record extraction duration
                duration_ms = int((time.time() - start_time) * 1000)
                page_content.extraction_duration_ms = duration_ms

                logger.debug(
                    f"Extracted: {url} - {page_content.word_count} words, "
                    f"{len(images)} images, {len(tables)} tables"
                )

                return page_content

            except Exception as e:
                logger.error(f"Error extracting {url}: {e}")
                return None

    async def _navigate_with_handling(self, page: Page, url: str) -> None:
        """Navigate to a URL with proper error handling and preparation.

        Args:
            page: Playwright page.
            url: URL to navigate to.
        """
        helper = PageInteractionHelper(page)

        # Navigate
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning(f"Navigation warning for {url}: {e}")
            # Try with shorter timeout
            await page.goto(url, wait_until="commit", timeout=30000)

        # Wait for initial load
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            # Network might not go idle on some sites
            pass

        # Dismiss cookie banners
        if self.config.extraction.dismiss_cookie_banners:
            await helper.dismiss_overlays(
                self.config.extraction.cookie_banner_selectors,
                self.config.extraction.cookie_dismiss_text,
            )

        # Execute any custom scripts
        for script in self.config.post_load_scripts:
            try:
                await page.evaluate(script)
            except Exception as e:
                logger.warning(f"Custom script failed: {e}")

    def _link_images_to_blocks(
        self,
        page_content: PageContent,
        images: list,
    ) -> None:
        """Link downloaded images to content blocks based on URL matching.

        This updates the media_assets in content blocks to include local_path
        from downloaded images, enabling the pitch generation to use local files.

        Args:
            page_content: The page content with content blocks to update.
            images: List of ImageAsset objects with local_path set.
        """
        # Build URL -> ImageAsset map for quick lookup
        image_map: dict[str, any] = {}
        for img in images:
            if img.url:
                # Normalize URL for matching (remove trailing slash, etc.)
                normalized_url = img.url.rstrip("/")
                image_map[normalized_url] = img
                image_map[img.url] = img

        # Update media_assets in content blocks
        for block in page_content.content_blocks:
            if not block.media_assets:
                continue

            for i, media in enumerate(block.media_assets):
                url = media.url
                if not url:
                    continue

                # Try to find matching downloaded image
                matched_image = image_map.get(url) or image_map.get(url.rstrip("/"))

                if matched_image and matched_image.local_path:
                    # Update the media asset with additional info from downloaded image
                    block.media_assets[i].local_path = matched_image.local_path
                    if matched_image.width:
                        block.media_assets[i].width = matched_image.width
                    if matched_image.height:
                        block.media_assets[i].height = matched_image.height
                    if matched_image.file_size:
                        block.media_assets[i].file_size = matched_image.file_size
                    if matched_image.mime_type:
                        block.media_assets[i].mime_type = matched_image.mime_type

        logger.debug(
            f"Linked images to content blocks for {page_content.url}"
        )

    async def quick_check(self, url: Optional[str] = None) -> dict:
        """Quick check to see if site content has changed.

        This is a lightweight check that doesn't do full extraction.
        Useful for scheduled monitoring.

        Args:
            url: URL to check (uses config URL if not provided).

        Returns:
            Dict with check results.
        """
        url = url or self.config.url

        async with self.browser_manager.get_page() as page:
            await self._navigate_with_handling(page, url)

            # Get page hash quickly
            page_hash = await page.evaluate("""
                () => {
                    const content = document.body.innerText || '';
                    let hash = 0;
                    for (let i = 0; i < content.length; i++) {
                        const char = content.charCodeAt(i);
                        hash = ((hash << 5) - hash) + char;
                        hash = hash & hash;
                    }
                    return hash.toString(16);
                }
            """)

            # Get last modified header if available
            last_modified = await page.evaluate("""
                () => {
                    // Check for meta tags indicating update time
                    const meta = document.querySelector(
                        'meta[property="article:modified_time"], ' +
                        'meta[name="last-modified"]'
                    );
                    return meta?.content || null;
                }
            """)

            return {
                "url": url,
                "content_hash": page_hash,
                "last_modified": last_modified,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }


class ExtractionProgress:
    """Helper class to track and report extraction progress."""

    def __init__(self, total_pages: int):
        """Initialize progress tracker.

        Args:
            total_pages: Total number of pages to extract.
        """
        self.total_pages = total_pages
        self.completed_pages = 0
        self.failed_pages = 0
        self.start_time = time.time()
        self.page_times: list[float] = []

    def page_completed(self, duration: float, success: bool = True) -> None:
        """Record a completed page.

        Args:
            duration: Time taken for the page.
            success: Whether extraction was successful.
        """
        if success:
            self.completed_pages += 1
        else:
            self.failed_pages += 1
        self.page_times.append(duration)

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        total_done = self.completed_pages + self.failed_pages
        return (total_done / self.total_pages * 100) if self.total_pages > 0 else 0

    @property
    def avg_page_time(self) -> float:
        """Get average time per page."""
        return sum(self.page_times) / len(self.page_times) if self.page_times else 0

    @property
    def estimated_remaining(self) -> float:
        """Get estimated remaining time in seconds."""
        remaining_pages = self.total_pages - self.completed_pages - self.failed_pages
        return self.avg_page_time * remaining_pages

    def get_status(self) -> dict:
        """Get current status as dict."""
        elapsed = time.time() - self.start_time
        return {
            "total_pages": self.total_pages,
            "completed": self.completed_pages,
            "failed": self.failed_pages,
            "progress_percent": round(self.progress_percent, 1),
            "elapsed_seconds": round(elapsed, 1),
            "avg_page_seconds": round(self.avg_page_time, 2),
            "estimated_remaining_seconds": round(self.estimated_remaining, 1),
        }
