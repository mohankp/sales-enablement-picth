"""Image content extractor for web pages."""

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp

from playwright.async_api import Page

from src.acquisition.extractors.base import BaseExtractor
from src.models.config import ExtractionConfig
from src.models.content import ImageAsset

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """
    Extracts and optionally downloads images from web pages.

    Features:
    - Filters out tiny images (icons, spacers)
    - Detects image types (logo, screenshot, diagram, etc.)
    - Downloads images with proper naming
    - Handles lazy-loaded images
    - Extracts alt text and captions
    """

    # Patterns to identify special image types
    LOGO_PATTERNS = [
        r"logo",
        r"brand",
        r"icon-logo",
    ]

    ICON_PATTERNS = [
        r"icon",
        r"favicon",
        r"glyph",
        r"sprite",
    ]

    SCREENSHOT_PATTERNS = [
        r"screenshot",
        r"screen-?shot",
        r"preview",
        r"demo",
        r"product-image",
    ]

    DIAGRAM_PATTERNS = [
        r"diagram",
        r"chart",
        r"graph",
        r"flow",
        r"architecture",
        r"workflow",
    ]

    # Image formats to extract
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif"}

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        download_dir: Optional[str] = None,
    ):
        """Initialize the image extractor.

        Args:
            config: Extraction configuration.
            download_dir: Directory to download images to.
        """
        super().__init__(config)
        self.download_dir = download_dir or "data/media/images"
        self._session: Optional[aiohttp.ClientSession] = None

    async def extract(self, page: Page) -> list[ImageAsset]:
        """Extract all images from the page.

        Args:
            page: The Playwright page to extract from.

        Returns:
            List of ImageAsset objects.
        """
        logger.debug("Starting image extraction")

        # Get base URL for resolving relative paths
        base_url = await page.evaluate("window.location.href")

        # Extract all image data using JavaScript
        raw_images = await self._extract_raw_images(page, base_url)

        # Filter and process images
        images = []
        for raw in raw_images:
            image = self._process_raw_image(raw, base_url)
            if image and self._should_include_image(image):
                images.append(image)

        logger.debug(f"Extracted {len(images)} images (filtered from {len(raw_images)})")

        # Download images if configured
        if self.config.download_images and images:
            await self._download_images(images)

        return images

    async def _extract_raw_images(
        self,
        page: Page,
        base_url: str,
    ) -> list[dict]:
        """Extract raw image data using JavaScript.

        Args:
            page: The Playwright page.
            base_url: Base URL for resolving relative paths.

        Returns:
            List of raw image dictionaries.
        """
        extraction_script = """
            () => {
                const images = [];
                const seen = new Set();

                // Helper to get absolute URL
                function getAbsoluteUrl(url) {
                    if (!url) return null;
                    try {
                        return new URL(url, window.location.href).href;
                    } catch (e) {
                        return null;
                    }
                }

                // Helper to get image dimensions
                function getDimensions(img) {
                    return {
                        width: img.naturalWidth || img.width || 0,
                        height: img.naturalHeight || img.height || 0,
                    };
                }

                // Helper to get selector
                function getSelector(el) {
                    if (el.id) return '#' + el.id;
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.split(' ').filter(c => c).slice(0, 2);
                        if (classes.length) return el.tagName.toLowerCase() + '.' + classes.join('.');
                    }
                    return el.tagName.toLowerCase();
                }

                // Extract from <img> elements
                document.querySelectorAll('img').forEach(img => {
                    // Try various sources for the URL
                    const src = img.src ||
                                img.dataset.src ||
                                img.dataset.lazySrc ||
                                img.dataset.original ||
                                img.getAttribute('data-src');

                    const url = getAbsoluteUrl(src);
                    if (!url || seen.has(url)) return;
                    seen.add(url);

                    const dims = getDimensions(img);

                    images.push({
                        url: url,
                        altText: img.alt || '',
                        title: img.title || '',
                        width: dims.width,
                        height: dims.height,
                        selector: getSelector(img),
                        className: img.className || '',
                        parentClass: img.parentElement?.className || '',
                    });
                });

                // Extract from <picture> elements
                document.querySelectorAll('picture source').forEach(source => {
                    const srcset = source.srcset;
                    if (!srcset) return;

                    // Get the largest image from srcset
                    const urls = srcset.split(',').map(s => s.trim().split(' ')[0]);
                    const url = getAbsoluteUrl(urls[urls.length - 1]);

                    if (!url || seen.has(url)) return;
                    seen.add(url);

                    const img = source.parentElement?.querySelector('img');
                    images.push({
                        url: url,
                        altText: img?.alt || '',
                        title: img?.title || '',
                        width: 0,
                        height: 0,
                        selector: getSelector(source),
                        className: source.className || '',
                        parentClass: source.parentElement?.className || '',
                    });
                });

                // Extract from CSS background images (important elements only)
                document.querySelectorAll('[style*="background"], [class*="hero"], [class*="banner"]').forEach(el => {
                    const style = window.getComputedStyle(el);
                    const bg = style.backgroundImage;

                    if (!bg || bg === 'none') return;

                    const match = bg.match(/url\\(["']?(.+?)["']?\\)/);
                    if (!match) return;

                    const url = getAbsoluteUrl(match[1]);
                    if (!url || seen.has(url)) return;
                    seen.add(url);

                    images.push({
                        url: url,
                        altText: '',
                        title: '',
                        width: el.offsetWidth || 0,
                        height: el.offsetHeight || 0,
                        selector: getSelector(el),
                        className: el.className || '',
                        parentClass: el.parentElement?.className || '',
                        isBackground: true,
                    });
                });

                // Look for figure captions
                document.querySelectorAll('figure').forEach(figure => {
                    const img = figure.querySelector('img');
                    if (!img || !img.src) return;

                    const caption = figure.querySelector('figcaption');
                    if (!caption) return;

                    // Update existing image entry with caption
                    const imgUrl = getAbsoluteUrl(img.src);
                    for (const imageData of images) {
                        if (imageData.url === imgUrl) {
                            imageData.caption = caption.textContent?.trim() || '';
                            break;
                        }
                    }
                });

                return images;
            }
        """

        try:
            return await page.evaluate(extraction_script)
        except Exception as e:
            logger.error(f"JavaScript image extraction failed: {e}")
            return []

    def _process_raw_image(
        self,
        raw: dict,
        base_url: str,
    ) -> Optional[ImageAsset]:
        """Process raw image data into an ImageAsset.

        Args:
            raw: Raw image dictionary from JavaScript.
            base_url: Base URL for the page.

        Returns:
            ImageAsset or None if invalid.
        """
        url = raw.get("url", "")
        if not url:
            return None

        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                url = urljoin(base_url, url)
        except Exception:
            return None

        # Check file extension
        path = urlparse(url).path.lower()
        ext = os.path.splitext(path)[1]
        if ext and ext not in self.VALID_EXTENSIONS:
            # Allow URLs without extensions (common for CDNs)
            if ext and not ext.startswith("."):
                return None

        # Determine image type
        class_info = (raw.get("className", "") + " " + raw.get("parentClass", "")).lower()
        url_lower = url.lower()

        is_logo = any(re.search(p, class_info + url_lower) for p in self.LOGO_PATTERNS)
        is_icon = any(re.search(p, class_info + url_lower) for p in self.ICON_PATTERNS)
        is_screenshot = any(re.search(p, class_info + url_lower) for p in self.SCREENSHOT_PATTERNS)
        is_diagram = any(re.search(p, class_info + url_lower) for p in self.DIAGRAM_PATTERNS)

        return ImageAsset(
            url=url,
            alt_text=raw.get("altText", ""),
            title=raw.get("title", ""),
            caption=raw.get("caption"),
            width=raw.get("width", 0),
            height=raw.get("height", 0),
            source_selector=raw.get("selector"),
            is_logo=is_logo,
            is_icon=is_icon,
            is_screenshot=is_screenshot,
            is_diagram=is_diagram,
        )

    def _should_include_image(self, image: ImageAsset) -> bool:
        """Determine if an image should be included.

        Args:
            image: The ImageAsset to check.

        Returns:
            True if the image should be included.
        """
        # Skip data URIs (too large to process efficiently)
        if image.url.startswith("data:"):
            return False

        # Skip known tracking pixels and analytics
        skip_domains = [
            "google-analytics",
            "facebook.com/tr",
            "pixel",
            "beacon",
            "tracking",
        ]
        if any(domain in image.url.lower() for domain in skip_domains):
            return False

        # Size filtering (but always include logos and specific types)
        if image.is_logo or image.is_diagram or image.is_screenshot:
            return True

        # Skip icons if they're too small
        if image.is_icon and image.width < 100 and image.height < 100:
            return False

        # Skip very small images (spacers, dots, etc.)
        if image.width > 0 and image.width < self.config.min_image_width:
            return False
        if image.height > 0 and image.height < self.config.min_image_height:
            return False

        return True

    async def _download_images(self, images: list[ImageAsset]) -> None:
        """Download images to local storage.

        Args:
            images: List of ImageAsset objects to download.
        """
        # Create download directory
        download_path = Path(self.download_dir)
        download_path.mkdir(parents=True, exist_ok=True)

        # Create session if needed
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Download in parallel with limit
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent downloads

        async def download_one(image: ImageAsset) -> None:
            async with semaphore:
                await self._download_single_image(image, download_path)

        tasks = [download_one(img) for img in images]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_single_image(
        self,
        image: ImageAsset,
        download_path: Path,
    ) -> None:
        """Download a single image.

        Args:
            image: ImageAsset to download.
            download_path: Directory to save to.
        """
        try:
            # Generate filename from URL hash
            url_hash = hashlib.md5(image.url.encode()).hexdigest()[:12]
            parsed_url = urlparse(image.url)
            ext = os.path.splitext(parsed_url.path)[1] or ".jpg"
            filename = f"{url_hash}{ext}"
            filepath = download_path / filename

            # Skip if already downloaded
            if filepath.exists():
                image.local_path = str(filepath)
                return

            # Download
            async with self._session.get(
                image.url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    logger.debug(f"Failed to download {image.url}: HTTP {response.status}")
                    return

                # Check file size
                content_length = response.headers.get("Content-Length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > self.config.max_image_size_mb:
                        logger.debug(f"Skipping large image: {image.url} ({size_mb:.1f}MB)")
                        return

                # Get content type
                content_type = response.headers.get("Content-Type", "")
                if content_type:
                    image.mime_type = content_type

                # Read and save
                content = await response.read()
                image.file_size = len(content)

                with open(filepath, "wb") as f:
                    f.write(content)

                image.local_path = str(filepath)
                logger.debug(f"Downloaded: {filename}")

        except asyncio.TimeoutError:
            logger.debug(f"Timeout downloading {image.url}")
        except Exception as e:
            logger.debug(f"Error downloading {image.url}: {e}")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
