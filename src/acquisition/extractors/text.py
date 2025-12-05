"""Text content extractor for web pages."""

import logging
import re
from typing import Optional

from playwright.async_api import Page

from src.acquisition.extractors.base import BaseExtractor
from src.models.config import ExtractionConfig
from src.models.content import ContentBlock, ContentType, MediaAsset

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """
    Extracts text content from web pages with semantic understanding.

    This extractor identifies different types of text content:
    - Headings (h1-h6)
    - Paragraphs
    - Lists (ordered and unordered)
    - Code blocks
    - Blockquotes
    - And more

    It preserves the hierarchical structure of the content
    for better understanding by the LLM.
    """

    # Selectors to exclude from extraction
    EXCLUDE_SELECTORS = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "nav",
        "header:not(article header)",
        "footer:not(article footer)",
        "[hidden]",
        "[aria-hidden='true']",
        ".sr-only",
        ".visually-hidden",
        "[class*='cookie']",
        "[class*='banner']",
        "[class*='popup']",
        "[class*='modal']",
        "[class*='overlay']",
        "[class*='advertisement']",
        "[class*='sidebar']",
        "[class*='widget']",
        "[class*='social-share']",
        "[class*='newsletter']",
        "[class*='comments']",
    ]

    # Content type detection patterns
    CONTENT_PATTERNS = {
        ContentType.FEATURE: [
            r"feature",
            r"capability",
            r"what you.+get",
            r"includes",
        ],
        ContentType.BENEFIT: [
            r"benefit",
            r"advantage",
            r"why choose",
            r"you.+will",
        ],
        ContentType.TESTIMONIAL: [
            r"testimonial",
            r"what.+customers.+say",
            r"review",
            r"case study",
        ],
        ContentType.PRICING: [
            r"pricing",
            r"plans?",
            r"packages?",
            r"\$\d+",
            r"per month",
            r"annually",
        ],
        ContentType.CTA: [
            r"get started",
            r"sign up",
            r"try.+free",
            r"request.+demo",
            r"contact.+us",
            r"learn more",
        ],
    }

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize the text extractor.

        Args:
            config: Extraction configuration.
        """
        super().__init__(config)
        self._current_parent_heading: Optional[str] = None

    async def extract(self, page: Page) -> list[ContentBlock]:
        """Extract all text content blocks from the page.

        Args:
            page: The Playwright page to extract from.

        Returns:
            List of ContentBlock objects with extracted text.
        """
        logger.debug("Starting text extraction")

        # Build exclusion selector
        exclude_selector = ", ".join(self.EXCLUDE_SELECTORS)

        # Extract content using JavaScript for better performance
        raw_content = await self._extract_raw_content(page, exclude_selector)

        # Process raw content into ContentBlocks
        content_blocks = self._process_raw_content(raw_content)

        # Add semantic classification
        content_blocks = self._classify_content_blocks(content_blocks)

        logger.debug(f"Extracted {len(content_blocks)} content blocks")
        return content_blocks

    async def _extract_raw_content(
        self,
        page: Page,
        exclude_selector: str,
    ) -> list[dict]:
        """Extract raw content using JavaScript.

        Args:
            page: The Playwright page.
            exclude_selector: CSS selector for elements to exclude.

        Returns:
            List of raw content dictionaries.
        """
        extraction_script = f"""
            () => {{
                const excludeSelector = `{exclude_selector}`;
                const results = [];
                let order = 0;

                // Helper to check if element should be excluded
                function shouldExclude(el) {{
                    if (!el || !el.parentElement) return false;
                    return el.closest(excludeSelector) !== null;
                }}

                // Helper to get clean text
                function getCleanText(el) {{
                    const clone = el.cloneNode(true);
                    // Remove excluded elements from clone
                    clone.querySelectorAll(excludeSelector).forEach(e => e.remove());
                    return (clone.textContent || '').trim();
                }}

                // Helper to get CSS selector path
                function getSelector(el) {{
                    if (el.id) return '#' + el.id;
                    if (el.className && typeof el.className === 'string') {{
                        const classes = el.className.split(' ').filter(c => c).slice(0, 2);
                        if (classes.length) return el.tagName.toLowerCase() + '.' + classes.join('.');
                    }}
                    return el.tagName.toLowerCase();
                }}

                // Helper to extract images from an element
                function extractImages(el) {{
                    const images = [];
                    el.querySelectorAll('img').forEach(img => {{
                        const src = img.src || img.dataset?.src || img.dataset?.lazySrc;
                        if (!src || src.startsWith('data:')) return;
                        images.push({{
                            url: src,
                            alt: img.alt || '',
                            title: img.title || '',
                            width: img.naturalWidth || img.width || null,
                            height: img.naturalHeight || img.height || null,
                            selector: getSelector(img)
                        }});
                    }});
                    return images;
                }}

                // Helper to extract links from an element
                function extractLinks(el) {{
                    const links = [];
                    el.querySelectorAll('a[href]').forEach(a => {{
                        const href = a.href;
                        if (!href || href.startsWith('javascript:') || href.startsWith('#')) return;
                        links.push({{
                            url: href,
                            text: a.textContent?.trim() || ''
                        }});
                    }});
                    return links;
                }}

                // Helper to find nearby/associated images for a content block
                function findNearbyImages(el) {{
                    const images = [];

                    // Check within the element itself
                    images.push(...extractImages(el));

                    // Check parent for images (e.g., figure > figcaption pattern)
                    if (el.parentElement) {{
                        const parent = el.parentElement;
                        if (parent.tagName === 'FIGURE' || parent.classList?.contains('card') ||
                            parent.classList?.contains('feature') || parent.classList?.contains('item')) {{
                            images.push(...extractImages(parent));
                        }}
                    }}

                    // Check previous sibling for images (common pattern: image then text)
                    const prevSibling = el.previousElementSibling;
                    if (prevSibling) {{
                        if (prevSibling.tagName === 'IMG') {{
                            const src = prevSibling.src || prevSibling.dataset?.src;
                            if (src && !src.startsWith('data:')) {{
                                images.push({{
                                    url: src,
                                    alt: prevSibling.alt || '',
                                    title: prevSibling.title || '',
                                    width: prevSibling.naturalWidth || prevSibling.width || null,
                                    height: prevSibling.naturalHeight || prevSibling.height || null,
                                    selector: getSelector(prevSibling)
                                }});
                            }}
                        }} else if (prevSibling.tagName === 'FIGURE') {{
                            images.push(...extractImages(prevSibling));
                        }}
                    }}

                    return images;
                }}

                // Extract headings
                document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(h => {{
                    if (shouldExclude(h)) return;
                    const text = getCleanText(h);
                    if (text.length < 2) return;

                    results.push({{
                        type: 'heading',
                        level: parseInt(h.tagName[1]),
                        text: text,
                        html: h.innerHTML,
                        selector: getSelector(h),
                        images: findNearbyImages(h),
                        links: extractLinks(h),
                        order: order++
                    }});
                }});

                // Extract paragraphs
                document.querySelectorAll('p').forEach(p => {{
                    if (shouldExclude(p)) return;
                    const text = getCleanText(p);
                    if (text.length < 10) return;

                    // Get parent heading
                    let parentHeading = null;
                    let prev = p.previousElementSibling;
                    while (prev) {{
                        if (/^H[1-6]$/.test(prev.tagName)) {{
                            parentHeading = prev.textContent?.trim();
                            break;
                        }}
                        prev = prev.previousElementSibling;
                    }}

                    results.push({{
                        type: 'paragraph',
                        text: text,
                        html: p.innerHTML,
                        selector: getSelector(p),
                        parentHeading: parentHeading,
                        images: findNearbyImages(p),
                        links: extractLinks(p),
                        order: order++
                    }});
                }});

                // Extract lists
                document.querySelectorAll('ul, ol').forEach(list => {{
                    if (shouldExclude(list)) return;

                    const items = [];
                    list.querySelectorAll(':scope > li').forEach(li => {{
                        const text = getCleanText(li);
                        if (text.length > 2) items.push(text);
                    }});

                    if (items.length === 0) return;

                    // Get parent heading
                    let parentHeading = null;
                    let prev = list.previousElementSibling;
                    while (prev) {{
                        if (/^H[1-6]$/.test(prev.tagName)) {{
                            parentHeading = prev.textContent?.trim();
                            break;
                        }}
                        prev = prev.previousElementSibling;
                    }}

                    results.push({{
                        type: 'list',
                        listType: list.tagName.toLowerCase(),
                        items: items,
                        text: items.join('\\n'),
                        selector: getSelector(list),
                        parentHeading: parentHeading,
                        images: findNearbyImages(list),
                        links: extractLinks(list),
                        order: order++
                    }});
                }});

                // Extract blockquotes
                document.querySelectorAll('blockquote').forEach(bq => {{
                    if (shouldExclude(bq)) return;
                    const text = getCleanText(bq);
                    if (text.length < 10) return;

                    results.push({{
                        type: 'quote',
                        text: text,
                        html: bq.innerHTML,
                        selector: getSelector(bq),
                        images: findNearbyImages(bq),
                        links: extractLinks(bq),
                        order: order++
                    }});
                }});

                // Extract code blocks
                document.querySelectorAll('pre, code').forEach(code => {{
                    if (shouldExclude(code)) return;
                    // Skip inline code that's just a few characters
                    if (code.tagName === 'CODE' && code.parentElement?.tagName === 'P') return;

                    const text = code.textContent?.trim() || '';
                    if (text.length < 5) return;

                    results.push({{
                        type: 'code',
                        text: text,
                        language: code.className?.match(/language-(\\w+)/)?.[1] || null,
                        selector: getSelector(code),
                        images: [],
                        links: [],
                        order: order++
                    }});
                }});

                // Extract definition lists
                document.querySelectorAll('dl').forEach(dl => {{
                    if (shouldExclude(dl)) return;

                    const items = [];
                    const dts = dl.querySelectorAll('dt');
                    dts.forEach(dt => {{
                        const term = getCleanText(dt);
                        const dd = dt.nextElementSibling;
                        const def = dd && dd.tagName === 'DD' ? getCleanText(dd) : '';
                        if (term) items.push(term + ': ' + def);
                    }});

                    if (items.length === 0) return;

                    results.push({{
                        type: 'list',
                        listType: 'definition',
                        items: items,
                        text: items.join('\\n'),
                        selector: getSelector(dl),
                        images: findNearbyImages(dl),
                        links: extractLinks(dl),
                        order: order++
                    }});
                }});

                // Sort by order
                results.sort((a, b) => a.order - b.order);

                return results;
            }}
        """

        try:
            return await page.evaluate(extraction_script)
        except Exception as e:
            logger.error(f"JavaScript extraction failed: {e}")
            return []

    def _process_raw_content(self, raw_content: list[dict]) -> list[ContentBlock]:
        """Process raw content into ContentBlock objects.

        Args:
            raw_content: Raw content from JavaScript extraction.

        Returns:
            List of ContentBlock objects.
        """
        blocks = []
        current_heading = None

        for item in raw_content:
            item_type = item.get("type", "")

            # Extract media assets and links from the item
            media_assets = self._process_images(item.get("images", []))
            links = item.get("links", [])

            if item_type == "heading":
                current_heading = item.get("text", "")
                block = ContentBlock(
                    content_type=ContentType.HEADING,
                    text=current_heading,
                    html=item.get("html"),
                    heading_level=item.get("level", 1),
                    source_selector=item.get("selector"),
                    order=item.get("order", 0),
                    media_assets=media_assets,
                    links=links,
                )
                blocks.append(block)

            elif item_type == "paragraph":
                block = ContentBlock(
                    content_type=ContentType.PARAGRAPH,
                    text=item.get("text", ""),
                    html=item.get("html"),
                    source_selector=item.get("selector"),
                    parent_section=item.get("parentHeading") or current_heading,
                    order=item.get("order", 0),
                    media_assets=media_assets,
                    links=links,
                )
                blocks.append(block)

            elif item_type == "list":
                block = ContentBlock(
                    content_type=ContentType.LIST,
                    text=item.get("text", ""),
                    list_items=item.get("items", []),
                    source_selector=item.get("selector"),
                    parent_section=item.get("parentHeading") or current_heading,
                    order=item.get("order", 0),
                    media_assets=media_assets,
                    links=links,
                )
                blocks.append(block)

            elif item_type == "quote":
                block = ContentBlock(
                    content_type=ContentType.QUOTE,
                    text=item.get("text", ""),
                    html=item.get("html"),
                    source_selector=item.get("selector"),
                    order=item.get("order", 0),
                    media_assets=media_assets,
                    links=links,
                )
                blocks.append(block)

            elif item_type == "code":
                block = ContentBlock(
                    content_type=ContentType.CODE,
                    text=item.get("text", ""),
                    source_selector=item.get("selector"),
                    order=item.get("order", 0),
                    media_assets=[],
                    links=[],
                )
                blocks.append(block)

        return blocks

    def _process_images(self, raw_images: list[dict]) -> list[MediaAsset]:
        """Convert raw image data to MediaAsset objects.

        Args:
            raw_images: List of raw image dictionaries from JavaScript extraction.

        Returns:
            List of MediaAsset objects.
        """
        assets = []
        seen_urls = set()  # Avoid duplicates

        for img in raw_images:
            url = img.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            asset = MediaAsset(
                url=url,
                alt_text=img.get("alt") or None,
                title=img.get("title") or None,
                width=img.get("width"),
                height=img.get("height"),
                source_selector=img.get("selector"),
            )
            assets.append(asset)

        return assets

    def _classify_content_blocks(
        self, blocks: list[ContentBlock]
    ) -> list[ContentBlock]:
        """Add semantic classification to content blocks.

        This attempts to identify content that represents:
        - Product features
        - Benefits
        - Testimonials
        - Pricing info
        - CTAs

        Args:
            blocks: List of content blocks to classify.

        Returns:
            Content blocks with updated content_type where applicable.
        """
        for block in blocks:
            # Don't reclassify headings
            if block.content_type == ContentType.HEADING:
                continue

            text_lower = block.text.lower()
            parent_lower = (block.parent_section or "").lower()
            combined = text_lower + " " + parent_lower

            for content_type, patterns in self.CONTENT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, combined, re.IGNORECASE):
                        block.content_type = content_type
                        break

        return blocks

    async def extract_metadata(self, page: Page) -> dict:
        """Extract page metadata.

        Args:
            page: The Playwright page.

        Returns:
            Dict with page metadata.
        """
        metadata = await page.evaluate("""
            () => {
                return {
                    title: document.title || '',
                    description: document.querySelector('meta[name="description"]')?.content || '',
                    keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                    canonical: document.querySelector('link[rel="canonical"]')?.href || '',
                    ogTitle: document.querySelector('meta[property="og:title"]')?.content || '',
                    ogDescription: document.querySelector('meta[property="og:description"]')?.content || '',
                    ogImage: document.querySelector('meta[property="og:image"]')?.content || '',
                    author: document.querySelector('meta[name="author"]')?.content || '',
                    publishedTime: document.querySelector('meta[property="article:published_time"]')?.content || '',
                };
            }
        """)

        return metadata

    async def extract_structured_data(self, page: Page) -> list[dict]:
        """Extract JSON-LD structured data from the page.

        Args:
            page: The Playwright page.

        Returns:
            List of parsed JSON-LD objects.
        """
        if not self.config.extract_structured_data:
            return []

        structured_data = await page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                const data = [];
                scripts.forEach(script => {
                    try {
                        const parsed = JSON.parse(script.textContent);
                        data.push(parsed);
                    } catch (e) {
                        // Invalid JSON, skip
                    }
                });
                return data;
            }
        """)

        return structured_data

    async def extract_links(self, page: Page) -> dict[str, list]:
        """Extract and categorize all links on the page.

        Args:
            page: The Playwright page.

        Returns:
            Dict with 'internal' and 'external' link lists.
        """
        current_host = await page.evaluate("window.location.host")

        links = await page.evaluate(f"""
            () => {{
                const currentHost = '{current_host}';
                const internal = [];
                const external = [];

                document.querySelectorAll('a[href]').forEach(a => {{
                    const href = a.href;
                    if (!href || href.startsWith('javascript:') || href.startsWith('#')) return;

                    const text = a.textContent?.trim() || '';
                    const link = {{ url: href, text: text }};

                    try {{
                        const url = new URL(href);
                        if (url.host === currentHost) {{
                            internal.push(link);
                        }} else {{
                            external.push(link);
                        }}
                    }} catch (e) {{
                        // Invalid URL, treat as internal
                        internal.push(link);
                    }}
                }});

                return {{ internal, external }};
            }}
        """)

        return links
