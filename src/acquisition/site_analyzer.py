"""Site analyzer for understanding website structure before extraction."""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page

from src.models.config import SiteConfig
from src.models.content import SiteStructure

logger = logging.getLogger(__name__)


class SiteAnalyzer:
    """
    Analyzes a website's structure to optimize extraction strategy.

    This analyzer performs reconnaissance on a website to understand:
    - Whether it's a SPA or traditional site
    - Navigation structure and important pages
    - Content layout and selectors
    - Interactive elements (tabs, accordions)
    - Loading patterns (lazy loading, infinite scroll)
    """

    # Patterns for identifying page types
    PAGE_TYPE_PATTERNS = {
        "features": [
            r"/features",
            r"/capabilities",
            r"/what-we-do",
            r"/product",
        ],
        "documentation": [
            r"/docs",
            r"/documentation",
            r"/guide",
            r"/manual",
            r"/help",
            r"/support",
        ],
        "pricing": [
            r"/pricing",
            r"/plans",
            r"/packages",
            r"/cost",
        ],
        "solutions": [
            r"/solutions",
            r"/use-cases",
            r"/industries",
            r"/for-",
        ],
        "integrations": [
            r"/integrations",
            r"/connect",
            r"/apps",
            r"/marketplace",
        ],
        "about": [
            r"/about",
            r"/company",
            r"/team",
            r"/story",
        ],
        "blog": [
            r"/blog",
            r"/news",
            r"/articles",
            r"/posts",
        ],
        "legal": [
            r"/privacy",
            r"/terms",
            r"/legal",
            r"/gdpr",
            r"/cookie",
        ],
        "careers": [
            r"/careers",
            r"/jobs",
            r"/join",
            r"/work-with-us",
        ],
    }

    # Common selectors for different page elements
    COMMON_SELECTORS = {
        "main_navigation": [
            "nav",
            "[role='navigation']",
            "header nav",
            ".navbar",
            ".nav-main",
            ".main-nav",
            "#navigation",
        ],
        "main_content": [
            "main",
            "[role='main']",
            "article",
            ".main-content",
            "#main-content",
            ".content",
            "#content",
            ".page-content",
        ],
        "sidebar": [
            "aside",
            "[role='complementary']",
            ".sidebar",
            "#sidebar",
            ".side-nav",
        ],
        "footer": [
            "footer",
            "[role='contentinfo']",
            ".footer",
            "#footer",
        ],
        "tabs": [
            "[role='tablist']",
            ".tabs",
            ".tab-list",
            ".nav-tabs",
        ],
        "accordions": [
            ".accordion",
            "[class*='accordion']",
            ".collapsible",
            ".expandable",
            "details",
        ],
        "cookie_banner": [
            "[class*='cookie']",
            "[class*='consent']",
            "[id*='cookie']",
            "[id*='gdpr']",
            ".cc-banner",
            "#onetrust-banner-sdk",
        ],
    }

    def __init__(self, config: Optional[SiteConfig] = None):
        """Initialize the site analyzer.

        Args:
            config: Site configuration with custom selectors.
        """
        self.config = config
        self._base_url: Optional[str] = None
        self._base_domain: Optional[str] = None
        self._base_path: Optional[str] = None

    async def analyze(self, page: Page, url: str) -> SiteStructure:
        """Analyze a website's structure.

        Args:
            page: The Playwright page (should already be navigated to the URL).
            url: The URL being analyzed.

        Returns:
            SiteStructure with analysis results.
        """
        logger.info(f"Analyzing site structure for: {url}")

        parsed_url = urlparse(url)
        self._base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self._base_domain = parsed_url.netloc
        # Store the base path to filter only child pages
        self._base_path = parsed_url.path.rstrip("/") or "/"

        structure = SiteStructure(base_url=self._base_url)

        # Run analysis tasks
        await asyncio.gather(
            self._analyze_spa_characteristics(page, structure),
            self._analyze_navigation(page, structure),
            self._analyze_content_structure(page, structure),
            self._analyze_interactive_elements(page, structure),
            self._analyze_loading_patterns(page, structure),
            self._discover_pages(page, structure),
        )

        # Calculate confidence score
        structure.analysis_confidence = self._calculate_confidence(structure)

        logger.info(
            f"Analysis complete: SPA={structure.is_spa}, "
            f"framework={structure.spa_framework}, "
            f"pages_found={len(structure.discovered_pages)}, "
            f"confidence={structure.analysis_confidence:.2f}"
        )

        return structure

    async def _analyze_spa_characteristics(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Analyze whether the site is a SPA and which framework it uses.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Check for SPA framework signatures
        spa_checks = await page.evaluate("""
            () => {
                const result = {
                    hasReact: false,
                    hasVue: false,
                    hasAngular: false,
                    hasSvelte: false,
                    hasNextJS: false,
                    hasNuxt: false,
                    requiresJS: false,
                };

                // React
                result.hasReact = !!(
                    document.querySelector('[data-reactroot]') ||
                    document.querySelector('[data-reactid]') ||
                    window.__REACT_DEVTOOLS_GLOBAL_HOOK__ ||
                    window.React
                );

                // Next.js
                result.hasNextJS = !!(
                    document.getElementById('__next') ||
                    window.__NEXT_DATA__
                );

                // Vue
                result.hasVue = !!(
                    document.querySelector('[data-v-app]') ||
                    document.querySelector('[data-v-]') ||
                    window.__VUE__ ||
                    window.Vue
                );

                // Nuxt
                result.hasNuxt = !!(
                    document.getElementById('__nuxt') ||
                    window.__NUXT__
                );

                // Angular
                result.hasAngular = !!(
                    document.querySelector('[ng-version]') ||
                    document.querySelector('[_ngcontent]') ||
                    window.ng
                );

                // Svelte
                result.hasSvelte = !!(
                    document.querySelector('[class*="svelte-"]')
                );

                // Check if JS is required (minimal initial content)
                const bodyText = document.body.innerText || '';
                result.requiresJS = bodyText.trim().length < 500;

                return result;
            }
        """)

        # Determine framework
        if spa_checks.get("hasNextJS"):
            structure.spa_framework = "nextjs"
            structure.is_spa = True
        elif spa_checks.get("hasNuxt"):
            structure.spa_framework = "nuxt"
            structure.is_spa = True
        elif spa_checks.get("hasReact"):
            structure.spa_framework = "react"
            structure.is_spa = True
        elif spa_checks.get("hasVue"):
            structure.spa_framework = "vue"
            structure.is_spa = True
        elif spa_checks.get("hasAngular"):
            structure.spa_framework = "angular"
            structure.is_spa = True
        elif spa_checks.get("hasSvelte"):
            structure.spa_framework = "svelte"
            structure.is_spa = True
        elif spa_checks.get("requiresJS"):
            structure.is_spa = True
            structure.spa_framework = "unknown"

        structure.requires_javascript = spa_checks.get("requiresJS", False)
        structure.has_dynamic_content = structure.is_spa

    async def _analyze_navigation(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Analyze the site's navigation structure.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Find main navigation
        for selector in self.COMMON_SELECTORS["main_navigation"]:
            try:
                nav = await page.query_selector(selector)
                if nav:
                    structure.main_navigation_selector = selector
                    break
            except Exception:
                continue

        # Extract navigation links
        if structure.main_navigation_selector:
            nav_links = await page.evaluate(
                f"""
                () => {{
                    const nav = document.querySelector('{structure.main_navigation_selector}');
                    if (!nav) return [];
                    return Array.from(nav.querySelectorAll('a[href]')).map(a => ({{
                        url: a.href,
                        text: a.textContent?.trim() || ''
                    }}));
                }}
            """
            )

            # Categorize navigation links
            for link in nav_links:
                url = link.get("url", "")
                if self._is_internal_url(url):
                    structure.discovered_pages.append(url)

    async def _analyze_content_structure(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Analyze the content layout and find key selectors.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Find main content area
        for selector in self.COMMON_SELECTORS["main_content"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    # Check if it has substantial content
                    text_length = await element.evaluate(
                        "el => (el.innerText || '').length"
                    )
                    if text_length > 100:
                        structure.main_content_selector = selector
                        break
            except Exception:
                continue

        # Find article selector if different from main
        article_selectors = ["article", ".article", ".post", ".entry"]
        for selector in article_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    structure.article_selector = selector
                    break
            except Exception:
                continue

        # Find sidebar
        for selector in self.COMMON_SELECTORS["sidebar"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    structure.sidebar_selectors.append(selector)
            except Exception:
                continue

        # Find footer
        for selector in self.COMMON_SELECTORS["footer"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    structure.footer_selector = selector
                    break
            except Exception:
                continue

    async def _analyze_interactive_elements(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Find tabs, accordions, and other interactive elements.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Find tabs and extract any linked URLs
        for selector in self.COMMON_SELECTORS["tabs"]:
            try:
                elements = await page.query_selector_all(f"{selector} [role='tab'], {selector} button, {selector} a")
                if elements and len(elements) > 1:
                    structure.tab_selectors.append(selector)
                    structure.requires_interaction = True
            except Exception:
                continue

        # Extract URLs from tab elements (tabs that link to child pages)
        await self._extract_tab_links(page, structure)

        # Find accordions
        for selector in self.COMMON_SELECTORS["accordions"]:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    structure.accordion_selectors.append(selector)
                    structure.requires_interaction = True
            except Exception:
                continue

        # Find modal triggers
        modal_triggers = await page.query_selector_all(
            "[data-toggle='modal'], [data-bs-toggle='modal'], "
            "[class*='modal-trigger'], button[class*='demo'], "
            "button[class*='video'], [class*='play-button']"
        )
        for trigger in modal_triggers:
            try:
                selector = await trigger.evaluate(
                    """
                    el => {
                        if (el.id) return '#' + el.id;
                        if (el.className) return '.' + el.className.split(' ')[0];
                        return null;
                    }
                """
                )
                if selector:
                    structure.modal_triggers.append(selector)
            except Exception:
                continue

        # Check for cookie banner
        for selector in self.COMMON_SELECTORS["cookie_banner"]:
            try:
                element = await page.query_selector(selector)
                if element:
                    is_visible = await element.is_visible()
                    if is_visible:
                        structure.has_cookie_banner = True
                        break
            except Exception:
                continue

    async def _extract_tab_links(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Extract URLs from tab elements that link to child pages.

        Many websites use tabs as navigation to child pages (e.g., product tabs,
        feature tabs). This method finds those tab links and adds them to
        discovered_pages.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Extended tab selectors that might contain links
        tab_link_selectors = [
            # Role-based tabs
            "[role='tablist'] a[href]",
            "[role='tablist'] [role='tab'][href]",
            "[role='tab'] a[href]",
            # Class-based tabs
            ".tabs a[href]",
            ".tab-list a[href]",
            ".nav-tabs a[href]",
            ".tab-nav a[href]",
            ".tab-menu a[href]",
            # Data attribute based
            "[data-tab] a[href]",
            "[data-tabs] a[href]",
            "a[data-tab][href]",
            "a[role='tab'][href]",
            # Common tab component patterns
            ".tab a[href]",
            ".tab-item a[href]",
            ".tab-link[href]",
            "a.tab-link[href]",
            # Navigation that acts like tabs
            ".nav-pills a[href]",
            ".pill-nav a[href]",
            # Sub-navigation patterns (often tab-like)
            ".subnav a[href]",
            ".sub-nav a[href]",
            ".secondary-nav a[href]",
            ".product-nav a[href]",
            ".feature-nav a[href]",
            # Section navigation
            ".section-nav a[href]",
            ".page-nav a[href]",
            ".content-nav a[href]",
        ]

        tab_links = set()

        for selector in tab_link_selectors:
            try:
                links = await page.evaluate(
                    f"""
                    () => {{
                        const elements = document.querySelectorAll('{selector}');
                        return Array.from(elements).map(el => {{
                            // Get href from the element itself or a child anchor
                            let href = el.href || el.getAttribute('href');
                            if (!href && el.tagName !== 'A') {{
                                const anchor = el.querySelector('a[href]');
                                href = anchor ? anchor.href : null;
                            }}
                            return href;
                        }}).filter(href => href && !href.startsWith('javascript:') && !href.startsWith('#'));
                    }}
                """
                )
                for url in links:
                    if self._is_internal_url(url):
                        tab_links.add(url)
            except Exception:
                continue

        # Also check for tabs with data-href or similar attributes
        data_href_selectors = [
            "[role='tab'][data-href]",
            "[role='tab'][data-url]",
            "[role='tab'][data-link]",
            ".tab[data-href]",
            ".tab[data-url]",
            ".tab-item[data-href]",
        ]

        for selector in data_href_selectors:
            try:
                data_links = await page.evaluate(
                    f"""
                    () => {{
                        const elements = document.querySelectorAll('{selector}');
                        return Array.from(elements).map(el => {{
                            return el.dataset.href || el.dataset.url || el.dataset.link;
                        }}).filter(href => href && !href.startsWith('javascript:') && !href.startsWith('#'));
                    }}
                """
                )
                for url in data_links:
                    # Convert relative URLs to absolute
                    if url.startswith("/"):
                        url = f"{self._base_url}{url}"
                    if self._is_internal_url(url):
                        tab_links.add(url)
            except Exception:
                continue

        # Add tab links to discovered pages
        if tab_links:
            logger.info(f"Found {len(tab_links)} URLs from tab elements")
            for url in tab_links:
                normalized = self._normalize_url(url)
                if normalized and normalized not in structure.discovered_pages:
                    structure.discovered_pages.append(normalized)

    async def _analyze_loading_patterns(
        self, page: Page, structure: SiteStructure
    ) -> None:
        """Analyze lazy loading and infinite scroll patterns.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Check for lazy loading
        lazy_load_check = await page.evaluate("""
            () => {
                // Check for common lazy loading patterns
                const lazyImages = document.querySelectorAll(
                    'img[loading="lazy"], img[data-src], img[data-lazy], ' +
                    '[class*="lazy"], [class*="lazyload"]'
                );

                // Check for intersection observer usage (common for lazy loading)
                const hasIntersectionObserver = 'IntersectionObserver' in window;

                // Check for infinite scroll patterns
                const hasInfiniteScroll = !!(
                    document.querySelector('[class*="infinite"]') ||
                    document.querySelector('[data-infinite]') ||
                    document.querySelector('[class*="load-more"]')
                );

                return {
                    lazyImagesCount: lazyImages.length,
                    hasIntersectionObserver: hasIntersectionObserver,
                    hasInfiniteScroll: hasInfiniteScroll
                };
            }
        """)

        structure.uses_lazy_loading = lazy_load_check.get("lazyImagesCount", 0) > 3
        structure.uses_infinite_scroll = lazy_load_check.get("hasInfiniteScroll", False)

    async def _discover_pages(self, page: Page, structure: SiteStructure) -> None:
        """Discover important pages on the site.

        Args:
            page: The Playwright page.
            structure: SiteStructure to populate.
        """
        # Get all links on the page
        all_links = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(href => href && !href.startsWith('javascript:') && !href.startsWith('#'));
            }
        """)

        # Categorize links
        seen_urls = set()
        for url in all_links:
            if not self._is_internal_url(url):
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Normalize URL
            normalized = self._normalize_url(url)
            if not normalized:
                continue

            # Categorize by page type
            page_type = self._categorize_url(normalized)

            if page_type == "features":
                structure.feature_pages.append(normalized)
            elif page_type == "documentation":
                structure.documentation_pages.append(normalized)
            elif page_type == "pricing":
                structure.pricing_pages.append(normalized)
            elif page_type in ["blog", "careers", "legal"]:
                # Skip these
                continue
            else:
                # General pages
                if normalized not in structure.discovered_pages:
                    structure.discovered_pages.append(normalized)

        # Deduplicate
        structure.discovered_pages = list(set(structure.discovered_pages))
        structure.feature_pages = list(set(structure.feature_pages))
        structure.documentation_pages = list(set(structure.documentation_pages))
        structure.pricing_pages = list(set(structure.pricing_pages))

    def _is_internal_url(self, url: str) -> bool:
        """Check if a URL is internal to the site and is a child of the base path.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is internal and a child of the base path.
        """
        if not url or not self._base_domain:
            return False

        try:
            parsed = urlparse(url)
            # Must be same domain (or relative URL)
            if parsed.netloc and parsed.netloc != self._base_domain:
                return False

            # Check if the URL path is a child of the base path
            url_path = parsed.path.rstrip("/") or "/"
            return self._is_child_path(url_path)
        except Exception:
            return False

    def _is_child_path(self, url_path: str) -> bool:
        """Check if a URL path is a child of the base path.

        Uses a hybrid approach to handle both directory-based and file-based URLs:
        1. Directory-based: /products/foo -> children are /products/foo/*
        2. File-based: /products/foo.aspx -> children are /products/foo/* (stem-based)

        Args:
            url_path: The URL path to check.

        Returns:
            True if the path is the base path or a child of it.
        """
        if not self._base_path:
            return True

        # Normalize paths
        base = self._base_path.rstrip("/")
        path = url_path.rstrip("/")

        # If base path is root, all paths are children
        if base == "" or base == "/":
            return True

        # Exact match is allowed (the starting page itself)
        if path == base:
            return True

        # Method 1: Directory-based matching
        # e.g., base="/products", valid children: "/products/widget", "/products/foo/bar"
        if path.startswith(base + "/"):
            return True

        # Method 2: File-stem-based matching for URLs with file extensions
        # e.g., base="/products/foo.aspx", stem="/products/foo"
        # valid children: "/products/foo/bar.aspx", "/products/foo/bar"
        file_extensions = ('.aspx', '.html', '.htm', '.php', '.jsp', '.do', '.action')
        base_lower = base.lower()

        for ext in file_extensions:
            if base_lower.endswith(ext):
                # Extract stem by removing the extension
                base_stem = base[:-len(ext)]
                # Check if the URL path is under the stem directory
                if path.startswith(base_stem + "/"):
                    return True
                break

        return False

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize a URL for comparison.

        Args:
            url: The URL to normalize.

        Returns:
            Normalized URL or None if invalid.
        """
        try:
            parsed = urlparse(url)

            # Remove trailing slash
            path = parsed.path.rstrip("/")

            # Remove common non-content paths
            skip_patterns = [
                r"^/$",  # Home page (handle separately)
                r"\.(js|css|ico|png|jpg|jpeg|gif|svg|woff|woff2|ttf|eot)$",
                r"^/api/",
                r"^/static/",
                r"^/assets/",
                r"^/_next/",
                r"^/__",
            ]

            for pattern in skip_patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    return None

            # Reconstruct URL without query params and fragments
            return f"{self._base_url}{path}" if path else self._base_url

        except Exception:
            return None

    def _categorize_url(self, url: str) -> Optional[str]:
        """Categorize a URL by its content type.

        Args:
            url: The URL to categorize.

        Returns:
            Category name or None if uncategorized.
        """
        path = urlparse(url).path.lower()

        for category, patterns in self.PAGE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    return category

        return None

    def _calculate_confidence(self, structure: SiteStructure) -> float:
        """Calculate confidence score for the analysis.

        Args:
            structure: The analyzed structure.

        Returns:
            Confidence score between 0 and 1.
        """
        score = 0.5  # Base score

        # Boost for finding main content
        if structure.main_content_selector:
            score += 0.1

        # Boost for finding navigation
        if structure.main_navigation_selector:
            score += 0.1

        # Boost for discovering pages
        if len(structure.discovered_pages) > 5:
            score += 0.1
        if len(structure.feature_pages) > 0:
            score += 0.1

        # Boost for identifying framework (helps with SPA handling)
        if structure.spa_framework and structure.spa_framework != "unknown":
            score += 0.1

        return min(score, 1.0)

    async def get_extraction_recommendations(
        self, structure: SiteStructure
    ) -> dict:
        """Generate extraction recommendations based on analysis.

        Args:
            structure: The analyzed site structure.

        Returns:
            Dict with extraction recommendations.
        """
        recommendations = {
            "priority_pages": [],
            "wait_strategy": "networkidle",
            "needs_scroll": structure.uses_lazy_loading,
            "needs_interaction": structure.requires_interaction,
            "estimated_pages": 0,
            "warnings": [],
        }

        # Prioritize pages
        priority_pages = []

        # Add feature pages first (most important for sales pitch)
        priority_pages.extend(structure.feature_pages[:10])

        # Then pricing
        priority_pages.extend(structure.pricing_pages[:2])

        # Then documentation (limited)
        priority_pages.extend(structure.documentation_pages[:5])

        # Then other discovered pages
        remaining_slots = 20 - len(priority_pages)
        if remaining_slots > 0:
            priority_pages.extend(structure.discovered_pages[:remaining_slots])

        recommendations["priority_pages"] = priority_pages
        recommendations["estimated_pages"] = len(priority_pages)

        # Wait strategy based on site type
        if structure.is_spa:
            recommendations["wait_strategy"] = "networkidle"
            if structure.spa_framework == "angular":
                recommendations["wait_strategy"] = "domcontentloaded"
                recommendations["warnings"].append(
                    "Angular detected - may need custom wait conditions"
                )

        # Warnings
        if structure.has_auth_wall:
            recommendations["warnings"].append(
                "Authentication may be required for some content"
            )

        if structure.uses_infinite_scroll:
            recommendations["warnings"].append(
                "Infinite scroll detected - content extraction may be limited"
            )

        if not structure.main_content_selector:
            recommendations["warnings"].append(
                "Could not identify main content area - extraction may include noise"
            )

        return recommendations
