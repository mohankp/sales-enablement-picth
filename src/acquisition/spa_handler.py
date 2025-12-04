"""Handler for Single Page Application (SPA) specific extraction challenges."""

import asyncio
import logging
import re
from typing import Optional

from playwright.async_api import Page

from src.models.config import SPAConfig, WaitStrategy

logger = logging.getLogger(__name__)


class SPAHandler:
    """
    Handles SPA-specific challenges in content extraction.

    Modern web applications use frameworks like React, Vue, Angular that
    render content dynamically via JavaScript. This handler provides
    strategies to ensure all content is fully loaded before extraction.

    Key challenges addressed:
    - JavaScript hydration
    - Lazy-loaded content
    - Tab/accordion content
    - Infinite scroll
    - Dynamic routing
    """

    # Framework detection patterns
    FRAMEWORK_SIGNATURES = {
        "react": [
            "[data-reactroot]",
            "[data-reactid]",
            "#__next",  # Next.js
            "script[src*='react']",
            "script[src*='next']",
        ],
        "vue": [
            "[data-v-app]",
            "[data-v-]",
            "#__nuxt",  # Nuxt.js
            "script[src*='vue']",
            "script[src*='nuxt']",
        ],
        "angular": [
            "[ng-version]",
            "[_ngcontent]",
            "script[src*='angular']",
            "[ng-app]",
        ],
        "svelte": [
            "script[src*='svelte']",
            "[class*='svelte-']",
        ],
        "ember": [
            "[id='ember']",
            "script[src*='ember']",
        ],
    }

    # Common SPA loading indicators
    LOADING_INDICATORS = [
        ".loading",
        ".spinner",
        "[class*='loading']",
        "[class*='spinner']",
        "[class*='skeleton']",
        "[aria-busy='true']",
        ".loader",
        "[data-loading]",
    ]

    # Content ready indicators
    CONTENT_READY_INDICATORS = [
        "main",
        "article",
        "[role='main']",
        ".content",
        "#content",
        ".main-content",
    ]

    def __init__(self, config: Optional[SPAConfig] = None):
        """Initialize the SPA handler.

        Args:
            config: SPA handling configuration. Uses defaults if not provided.
        """
        self.config = config or SPAConfig()
        self._detected_framework: Optional[str] = None

    async def detect_framework(self, page: Page) -> Optional[str]:
        """Detect which SPA framework the page uses, if any.

        Args:
            page: The Playwright page to analyze.

        Returns:
            Framework name (react, vue, angular, etc.) or None if not detected.
        """
        if self.config.framework_hint:
            logger.debug(f"Using framework hint: {self.config.framework_hint}")
            self._detected_framework = self.config.framework_hint
            return self.config.framework_hint

        if not self.config.detect_framework:
            return None

        for framework, selectors in self.FRAMEWORK_SIGNATURES.items():
            for selector in selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        logger.info(f"Detected SPA framework: {framework}")
                        self._detected_framework = framework
                        return framework
                except Exception:
                    continue

        # Check for general SPA indicators
        is_spa = await self._check_spa_characteristics(page)
        if is_spa:
            logger.info("Detected SPA characteristics but unknown framework")
            self._detected_framework = "unknown"
            return "unknown"

        logger.debug("No SPA framework detected - appears to be static/SSR site")
        return None

    async def _check_spa_characteristics(self, page: Page) -> bool:
        """Check if the page has general SPA characteristics.

        Args:
            page: The Playwright page to analyze.

        Returns:
            True if the page appears to be an SPA.
        """
        # Check for minimal initial HTML that gets populated by JS
        initial_content = await page.evaluate("""
            () => {
                const body = document.body;
                const textContent = body.textContent?.trim() || '';
                const childCount = body.children.length;
                return {
                    textLength: textContent.length,
                    childCount: childCount,
                    hasAppRoot: !!document.querySelector('#app, #root, #__app, #__root')
                };
            }
        """)

        # SPAs often have minimal initial content and an app root
        return initial_content.get("hasAppRoot", False)

    async def wait_for_content(self, page: Page) -> bool:
        """Wait for the SPA content to be fully loaded.

        This is the main method that orchestrates all waiting strategies.

        Args:
            page: The Playwright page.

        Returns:
            True if content is ready, False if timeout occurred.
        """
        logger.debug("Waiting for SPA content to load...")

        # Initial wait for basic JavaScript execution
        await asyncio.sleep(self.config.initial_wait_ms / 1000)

        # Primary wait strategy
        success = await self._apply_wait_strategy(
            page, self.config.primary_wait_strategy
        )

        if not success and self.config.secondary_wait_strategy:
            logger.debug("Primary wait strategy incomplete, trying secondary")
            success = await self._apply_wait_strategy(
                page, self.config.secondary_wait_strategy
            )

        # Wait for hydration if enabled
        if self.config.wait_for_hydration:
            await self._wait_for_hydration(page)

        # Wait for loading indicators to disappear
        await self._wait_for_loading_complete(page)

        # Verify content is actually present
        has_content = await self._verify_content_present(page)

        if has_content:
            logger.debug("SPA content loaded successfully")
        else:
            logger.warning("Content verification failed - page may be incomplete")

        return has_content

    async def _apply_wait_strategy(
        self, page: Page, strategy: WaitStrategy
    ) -> bool:
        """Apply a specific wait strategy.

        Args:
            page: The Playwright page.
            strategy: The wait strategy to apply.

        Returns:
            True if wait completed successfully.
        """
        try:
            if strategy == WaitStrategy.LOAD:
                await page.wait_for_load_state("load", timeout=self.config.max_wait_ms)
                return True

            elif strategy == WaitStrategy.DOMCONTENTLOADED:
                await page.wait_for_load_state(
                    "domcontentloaded", timeout=self.config.max_wait_ms
                )
                return True

            elif strategy == WaitStrategy.NETWORKIDLE:
                await page.wait_for_load_state(
                    "networkidle", timeout=self.config.max_wait_ms
                )
                return True

            elif strategy == WaitStrategy.SELECTOR:
                if self.config.content_ready_selector:
                    await page.wait_for_selector(
                        self.config.content_ready_selector,
                        timeout=self.config.max_wait_ms,
                    )
                    return True
                else:
                    # Try common content selectors
                    for selector in self.CONTENT_READY_INDICATORS:
                        try:
                            await page.wait_for_selector(selector, timeout=3000)
                            return True
                        except Exception:
                            continue
                return False

            elif strategy == WaitStrategy.FUNCTION:
                if self.config.content_ready_function:
                    await page.wait_for_function(
                        self.config.content_ready_function,
                        timeout=self.config.max_wait_ms,
                    )
                    return True
                return False

            elif strategy == WaitStrategy.TIMEOUT:
                await asyncio.sleep(self.config.max_wait_ms / 1000)
                return True

        except Exception as e:
            logger.debug(f"Wait strategy {strategy} failed: {e}")
            return False

        return False

    async def _wait_for_hydration(self, page: Page) -> None:
        """Wait for React/Vue/Angular hydration to complete.

        Hydration is when the JS framework "takes over" the server-rendered HTML.

        Args:
            page: The Playwright page.
        """
        # Check for hydration indicators
        for indicator in self.config.hydration_indicators:
            try:
                element = await page.query_selector(indicator)
                if element:
                    logger.debug(f"Found hydration indicator: {indicator}")
                    # Give it a moment to hydrate
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                continue

        # For React specifically, wait for state to be populated
        if self._detected_framework == "react":
            await self._wait_for_react_hydration(page)

    async def _wait_for_react_hydration(self, page: Page) -> None:
        """React-specific hydration waiting.

        Args:
            page: The Playwright page.
        """
        try:
            await page.wait_for_function(
                """
                () => {
                    // Check if React has hydrated by looking for React fiber
                    const root = document.querySelector('#root, #__next, [data-reactroot]');
                    if (!root) return true;  // No React root found

                    // Check for React 18+ root
                    if (root._reactRootContainer || root.__reactContainer$) {
                        return true;
                    }

                    // Fallback: check if content is present
                    return root.children.length > 0;
                }
                """,
                timeout=5000,
            )
        except Exception as e:
            logger.debug(f"React hydration wait failed: {e}")

    async def _wait_for_loading_complete(self, page: Page) -> None:
        """Wait for loading indicators to disappear.

        Args:
            page: The Playwright page.
        """
        for indicator in self.LOADING_INDICATORS:
            try:
                # Wait for loading indicator to be hidden or removed
                await page.wait_for_selector(
                    indicator,
                    state="hidden",
                    timeout=2000,
                )
            except Exception:
                # Indicator might not exist, which is fine
                continue

    async def _verify_content_present(self, page: Page) -> bool:
        """Verify that actual content is present on the page.

        Args:
            page: The Playwright page.

        Returns:
            True if substantial content is found.
        """
        content_check = await page.evaluate("""
            () => {
                const body = document.body;
                const text = body.innerText || '';
                const wordCount = text.split(/\\s+/).filter(w => w.length > 0).length;
                const hasMainContent = !!(
                    document.querySelector('main, article, [role="main"]') ||
                    document.querySelector('.content, #content, .main-content')
                );
                return {
                    wordCount: wordCount,
                    hasMainContent: hasMainContent,
                    hasImages: document.querySelectorAll('img').length,
                    hasHeadings: document.querySelectorAll('h1, h2, h3').length
                };
            }
        """)

        # Consider content present if we have reasonable indicators
        return (
            content_check.get("wordCount", 0) > 50
            or content_check.get("hasMainContent", False)
            or content_check.get("hasHeadings", 0) > 2
        )

    async def handle_lazy_loading(self, page: Page) -> int:
        """Handle lazy-loaded content by scrolling the page.

        Args:
            page: The Playwright page.

        Returns:
            Number of scroll iterations performed.
        """
        if not self.config.scroll_to_load:
            return 0

        logger.debug("Handling lazy-loaded content via scrolling")

        iterations = 0
        previous_height = 0

        for i in range(self.config.max_scroll_iterations):
            # Get current page height
            current_height = await page.evaluate("document.body.scrollHeight")

            if current_height == previous_height:
                # No new content loaded
                break

            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for content to load
            await asyncio.sleep(self.config.scroll_pause_ms / 1000)

            # Wait for network to settle
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass

            previous_height = current_height
            iterations += 1

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")

        logger.debug(f"Completed {iterations} scroll iterations for lazy loading")
        return iterations

    async def expand_tabs(self, page: Page) -> list[dict]:
        """Expand all tabs on the page and extract their content.

        Args:
            page: The Playwright page.

        Returns:
            List of dicts containing tab info and whether it was expanded.
        """
        if not self.config.expand_tabs:
            return []

        logger.debug("Expanding tabs to reveal hidden content")

        # Common tab selectors
        tab_selectors = [
            "[role='tab']",
            ".tab",
            ".nav-tab",
            "[data-toggle='tab']",
            "[data-bs-toggle='tab']",
            ".tabs button",
            ".tab-button",
            "[class*='tab-trigger']",
        ]

        expanded_tabs = []

        for selector in tab_selectors:
            try:
                tabs = await page.query_selector_all(selector)

                for i, tab in enumerate(tabs):
                    try:
                        # Check if tab is already selected
                        is_selected = await tab.get_attribute("aria-selected")
                        if is_selected == "true":
                            continue

                        # Get tab text for identification
                        tab_text = await tab.text_content()

                        # Click the tab
                        await tab.click()
                        await asyncio.sleep(self.config.click_delay_ms / 1000)

                        expanded_tabs.append({
                            "selector": selector,
                            "index": i,
                            "text": tab_text.strip() if tab_text else f"Tab {i}",
                            "expanded": True,
                        })

                        logger.debug(f"Expanded tab: {tab_text}")

                    except Exception as e:
                        logger.debug(f"Could not expand tab: {e}")

            except Exception:
                continue

        logger.debug(f"Expanded {len(expanded_tabs)} tabs")
        return expanded_tabs

    async def expand_accordions(self, page: Page) -> list[dict]:
        """Expand all accordions on the page.

        Args:
            page: The Playwright page.

        Returns:
            List of dicts containing accordion info and expansion status.
        """
        if not self.config.expand_accordions:
            return []

        logger.debug("Expanding accordions to reveal hidden content")

        # Common accordion selectors
        accordion_selectors = [
            "[data-toggle='collapse']:not(.show)",
            "[data-bs-toggle='collapse']",
            ".accordion-button:not(.show)",
            ".accordion-header button",
            "[class*='accordion'] button",
            ".collapsible:not(.active)",
            ".expandable:not(.expanded)",
            "details:not([open])",
            "[aria-expanded='false']",
        ]

        expanded_accordions = []

        for selector in accordion_selectors:
            try:
                elements = await page.query_selector_all(selector)

                for i, element in enumerate(elements):
                    try:
                        # Get element text
                        element_text = await element.text_content()

                        # Special handling for <details> elements
                        tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                        if tag_name == "details":
                            await element.evaluate("el => el.open = true")
                        else:
                            await element.click()

                        await asyncio.sleep(self.config.click_delay_ms / 1000)

                        expanded_accordions.append({
                            "selector": selector,
                            "index": i,
                            "text": (element_text.strip()[:50] if element_text else f"Accordion {i}"),
                            "expanded": True,
                        })

                    except Exception as e:
                        logger.debug(f"Could not expand accordion: {e}")

            except Exception:
                continue

        logger.debug(f"Expanded {len(expanded_accordions)} accordions")
        return expanded_accordions

    async def handle_infinite_scroll(
        self,
        page: Page,
        max_items: int = 100,
    ) -> int:
        """Handle infinite scroll pages.

        Args:
            page: The Playwright page.
            max_items: Maximum number of items to load.

        Returns:
            Estimated number of items loaded.
        """
        logger.debug("Handling infinite scroll")

        # Common selectors for scrollable item containers
        item_selectors = [
            "article",
            "[class*='card']",
            "[class*='item']",
            "li",
            "[class*='post']",
        ]

        initial_count = 0
        for selector in item_selectors:
            count = await page.evaluate(
                f"document.querySelectorAll('{selector}').length"
            )
            if count > 5:  # Found likely container
                initial_count = count
                item_selector = selector
                break
        else:
            return 0

        current_count = initial_count
        iterations = 0
        max_iterations = 20

        while current_count < max_items and iterations < max_iterations:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self.config.scroll_pause_ms / 1000)

            # Check for new items
            new_count = await page.evaluate(
                f"document.querySelectorAll('{item_selector}').length"
            )

            if new_count == current_count:
                # No new items loaded
                break

            current_count = new_count
            iterations += 1

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")

        logger.debug(f"Infinite scroll: loaded {current_count} items in {iterations} iterations")
        return current_count

    async def prepare_page_for_extraction(self, page: Page) -> dict:
        """Prepare a SPA page for content extraction.

        This is the main orchestration method that combines all SPA handling.

        Args:
            page: The Playwright page.

        Returns:
            Dict with preparation results and statistics.
        """
        results = {
            "framework": None,
            "content_ready": False,
            "scroll_iterations": 0,
            "tabs_expanded": 0,
            "accordions_expanded": 0,
            "items_loaded": 0,
        }

        # Detect framework
        results["framework"] = await self.detect_framework(page)

        # Wait for initial content
        results["content_ready"] = await self.wait_for_content(page)

        # Handle lazy loading
        results["scroll_iterations"] = await self.handle_lazy_loading(page)

        # Expand tabs and accordions
        tabs = await self.expand_tabs(page)
        results["tabs_expanded"] = len(tabs)

        accordions = await self.expand_accordions(page)
        results["accordions_expanded"] = len(accordions)

        # Final wait for any animations
        await asyncio.sleep(0.5)

        logger.info(
            f"SPA preparation complete: framework={results['framework']}, "
            f"tabs={results['tabs_expanded']}, accordions={results['accordions_expanded']}"
        )

        return results
