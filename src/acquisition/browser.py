"""Browser management for Playwright-based web scraping."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from src.models.config import BrowserConfig

logger = logging.getLogger(__name__)


# Stealth mode JavaScript to avoid bot detection
STEALTH_JS = """
() => {
    // Overwrite the 'webdriver' property
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
    });

    // Overwrite the 'plugins' property to appear more realistic
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
    });

    // Overwrite the 'languages' property
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
    });

    // Override chrome runtime
    window.chrome = {
        runtime: {},
    };

    // Override permissions
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );

    // Prevent detection via iframe contentWindow
    try {
        Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
            get: function() {
                return window;
            }
        });
    } catch (e) {}
}
"""


class BrowserManager:
    """
    Manages Playwright browser instances with proper lifecycle handling.

    Features:
    - Configurable browser options
    - Stealth mode for anti-bot detection
    - Resource blocking for faster loading
    - Connection pooling for multiple pages
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        """Initialize the browser manager.

        Args:
            config: Browser configuration. Uses defaults if not provided.
        """
        self.config = config or BrowserConfig()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._contexts: list[BrowserContext] = []

    async def start(self) -> None:
        """Start the Playwright instance and launch browser."""
        if self._playwright is not None:
            logger.warning("Browser manager already started")
            return

        logger.info(f"Starting {self.config.browser_type} browser (headless={self.config.headless})")
        self._playwright = await async_playwright().start()

        # Select browser type
        browser_type = getattr(self._playwright, self.config.browser_type)

        # Prepare launch options
        launch_options = {
            "headless": self.config.headless,
            "slow_mo": self.config.slow_mo,
        }

        # Add proxy if configured
        if self.config.proxy_server:
            launch_options["proxy"] = {
                "server": self.config.proxy_server,
            }
            if self.config.proxy_username:
                launch_options["proxy"]["username"] = self.config.proxy_username
                launch_options["proxy"]["password"] = self.config.proxy_password

        self._browser = await browser_type.launch(**launch_options)
        logger.info("Browser launched successfully")

    async def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        logger.info("Stopping browser manager")

        # Close all contexts
        for context in self._contexts:
            try:
                await context.close()
            except Exception as e:
                logger.warning(f"Error closing context: {e}")
        self._contexts.clear()

        # Close browser
        if self._browser:
            await self._browser.close()
            self._browser = None

        # Stop playwright
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser manager stopped")

    async def create_context(self) -> BrowserContext:
        """Create a new browser context with configured options.

        Returns:
            A new BrowserContext instance.

        Raises:
            RuntimeError: If browser is not started.
        """
        if not self._browser:
            raise RuntimeError("Browser not started. Call start() first.")

        context_options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "device_scale_factor": self.config.device_scale_factor,
            "is_mobile": self.config.is_mobile,
            "locale": self.config.locale,
            "timezone_id": self.config.timezone_id,
            "ignore_https_errors": self.config.ignore_https_errors,
            "java_script_enabled": self.config.java_script_enabled,
            "bypass_csp": self.config.bypass_csp,
            "accept_downloads": self.config.accept_downloads,
        }

        if self.config.user_agent:
            context_options["user_agent"] = self.config.user_agent

        if self.config.extra_http_headers:
            context_options["extra_http_headers"] = self.config.extra_http_headers

        if self.config.storage_state:
            context_options["storage_state"] = self.config.storage_state

        context = await self._browser.new_context(**context_options)

        # Apply stealth mode if enabled
        if self.config.stealth_mode:
            await context.add_init_script(STEALTH_JS)

        # Set up resource blocking if configured
        if self.config.block_resources:
            await self._setup_resource_blocking(context)

        self._contexts.append(context)
        return context

    async def _setup_resource_blocking(self, context: BrowserContext) -> None:
        """Set up resource blocking for faster page loads.

        Args:
            context: The browser context to configure.
        """
        blocked_types = set(self.config.block_resources)

        async def route_handler(route):
            if route.request.resource_type in blocked_types:
                await route.abort()
            else:
                await route.continue_()

        await context.route("**/*", route_handler)

    async def create_page(self, context: Optional[BrowserContext] = None) -> Page:
        """Create a new page in the given or a new context.

        Args:
            context: Optional context to create page in. Creates new if None.

        Returns:
            A new Page instance.
        """
        if context is None:
            context = await self.create_context()

        page = await context.new_page()

        # Set default timeouts
        page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        page.set_default_timeout(self.config.default_timeout_ms)

        return page

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        """Context manager for getting a page with automatic cleanup.

        Yields:
            A Page instance that will be closed on exit.

        Example:
            async with browser_manager.get_page() as page:
                await page.goto("https://example.com")
                content = await page.content()
        """
        context = await self.create_context()
        page = await context.new_page()

        # Set default timeouts
        page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        page.set_default_timeout(self.config.default_timeout_ms)

        try:
            yield page
        finally:
            await page.close()
            await context.close()
            if context in self._contexts:
                self._contexts.remove(context)

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


class PageInteractionHelper:
    """Helper class for common page interactions."""

    def __init__(self, page: Page):
        """Initialize with a page instance.

        Args:
            page: Playwright Page instance.
        """
        self.page = page

    async def safe_click(
        self,
        selector: str,
        timeout_ms: int = 5000,
        ignore_not_found: bool = True,
    ) -> bool:
        """Safely click an element, handling common errors.

        Args:
            selector: CSS selector for the element.
            timeout_ms: Timeout for finding the element.
            ignore_not_found: If True, don't raise error if element not found.

        Returns:
            True if click was successful, False otherwise.
        """
        try:
            await self.page.click(selector, timeout=timeout_ms)
            return True
        except Exception as e:
            if ignore_not_found:
                logger.debug(f"Element not found or not clickable: {selector}")
                return False
            raise

    async def safe_fill(
        self,
        selector: str,
        value: str,
        timeout_ms: int = 5000,
    ) -> bool:
        """Safely fill an input field.

        Args:
            selector: CSS selector for the input.
            value: Value to fill.
            timeout_ms: Timeout for finding the element.

        Returns:
            True if fill was successful, False otherwise.
        """
        try:
            await self.page.fill(selector, value, timeout=timeout_ms)
            return True
        except Exception as e:
            logger.debug(f"Could not fill element: {selector} - {e}")
            return False

    async def wait_for_content(
        self,
        selector: str,
        timeout_ms: int = 10000,
    ) -> bool:
        """Wait for specific content to appear.

        Args:
            selector: CSS selector for the content.
            timeout_ms: Maximum time to wait.

        Returns:
            True if content appeared, False if timeout.
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout_ms)
            return True
        except Exception:
            return False

    async def scroll_to_bottom(
        self,
        pause_ms: int = 500,
        max_iterations: int = 10,
    ) -> int:
        """Scroll to the bottom of the page, loading lazy content.

        Args:
            pause_ms: Pause between scrolls for content to load.
            max_iterations: Maximum scroll iterations.

        Returns:
            Number of scroll iterations performed.
        """
        iterations = 0
        last_height = 0

        for i in range(max_iterations):
            # Get current scroll height
            current_height = await self.page.evaluate("document.body.scrollHeight")

            if current_height == last_height:
                # No more content to load
                break

            # Scroll to bottom
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(pause_ms / 1000)

            last_height = current_height
            iterations += 1

        # Scroll back to top
        await self.page.evaluate("window.scrollTo(0, 0)")

        logger.debug(f"Scrolled {iterations} times to load content")
        return iterations

    async def dismiss_overlays(
        self,
        selectors: list[str],
        button_texts: list[str],
    ) -> int:
        """Attempt to dismiss cookie banners and other overlays.

        Args:
            selectors: CSS selectors for potential dismiss buttons.
            button_texts: Text patterns for accept/dismiss buttons.

        Returns:
            Number of overlays dismissed.
        """
        dismissed = 0

        for selector in selectors:
            try:
                buttons = await self.page.query_selector_all(selector)
                for button in buttons:
                    text = await button.text_content()
                    if text:
                        text_lower = text.lower().strip()
                        if any(btn_text in text_lower for btn_text in button_texts):
                            await button.click()
                            dismissed += 1
                            await asyncio.sleep(0.3)  # Wait for overlay to close
                            break
            except Exception as e:
                logger.debug(f"Could not dismiss overlay with {selector}: {e}")

        if dismissed > 0:
            logger.info(f"Dismissed {dismissed} overlay(s)")

        return dismissed

    async def extract_element_text(self, selector: str) -> Optional[str]:
        """Extract text content from an element.

        Args:
            selector: CSS selector for the element.

        Returns:
            Text content or None if not found.
        """
        try:
            element = await self.page.query_selector(selector)
            if element:
                return await element.text_content()
        except Exception as e:
            logger.debug(f"Could not extract text from {selector}: {e}")
        return None

    async def get_all_links(self) -> list[dict[str, str]]:
        """Extract all links from the page.

        Returns:
            List of dicts with 'url' and 'text' keys.
        """
        links = await self.page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    links.push({
                        url: a.href,
                        text: a.textContent?.trim() || ''
                    });
                });
                return links;
            }
        """)
        return links

    async def take_screenshot(self, path: str, full_page: bool = True) -> str:
        """Take a screenshot of the page.

        Args:
            path: Path to save the screenshot.
            full_page: Whether to capture the full page.

        Returns:
            Path to the saved screenshot.
        """
        await self.page.screenshot(path=path, full_page=full_page)
        return path
