"""Base class for content extractors."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from playwright.async_api import Page

from src.models.config import ExtractionConfig

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for content extractors.

    All extractors should inherit from this class and implement
    the extract method.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize the extractor.

        Args:
            config: Extraction configuration.
        """
        self.config = config or ExtractionConfig()

    @abstractmethod
    async def extract(self, page: Page) -> Any:
        """Extract content from the page.

        Args:
            page: The Playwright page to extract from.

        Returns:
            Extracted content (type depends on extractor).
        """
        pass

    async def _safe_evaluate(
        self,
        page: Page,
        script: str,
        default: Any = None,
    ) -> Any:
        """Safely evaluate JavaScript on the page.

        Args:
            page: The Playwright page.
            script: JavaScript to evaluate.
            default: Default value if evaluation fails.

        Returns:
            Evaluation result or default value.
        """
        try:
            return await page.evaluate(script)
        except Exception as e:
            logger.debug(f"JavaScript evaluation failed: {e}")
            return default

    async def _get_element_text(
        self,
        page: Page,
        selector: str,
    ) -> Optional[str]:
        """Get text content from an element.

        Args:
            page: The Playwright page.
            selector: CSS selector for the element.

        Returns:
            Text content or None if not found.
        """
        try:
            element = await page.query_selector(selector)
            if element:
                return await element.text_content()
        except Exception as e:
            logger.debug(f"Could not get text from {selector}: {e}")
        return None

    async def _get_elements(self, page: Page, selector: str) -> list:
        """Get all elements matching a selector.

        Args:
            page: The Playwright page.
            selector: CSS selector.

        Returns:
            List of matching elements.
        """
        try:
            return await page.query_selector_all(selector)
        except Exception as e:
            logger.debug(f"Could not get elements for {selector}: {e}")
            return []
