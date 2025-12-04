"""Table content extractor for web pages."""

import logging
from typing import Optional

from playwright.async_api import Page

from src.acquisition.extractors.base import BaseExtractor
from src.models.config import ExtractionConfig
from src.models.content import TableData

logger = logging.getLogger(__name__)


class TableExtractor(BaseExtractor):
    """
    Extracts table data from web pages.

    Handles:
    - Standard HTML tables
    - Comparison/pricing tables
    - Feature matrices
    - Data grids
    - Tables with merged cells

    Important for sales pitches as tables often contain:
    - Pricing information
    - Feature comparisons
    - Plan details
    - Technical specifications
    """

    # Selectors to exclude (navigation tables, layout tables, etc.)
    EXCLUDE_PATTERNS = [
        "nav table",
        "table[role='presentation']",
        "table.layout",
        "[class*='menu'] table",
    ]

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize the table extractor.

        Args:
            config: Extraction configuration.
        """
        super().__init__(config)

    async def extract(self, page: Page) -> list[TableData]:
        """Extract all tables from the page.

        Args:
            page: The Playwright page to extract from.

        Returns:
            List of TableData objects.
        """
        logger.debug("Starting table extraction")

        # Build exclusion selector
        exclude_selector = ", ".join(self.EXCLUDE_PATTERNS)

        # Extract tables using JavaScript
        raw_tables = await self._extract_raw_tables(page, exclude_selector)

        # Process into TableData objects
        tables = []
        for raw in raw_tables:
            table = self._process_raw_table(raw)
            if table and self._is_meaningful_table(table):
                tables.append(table)

        logger.debug(f"Extracted {len(tables)} tables")
        return tables

    async def _extract_raw_tables(
        self,
        page: Page,
        exclude_selector: str,
    ) -> list[dict]:
        """Extract raw table data using JavaScript.

        Args:
            page: The Playwright page.
            exclude_selector: CSS selector for tables to exclude.

        Returns:
            List of raw table dictionaries.
        """
        extraction_script = f"""
            () => {{
                const excludeSelector = `{exclude_selector}`;
                const tables = [];

                // Helper to get selector
                function getSelector(el) {{
                    if (el.id) return '#' + el.id;
                    if (el.className && typeof el.className === 'string') {{
                        const classes = el.className.split(' ').filter(c => c).slice(0, 2);
                        if (classes.length) return el.tagName.toLowerCase() + '.' + classes.join('.');
                    }}
                    return el.tagName.toLowerCase();
                }}

                // Helper to check if should exclude
                function shouldExclude(table) {{
                    if (!excludeSelector) return false;
                    return table.matches(excludeSelector) || !!table.closest(excludeSelector);
                }}

                // Helper to get clean cell text
                function getCellText(cell) {{
                    // Remove hidden elements
                    const clone = cell.cloneNode(true);
                    clone.querySelectorAll('[hidden], .sr-only, .visually-hidden').forEach(el => el.remove());
                    return (clone.textContent || '').trim().replace(/\\s+/g, ' ');
                }}

                // Process each table
                document.querySelectorAll('table').forEach((table, tableIndex) => {{
                    if (shouldExclude(table)) return;

                    const tableData = {{
                        selector: getSelector(table),
                        caption: '',
                        headers: [],
                        rows: [],
                        className: table.className || '',
                    }};

                    // Get caption
                    const caption = table.querySelector('caption');
                    if (caption) {{
                        tableData.caption = caption.textContent?.trim() || '';
                    }}

                    // Get headers from thead or first row
                    const headerCells = table.querySelectorAll('thead th, thead td');
                    if (headerCells.length > 0) {{
                        tableData.headers = Array.from(headerCells).map(getCellText);
                    }} else {{
                        // Try first row if no thead
                        const firstRow = table.querySelector('tr');
                        if (firstRow) {{
                            const cells = firstRow.querySelectorAll('th, td');
                            const hasHeaders = firstRow.querySelector('th') !== null;
                            if (hasHeaders) {{
                                tableData.headers = Array.from(cells).map(getCellText);
                            }}
                        }}
                    }}

                    // Get body rows
                    const tbody = table.querySelector('tbody') || table;
                    const rows = tbody.querySelectorAll('tr');

                    rows.forEach((row, rowIndex) => {{
                        // Skip header row if it was already processed
                        if (rowIndex === 0 && tableData.headers.length > 0 && !table.querySelector('thead')) {{
                            return;
                        }}

                        const cells = row.querySelectorAll('td, th');
                        if (cells.length === 0) return;

                        const rowData = Array.from(cells).map(cell => {{
                            // Handle colspan/rowspan
                            const colspan = parseInt(cell.getAttribute('colspan') || '1');
                            const text = getCellText(cell);

                            // If colspan > 1, repeat the text
                            if (colspan > 1) {{
                                return Array(colspan).fill(text);
                            }}
                            return text;
                        }}).flat();

                        if (rowData.some(text => text.length > 0)) {{
                            tableData.rows.push(rowData);
                        }}
                    }});

                    // Only include tables with actual content
                    if (tableData.rows.length > 0 || tableData.headers.length > 0) {{
                        tables.push(tableData);
                    }}
                }});

                return tables;
            }}
        """

        try:
            return await page.evaluate(extraction_script)
        except Exception as e:
            logger.error(f"JavaScript table extraction failed: {e}")
            return []

    def _process_raw_table(self, raw: dict) -> Optional[TableData]:
        """Process raw table data into a TableData object.

        Args:
            raw: Raw table dictionary from JavaScript.

        Returns:
            TableData or None if invalid.
        """
        headers = raw.get("headers", [])
        rows = raw.get("rows", [])

        # Normalize row lengths to match header count
        if headers:
            max_cols = len(headers)
            rows = [
                row[:max_cols] + [""] * (max_cols - len(row))
                if len(row) != max_cols
                else row
                for row in rows
            ]
        elif rows:
            # Infer column count from longest row
            max_cols = max(len(row) for row in rows)
            rows = [
                row + [""] * (max_cols - len(row))
                for row in rows
            ]

        return TableData(
            headers=headers,
            rows=rows,
            caption=raw.get("caption", ""),
            source_selector=raw.get("selector"),
        )

    def _is_meaningful_table(self, table: TableData) -> bool:
        """Check if a table contains meaningful content.

        Filters out:
        - Empty tables
        - Tables with only 1 cell
        - Tables that are clearly layout/navigation

        Args:
            table: The TableData to check.

        Returns:
            True if the table has meaningful content.
        """
        # Must have either headers or rows
        if not table.headers and not table.rows:
            return False

        # Must have more than 1 column
        col_count = len(table.headers) if table.headers else (
            len(table.rows[0]) if table.rows else 0
        )
        if col_count < 2:
            return False

        # Must have at least 1 data row (excluding headers)
        if not table.rows:
            return False

        # Check that cells actually contain text
        total_text = 0
        for row in table.rows:
            for cell in row:
                total_text += len(cell)

        # At least 20 characters of actual content
        if total_text < 20:
            return False

        return True

    async def extract_comparison_tables(
        self,
        page: Page,
    ) -> list[TableData]:
        """Extract tables that appear to be comparison/pricing tables.

        These are typically more valuable for sales pitches.

        Args:
            page: The Playwright page.

        Returns:
            List of comparison-style TableData objects.
        """
        all_tables = await self.extract(page)

        comparison_indicators = [
            "compare",
            "comparison",
            "pricing",
            "plan",
            "tier",
            "feature",
            "basic",
            "pro",
            "enterprise",
            "free",
            "premium",
            "standard",
        ]

        comparison_tables = []
        for table in all_tables:
            # Check caption
            caption_lower = table.caption.lower() if table.caption else ""

            # Check headers
            headers_lower = " ".join(table.headers).lower() if table.headers else ""

            # Check if it looks like a comparison table
            combined = caption_lower + " " + headers_lower
            if any(indicator in combined for indicator in comparison_indicators):
                comparison_tables.append(table)
                continue

            # Also check for tables with checkmarks/crosses (common in feature comparisons)
            checkmark_indicators = ["✓", "✗", "✔", "✘", "yes", "no", "included", "×"]
            row_text = " ".join(
                cell.lower() for row in table.rows for cell in row
            )
            if any(indicator in row_text for indicator in checkmark_indicators):
                comparison_tables.append(table)

        return comparison_tables

    def table_to_markdown(self, table: TableData) -> str:
        """Convert a TableData to Markdown format.

        Args:
            table: The TableData to convert.

        Returns:
            Markdown string representation.
        """
        lines = []

        # Add caption if present
        if table.caption:
            lines.append(f"**{table.caption}**")
            lines.append("")

        # Add headers
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        elif table.rows:
            # No headers, create placeholder
            col_count = len(table.rows[0])
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

        # Add rows
        for row in table.rows:
            # Escape pipe characters in cells
            escaped_row = [cell.replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(lines)
