"""Visual inventory builder for processing extracted visual assets."""

import hashlib
import re
from typing import Optional

from ..models.content import ExtractedContent, ImageAsset, TableData, VideoAsset
from ..models.processed import VisualAssetReference, VisualInventory


class VisualInventoryBuilder:
    """
    Builds a visual inventory from extracted content.

    Converts raw ImageAsset, TableData, and VideoAsset objects into
    VisualAssetReference objects with classifications and metadata
    suitable for matching to pitch sections.
    """

    # Keywords for detecting comparison tables
    COMPARISON_KEYWORDS = [
        "compare",
        "comparison",
        "vs",
        "versus",
        "differences",
        "feature",
        "plan",
        "tier",
    ]

    # Keywords for detecting pricing tables
    PRICING_KEYWORDS = [
        "price",
        "pricing",
        "cost",
        "plan",
        "tier",
        "monthly",
        "annually",
        "year",
        "free",
        "pro",
        "enterprise",
        "basic",
        "premium",
        "starter",
        "$",
        "€",
        "£",
    ]

    # Patterns that indicate a pricing value
    PRICE_PATTERNS = [
        r"\$\d+",
        r"€\d+",
        r"£\d+",
        r"\d+/mo",
        r"\d+/month",
        r"\d+/year",
        r"free",
        r"contact",
    ]

    def build_inventory(self, content: ExtractedContent) -> VisualInventory:
        """
        Build a complete visual inventory from extracted content.

        Args:
            content: ExtractedContent with images, tables, and videos

        Returns:
            VisualInventory with all assets categorized
        """
        inventory = VisualInventory()

        # Process images
        for image in content.all_images:
            ref = self._process_image(image)
            if ref:
                inventory.images.append(ref)
                # Categorize by type
                if ref.is_screenshot:
                    inventory.screenshots.append(ref.asset_id)
                if ref.is_diagram:
                    inventory.diagrams.append(ref.asset_id)
                if ref.is_logo:
                    inventory.logos.append(ref.asset_id)

        # Process tables
        for table in content.all_tables:
            ref = self._process_table(table)
            if ref:
                inventory.tables.append(ref)
                # Categorize by type
                if ref.is_comparison_table:
                    inventory.comparison_tables.append(ref.asset_id)
                if ref.is_pricing_table:
                    inventory.pricing_tables.append(ref.asset_id)

        # Process videos
        for video in content.all_videos:
            ref = self._process_video(video)
            if ref:
                inventory.videos.append(ref)

        return inventory

    def _process_image(self, image: ImageAsset) -> Optional[VisualAssetReference]:
        """
        Convert an ImageAsset to a VisualAssetReference.

        Args:
            image: Raw image asset from extraction

        Returns:
            VisualAssetReference or None if image should be excluded
        """
        # Skip tracking pixels and very small images (unless they're logos/icons)
        if not image.is_logo and not image.is_icon:
            if image.width and image.height:
                if image.width < 50 or image.height < 50:
                    return None

        # Generate asset ID from URL
        asset_id = self._generate_asset_id(image.url)

        return VisualAssetReference(
            asset_id=asset_id,
            asset_type="image",
            url=image.url,
            local_path=image.local_path,
            alt_text=image.alt_text,
            caption=image.caption,
            title=image.title,
            is_logo=image.is_logo,
            is_screenshot=image.is_screenshot,
            is_diagram=image.is_diagram,
            is_icon=image.is_icon,
            width=image.width,
            height=image.height,
        )

    def _process_table(self, table: TableData) -> Optional[VisualAssetReference]:
        """
        Convert a TableData to a VisualAssetReference.

        Args:
            table: Raw table data from extraction

        Returns:
            VisualAssetReference or None if table should be excluded
        """
        # Skip empty tables
        if not table.rows or (not table.headers and len(table.rows) < 2):
            return None

        # Generate asset ID from table content
        table_content = str(table.headers) + str(table.rows[:3])
        asset_id = self._generate_asset_id(table_content)

        # Detect table type
        is_comparison = self._detect_comparison_table(table)
        is_pricing = self._detect_pricing_table(table)

        # Generate markdown representation
        table_markdown = self._table_to_markdown(table)

        # Build caption from table content if not provided
        caption = table.caption
        if not caption and table.headers:
            caption = f"Table: {', '.join(table.headers[:3])}"
            if len(table.headers) > 3:
                caption += "..."

        return VisualAssetReference(
            asset_id=asset_id,
            asset_type="table",
            url="",  # Tables don't have URLs
            caption=caption,
            table_headers=table.headers,
            table_row_count=len(table.rows),
            table_markdown=table_markdown,
            is_comparison_table=is_comparison,
            is_pricing_table=is_pricing,
        )

    def _process_video(self, video: VideoAsset) -> Optional[VisualAssetReference]:
        """
        Convert a VideoAsset to a VisualAssetReference.

        Args:
            video: Raw video asset from extraction

        Returns:
            VisualAssetReference or None if video should be excluded
        """
        # Generate asset ID from URL
        asset_id = self._generate_asset_id(video.url)

        return VisualAssetReference(
            asset_id=asset_id,
            asset_type="video",
            url=video.embed_url or video.url,
            local_path=video.local_path,
            alt_text=video.alt_text,
            caption=video.caption,
            title=video.title,
            video_platform=video.platform,
            video_duration=video.duration_seconds,
            thumbnail_url=video.thumbnail_url,
        )

    def _detect_comparison_table(self, table: TableData) -> bool:
        """
        Detect if a table is a comparison/feature matrix table.

        Args:
            table: Table data to analyze

        Returns:
            True if table appears to be a comparison table
        """
        # Check caption for comparison keywords
        if table.caption:
            caption_lower = table.caption.lower()
            if any(kw in caption_lower for kw in self.COMPARISON_KEYWORDS):
                return True

        # Check headers for comparison keywords
        headers_text = " ".join(table.headers).lower()
        if any(kw in headers_text for kw in self.COMPARISON_KEYWORDS):
            return True

        # Look for check marks, x marks, yes/no patterns in cells
        comparison_indicators = ["✓", "✗", "✔", "✘", "yes", "no", "included", "•"]
        indicator_count = 0
        total_cells = 0

        for row in table.rows[:5]:  # Check first 5 rows
            for cell in row:
                total_cells += 1
                cell_lower = cell.lower().strip()
                if cell_lower in comparison_indicators or cell_lower in ["x", "-"]:
                    indicator_count += 1

        # If more than 30% of cells are comparison indicators
        if total_cells > 0 and indicator_count / total_cells > 0.3:
            return True

        return False

    def _detect_pricing_table(self, table: TableData) -> bool:
        """
        Detect if a table is a pricing table.

        Args:
            table: Table data to analyze

        Returns:
            True if table appears to be a pricing table
        """
        # Check caption for pricing keywords
        if table.caption:
            caption_lower = table.caption.lower()
            if any(kw in caption_lower for kw in self.PRICING_KEYWORDS):
                return True

        # Check headers for pricing keywords
        headers_text = " ".join(table.headers).lower()
        if any(kw in headers_text for kw in self.PRICING_KEYWORDS):
            return True

        # Look for price patterns in cells
        all_text = " ".join(
            cell for row in table.rows for cell in row
        ).lower()

        price_pattern_count = 0
        for pattern in self.PRICE_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                price_pattern_count += 1

        # If multiple price patterns found, likely a pricing table
        if price_pattern_count >= 2:
            return True

        return False

    def _table_to_markdown(self, table: TableData) -> str:
        """
        Convert a TableData to markdown format.

        Args:
            table: Table data to convert

        Returns:
            Markdown string representation of the table
        """
        lines = []

        # Headers
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        elif table.rows:
            # Use first row as header if no headers
            first_row = table.rows[0]
            lines.append("| " + " | ".join(first_row) + " |")
            lines.append("| " + " | ".join(["---"] * len(first_row)) + " |")
            table_rows = table.rows[1:]
        else:
            return ""

        # Data rows
        table_rows = table.rows if table.headers else table.rows[1:]
        for row in table_rows:
            # Ensure row has same number of columns as headers
            if table.headers:
                while len(row) < len(table.headers):
                    row.append("")
                row = row[: len(table.headers)]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _generate_asset_id(self, content: str) -> str:
        """
        Generate a unique asset ID from content.

        Args:
            content: Content to hash (URL or other unique data)

        Returns:
            Short hash string as asset ID
        """
        return hashlib.md5(content.encode()).hexdigest()[:12]
