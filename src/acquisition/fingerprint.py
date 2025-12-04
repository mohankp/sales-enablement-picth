"""Content fingerprinting for change detection."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from src.models.content import ContentBlock, ExtractedContent, PageContent

logger = logging.getLogger(__name__)


class ContentFingerprinter:
    """
    Creates fingerprints (hashes) for content to detect changes.

    This is crucial for the incremental update feature - we need to know
    what has changed on the product website since the last extraction.

    Fingerprinting strategy:
    1. Page-level hash: Quick check if any content changed
    2. Section-level hashes: Identify which sections changed
    3. Semantic fingerprints: Content meaning-based comparison

    This allows us to:
    - Skip re-extraction if nothing changed
    - Identify new content to add to the pitch
    - Identify removed content that may need cleanup
    """

    @staticmethod
    def hash_text(text: str) -> str:
        """Create a hash of text content.

        Args:
            text: Text to hash.

        Returns:
            SHA256 hash string.
        """
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def hash_content_block(block: ContentBlock) -> str:
        """Create a hash for a content block.

        Args:
            block: The ContentBlock to hash.

        Returns:
            Hash string.
        """
        content = f"{block.content_type}:{block.text}"
        if block.list_items:
            content += ":" + "|".join(block.list_items)
        return ContentFingerprinter.hash_text(content)

    @classmethod
    def fingerprint_page(cls, page: PageContent) -> str:
        """Create a fingerprint for an entire page.

        Args:
            page: The PageContent to fingerprint.

        Returns:
            Combined hash string.
        """
        components = [
            page.title or "",
            page.meta_description or "",
        ]

        # Add all content block hashes
        for block in page.content_blocks:
            components.append(cls.hash_content_block(block))

        # Add table hashes
        for table in page.tables:
            table_str = json.dumps({"h": table.headers, "r": table.rows})
            components.append(cls.hash_text(table_str))

        combined = "|".join(components)
        page_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]

        # Update the page object
        page.content_hash = page_hash

        return page_hash

    @classmethod
    def fingerprint_extraction(cls, extraction: ExtractedContent) -> str:
        """Create a fingerprint for an entire extraction.

        Args:
            extraction: The ExtractedContent to fingerprint.

        Returns:
            Combined hash string.
        """
        page_hashes = []

        for page in extraction.pages:
            if not page.content_hash:
                cls.fingerprint_page(page)
            page_hashes.append(f"{page.url}:{page.content_hash}")

        combined = "\n".join(sorted(page_hashes))
        extraction_hash = hashlib.sha256(combined.encode()).hexdigest()

        extraction.content_hash = extraction_hash
        return extraction_hash

    @classmethod
    def compare_extractions(
        cls,
        old: ExtractedContent,
        new: ExtractedContent,
    ) -> dict:
        """Compare two extractions and identify changes.

        Args:
            old: Previous extraction.
            new: New extraction.

        Returns:
            Dict with 'added', 'removed', 'modified', 'unchanged' page lists.
        """
        result = {
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": [],
            "has_changes": False,
        }

        # Build lookup of old pages by URL
        old_pages = {p.url: p for p in old.pages}
        new_pages = {p.url: p for p in new.pages}

        old_urls = set(old_pages.keys())
        new_urls = set(new_pages.keys())

        # Find added pages
        result["added"] = list(new_urls - old_urls)

        # Find removed pages
        result["removed"] = list(old_urls - new_urls)

        # Compare common pages
        common_urls = old_urls & new_urls
        for url in common_urls:
            old_page = old_pages[url]
            new_page = new_pages[url]

            # Ensure fingerprints exist
            if not old_page.content_hash:
                cls.fingerprint_page(old_page)
            if not new_page.content_hash:
                cls.fingerprint_page(new_page)

            if old_page.content_hash != new_page.content_hash:
                result["modified"].append(url)
            else:
                result["unchanged"].append(url)

        result["has_changes"] = bool(
            result["added"] or result["removed"] or result["modified"]
        )

        return result

    @classmethod
    def get_content_diff(
        cls,
        old_page: PageContent,
        new_page: PageContent,
    ) -> dict:
        """Get detailed diff between two versions of a page.

        Args:
            old_page: Previous version.
            new_page: New version.

        Returns:
            Dict with detailed change information.
        """
        diff = {
            "url": new_page.url,
            "title_changed": old_page.title != new_page.title,
            "new_blocks": [],
            "removed_blocks": [],
            "modified_blocks": [],
        }

        # Build block hash -> block mapping
        old_block_hashes = {
            cls.hash_content_block(b): b for b in old_page.content_blocks
        }
        new_block_hashes = {
            cls.hash_content_block(b): b for b in new_page.content_blocks
        }

        old_hashes = set(old_block_hashes.keys())
        new_hashes = set(new_block_hashes.keys())

        # New blocks
        for hash_val in new_hashes - old_hashes:
            block = new_block_hashes[hash_val]
            diff["new_blocks"].append({
                "type": block.content_type,
                "text": block.text[:200] + "..." if len(block.text) > 200 else block.text,
                "section": block.parent_section,
            })

        # Removed blocks
        for hash_val in old_hashes - new_hashes:
            block = old_block_hashes[hash_val]
            diff["removed_blocks"].append({
                "type": block.content_type,
                "text": block.text[:200] + "..." if len(block.text) > 200 else block.text,
                "section": block.parent_section,
            })

        return diff


class ExtractionStore:
    """
    Simple storage for extraction history with fingerprint comparison.

    In a production system, this would be backed by a proper database.
    For now, it uses JSON files for simplicity.
    """

    def __init__(self, storage_path: str = "data/extractions"):
        """Initialize the extraction store.

        Args:
            storage_path: Directory to store extraction data.
        """
        self.storage_path = storage_path

    def save_extraction(
        self,
        extraction: ExtractedContent,
        product_id: str,
    ) -> str:
        """Save an extraction to storage.

        Args:
            extraction: The extraction to save.
            product_id: Identifier for the product.

        Returns:
            Extraction ID.
        """
        import os

        # Generate extraction ID
        extraction_id = f"{product_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        extraction.extraction_id = extraction_id

        # Ensure fingerprint
        ContentFingerprinter.fingerprint_extraction(extraction)

        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)

        # Save as JSON
        filepath = os.path.join(self.storage_path, f"{extraction_id}.json")
        with open(filepath, "w") as f:
            json.dump(extraction.model_dump(), f, default=str, indent=2)

        logger.info(f"Saved extraction: {extraction_id}")
        return extraction_id

    def load_extraction(self, extraction_id: str) -> Optional[ExtractedContent]:
        """Load an extraction from storage.

        Args:
            extraction_id: The extraction ID.

        Returns:
            ExtractedContent or None if not found.
        """
        import os

        filepath = os.path.join(self.storage_path, f"{extraction_id}.json")

        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        return ExtractedContent(**data)

    def get_latest_extraction(self, product_id: str) -> Optional[ExtractedContent]:
        """Get the most recent extraction for a product.

        Args:
            product_id: The product identifier.

        Returns:
            Most recent ExtractedContent or None.
        """
        import os

        if not os.path.exists(self.storage_path):
            return None

        # Find all extractions for this product
        pattern = f"{product_id}_"
        matching_files = [
            f for f in os.listdir(self.storage_path)
            if f.startswith(pattern) and f.endswith(".json")
        ]

        if not matching_files:
            return None

        # Get most recent
        latest = sorted(matching_files)[-1]
        extraction_id = latest.replace(".json", "")

        return self.load_extraction(extraction_id)
