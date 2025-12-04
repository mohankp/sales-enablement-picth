"""Content chunking strategies for handling large documents."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.models.content import ContentBlock, ContentType, ExtractedContent, PageContent

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"  # Fixed token/character count
    SEMANTIC = "semantic"  # Based on content structure (headings, sections)
    PAGE_BASED = "page_based"  # One chunk per page
    HYBRID = "hybrid"  # Combination of semantic with size limits


@dataclass
class ContentChunk:
    """A chunk of content ready for LLM processing."""

    content: str
    chunk_index: int
    total_chunks: int
    source_urls: list[str] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    section_headers: list[str] = field(default_factory=list)
    char_count: int = 0
    estimated_tokens: int = 0
    chunk_hash: str = ""

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.char_count = len(self.content)
        # Rough token estimate: ~4 chars per token for English
        self.estimated_tokens = self.char_count // 4
        self.chunk_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]


class ContentChunker:
    """
    Chunks extracted content for LLM processing.

    Handles large documents that exceed context windows by intelligently
    splitting content while preserving semantic structure.
    """

    # Default limits (conservative to leave room for prompts and responses)
    DEFAULT_MAX_CHARS = 50_000  # ~12,500 tokens
    DEFAULT_OVERLAP_CHARS = 500  # Overlap between chunks for context
    DEFAULT_MIN_CHUNK_CHARS = 1_000  # Minimum meaningful chunk size

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        self.strategy = strategy
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_extracted_content(
        self,
        content: ExtractedContent,
    ) -> list[ContentChunk]:
        """
        Chunk extracted content into processable pieces.

        Args:
            content: The extracted content to chunk

        Returns:
            List of content chunks
        """
        if self.strategy == ChunkingStrategy.PAGE_BASED:
            return self._chunk_by_page(content)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(content)
        elif self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(content)
        else:  # HYBRID
            return self._chunk_hybrid(content)

    def chunk_text(self, text: str, source_url: str = "") -> list[ContentChunk]:
        """
        Chunk plain text content.

        Args:
            text: Text to chunk
            source_url: Source URL for attribution

        Returns:
            List of content chunks
        """
        if len(text) <= self.max_chars:
            return [
                ContentChunk(
                    content=text,
                    chunk_index=0,
                    total_chunks=1,
                    source_urls=[source_url] if source_url else [],
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.max_chars

            # Find a good break point (end of sentence or paragraph)
            if end < len(text):
                end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    ContentChunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will update after
                        source_urls=[source_url] if source_url else [],
                    )
                )
                chunk_index += 1

            # Move start with overlap
            start = end - self.overlap_chars if end < len(text) else end

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_by_page(self, content: ExtractedContent) -> list[ContentChunk]:
        """Create one chunk per page."""
        chunks = []

        for i, page in enumerate(content.pages):
            page_text = self._page_to_text(page)

            # If page is too large, sub-chunk it
            if len(page_text) > self.max_chars:
                sub_chunks = self.chunk_text(page_text, page.url)
                for sub_chunk in sub_chunks:
                    sub_chunk.source_urls = [page.url]
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    ContentChunk(
                        content=page_text,
                        chunk_index=i,
                        total_chunks=len(content.pages),
                        source_urls=[page.url],
                        section_headers=self._extract_headers(page),
                    )
                )

        # Renumber if we had sub-chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_semantic(self, content: ExtractedContent) -> list[ContentChunk]:
        """Chunk based on semantic structure (sections, headings)."""
        # Gather all content organized by section
        sections: list[dict] = []
        current_section: dict = {
            "header": "Introduction",
            "content": [],
            "urls": set(),
        }

        for page in content.pages:
            for block in page.content_blocks:
                if block.content_type == ContentType.HEADING and block.heading_level in (1, 2):
                    # Start new section
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {
                        "header": block.text,
                        "content": [],
                        "urls": {page.url},
                    }
                else:
                    current_section["content"].append(self._block_to_text(block))
                    current_section["urls"].add(page.url)

        # Add final section
        if current_section["content"]:
            sections.append(current_section)

        # Build chunks from sections
        chunks = []
        current_chunk_text = ""
        current_headers: list[str] = []
        current_urls: set[str] = set()

        for section in sections:
            section_text = f"\n\n## {section['header']}\n\n" + "\n\n".join(section["content"])

            if len(current_chunk_text) + len(section_text) > self.max_chars:
                # Save current chunk
                if current_chunk_text:
                    chunks.append(
                        ContentChunk(
                            content=current_chunk_text.strip(),
                            chunk_index=len(chunks),
                            total_chunks=0,
                            source_urls=list(current_urls),
                            section_headers=current_headers,
                        )
                    )
                current_chunk_text = section_text
                current_headers = [section["header"]]
                current_urls = set(section["urls"])
            else:
                current_chunk_text += section_text
                current_headers.append(section["header"])
                current_urls.update(section["urls"])

        # Add final chunk
        if current_chunk_text:
            chunks.append(
                ContentChunk(
                    content=current_chunk_text.strip(),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    source_urls=list(current_urls),
                    section_headers=current_headers,
                )
            )

        # Update totals
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_fixed_size(self, content: ExtractedContent) -> list[ContentChunk]:
        """Simple fixed-size chunking."""
        full_text = self._content_to_text(content)
        all_urls = [page.url for page in content.pages]

        chunks = self.chunk_text(full_text)
        for chunk in chunks:
            chunk.source_urls = all_urls

        return chunks

    def _chunk_hybrid(self, content: ExtractedContent) -> list[ContentChunk]:
        """
        Hybrid chunking: semantic structure with size limits.

        Tries to preserve semantic structure but enforces size limits.
        """
        # First try semantic chunking
        semantic_chunks = self._chunk_semantic(content)

        # Check if any chunks exceed limits and split them
        final_chunks = []
        for chunk in semantic_chunks:
            if chunk.char_count > self.max_chars:
                # Split oversized chunk
                sub_chunks = self.chunk_text(chunk.content, chunk.source_urls[0] if chunk.source_urls else "")
                for sub_chunk in sub_chunks:
                    sub_chunk.source_urls = chunk.source_urls
                    sub_chunk.section_headers = chunk.section_headers
                final_chunks.extend(sub_chunks)
            elif chunk.char_count >= self.min_chunk_chars:
                final_chunks.append(chunk)
            else:
                # Try to merge small chunks
                if final_chunks and final_chunks[-1].char_count + chunk.char_count < self.max_chars:
                    final_chunks[-1].content += "\n\n" + chunk.content
                    final_chunks[-1].source_urls.extend(chunk.source_urls)
                    final_chunks[-1].section_headers.extend(chunk.section_headers)
                    final_chunks[-1].char_count = len(final_chunks[-1].content)
                else:
                    final_chunks.append(chunk)

        # Renumber
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(final_chunks)

        return final_chunks

    def _page_to_text(self, page: PageContent) -> str:
        """Convert a page to plain text."""
        parts = []

        if page.title:
            parts.append(f"# {page.title}")

        if page.meta_description:
            parts.append(f"*{page.meta_description}*")

        for block in page.content_blocks:
            text = self._block_to_text(block)
            if text:
                parts.append(text)

        return "\n\n".join(parts)

    def _block_to_text(self, block: ContentBlock) -> str:
        """Convert a content block to text."""
        if block.content_type == ContentType.HEADING:
            level = block.heading_level or 2
            prefix = "#" * min(level, 6)
            return f"{prefix} {block.text}"

        if block.content_type == ContentType.LIST:
            items = block.list_items or [block.text]
            return "\n".join(f"- {item}" for item in items)

        if block.content_type == ContentType.TABLE and block.table_data:
            return self._table_to_text(block.table_data)

        if block.content_type == ContentType.CODE:
            return f"```\n{block.text}\n```"

        if block.content_type == ContentType.QUOTE:
            return f"> {block.text}"

        return block.text

    def _table_to_text(self, table) -> str:
        """Convert table data to markdown text."""
        lines = []

        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")

        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _content_to_text(self, content: ExtractedContent) -> str:
        """Convert full extracted content to text."""
        parts = []

        if content.product_name:
            parts.append(f"# {content.product_name}")

        for page in content.pages:
            parts.append(self._page_to_text(page))

        return "\n\n---\n\n".join(parts)

    def _extract_headers(self, page: PageContent) -> list[str]:
        """Extract section headers from a page."""
        headers = []
        for block in page.content_blocks:
            if block.content_type == ContentType.HEADING:
                headers.append(block.text)
        return headers

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point near the end position."""
        # Try to break at paragraph boundary
        para_break = text.rfind("\n\n", start, end)
        if para_break > start + self.min_chunk_chars:
            return para_break + 2

        # Try sentence boundary
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        best_break = end

        for ending in sentence_endings:
            pos = text.rfind(ending, start + self.min_chunk_chars, end)
            if pos > 0:
                best_break = min(best_break, pos + len(ending))
                break

        # Fallback to word boundary
        if best_break == end:
            space_pos = text.rfind(" ", start + self.min_chunk_chars, end)
            if space_pos > 0:
                best_break = space_pos + 1

        return best_break

    def estimate_total_tokens(self, content: ExtractedContent) -> int:
        """Estimate total tokens in the content."""
        total_chars = sum(
            len(self._page_to_text(page)) for page in content.pages
        )
        return total_chars // 4  # Rough estimate
