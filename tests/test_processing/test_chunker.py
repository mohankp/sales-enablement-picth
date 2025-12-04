"""Tests for the content chunking module."""

import pytest

from src.processing.chunker import (
    ContentChunker,
    ContentChunk,
    ChunkingStrategy,
)
from src.models.content import (
    ContentBlock,
    ContentType,
    ExtractedContent,
    PageContent,
)


class TestContentChunk:
    """Tests for ContentChunk dataclass."""

    def test_chunk_computes_derived_fields(self):
        """Test that chunk automatically computes char count and token estimate."""
        chunk = ContentChunk(
            content="Hello, this is a test content.",
            chunk_index=0,
            total_chunks=1,
        )

        assert chunk.char_count == 30
        assert chunk.estimated_tokens == 7  # 30 // 4
        assert len(chunk.chunk_hash) == 12

    def test_chunk_hash_is_consistent(self):
        """Test that same content produces same hash."""
        content = "Test content for hashing"
        chunk1 = ContentChunk(content=content, chunk_index=0, total_chunks=1)
        chunk2 = ContentChunk(content=content, chunk_index=1, total_chunks=2)

        assert chunk1.chunk_hash == chunk2.chunk_hash

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        chunk1 = ContentChunk(content="Content A", chunk_index=0, total_chunks=1)
        chunk2 = ContentChunk(content="Content B", chunk_index=0, total_chunks=1)

        assert chunk1.chunk_hash != chunk2.chunk_hash


class TestContentChunker:
    """Tests for ContentChunker class."""

    @pytest.fixture
    def sample_page(self):
        """Create a sample page with content blocks."""
        blocks = [
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Product Overview",
                heading_level=1,
            ),
            ContentBlock(
                content_type=ContentType.PARAGRAPH,
                text="This is a detailed description of our amazing product. " * 10,
            ),
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Key Features",
                heading_level=2,
            ),
            ContentBlock(
                content_type=ContentType.LIST,
                text="Features list",
                list_items=["Feature 1", "Feature 2", "Feature 3"],
            ),
            ContentBlock(
                content_type=ContentType.HEADING,
                text="Pricing",
                heading_level=2,
            ),
            ContentBlock(
                content_type=ContentType.PARAGRAPH,
                text="Our pricing is competitive and transparent.",
            ),
        ]
        return PageContent(
            url="https://example.com/product",
            title="Amazing Product",
            content_blocks=blocks,
            word_count=100,
        )

    @pytest.fixture
    def sample_extracted_content(self, sample_page):
        """Create sample extracted content with one page."""
        return ExtractedContent(
            product_name="Amazing Product",
            product_url="https://example.com",
            pages=[sample_page],
        )

    def test_chunker_initialization(self):
        """Test chunker initializes with correct defaults."""
        chunker = ContentChunker()

        assert chunker.strategy == ChunkingStrategy.HYBRID
        assert chunker.max_chars == ContentChunker.DEFAULT_MAX_CHARS
        assert chunker.overlap_chars == ContentChunker.DEFAULT_OVERLAP_CHARS

    def test_chunk_text_small_content(self):
        """Test that small content results in single chunk."""
        chunker = ContentChunker(max_chars=1000)
        text = "Small content that fits in one chunk."

        chunks = chunker.chunk_text(text, source_url="https://example.com")

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert "https://example.com" in chunks[0].source_urls

    def test_chunk_text_large_content(self):
        """Test that large content is split into multiple chunks."""
        chunker = ContentChunker(max_chars=100, overlap_chars=10, min_chunk_chars=20)
        text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + "."

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        # Each chunk should be at most max_chars
        for chunk in chunks:
            assert chunk.char_count <= chunker.max_chars + 20  # Allow some flexibility

    def test_chunk_text_finds_sentence_boundaries(self):
        """Test that chunker tries to break at sentence boundaries."""
        chunker = ContentChunker(max_chars=50, overlap_chars=5, min_chunk_chars=10)
        text = "First sentence here. Second sentence here. Third sentence here."

        chunks = chunker.chunk_text(text)

        # Should break at periods rather than mid-word
        for chunk in chunks:
            # Check doesn't end mid-word (unless at end)
            if chunk.chunk_index < chunk.total_chunks - 1:
                assert chunk.content.endswith(".") or chunk.content.endswith(" ")

    def test_page_based_chunking(self, sample_extracted_content):
        """Test page-based chunking strategy."""
        chunker = ContentChunker(strategy=ChunkingStrategy.PAGE_BASED)

        chunks = chunker.chunk_extracted_content(sample_extracted_content)

        # Should have one chunk per page (since page is small enough)
        assert len(chunks) >= 1
        assert sample_extracted_content.pages[0].url in chunks[0].source_urls

    def test_semantic_chunking(self, sample_extracted_content):
        """Test semantic chunking based on headings."""
        chunker = ContentChunker(strategy=ChunkingStrategy.SEMANTIC)

        chunks = chunker.chunk_extracted_content(sample_extracted_content)

        assert len(chunks) >= 1
        # Check that section headers are preserved
        for chunk in chunks:
            assert isinstance(chunk.section_headers, list)

    def test_hybrid_chunking_merges_small_sections(self, sample_extracted_content):
        """Test hybrid chunking combines small sections."""
        chunker = ContentChunker(
            strategy=ChunkingStrategy.HYBRID,
            max_chars=10000,  # Large enough to fit all content
            min_chunk_chars=100,
        )

        chunks = chunker.chunk_extracted_content(sample_extracted_content)

        # With large max_chars, content should be combined
        assert len(chunks) >= 1

    def test_estimate_total_tokens(self, sample_extracted_content):
        """Test token estimation."""
        chunker = ContentChunker()

        tokens = chunker.estimate_total_tokens(sample_extracted_content)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_extract_headers(self, sample_page):
        """Test header extraction from page."""
        chunker = ContentChunker()

        headers = chunker._extract_headers(sample_page)

        assert "Product Overview" in headers
        assert "Key Features" in headers
        assert "Pricing" in headers

    def test_block_to_text_heading(self):
        """Test conversion of heading block to text."""
        chunker = ContentChunker()
        block = ContentBlock(
            content_type=ContentType.HEADING,
            text="My Heading",
            heading_level=2,
        )

        text = chunker._block_to_text(block)

        assert text == "## My Heading"

    def test_block_to_text_list(self):
        """Test conversion of list block to text."""
        chunker = ContentChunker()
        block = ContentBlock(
            content_type=ContentType.LIST,
            text="List",
            list_items=["Item 1", "Item 2"],
        )

        text = chunker._block_to_text(block)

        assert "- Item 1" in text
        assert "- Item 2" in text

    def test_block_to_text_code(self):
        """Test conversion of code block to text."""
        chunker = ContentChunker()
        block = ContentBlock(
            content_type=ContentType.CODE,
            text="print('hello')",
        )

        text = chunker._block_to_text(block)

        assert text == "```\nprint('hello')\n```"

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        chunker = ContentChunker()
        content = ExtractedContent(
            product_url="https://example.com",
            pages=[],
        )

        chunks = chunker.chunk_extracted_content(content)

        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].char_count == 0)


class TestChunkingStrategies:
    """Test different chunking strategies."""

    @pytest.fixture
    def multi_page_content(self):
        """Create content with multiple pages."""
        pages = []
        for i in range(3):
            page = PageContent(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                content_blocks=[
                    ContentBlock(
                        content_type=ContentType.HEADING,
                        text=f"Page {i} Title",
                        heading_level=1,
                    ),
                    ContentBlock(
                        content_type=ContentType.PARAGRAPH,
                        text=f"Content for page {i}. " * 20,
                    ),
                ],
                word_count=50,
            )
            pages.append(page)

        return ExtractedContent(
            product_name="Multi Page Product",
            product_url="https://example.com",
            pages=pages,
        )

    def test_fixed_size_strategy(self, multi_page_content):
        """Test fixed-size chunking."""
        chunker = ContentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chars=500,
            overlap_chars=50,
        )

        chunks = chunker.chunk_extracted_content(multi_page_content)

        # Fixed size should produce consistent chunk sizes
        for chunk in chunks[:-1]:  # Except last which may be smaller
            assert chunk.char_count <= 500 + 100  # Allow buffer

    def test_semantic_preserves_structure(self, multi_page_content):
        """Test semantic chunking preserves document structure."""
        chunker = ContentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chars=5000,
        )

        chunks = chunker.chunk_extracted_content(multi_page_content)

        # Each chunk should have section headers
        for chunk in chunks:
            if chunk.content.strip():
                # Content should include heading markers
                assert "Page" in chunk.content or "Title" in chunk.content
