"""Tests for the processing cache module."""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

from pydantic import BaseModel

from src.processing.cache import (
    CacheEntry,
    CacheStats,
    ProcessingCache,
    CachedProcessor,
)


class SampleModel(BaseModel):
    """Sample model for testing."""
    name: str
    value: int


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_not_expired_without_expiry(self):
        """Entry without expiry never expires."""
        entry = CacheEntry(
            key="test",
            data={"value": 1},
            content_hash="abc123",
            created_at=time.time(),
            expires_at=None,
        )
        assert entry.is_expired is False

    def test_entry_not_expired_within_ttl(self):
        """Entry within TTL is not expired."""
        entry = CacheEntry(
            key="test",
            data={"value": 1},
            content_hash="abc123",
            created_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour from now
        )
        assert entry.is_expired is False

    def test_entry_expired_after_ttl(self):
        """Entry is expired after TTL."""
        entry = CacheEntry(
            key="test",
            data={"value": 1},
            content_hash="abc123",
            created_at=time.time() - 3600,
            expires_at=time.time() - 1,  # 1 second ago
        )
        assert entry.is_expired is True

    def test_touch_updates_access_time(self):
        """Touch updates access time and hit count."""
        entry = CacheEntry(
            key="test",
            data={"value": 1},
            content_hash="abc123",
            created_at=time.time(),
        )
        old_access_time = entry.last_accessed
        old_hit_count = entry.hit_count

        time.sleep(0.01)
        entry.touch()

        assert entry.last_accessed > old_access_time
        assert entry.hit_count == old_hit_count + 1


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_with_hits_and_misses(self):
        """Calculate hit rate correctly."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 0.8

    def test_hit_rate_with_no_requests(self):
        """Hit rate is 0 with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Convert stats to dictionary."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            size_bytes=1024,
            entry_count=15,
        )
        result = stats.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["hit_rate"] == "66.7%"
        assert result["evictions"] == 2


class TestProcessingCacheMemory:
    """Tests for ProcessingCache memory cache."""

    def test_set_and_get(self):
        """Basic set and get operations."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("features", "hash123", {"name": "test"})
        result = cache.get("features", "hash123")

        assert result == {"name": "test"}

    def test_get_with_model_class(self):
        """Get with Pydantic model validation."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("test", "hash123", {"name": "test", "value": 42})
        result = cache.get("test", "hash123", model_class=SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        cache = ProcessingCache(enable_disk_cache=False)

        result = cache.get("features", "nonexistent")

        assert result is None

    def test_stats_track_hits_and_misses(self):
        """Stats track cache hits and misses."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("test", "hash1", {"value": 1})
        cache.get("test", "hash1")  # Hit
        cache.get("test", "hash1")  # Hit
        cache.get("test", "missing")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1

    def test_expired_entries_not_returned(self):
        """Expired entries are not returned."""
        cache = ProcessingCache(
            enable_disk_cache=False,
            default_ttl_seconds=1,
        )

        # Set with TTL that puts expiry in the past
        cache.set("test", "hash1", {"value": 1}, ttl_seconds=-1)

        result = cache.get("test", "hash1")
        assert result is None

    def test_lru_eviction(self):
        """LRU eviction when at capacity."""
        cache = ProcessingCache(
            enable_disk_cache=False,
            max_memory_entries=3,
        )

        cache.set("test", "hash1", {"value": 1})
        cache.set("test", "hash2", {"value": 2})
        cache.set("test", "hash3", {"value": 3})

        # Access hash1 and hash2 to make them recently used
        cache.get("test", "hash1")
        cache.get("test", "hash2")

        # Adding hash4 should evict hash3 (least recently used)
        cache.set("test", "hash4", {"value": 4})

        assert cache.get("test", "hash1") is not None
        assert cache.get("test", "hash2") is not None
        assert cache.get("test", "hash4") is not None

    def test_invalidate_specific_entry(self):
        """Invalidate a specific cache entry."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("features", "hash1", {"value": 1})
        cache.set("benefits", "hash1", {"value": 2})

        result = cache.invalidate("features", "hash1")

        assert result is True
        assert cache.get("features", "hash1") is None
        assert cache.get("benefits", "hash1") is not None

    def test_invalidate_content(self):
        """Invalidate all entries for a content hash."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("features", "hash1", {"value": 1})
        cache.set("benefits", "hash1", {"value": 2})
        cache.set("features", "hash2", {"value": 3})

        count = cache.invalidate_content("hash1")

        assert count == 2
        assert cache.get("features", "hash1") is None
        assert cache.get("benefits", "hash1") is None
        assert cache.get("features", "hash2") is not None

    def test_clear_removes_all(self):
        """Clear removes all entries."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("features", "hash1", {"value": 1})
        cache.set("benefits", "hash2", {"value": 2})
        cache.clear()

        assert cache.get("features", "hash1") is None
        assert cache.get("benefits", "hash2") is None

    def test_get_cached_aspects(self):
        """Get list of cached aspects for a content hash."""
        cache = ProcessingCache(enable_disk_cache=False)

        cache.set("features", "hash1", {"value": 1})
        cache.set("benefits", "hash1", {"value": 2})
        cache.set("pricing", "hash2", {"value": 3})

        aspects = cache.get_cached_aspects("hash1")

        assert "features" in aspects
        assert "benefits" in aspects
        assert "pricing" not in aspects


class TestProcessingCacheDisk:
    """Tests for ProcessingCache disk cache."""

    def test_disk_cache_persists(self):
        """Disk cache persists between instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache and add entry
            cache1 = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=False,
            )
            cache1.set("features", "hash1", {"name": "test", "value": 42})

            # Create new cache instance
            cache2 = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=False,
            )
            result = cache2.get("features", "hash1")

            assert result == {"name": "test", "value": 42}

    def test_disk_cache_promotes_to_memory(self):
        """Disk cache entries are promoted to memory on access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create entry on disk only
            cache1 = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=False,
            )
            cache1.set("features", "hash1", {"value": 1})

            # New cache with memory enabled
            cache2 = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=True,
            )

            # First access loads from disk
            result1 = cache2.get("features", "hash1")
            assert result1 is not None

            # Verify it's now in memory (check internal state)
            key = cache2._make_key("features", "hash1")
            assert key in cache2._memory_cache

    def test_disk_cache_respects_ttl(self):
        """Disk cache entries expire based on TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=False,
                default_ttl_seconds=1,
            )

            # Set with TTL that puts expiry in the past
            cache.set("features", "hash1", {"value": 1}, ttl_seconds=-1)

            result = cache.get("features", "hash1")
            assert result is None

    def test_disk_cleanup_expired(self):
        """Cleanup removes expired disk entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessingCache(
                cache_dir=tmpdir,
                enable_memory_cache=False,
            )

            # Set one entry with expiry in the past
            cache.set("old", "hash1", {"value": 1}, ttl_seconds=-1)
            cache.set("new", "hash2", {"value": 2}, ttl_seconds=3600)

            count = cache.cleanup_expired()

            assert count >= 1
            assert cache.get("old", "hash1") is None
            assert cache.get("new", "hash2") is not None


class TestCachedProcessor:
    """Tests for CachedProcessor wrapper."""

    @pytest.mark.asyncio
    async def test_cached_call_returns_cached(self):
        """Cached call returns cached value."""
        cache = ProcessingCache(enable_disk_cache=False)
        processor = CachedProcessor(cache)

        # Pre-populate cache
        cache.set("features", "hash1", {"name": "cached", "value": 100})

        # Mock processor that should NOT be called
        mock_func = AsyncMock(return_value={"name": "fresh", "value": 200})

        result = await processor.cached_call(
            "features",
            "hash1",
            mock_func,
            model_class=SampleModel,
        )

        mock_func.assert_not_called()
        assert isinstance(result, SampleModel)
        assert result.name == "cached"
        assert result.value == 100

    @pytest.mark.asyncio
    async def test_cached_call_calls_func_on_miss(self):
        """Cached call executes function on cache miss."""
        cache = ProcessingCache(enable_disk_cache=False)
        processor = CachedProcessor(cache)

        mock_func = AsyncMock(return_value={"name": "fresh", "value": 200})

        result = await processor.cached_call(
            "features",
            "hash1",
            mock_func,
        )

        mock_func.assert_called_once()
        assert result == {"name": "fresh", "value": 200}

    @pytest.mark.asyncio
    async def test_cached_call_caches_result(self):
        """Cached call caches the result after computation."""
        cache = ProcessingCache(enable_disk_cache=False)
        processor = CachedProcessor(cache)

        mock_func = AsyncMock(return_value={"name": "computed", "value": 300})

        # First call
        await processor.cached_call("features", "hash1", mock_func)

        # Verify cached
        cached = cache.get("features", "hash1")
        assert cached == {"name": "computed", "value": 300}

    @pytest.mark.asyncio
    async def test_cached_call_passes_args(self):
        """Cached call passes arguments to function."""
        cache = ProcessingCache(enable_disk_cache=False)
        processor = CachedProcessor(cache)

        mock_func = AsyncMock(return_value={"value": 1})

        await processor.cached_call(
            "features",
            "hash1",
            mock_func,
            "arg1",
            "arg2",
            kwarg1="value1",
        )

        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
