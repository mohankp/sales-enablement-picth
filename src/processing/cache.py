"""Caching layer for LLM processing results."""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    data: dict[str, Any]
    content_hash: str
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access time and hit count."""
        self.last_accessed = time.time()
        self.hit_count += 1


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
        }


class ProcessingCache:
    """
    Cache for LLM processing results.

    Provides both in-memory and persistent file-based caching
    with content-hash-based invalidation.

    Usage:
        cache = ProcessingCache(cache_dir="./cache")

        # Check cache
        result = cache.get("features", content_hash)
        if result:
            return FeatureSet.model_validate(result)

        # Process and cache
        features = await extract_features(content)
        cache.set("features", content_hash, features.model_dump())
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_entries: int = 100,
        max_disk_entries: int = 1000,
        default_ttl_seconds: int = 86400,  # 24 hours
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_memory_cache = enable_memory_cache
        self.enable_disk_cache = enable_disk_cache and cache_dir is not None

        self._memory_cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

        # Initialize disk cache directory
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_index()

    def _make_key(self, aspect: str, content_hash: str) -> str:
        """Create a cache key from aspect and content hash."""
        return f"{aspect}:{content_hash}"

    def _make_disk_path(self, key: str) -> Path:
        """Create file path for disk cache entry."""
        # Use hash of key for filename to avoid special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def _load_disk_index(self) -> None:
        """Load index of disk cache entries."""
        if not self.cache_dir or not self.cache_dir.exists():
            return

        # Count entries and compute size
        entries = list(self.cache_dir.glob("*.json"))
        self._stats.entry_count = len(entries)
        self._stats.size_bytes = sum(f.stat().st_size for f in entries)

    def get(
        self,
        aspect: str,
        content_hash: str,
        model_class: Optional[type[T]] = None,
    ) -> Optional[Any]:
        """
        Retrieve a cached result.

        Args:
            aspect: The processing aspect (e.g., "features", "benefits")
            content_hash: Hash of the source content
            model_class: Optional Pydantic model class to validate result

        Returns:
            Cached data if found and valid, None otherwise
        """
        key = self._make_key(aspect, content_hash)

        # Try memory cache first
        if self.enable_memory_cache and key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                entry.touch()
                self._stats.hits += 1
                logger.debug(f"Cache hit (memory): {aspect}")
                data = entry.data
                if model_class:
                    return model_class.model_validate(data)
                return data
            else:
                # Remove expired entry
                del self._memory_cache[key]

        # Try disk cache
        if self.enable_disk_cache:
            disk_path = self._make_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path) as f:
                        entry_data = json.load(f)

                    # Check expiration
                    expires_at = entry_data.get("expires_at")
                    if expires_at and time.time() > expires_at:
                        # Remove expired entry
                        disk_path.unlink()
                        self._stats.evictions += 1
                    else:
                        self._stats.hits += 1
                        logger.debug(f"Cache hit (disk): {aspect}")

                        # Promote to memory cache
                        data = entry_data["data"]
                        if self.enable_memory_cache:
                            self._add_to_memory(key, data, content_hash, expires_at)

                        if model_class:
                            return model_class.model_validate(data)
                        return data

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Corrupted cache entry {key}: {e}")
                    disk_path.unlink(missing_ok=True)

        self._stats.misses += 1
        logger.debug(f"Cache miss: {aspect}")
        return None

    def set(
        self,
        aspect: str,
        content_hash: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            aspect: The processing aspect
            content_hash: Hash of the source content
            data: Data to cache (dict or Pydantic model)
            ttl_seconds: Time-to-live in seconds (uses default if not specified)
        """
        key = self._make_key(aspect, content_hash)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        # Handle TTL: positive = future expiry, 0 or negative = already expired, None = never
        if ttl < 0:
            expires_at = time.time() - 1  # Already expired
        elif ttl == 0:
            expires_at = None  # Never expires
        else:
            expires_at = time.time() + ttl

        # Convert Pydantic model to dict if needed
        if isinstance(data, BaseModel):
            data = data.model_dump()

        # Store in memory cache
        if self.enable_memory_cache:
            self._add_to_memory(key, data, content_hash, expires_at)

        # Store on disk
        if self.enable_disk_cache:
            self._write_to_disk(key, data, content_hash, expires_at)

        logger.debug(f"Cached: {aspect} (ttl={ttl}s)")

    def _add_to_memory(
        self,
        key: str,
        data: dict,
        content_hash: str,
        expires_at: Optional[float],
    ) -> None:
        """Add entry to memory cache with eviction if needed."""
        # Evict oldest entries if at capacity
        while len(self._memory_cache) >= self.max_memory_entries:
            self._evict_lru()

        entry = CacheEntry(
            key=key,
            data=data,
            content_hash=content_hash,
            created_at=time.time(),
            expires_at=expires_at,
        )
        self._memory_cache[key] = entry

    def _evict_lru(self) -> None:
        """Evict least recently used entry from memory cache."""
        if not self._memory_cache:
            return

        # Find LRU entry
        lru_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].last_accessed,
        )
        del self._memory_cache[lru_key]
        self._stats.evictions += 1

    def _write_to_disk(
        self,
        key: str,
        data: dict,
        content_hash: str,
        expires_at: Optional[float],
    ) -> None:
        """Write cache entry to disk."""
        if not self.cache_dir:
            return

        # Check disk entry limit
        self._enforce_disk_limit()

        disk_path = self._make_disk_path(key)
        entry_data = {
            "key": key,
            "content_hash": content_hash,
            "data": data,
            "created_at": time.time(),
            "expires_at": expires_at,
        }

        try:
            with open(disk_path, "w") as f:
                json.dump(entry_data, f, default=str)
            self._stats.entry_count += 1
            self._stats.size_bytes += disk_path.stat().st_size
        except IOError as e:
            logger.warning(f"Failed to write cache entry: {e}")

    def _enforce_disk_limit(self) -> None:
        """Enforce maximum disk entries by removing oldest."""
        if not self.cache_dir:
            return

        entries = list(self.cache_dir.glob("*.json"))
        if len(entries) >= self.max_disk_entries:
            # Sort by modification time, remove oldest
            entries.sort(key=lambda p: p.stat().st_mtime)
            for entry in entries[: len(entries) - self.max_disk_entries + 1]:
                entry.unlink()
                self._stats.evictions += 1

    def invalidate(self, aspect: str, content_hash: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            aspect: The processing aspect
            content_hash: Hash of the source content

        Returns:
            True if entry was found and removed
        """
        key = self._make_key(aspect, content_hash)
        found = False

        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            found = True

        # Remove from disk
        if self.enable_disk_cache:
            disk_path = self._make_disk_path(key)
            if disk_path.exists():
                disk_path.unlink()
                found = True

        return found

    def invalidate_content(self, content_hash: str) -> int:
        """
        Invalidate all cache entries for a content hash.

        Args:
            content_hash: Hash of the source content

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Remove from memory
        keys_to_remove = [
            k for k in self._memory_cache if k.endswith(f":{content_hash}")
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]
            count += 1

        # Remove from disk
        if self.enable_disk_cache and self.cache_dir:
            for disk_path in self.cache_dir.glob("*.json"):
                try:
                    with open(disk_path) as f:
                        entry_data = json.load(f)
                    if entry_data.get("content_hash") == content_hash:
                        disk_path.unlink()
                        count += 1
                except (json.JSONDecodeError, IOError):
                    pass

        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()

        if self.enable_disk_cache and self.cache_dir:
            for disk_path in self.cache_dir.glob("*.json"):
                disk_path.unlink()

        self._stats = CacheStats()
        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        count = 0
        now = time.time()

        # Clean memory cache
        expired_keys = [
            k for k, v in self._memory_cache.items() if v.is_expired
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            count += 1

        # Clean disk cache
        if self.enable_disk_cache and self.cache_dir:
            for disk_path in self.cache_dir.glob("*.json"):
                try:
                    with open(disk_path) as f:
                        entry_data = json.load(f)
                    expires_at = entry_data.get("expires_at")
                    if expires_at and now > expires_at:
                        disk_path.unlink()
                        count += 1
                except (json.JSONDecodeError, IOError):
                    # Remove corrupted entries
                    disk_path.unlink()
                    count += 1

        self._stats.evictions += count
        logger.info(f"Cleaned up {count} expired entries")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_cached_aspects(self, content_hash: str) -> list[str]:
        """
        Get list of aspects that are cached for a content hash.

        Args:
            content_hash: Hash of the source content

        Returns:
            List of cached aspect names
        """
        aspects = set()

        # Check memory
        for key in self._memory_cache:
            if key.endswith(f":{content_hash}"):
                aspect = key.split(":")[0]
                aspects.add(aspect)

        # Check disk
        if self.enable_disk_cache and self.cache_dir:
            for disk_path in self.cache_dir.glob("*.json"):
                try:
                    with open(disk_path) as f:
                        entry_data = json.load(f)
                    if entry_data.get("content_hash") == content_hash:
                        key = entry_data.get("key", "")
                        aspect = key.split(":")[0] if ":" in key else key
                        aspects.add(aspect)
                except (json.JSONDecodeError, IOError):
                    pass

        return list(aspects)


class CachedProcessor:
    """
    Mixin/wrapper for adding caching to processing operations.

    Can be used to wrap individual processing functions with caching.
    """

    def __init__(self, cache: ProcessingCache):
        self.cache = cache

    async def cached_call(
        self,
        aspect: str,
        content_hash: str,
        processor_func,
        *args,
        model_class: Optional[type[T]] = None,
        ttl_seconds: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a processing function with caching.

        Args:
            aspect: Cache aspect name
            content_hash: Content hash for cache key
            processor_func: Async function to call on cache miss
            model_class: Optional Pydantic model for validation
            ttl_seconds: Cache TTL
            *args, **kwargs: Arguments for processor_func

        Returns:
            Cached or freshly computed result
        """
        # Try cache first
        cached = self.cache.get(aspect, content_hash, model_class)
        if cached is not None:
            return cached

        # Call processor
        result = await processor_func(*args, **kwargs)

        # Cache result
        if result is not None:
            self.cache.set(aspect, content_hash, result, ttl_seconds)

        return result
