"""
Cache utilities for optimizing expensive computations and data retrieval.

This module provides a flexible caching system with:
1. Memory and disk-based caching options
2. Configurable expiration and size limits
3. Automatic cache invalidation
4. Thread-safe operations
5. Monitoring and statistics
"""

import functools
import threading
import time
import pickle
import os
from typing import Any, Dict, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheStats:
    """Tracks cache performance metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __str__(self) -> str:
        return (f"Cache Stats - Hits: {self.hits}, Misses: {self.misses}, "
                f"Evictions: {self.evictions}, Hit Rate: {self.hit_rate:.2%}")

class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, value: Any, expiry: Optional[datetime] = None):
        self.value = value
        self.expiry = expiry
        self.last_accessed = datetime.now()
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
    
    def access(self):
        """Update entry metadata on access."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class Cache:
    """
    Thread-safe cache implementation with memory and disk backing.
    
    Features:
    - LRU eviction policy
    - Time-based expiration
    - Size limits
    - Optional disk persistence
    - Statistics tracking
    """
    
    def __init__(self, max_size: int = 1000, 
                 default_ttl: Optional[timedelta] = timedelta(hours=1),
                 disk_path: Optional[Path] = None):
        """
        Initialize cache with configuration parameters.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live for entries
            disk_path: Optional path for disk persistence
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.disk_path = disk_path
        self.stats = CacheStats()
        
        # Create disk cache directory if needed
        if disk_path:
            disk_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Thread-safe retrieval with automatic expiration handling.
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                # Check disk cache if configured
                if self.disk_path:
                    entry = self._load_from_disk(key)
                if entry is None:
                    self.stats.record_miss()
                    return None
            
            # Check expiration
            if entry.is_expired():
                self._remove(key)
                self.stats.record_miss()
                return None
            
            # Update access metadata
            entry.access()
            self.stats.record_hit()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional time-to-live override
        """
        with self._lock:
            # Calculate expiry
            expiry = None
            if ttl or self.default_ttl:
                expiry = datetime.now() + (ttl or self.default_ttl)
            
            # Create entry
            entry = CacheEntry(value, expiry)
            
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store in memory
            self._cache[key] = entry
            
            # Store to disk if configured
            if self.disk_path:
                self._save_to_disk(key, entry)
    
    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._remove(key)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            if self.disk_path:
                for file in self.disk_path.glob("*.cache"):
                    file.unlink()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(self._cache.items(), 
                     key=lambda x: x[1].last_accessed)[0]
        
        # Remove it
        self._remove(lru_key)
        self.stats.record_eviction()
    
    def _remove(self, key: str) -> None:
        """Remove an entry from both memory and disk."""
        self._cache.pop(key, None)
        if self.disk_path:
            cache_file = self.disk_path / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
    
    def _save_to_disk(self, key: str, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        try:
            cache_file = self.disk_path / f"{key}.cache"
            with cache_file.open('wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk."""
        try:
            cache_file = self.disk_path / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            with cache_file.open('rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache entry from disk: {e}")
            return None

# Decorator for function-level caching
def cached(ttl: Optional[timedelta] = None, 
          key_prefix: str = "",
          cache_instance: Optional[Cache] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Optional time-to-live override
        key_prefix: Optional prefix for cache keys
        cache_instance: Optional specific cache instance to use
    """
    def decorator(func: Callable):
        # Use provided cache or create a default one
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = Cache()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Check cache
            result = cache_instance.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Create cache instance
    cache = Cache(max_size=100, 
                 default_ttl=timedelta(minutes=30),
                 disk_path=Path("cache"))
    
    # Example cached function
    @cached(ttl=timedelta(minutes=5), cache_instance=cache)
    def expensive_computation(x: int, y: int) -> int:
        time.sleep(1)  # Simulate expensive work
        return x + y
    
    # Test caching
    start = time.time()
    result1 = expensive_computation(1, 2)
    time1 = time.time() - start
    
    start = time.time()
    result2 = expensive_computation(1, 2)  # Should be cached
    time2 = time.time() - start
    
    print(f"First call: {time1:.2f}s")
    print(f"Second call: {time2:.2f}s")
    print(f"Cache stats: {cache.stats}")