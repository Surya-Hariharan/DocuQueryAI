"""
Utility functions for monitoring, caching, and performance optimization.
"""

import time
import hashlib
import logging
import functools
import threading
from typing import Any, Callable, Optional
from collections import OrderedDict

# === Logging Setup ===
logger = logging.getLogger("utils")

# === Performance Monitoring Decorator ===
def monitor_performance(func_name: Optional[str] = None):
    """
    Decorator to monitor and log function execution time.
    
    Usage:
        @monitor_performance()
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"⏱️  {name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {name} failed after {elapsed:.3f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"⏱️  {name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {name} failed after {elapsed:.3f}s: {e}")
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# === In-Memory LRU Cache ===
class LRUCache:
    """
    Thread-safe LRU cache for embeddings and query results.
    """
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                # Update and move to end
                self.cache.move_to_end(key)
            self.cache[key] = value

            # Remove oldest if exceeding max size
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear all cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2f}%"
            }


# === Global Cache Instances ===
embedding_cache = LRUCache(max_size=5000)
query_cache = LRUCache(max_size=1000)


# === Text Hashing for Deduplication ===
def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text for deduplication.
    
    Args:
        text: Input text to hash
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of raw bytes — used for whole-document checksums
    (e.g. a downloaded file), as opposed to compute_text_hash for chunks.

    Args:
        data: Raw bytes to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data).hexdigest()


# === Retry Decorator ===
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


# === Batch Processing Helper ===
def batch_items(items: list, batch_size: int):
    """
    Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
