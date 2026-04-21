"""In-memory TTL cache with SQLite persistence fallback.

The ``DataCache`` class implements a thread-safe OrderedDict backed
cache with configurable TTL per key.  When a SQLite path is provided,
expired entries are also persisted / restored from disk.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Exceptions                                                         #
# ------------------------------------------------------------------ #

class CacheError(Exception):
    """Base exception for cache operations."""


class CacheMissError(CacheError):
    """Raised when a key is not found in the cache."""


# ------------------------------------------------------------------ #
#  DataCache                                                          #
# ------------------------------------------------------------------ #

class DataCache:
    """TTL-based in-memory cache with optional SQLite persistence.

    Each entry stores ``(value, expiry_timestamp)``.  Expired entries
    are lazily removed on access and eagerly cleaned during
    ``cleanup()``.

    Args:
        default_ttl: Default time-to-live in seconds.
        sqlite_path: If provided, expired entries are persisted here.

    Attributes:
        default_ttl: Configured TTL in seconds.
    """

    def __init__(self, default_ttl: float = 60.0, sqlite_path: str | None = None) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self.default_ttl = default_ttl
        self._sqlite_path = sqlite_path
        self._sqlite_conn: sqlite3.Connection | None = None

        if sqlite_path:
            self._init_sqlite(sqlite_path)

    # -- lifecycle -----------------------------------------------------

    def _init_sqlite(self, path: str) -> None:
        self._sqlite_conn = sqlite3.connect(path)
        self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key   TEXT PRIMARY KEY,
                value BLOB,
                ttl   REAL
            )
        """)
        self._sqlite_conn.commit()
        logger.info("SQLite cache backend initialised at %s", path)

    def close(self) -> None:
        if self._sqlite_conn:
            self._sqlite_conn.close()
            self._sqlite_conn = None

    # -- core operations ------------------------------------------------

    def get(self, key: str) -> Any:
        """Return the cached value or ``None`` if expired / missing."""
        with self._lock:
            if key not in self._store:
                # Check SQLite fallback
                if self._sqlite_conn:
                    val = self._get_from_sqlite(key)
                    if val is not None:
                        logger.debug("Cache HIT (SQLite fallback) for key '%s'", key)
                        return val
                logger.debug("Cache MISS for key '%s'", key)
                return None

            value, expiry = self._store[key]
            if time.time() > expiry:
                del self._store[key]
                self._delete_from_sqlite(key)
                return None

            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            logger.debug("Cache HIT for key '%s'", key)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store *value* under *key* with the given *ttl* (seconds)."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + effective_ttl

        with self._lock:
            self._store[key] = (value, expiry)
            self._store.move_to_end(key)

        if self._sqlite_conn:
            self._set_to_sqlite(key, value, effective_ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
        if self._sqlite_conn:
            self._delete_from_sqlite(key)

    def cleanup(self) -> int:
        """Remove all expired entries. Returns the count of removed entries."""
        now = time.time()
        expired_keys: list[str] = []
        with self._lock:
            for k, (_, exp) in list(self._store.items()):
                if now > exp:
                    expired_keys.append(k)
            for k in expired_keys:
                del self._store[k]
        for k in expired_keys:
            if self._sqlite_conn:
                self._delete_from_sqlite(k)
        if expired_keys:
            logger.info("Cleaned up %d expired cache entries", len(expired_keys))
        return len(expired_keys)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        if self._sqlite_conn:
            self._sqlite_conn.execute("DELETE FROM cache_entries")
            self._sqlite_conn.commit()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    # -- SQLite helpers ------------------------------------------------

    def _set_to_sqlite(self, key: str, value: Any, ttl: float) -> None:
        try:
            self._sqlite_conn.execute(
                "INSERT OR REPLACE INTO cache_entries (key, value, ttl) VALUES (?, ?, ?)",
                (key, bytes(str(value), "utf-8"), ttl),
            )
            self._sqlite_conn.commit()
        except sqlite3.Error as exc:
            logger.warning("SQLite write error for key '%s': %s", key, exc)

    def _get_from_sqlite(self, key: str) -> Any | None:
        try:
            row = self._sqlite_conn.execute(
                "SELECT value FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return str(row[0], "utf-8")
            return None
        except sqlite3.Error as exc:
            logger.warning("SQLite read error for key '%s': %s", key, exc)
            return None

    def _delete_from_sqlite(self, key: str) -> None:
        try:
            self._sqlite_conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._sqlite_conn.commit()
        except sqlite3.Error:
            pass

    # -- context manager -----------------------------------------------

    def __enter__(self) -> DataCache:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ------------------------------------------------------------------ #
#  cache decorator                                                    #
# ------------------------------------------------------------------ #

def cached(ttl: float = 60.0, cache: DataCache | None = None):
    """Decorator that caches the result of an async or sync function.

    Args:
        ttl: Time-to-live in seconds.
        cache: Shared ``DataCache`` instance.  If ``None`` a new one
               with *ttl* as default is created per decorated function.
    """

    def decorator(func):
        cache_instance = cache or DataCache(default_ttl=ttl)

        @functools.wraps(func)
        def _sync_wrapper(*args, **kwargs):
            key = _make_key(func, args, kwargs)
            hit = cache_instance.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            cache_instance.set(key, result, ttl=ttl)
            return result

        async def _async_wrapper(*args, **kwargs):
            key = _make_key(func, args, kwargs)
            hit = cache_instance.get(key)
            if hit is not None:
                return hit
            result = await func(*args, **kwargs)
            cache_instance.set(key, result, ttl=ttl)
            return result

        # Wrap to detect async at call site
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # If func is a coroutine function use the async path
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    return await _async_wrapper(*args, **kwargs)
                else:
                    return _sync_wrapper(*args, **kwargs)
            except Exception:
                logger.exception("cache error in %s", func.__name__)
                raise

        wrapper.cache = cache_instance  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _make_key(func, args, kwargs):
    """Create a hashable cache key from function name + arguments."""
    parts = [func.__qualname__]
    for a in args:
        parts.append(repr(a))
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={repr(v)}")
    return "|".join(parts)
