"""
Cache Manager for Multi-Source Retrieval
Provides in-memory and persistent disk caching for retrieved web pages and external searches.
"""

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any

from ..config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None, default_ttl_hours: int = 24):
        self.cache_dir = cache_dir or settings.WEB_CACHE_DIR
        self.default_ttl_seconds = default_ttl_hours * 3600
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.strip().lower().encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        h = self._hash_key(key)
        now = time.time()

        # Check memory cache first
        if h in self._memory_cache:
            entry = self._memory_cache[h]
            if entry["expires_at"] > now:
                return entry["data"]
            else:
                del self._memory_cache[h]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if entry.get("expires_at", 0) > now:
                    self._memory_cache[h] = entry
                    return entry.get("data")
                else:
                    os.remove(cache_file)
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")

        return None

    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        h = self._hash_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = time.time() + ttl

        entry = {
            "key": key,
            "data": data,
            "cached_at": time.time(),
            "expires_at": expires_at
        }

        # Save to memory
        self._memory_cache[h] = entry

        # Save to disk
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache file {cache_file}: {e}")

cache_manager = CacheManager()
