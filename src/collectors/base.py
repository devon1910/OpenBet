import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when an API returns HTTP 429 (rate limit exceeded)."""

    def __init__(self, url: str, retry_after: int | None = None):
        self.url = url
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {url}"
        if retry_after:
            msg += f" (retry after {retry_after}s)"
        super().__init__(msg)


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self._last_call = 0.0

    async def acquire(self):
        now = time.monotonic()
        wait = self._last_call + self.interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


# Simple in-memory response cache (lives for one process lifecycle)
_response_cache: dict[str, tuple[float, dict | list]] = {}
DEFAULT_CACHE_TTL = 300  # 5 minutes


class BaseCollector:
    """Base async HTTP client with rate limiting and caching."""

    def __init__(self, base_url: str, headers: dict, calls_per_minute: int = 10):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.limiter = RateLimiter(calls_per_minute)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def get(self, path: str, params: dict | None = None, cache_ttl: int = 0) -> dict:
        """Make a GET request with rate limiting.

        Args:
            path: API path
            params: query parameters
            cache_ttl: seconds to cache the response (0 = no caching)
        """
        # Build cache key
        url = f"{self.base_url}{path}"
        cache_key = f"{url}?{sorted(params.items()) if params else ''}"

        # Check cache
        if cache_ttl > 0 and cache_key in _response_cache:
            cached_at, cached_data = _response_cache[cache_key]
            if time.time() - cached_at < cache_ttl:
                logger.debug("Cache hit for %s", url)
                return cached_data

        await self.limiter.acquire()
        session = await self._get_session()
        logger.debug("GET %s", url)
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                retry_after = resp.headers.get("Retry-After")
                retry_secs = int(retry_after) if retry_after and retry_after.isdigit() else None
                raise RateLimitError(url, retry_secs)
            if resp.status in (403, 404):
                logger.warning("Skipping %s — HTTP %s (subscription tier or not found)", url, resp.status)
                return {}
            resp.raise_for_status()
            data = await resp.json()

        # Store in cache
        if cache_ttl > 0:
            _response_cache[cache_key] = (time.time(), data)

        return data

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
