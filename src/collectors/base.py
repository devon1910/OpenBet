import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


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


class BaseCollector:
    """Base async HTTP client with rate limiting."""

    def __init__(self, base_url: str, headers: dict, calls_per_minute: int = 10):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.limiter = RateLimiter(calls_per_minute)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def get(self, path: str, params: dict | None = None) -> dict:
        await self.limiter.acquire()
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        logger.debug("GET %s", url)
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
