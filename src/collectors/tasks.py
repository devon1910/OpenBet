"""Celery tasks for data collection."""

import asyncio
import logging

from src.collectors.football_data import FootballDataCollector, COMPETITIONS
from src.collectors.api_football import ApiFootballCollector, LEAGUE_MAP
from src.database import async_session

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _sync_all_data():
    collector = FootballDataCollector()
    try:
        async with async_session() as session:
            await collector.sync_all(session)
    finally:
        await collector.close()


async def _enrich_xg_data():
    from datetime import date, timedelta

    collector = ApiFootballCollector()
    yesterday = date.today() - timedelta(days=1)
    try:
        async with async_session() as session:
            for code, league_id in LEAGUE_MAP.items():
                await collector.enrich_xg(session, yesterday, league_id)
    finally:
        await collector.close()


def sync_football_data():
    """Sync all fixtures, teams, and standings from football-data.org."""
    logger.info("Starting football data sync")
    _run_async(_sync_all_data())
    logger.info("Football data sync complete")


def enrich_xg():
    """Enrich yesterday's matches with xG data from API-Football."""
    logger.info("Starting xG enrichment")
    _run_async(_enrich_xg_data())
    logger.info("xG enrichment complete")
