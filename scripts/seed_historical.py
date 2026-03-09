"""Seed historical data from football-data.org.

Backfills 2-3 seasons of match data, then processes Elo ratings
and builds features for all finished matches.
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

from src.collectors.football_data import FootballDataCollector, COMPETITIONS
from src.database import async_session, engine, Base
from src.features.elo import process_all_matches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seasons to backfill (football-data.org uses start year)
SEASONS = ["2023", "2024", "2025"]


async def main():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    collector = FootballDataCollector()

    try:
        async with async_session() as session:
            # Sync competitions
            logger.info("Syncing competitions...")
            await collector.sync_competitions(session)

            # Sync teams and matches for each competition and season
            for code in COMPETITIONS:
                logger.info("Syncing teams for %s...", code)
                await collector.sync_teams(session, code)

                for season in SEASONS:
                    logger.info("Syncing matches for %s season %s...", code, season)
                    await collector.sync_matches(session, code, season=season)

            # Process Elo ratings
            logger.info("Processing Elo ratings...")
            await process_all_matches(session)

    finally:
        await collector.close()

    logger.info("Historical data seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
