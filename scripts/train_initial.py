"""Initial model training script.

Builds features for all finished matches, then trains the XGBoost model.
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

from src.database import async_session
from src.features.builder import build_features_for_match
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models_ml.training import train_and_evaluate
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def build_historical_features():
    """Build features for all finished matches that don't have features yet."""
    async with async_session() as session:
        stmt = (
            select(Match)
            .where(Match.status == "FINISHED")
            .outerjoin(MatchFeature, MatchFeature.match_id == Match.id)
            .where(MatchFeature.id.is_(None))
            .order_by(Match.match_date.asc())
        )
        result = await session.execute(stmt)
        matches = result.scalars().all()

        logger.info("Building features for %d historical matches...", len(matches))
        count = 0
        for match in matches:
            try:
                feature = await build_features_for_match(session, match)
                session.add(feature)
                count += 1
                if count % 100 == 0:
                    await session.commit()
                    logger.info("Progress: %d/%d", count, len(matches))
            except Exception:
                logger.exception("Failed for match %s", match.id)

        await session.commit()
        logger.info("Built features for %d matches", count)


async def main():
    # Build features
    await build_historical_features()

    # Train model
    async with async_session() as session:
        metrics = await train_and_evaluate(session, version="v1")
        logger.info("Training metrics: %s", metrics)


if __name__ == "__main__":
    asyncio.run(main())
