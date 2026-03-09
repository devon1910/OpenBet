"""Celery beat schedule and task definitions."""

import asyncio
import logging

from celery import shared_task
from celery.schedules import crontab

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(name="openbet.sync_data")
def task_sync_data():
    """Sync football data from APIs."""
    from src.collectors.tasks import sync_football_data, enrich_xg
    sync_football_data()
    enrich_xg()


@shared_task(name="openbet.build_features")
def task_build_features():
    """Build features for upcoming matches."""
    from src.database import async_session
    from src.features.builder import build_features_for_upcoming
    from src.features.elo import process_all_matches

    async def _run():
        async with async_session() as session:
            await process_all_matches(session)
            await build_features_for_upcoming(session)

    _run_async(_run())


@shared_task(name="openbet.generate_predictions")
def task_generate_predictions():
    """Generate predictions and picks for upcoming matches."""
    from src.database import async_session
    from src.engine.picks import generate_predictions_and_picks
    from src.config import settings

    async def _run():
        async with async_session() as session:
            await generate_predictions_and_picks(session, settings.model_version)

    _run_async(_run())


@shared_task(name="openbet.update_outcomes")
def task_update_outcomes():
    """Update pick outcomes after matches finish."""
    from src.database import async_session
    from src.learning.tracker import update_outcomes

    async def _run():
        async with async_session() as session:
            await update_outcomes(session)

    _run_async(_run())


@shared_task(name="openbet.evaluate_performance")
def task_evaluate_performance():
    """Evaluate model performance and retrain if needed."""
    from src.database import async_session
    from src.learning.retrainer import check_and_retrain
    from src.config import settings

    async def _run():
        async with async_session() as session:
            await check_and_retrain(session, settings.model_version)

    _run_async(_run())


# Beat schedule
celery_app.conf.beat_schedule = {
    "sync-data-daily": {
        "task": "openbet.sync_data",
        "schedule": crontab(hour=6, minute=0),  # Daily at 06:00 UTC
    },
    "build-features-daily": {
        "task": "openbet.build_features",
        "schedule": crontab(hour=7, minute=0),  # Daily at 07:00 UTC
    },
    "generate-predictions-daily": {
        "task": "openbet.generate_predictions",
        "schedule": crontab(hour=8, minute=0),  # Daily at 08:00 UTC
    },
    "update-outcomes-daily": {
        "task": "openbet.update_outcomes",
        "schedule": crontab(hour=23, minute=30),  # Daily at 23:30 UTC
    },
    "evaluate-performance-weekly": {
        "task": "openbet.evaluate_performance",
        "schedule": crontab(hour=4, minute=0, day_of_week=1),  # Monday 04:00 UTC
    },
}
