"""APScheduler-based background job scheduler.

Replaces Celery Beat for Azure deployment — no Redis required.
Runs inside the FastAPI process using AsyncIOScheduler.

Pipeline (every 6 hours):
  1. Sync match data
  2. Build features + ELO
  3. Fetch odds
  4. Generate predictions/picks

Outcome resolution runs nightly at 23:30 UTC.
Model evaluation + retraining runs every Monday at 04:00 UTC.
On startup, the pipeline runs once after a short delay.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="UTC")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


async def _set_pipeline_status(key: str, status: str, **extra):
    """Persist pipeline status to DB via JobStatus table."""
    from src.database import async_session
    from src.models.job_status import JobStatus

    action = f"pipeline:{key}"
    extra_json = json.dumps(extra) if extra else ""
    try:
        async with async_session() as session:
            row = await session.get(JobStatus, action)
            if row is None:
                row = JobStatus(action=action)
                session.add(row)
            row.status = status
            row.message = extra.get("message", "")
            row.extra_json = extra_json
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
    except Exception as e:
        logger.warning("Could not write pipeline status to DB: %s", e)


async def get_pipeline_status() -> dict:
    """Read pipeline status from DB."""
    from src.database import async_session
    from src.models.job_status import JobStatus
    from sqlalchemy import select

    result = {
        "pipeline": {"last_run": None, "status": None, "next_run": None},
        "retrain":  {"last_run": None, "status": None, "action": None},
        "outcomes": {"last_run": None, "status": None},
    }

    try:
        async with async_session() as session:
            stmt = select(JobStatus).where(JobStatus.action.like("pipeline:%"))
            rows = (await session.execute(stmt)).scalars().all()
            for row in rows:
                key = row.action.replace("pipeline:", "")
                if key in result:
                    result[key]["last_run"] = row.updated_at.strftime("%Y-%m-%d %H:%M UTC") if row.updated_at else None
                    result[key]["status"] = row.status
                    if row.extra_json:
                        try:
                            extras = json.loads(row.extra_json)
                            if key == "retrain" and "action" in extras:
                                result[key]["action"] = extras["action"]
                        except Exception:
                            pass
    except Exception as e:
        logger.warning("Could not read pipeline status from DB: %s", e)

    # Always populate next_run from scheduler
    job = scheduler.get_job("pipeline_6h")
    if job and job.next_run_time:
        result["pipeline"]["next_run"] = job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")

    return result


async def _sync_data():
    from src.collectors.football_data import FootballDataCollector, COMPETITIONS
    from src.database import async_session

    logger.info("[scheduler] Syncing match data...")
    collector = FootballDataCollector()
    try:
        async with async_session() as session:
            await collector.sync_competitions(session)
            for code in COMPETITIONS:
                await collector.sync_teams(session, code)
                for season in ["2024", "2025"]:
                    await collector.sync_matches(session, code, season=season)
    finally:
        await collector.close()
    logger.info("[scheduler] Match data sync complete.")


async def _fetch_odds():
    from src.collectors.odds_api import OddsApiCollector
    from src.database import async_session

    logger.info("[scheduler] Fetching odds...")
    collector = OddsApiCollector()
    try:
        async with async_session() as session:
            count = await collector.enrich_odds(session)
        logger.info("[scheduler] Odds updated for %d matches.", count)
    finally:
        await collector.close()


async def _build_features():
    from src.database import async_session
    from src.features.bulk_builder import bulk_build_features
    from src.features.elo import process_all_matches
    from src.models.feature import MatchFeature
    from src.models.match import Match
    from sqlalchemy import select

    logger.info("[scheduler] Building features...")

    # Elo in its own session
    async with async_session() as session:
        await process_all_matches(session)

    # Historical features (finished matches without features)
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
        count = await bulk_build_features(session, matches)

    # Upcoming features
    async with async_session() as session:
        stmt = (
            select(Match)
            .where(Match.status.in_(["SCHEDULED", "TIMED"]))
            .outerjoin(MatchFeature, MatchFeature.match_id == Match.id)
            .where(MatchFeature.id.is_(None))
        )
        result = await session.execute(stmt)
        upcoming_matches = result.scalars().all()
        upcoming = await bulk_build_features(session, upcoming_matches)

    logger.info("[scheduler] Features built: %d historical + %d upcoming.", count, upcoming)


async def _generate_predictions():
    from src.database import async_session
    from src.engine.picks import generate_predictions_and_picks
    from src.config import settings

    logger.info("[scheduler] Generating predictions and picks...")
    async with async_session() as session:
        await generate_predictions_and_picks(session, settings.model_version)
    logger.info("[scheduler] Predictions generated.")


async def _update_outcomes():
    from src.database import async_session
    from src.learning.tracker import update_outcomes

    logger.info("[scheduler] Resolving pick outcomes...")
    try:
        async with async_session() as session:
            count = await update_outcomes(session)
        await _set_pipeline_status("outcomes", "ok", message=f"Resolved {count} outcomes.")
        logger.info("[scheduler] Resolved %d outcomes.", count)
    except Exception:
        logger.exception("[scheduler] Outcome resolution failed")
        await _set_pipeline_status("outcomes", "error", message="Outcome resolution failed")


async def _evaluate_and_retrain():
    from src.database import async_session
    from src.learning.retrainer import check_and_retrain
    from src.config import settings

    logger.info("[scheduler] Evaluating model performance...")
    try:
        async with async_session() as session:
            result = await check_and_retrain(session, settings.model_version)
        action = result.get("action", "none")
        await _set_pipeline_status("retrain", "ok", action=action, message=f"Evaluation complete: {action}")
        logger.info("[scheduler] Evaluation result: %s", action)
    except Exception:
        logger.exception("[scheduler] Evaluation/retraining failed")
        await _set_pipeline_status("retrain", "error", message="Evaluation/retraining failed")


async def _run_pipeline():
    """Full 6-hour pipeline: sync → features → odds → predictions."""
    await _set_pipeline_status("pipeline", "running", message="Pipeline running...")
    try:
        await _sync_data()
        await _build_features()
        await _fetch_odds()
        await _generate_predictions()
        await _set_pipeline_status("pipeline", "ok", message="Pipeline completed successfully.")
    except Exception:
        logger.exception("[scheduler] Pipeline run failed")
        await _set_pipeline_status("pipeline", "error", message="Pipeline run failed")


def start_scheduler():
    """Register all jobs and start the scheduler."""
    # Full pipeline every 6 hours: 00:00, 06:00, 12:00, 18:00 UTC
    scheduler.add_job(
        _run_pipeline,
        CronTrigger(hour="0,6,12,18", minute=0),
        id="pipeline_6h",
        name="6-hour data pipeline",
        replace_existing=True,
    )

    # Resolve outcomes nightly after late matches finish
    scheduler.add_job(
        _update_outcomes,
        CronTrigger(hour=23, minute=30),
        id="resolve_outcomes",
        name="Nightly outcome resolution",
        replace_existing=True,
    )

    # Evaluate performance and retrain if needed — every Monday 04:00 UTC
    scheduler.add_job(
        _evaluate_and_retrain,
        CronTrigger(day_of_week="mon", hour=4, minute=0),
        id="evaluate_retrain",
        name="Weekly model evaluation + retraining",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("[scheduler] APScheduler started. Jobs: %s", [j.id for j in scheduler.get_jobs()])

    # Run the pipeline once on startup (30s delay so the app is fully ready)
    asyncio.get_event_loop().call_later(30, lambda: asyncio.ensure_future(_run_pipeline()))
    logger.info("[scheduler] Startup pipeline run scheduled in 30 seconds.")
