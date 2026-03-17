"""Admin endpoints - trigger data sync, training, and other operations from the UI."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import require_admin
from src.api.deps import get_db

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_admin)])

# Track background job status
_jobs: dict[str, dict] = {}


def _job_status(action: str) -> dict:
    return _jobs.get(action, {"status": "idle"})


def _set_job(action: str, status: str, message: str = "", **extra):
    _jobs[action] = {
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


async def _run_sync():
    from src.collectors.football_data import FootballDataCollector, COMPETITIONS
    from src.database import async_session

    _set_job("sync-data", "running", "Syncing match data...")
    collector = FootballDataCollector()
    try:
        async with async_session() as session:
            await collector.sync_competitions(session)
            for code in COMPETITIONS:
                await collector.sync_teams(session, code)
                for season in ["2023", "2024", "2025"]:
                    await collector.sync_matches(session, code, season=season)
        _set_job("sync-data", "ok", "Football data synced successfully.")
    except Exception as e:
        logger.exception("Sync failed")
        _set_job("sync-data", "error", str(e))
    finally:
        await collector.close()


async def _run_fetch_odds():
    from src.collectors.odds_api import OddsApiCollector
    from src.database import async_session

    _set_job("fetch-odds", "running", "Fetching odds...")
    collector = OddsApiCollector()
    try:
        async with async_session() as session:
            await collector.enrich_odds(session)
        _set_job("fetch-odds", "ok", "Odds fetched successfully.")
    except Exception as e:
        logger.exception("Odds fetch failed")
        _set_job("fetch-odds", "error", str(e))
    finally:
        await collector.close()


async def _run_build_features():
    from src.database import async_session
    from src.features.builder import build_features_for_match, build_features_for_upcoming
    from src.features.elo import process_all_matches
    from src.models.feature import MatchFeature
    from src.models.match import Match
    from sqlalchemy import select

    _set_job("build-features", "running", "Building features...")
    try:
        async with async_session() as session:
            await process_all_matches(session)
            stmt = (
                select(Match)
                .where(Match.status == "FINISHED")
                .outerjoin(MatchFeature, MatchFeature.match_id == Match.id)
                .where(MatchFeature.id.is_(None))
                .order_by(Match.match_date.asc())
            )
            result = await session.execute(stmt)
            matches = result.scalars().all()

            count = 0
            for match in matches:
                try:
                    feature = await build_features_for_match(session, match)
                    session.add(feature)
                    count += 1
                    if count % 100 == 0:
                        await session.commit()
                except Exception:
                    logger.exception("Failed for match %s", match.id)

            await session.commit()
            upcoming = await build_features_for_upcoming(session)

        _set_job("build-features", "ok", f"Built features for {count} historical + {upcoming} upcoming matches.")
    except Exception as e:
        logger.exception("Feature building failed")
        _set_job("build-features", "error", str(e))


async def _run_train():
    import traceback
    from src.database import async_session
    from src.models_ml.training import train_and_evaluate

    _set_job("train", "running", "Training model...")
    try:
        async with async_session() as session:
            metrics = await train_and_evaluate(session, version="v1")
    except Exception as e:
        logger.exception("Training crashed")
        _set_job("train", "error", f"Training crashed: {type(e).__name__}: {e}")
        return

    if "error" in metrics:
        _set_job("train", "error", f"Training failed: {metrics['error']}", metrics=metrics)
        return

    try:
        from src.api.routes import picks
        from src.models_ml.xgboost_model import load_model
        picks._xgb_model = load_model("v1")
        logger.info("Reloaded XGBoost model after training")
    except Exception:
        logger.warning("Could not reload model — restart the server to use new model")

    _set_job("train", "ok", "Training complete.", metrics=metrics)


async def _run_resolve():
    from src.database import async_session
    from src.learning.tracker import update_outcomes

    _set_job("resolve-outcomes", "running", "Resolving outcomes...")
    try:
        async with async_session() as session:
            count = await update_outcomes(session)
        _set_job("resolve-outcomes", "ok", f"Resolved outcomes for {count} picks.")
    except Exception as e:
        logger.exception("Outcome resolution failed")
        _set_job("resolve-outcomes", "error", str(e))


async def _run_backtest():
    from datetime import date, timedelta
    from src.database import async_session
    from src.learning.backtester import backtest

    _set_job("backtest", "running", "Running backtest...")
    end = date.today()
    start = end - timedelta(weeks=8)

    try:
        async with async_session() as session:
            results = await backtest(session, start, end)
    except Exception as e:
        logger.exception("Backtest failed")
        _set_job("backtest", "error", str(e))
        return

    roi_str = f"{results['roi']:.1f}%" if results.get('roi') is not None else "N/A"
    msg = (f"Backtest complete: {results['total_picks']} picks, "
           f"{results['accuracy']*100:.1f}% accuracy, "
           f"ROI: {roi_str}, "
           f"Perfect days: {results['perfect_matchdays']}/{results['total_matchdays']}")
    _set_job("backtest", "ok", msg, results=results)


_RUNNERS = {
    "sync-data": _run_sync,
    "fetch-odds": _run_fetch_odds,
    "build-features": _run_build_features,
    "train": _run_train,
    "resolve-outcomes": _run_resolve,
    "backtest": _run_backtest,
}


@router.post("/{action}")
async def run_action(action: str):
    """Kick off an admin action as a background task. Returns immediately."""
    if action == "status":
        return await system_status()

    runner = _RUNNERS.get(action)
    if not runner:
        return {"status": "error", "message": f"Unknown action: {action}"}

    current = _job_status(action)
    if current.get("status") == "running":
        return {"status": "running", "message": f"{action} is already running."}

    asyncio.create_task(runner())
    return {"status": "accepted", "message": f"{action} started in background."}


@router.get("/job-status/{action}")
async def get_job_status(action: str):
    """Poll the status of a background job."""
    return _job_status(action)


@router.get("/status")
async def system_status():
    """Check system status — DB connection, model availability, data counts."""
    from src.database import async_session
    from src.models.match import Match
    from src.models.feature import MatchFeature
    from sqlalchemy import select, func

    info = {}

    try:
        async with async_session() as session:
            total = (await session.execute(select(func.count(Match.id)))).scalar() or 0
            finished = (await session.execute(
                select(func.count(Match.id)).where(Match.status == "FINISHED")
            )).scalar() or 0
            features = (await session.execute(select(func.count(MatchFeature.id)))).scalar() or 0
            with_odds = (await session.execute(
                select(func.count(MatchFeature.id)).where(MatchFeature.odds_home.is_not(None))
            )).scalar() or 0

        info["db"] = "connected"
        info["total_matches"] = total
        info["finished_matches"] = finished
        info["features_built"] = features
        info["with_odds"] = with_odds
    except Exception as e:
        info["db"] = f"error: {e}"

    from pathlib import Path
    info["xgb_model"] = Path("trained_models/xgboost_v1.joblib").exists()
    info["meta_learner"] = Path("trained_models/meta_learner_v1.joblib").exists()
    info["jobs"] = {k: v for k, v in _jobs.items()}

    return info
