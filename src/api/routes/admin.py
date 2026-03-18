"""Admin endpoints - trigger data sync, training, and other operations from the UI."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.auth import require_admin
from src.api.deps import get_db

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_admin)])


async def _get_job_status(action: str) -> dict:
    from src.database import async_session
    from src.models.job_status import JobStatus

    try:
        async with async_session() as session:
            row = await session.get(JobStatus, action)
            if row is None:
                return {"status": "idle"}
            result = {
                "status": row.status,
                "message": row.message,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            if row.extra_json:
                try:
                    result.update(json.loads(row.extra_json))
                except Exception:
                    pass
            return result
    except Exception as e:
        logger.warning("Could not read job status from DB: %s", e)
        return {"status": "idle"}


async def _set_job_status(action: str, status: str, message: str = "", **extra):
    from src.database import async_session
    from src.models.job_status import JobStatus

    extra_json = json.dumps(extra) if extra else ""
    try:
        async with async_session() as session:
            row = await session.get(JobStatus, action)
            if row is None:
                row = JobStatus(action=action)
                session.add(row)
            row.status = status
            row.message = message
            row.extra_json = extra_json
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
    except Exception as e:
        logger.warning("Could not write job status to DB: %s", e)


async def _run_sync():
    from src.collectors.football_data import FootballDataCollector, COMPETITIONS
    from src.database import async_session

    await _set_job_status("sync-data", "running", "Syncing match data...")
    collector = FootballDataCollector()
    try:
        async with async_session() as session:
            await collector.sync_competitions(session)
            for code in COMPETITIONS:
                await collector.sync_teams(session, code)
            # Sync matches after all teams are loaded
            for code in COMPETITIONS:
                for season in ["2024", "2025"]:
                    await collector.sync_matches(session, code, season=season)
        await _set_job_status("sync-data", "ok", "Football data synced successfully.")
    except Exception as e:
        logger.exception("Sync failed")
        await _set_job_status("sync-data", "error", str(e))
    finally:
        await collector.close()


async def _run_fetch_odds():
    from src.collectors.odds_api import OddsApiCollector
    from src.database import async_session

    await _set_job_status("fetch-odds", "running", "Fetching odds...")
    collector = OddsApiCollector()
    try:
        async with async_session() as session:
            await collector.enrich_odds(session)
        await _set_job_status("fetch-odds", "ok", "Odds fetched successfully.")
    except Exception as e:
        logger.exception("Odds fetch failed")
        await _set_job_status("fetch-odds", "error", str(e))
    finally:
        await collector.close()


async def _run_build_features():
    from src.database import async_session
    from src.features.builder import build_features_for_match, build_features_for_upcoming
    from src.features.elo import process_all_matches
    from src.models.feature import MatchFeature
    from src.models.match import Match
    from sqlalchemy import select

    await _set_job_status("build-features", "running", "Building features...")
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

            total = len(matches)
            count = 0
            for match in matches:
                try:
                    feature = await build_features_for_match(session, match)
                    session.add(feature)
                    count += 1
                    if count % 50 == 0:
                        await session.commit()
                        await _set_job_status("build-features", "running", f"Built {count}/{total} features...")
                except Exception:
                    logger.exception("Failed for match %s", match.id)

            await session.commit()
            upcoming = await build_features_for_upcoming(session)

        await _set_job_status("build-features", "ok", f"Built features for {count} historical + {upcoming} upcoming matches.")
    except Exception as e:
        logger.exception("Feature building failed")
        await _set_job_status("build-features", "error", str(e))


async def _run_train():
    import time
    from src.database import async_session
    from src.models_ml.training import load_training_data, train_from_dataframe

    await _set_job_status("train", "running", "Training model — loading data...")
    try:
        # Step 1: Load data from DB (async)
        t0 = time.time()
        async with async_session() as session:
            df = await load_training_data(session)
        logger.info("[train] Data loaded: %d rows in %.1fs", len(df), time.time() - t0)

        if len(df) < 50:
            await _set_job_status("train", "error", f"Not enough data: {len(df)} matches", metrics={"n_matches": len(df)})
            return

        await _set_job_status("train", "running", f"Training on {len(df)} matches...")

        # Step 2: Run CPU-bound training in a thread so it doesn't block the event loop
        metrics = await asyncio.to_thread(train_from_dataframe, df, "v1")
    except Exception as e:
        logger.exception("Training crashed")
        await _set_job_status("train", "error", f"Training crashed: {type(e).__name__}: {e}")
        return

    if "error" in metrics:
        await _set_job_status("train", "error", f"Training failed: {metrics['error']}", metrics=metrics)
        return

    try:
        from src.api.routes import picks
        from src.models_ml.xgboost_model import load_model
        picks._xgb_model = load_model("v1")
        logger.info("Reloaded XGBoost model after training")
    except Exception:
        logger.warning("Could not reload model — restart the server to use new model")

    await _set_job_status("train", "ok", "Training complete.", metrics=metrics)


async def _run_resolve():
    from src.database import async_session
    from src.learning.tracker import update_outcomes

    await _set_job_status("resolve-outcomes", "running", "Resolving outcomes...")
    try:
        async with async_session() as session:
            count = await update_outcomes(session)
        await _set_job_status("resolve-outcomes", "ok", f"Resolved outcomes for {count} picks.")
    except Exception as e:
        logger.exception("Outcome resolution failed")
        await _set_job_status("resolve-outcomes", "error", str(e))


async def _run_backtest():
    from datetime import date, timedelta
    from src.database import async_session
    from src.learning.backtester import backtest

    await _set_job_status("backtest", "running", "Running backtest...")
    end = date.today()
    start = end - timedelta(weeks=8)

    try:
        async with async_session() as session:
            results = await backtest(session, start, end)
    except Exception as e:
        logger.exception("Backtest failed")
        await _set_job_status("backtest", "error", str(e))
        return

    roi_str = f"{results['roi']:.1f}%" if results.get('roi') is not None else "N/A"
    msg = (f"Backtest complete: {results['total_picks']} picks, "
           f"{results['accuracy']*100:.1f}% accuracy, "
           f"ROI: {roi_str}, "
           f"Perfect days: {results['perfect_matchdays']}/{results['total_matchdays']}")
    await _set_job_status("backtest", "ok", msg, results=results)


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

    current = await _get_job_status(action)
    if current.get("status") == "running":
        # Allow restart if job has been "running" for more than 10 minutes (stale/crashed)
        updated = current.get("updated_at")
        stale = False
        if updated:
            try:
                from datetime import datetime, timezone
                if isinstance(updated, str):
                    updated_dt = datetime.fromisoformat(updated)
                else:
                    updated_dt = updated
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                stale = (datetime.now(timezone.utc) - updated_dt).total_seconds() > 600
            except Exception:
                pass
        if not stale:
            return {"status": "running", "message": f"{action} is already running."}

    asyncio.create_task(runner())
    return {"status": "accepted", "message": f"{action} started in background."}


@router.get("/job-status/{action}")
async def get_job_status(action: str):
    """Poll the status of a background job."""
    return await _get_job_status(action)


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

    # Read all job statuses from DB
    try:
        from src.database import async_session
        from src.models.job_status import JobStatus
        async with async_session() as session:
            rows = (await session.execute(select(JobStatus))).scalars().all()
            info["jobs"] = {
                r.action: {"status": r.status, "message": r.message}
                for r in rows
            }
    except Exception:
        info["jobs"] = {}

    return info
