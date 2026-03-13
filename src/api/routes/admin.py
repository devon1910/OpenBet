"""Admin endpoints - trigger data sync, training, and other operations from the UI."""

import asyncio
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sync-data")
async def sync_data():
    """Sync match data from football-data.org."""
    from src.collectors.football_data import FootballDataCollector, COMPETITIONS
    from src.database import async_session

    collector = FootballDataCollector()
    try:
        async with async_session() as session:
            await collector.sync_competitions(session)
            for code in COMPETITIONS:
                await collector.sync_teams(session, code)
                for season in ["2023", "2024", "2025"]:
                    await collector.sync_matches(session, code, season=season)
        return {"status": "ok", "message": "Football data synced successfully."}
    except Exception as e:
        logger.exception("Sync failed")
        return {"status": "error", "message": str(e)}
    finally:
        await collector.close()


@router.post("/fetch-odds")
async def fetch_odds():
    """Fetch bookmaker odds from The Odds API."""
    from src.collectors.odds_api import OddsApiCollector
    from src.database import async_session

    collector = OddsApiCollector()
    try:
        async with async_session() as session:
            await collector.enrich_odds(session)
        return {"status": "ok", "message": "Odds fetched successfully."}
    except Exception as e:
        logger.exception("Odds fetch failed")
        return {"status": "error", "message": str(e)}
    finally:
        await collector.close()


@router.post("/build-features")
async def build_features():
    """Build features for all matches missing them."""
    from src.database import async_session
    from src.features.builder import build_features_for_match
    from src.features.elo import process_all_matches
    from src.models.feature import MatchFeature
    from src.models.match import Match
    from sqlalchemy import select

    async with async_session() as session:
        # Process Elo ratings first
        await process_all_matches(session)

        # Build features for finished matches without features
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

        # Also build for upcoming matches
        from src.features.builder import build_features_for_upcoming
        upcoming = await build_features_for_upcoming(session)

    return {
        "status": "ok",
        "message": f"Built features for {count} historical + {upcoming} upcoming matches.",
    }


@router.post("/train")
async def train_model():
    """Train the XGBoost model and meta-learner."""
    import traceback
    from src.database import async_session
    from src.models_ml.training import train_and_evaluate

    try:
        async with async_session() as session:
            metrics = await train_and_evaluate(session, version="v1")
    except Exception as e:
        logger.exception("Training crashed")
        return {"status": "error", "message": f"Training crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

    if "error" in metrics:
        return {"status": "error", "message": f"Training failed: {metrics['error']} ({metrics.get('n_matches', 0)} matches)", "metrics": metrics}

    # Reload the model in the picks route
    try:
        from src.api.routes import picks
        from src.models_ml.xgboost_model import load_model
        picks._xgb_model = load_model("v1")
        logger.info("Reloaded XGBoost model after training")
    except Exception:
        logger.warning("Could not reload model — restart the server to use new model")

    return {"status": "ok", "message": "Training complete.", "metrics": metrics}


@router.post("/backtest")
async def run_backtest():
    """Run a walk-forward backtest over recent historical data."""
    from datetime import date, timedelta
    from src.database import async_session
    from src.learning.backtester import backtest

    end = date.today()
    start = end - timedelta(weeks=8)

    try:
        async with async_session() as session:
            results = await backtest(session, start, end)
    except Exception as e:
        logger.exception("Backtest failed")
        return {"status": "error", "message": str(e)}

    roi_str = f"{results['roi']:.1f}%" if results.get('roi') is not None else "N/A"
    msg = (f"Backtest complete: {results['total_picks']} picks, "
           f"{results['accuracy']*100:.1f}% accuracy, "
           f"ROI: {roi_str}, "
           f"Perfect days: {results['perfect_matchdays']}/{results['total_matchdays']}")
    return {"status": "ok", "message": msg, "results": results}


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

    # Check model
    from pathlib import Path
    info["xgb_model"] = Path("trained_models/xgboost_v1.joblib").exists()
    info["meta_learner"] = Path("trained_models/meta_learner_v1.joblib").exists()

    return info
