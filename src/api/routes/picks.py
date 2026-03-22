"""Picks endpoints - betting recommendations by date."""

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.api.deps import get_db
from src.collectors.football_data import FootballDataCollector
from src.collectors.odds_api import OddsApiCollector
from src.engine.picks import generate_predictions_and_picks
from src.features.bulk_builder import bulk_build_features
from src.features.elo import process_all_matches
from src.models.match import Match
from src.models.prediction import Pick, Prediction

import logging

logger = logging.getLogger(__name__)

router = APIRouter()



def _format_pick(pick: Pick, match: Match) -> dict:
    """Format a pick for API response."""
    return {
        "id": pick.id,
        "match": f"{match.home_team.name} vs {match.away_team.name}",
        "home_team": match.home_team.name,
        "away_team": match.away_team.name,
        "competition": match.competition.name if match.competition else "",
        "match_date": match.match_date.isoformat() if match.match_date else None,
        "kick_off": match.match_date.strftime("%H:%M") if match.match_date else None,
        "pick_type": pick.pick_type,
        "pick_value": pick.pick_value,
        "confidence": round(pick.confidence, 3),
        "reasoning": pick.reasoning,
        "probabilities": {
            "home_win": round(pick.prediction.prob_home, 3) if pick.prediction and pick.prediction.prob_home else None,
            "draw": round(pick.prediction.prob_draw, 3) if pick.prediction and pick.prediction.prob_draw else None,
            "away_win": round(pick.prediction.prob_away, 3) if pick.prediction and pick.prediction.prob_away else None,
        },
    }


def _date_range(target_date: date) -> tuple[datetime, datetime]:
    """Convert a date to a UTC datetime range (start, end)."""
    start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


async def _get_picks_for_date(target_date: date, db: AsyncSession) -> list[dict]:
    """Get existing picks for a given date."""
    start, end = _date_range(target_date)

    stmt = (
        select(Pick)
        .join(Match, Match.id == Pick.match_id)
        .where(
            Match.match_date >= start,
            Match.match_date < end,
        )
        .options(joinedload(Pick.prediction))
        .order_by(Pick.confidence.desc())
    )
    result = await db.execute(stmt)
    picks = result.scalars().unique().all()

    output = []
    for pick in picks:
        match = (await db.execute(
            select(Match)
            .where(Match.id == pick.match_id)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.competition),
            )
        )).unique().scalar_one()
        output.append(_format_pick(pick, match))

    return output


async def _run_predictions_for_date(target_date: date, db: AsyncSession) -> list[dict]:
    """Run the full prediction engine for matches on a specific date.

    Uses the same pipeline as the scheduler: ensemble + Claude reasoning.
    """
    pick_objects = await generate_predictions_and_picks(db, target_date=target_date)

    if not pick_objects:
        return []

    # Re-fetch picks with relationships eagerly loaded for formatting
    output = []
    pick_ids = [p.id for p in pick_objects]
    fresh_picks = (await db.execute(
        select(Pick)
        .where(Pick.id.in_(pick_ids))
        .options(joinedload(Pick.prediction))
        .order_by(Pick.confidence.desc())
    )).scalars().unique().all()

    for pick in fresh_picks:
        match = (await db.execute(
            select(Match)
            .where(Match.id == pick.match_id)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.competition),
            )
        )).unique().scalar_one()
        output.append(_format_pick(pick, match))

    return output


@router.get("/date/{target_date}")
async def get_picks_by_date(target_date: str, force: bool = False, db: AsyncSession = Depends(get_db)):
    """Get predictions for a specific date. Runs the engine if no picks exist yet.

    Date format: YYYY-MM-DD
    Pass ?force=true to regenerate predictions.
    """
    try:
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD.", "picks": [], "count": 0}

    try:
        return await _get_picks_by_date_inner(parsed_date, force, db)
    except Exception:
        logger.exception("Failed to get picks for %s (force=%s)", target_date, force)
        raise


async def _get_picks_by_date_inner(parsed_date: date, force: bool, db: AsyncSession):
    if force:
        # 1. Light sync — fetch latest match data (status changes, new matches)
        try:
            sync_collector = FootballDataCollector()
            try:
                await sync_collector.sync_matches_only(db)
                logger.info("Light sync complete")
            finally:
                await sync_collector.close()
        except Exception:
            logger.warning("Light sync failed — continuing with existing data")

        # 2. Elo update (incremental, fast if already current)
        try:
            await process_all_matches(db)
        except Exception:
            logger.warning("Elo update failed — continuing without Elo refresh")

        # 3. Build/rebuild features for matches on this date
        start, end = _date_range(parsed_date)
        try:
            date_matches = (await db.execute(
                select(Match).where(
                    Match.match_date >= start,
                    Match.match_date < end,
                )
            )).scalars().all()
            if date_matches:
                built = await bulk_build_features(db, date_matches)
                logger.info("Built features for %d matches on %s", built, parsed_date)
        except Exception:
            logger.warning("Feature building failed — continuing with existing features")

        # 4. Fetch latest odds
        try:
            odds_collector = OddsApiCollector()
            try:
                updated = await odds_collector.enrich_odds(db)
                logger.info("Refreshed odds for %d matches", updated)
            finally:
                await odds_collector.close()
        except Exception:
            logger.warning("Failed to refresh odds — using cached odds")

        # 5. Delete existing unresolved picks/predictions
        match_ids_stmt = select(Match.id).where(
            Match.match_date >= start,
            Match.match_date < end,
        )
        match_ids = (await db.execute(match_ids_stmt)).scalars().all()
        if match_ids:
            from sqlalchemy import delete
            await db.execute(
                delete(Pick).where(
                    Pick.match_id.in_(match_ids),
                    Pick.outcome.is_(None),
                )
            )
            pred_with_picks = select(Pick.prediction_id).where(Pick.prediction_id.is_not(None))
            await db.execute(
                delete(Prediction).where(
                    Prediction.match_id.in_(match_ids),
                    Prediction.id.notin_(pred_with_picks),
                )
            )
            await db.commit()

    # Check for existing picks first
    picks = await _get_picks_for_date(parsed_date, db)

    if not picks:
        # Run the prediction engine for this date
        picks = await _run_predictions_for_date(parsed_date, db)

    if not picks:
        return {
            "date": str(parsed_date),
            "picks": [],
            "count": 0,
            "message": "No available games on this date. Please check back later.",
        }

    return {"date": str(parsed_date), "picks": picks, "count": len(picks)}


@router.get("/today")
async def get_today_picks(db: AsyncSession = Depends(get_db)):
    """Get today's betting picks."""
    return await get_picks_by_date(date.today().isoformat(), db)


@router.get("/history")
async def get_pick_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get historical picks with outcomes."""
    stmt = (
        select(Pick)
        .where(Pick.outcome.is_not(None))
        .order_by(Pick.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    picks = result.scalars().all()

    output = []
    for pick in picks:
        match = (await db.execute(
            select(Match)
            .where(Match.id == pick.match_id)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
            )
        )).scalar_one()

        output.append({
            "id": pick.id,
            "match": f"{match.home_team.name} vs {match.away_team.name}",
            "pick_type": pick.pick_type,
            "pick_value": pick.pick_value,
            "confidence": round(pick.confidence, 3),
            "outcome": pick.outcome,
        })

    return {"picks": output, "count": len(output)}
