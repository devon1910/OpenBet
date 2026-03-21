"""Picks endpoints - betting recommendations by date."""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.api.deps import get_db
from src.collectors.odds_api import OddsApiCollector
from src.engine.betting import evaluate_betting_opportunity
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.prediction import Pick, Prediction
from src.models_ml.ensemble import ensemble_predict
from src.models_ml.poisson import match_outcome_probabilities
from src.models_ml.xgboost_model import FEATURE_COLUMNS, load_model, prepare_features

import logging
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()

# Try to load the XGBoost model at startup
_xgb_model = None
try:
    _xgb_model = load_model("v1")
    logger.info("XGBoost model v1 loaded successfully")
except Exception:
    logger.warning("XGBoost model not found — predictions will use Poisson only")


def _build_reasoning(pick_type: str, pick_value: str, ctx: dict) -> str:
    """Generate a short human-readable reason for a pick."""
    home = ctx["home"]
    away = ctx["away"]
    ph = ctx["prob_home"]
    pd_ = ctx["prob_draw"]
    pa = ctx["prob_away"]
    feat = ctx.get("feature")

    parts = []

    # Core probability statement
    if pick_type == "STRAIGHT_WIN" and pick_value == "HOME":
        parts.append(f"{home} have a {ph:.0%} win probability, making them clear favourites.")
    elif pick_type == "STRAIGHT_WIN" and pick_value == "AWAY":
        parts.append(f"{away} have a {pa:.0%} win probability, making them clear favourites.")
    elif pick_value == "1X":
        parts.append(f"{home} have a {ph + pd_:.0%} combined chance of winning or drawing.")
    elif pick_value == "X2":
        parts.append(f"{away} have a {pa + pd_:.0%} combined chance of winning or drawing.")
    else:
        parts.append(f"Probabilities: {home} {ph:.0%}, Draw {pd_:.0%}, {away} {pa:.0%}.")

    # Supporting factors
    factors = []
    if feat:
        if feat.elo_diff is not None:
            diff = feat.elo_diff
            if abs(diff) > 80:
                stronger = home if diff > 0 else away
                factors.append(f"{stronger} are Elo-rated significantly higher")
            elif abs(diff) < 30:
                factors.append("both teams are closely rated by Elo")

        if feat.home_form is not None and feat.away_form is not None:
            fd = feat.home_form - feat.away_form
            if fd > 5:
                factors.append(f"{home} are in stronger recent form")
            elif fd < -5:
                factors.append(f"{away} are in stronger recent form")

        if feat.home_advantage is not None and feat.home_advantage > 0.55:
            factors.append(f"{home} have a strong home record")

        if feat.h2h_home_wins is not None:
            total = (feat.h2h_home_wins or 0) + (feat.h2h_draws or 0) + (feat.h2h_away_wins or 0)
            if total >= 3:
                if feat.h2h_home_wins / total > 0.6:
                    factors.append(f"{home} dominate the head-to-head record")
                elif feat.h2h_away_wins / total > 0.6:
                    factors.append(f"{away} dominate the head-to-head record")

    if factors:
        parts.append(" ".join(f.capitalize() if i == 0 else f for i, f in enumerate(factors[:2])) + ".")

    return " ".join(parts)


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
    """Run the prediction engine for matches on a specific date.

    Uses Poisson model directly for on-the-fly predictions when no
    pre-computed picks exist yet.
    """
    start, end = _date_range(target_date)

    # Find matches on this date that haven't finished yet
    stmt = (
        select(Match)
        .where(
            Match.match_date >= start,
            Match.match_date < end,
            Match.status.notin_(["FINISHED"]),
        )
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.competition),
            joinedload(Match.features),
        )
    )
    result = await db.execute(stmt)
    matches = result.unique().scalars().all()

    if not matches:
        return []

    all_candidates = []

    for match in matches:
        feature = match.features

        # Build feature dict for this match
        feat_dict = {}
        if feature:
            feat_dict = {col: getattr(feature, col, None) for col in FEATURE_COLUMNS}
        else:
            feat_dict = {col: None for col in FEATURE_COLUMNS}

        # Poisson baseline (always available)
        home_attack = feat_dict.get("home_attack_strength") or 1.0
        home_defense = feat_dict.get("home_defense_strength") or 1.0
        away_attack = feat_dict.get("away_attack_strength") or 1.0
        away_defense = feat_dict.get("away_defense_strength") or 1.0

        poisson_result = match_outcome_probabilities(
            home_attack=home_attack,
            home_defense=home_defense,
            away_attack=away_attack,
            away_defense=away_defense,
        )

        # Try stacking ensemble (Poisson + XGBoost + odds via meta-learner)
        prob_home = poisson_result["home_win"]
        prob_draw = poisson_result["draw"]
        prob_away = poisson_result["away_win"]

        if _xgb_model is not None and feature:
            try:
                feat_df = pd.DataFrame([feat_dict])
                ens_results = ensemble_predict(feat_df, model_version="v1")
                if ens_results:
                    prob_home = ens_results[0]["ensemble_home"]
                    prob_draw = ens_results[0]["ensemble_draw"]
                    prob_away = ens_results[0]["ensemble_away"]
            except Exception:
                logger.debug("Ensemble failed for match %s, using Poisson only", match.id)

        # Save prediction
        prediction = Prediction(
            match_id=match.id,
            model_version="live",
            poisson_home=poisson_result["home_win"],
            poisson_draw=poisson_result["draw"],
            poisson_away=poisson_result["away_win"],
            ensemble_home=prob_home,
            ensemble_draw=prob_draw,
            ensemble_away=prob_away,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
        )
        db.add(prediction)
        await db.flush()

        # Build context dict for reasoning
        ctx = {
            "home": match.home_team.name,
            "away": match.away_team.name,
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "feature": feature,
        }

        # Evaluate betting opportunities with value edge filtering
        odds_home = feature.odds_home if feature else None
        odds_draw = feature.odds_draw if feature else None
        odds_away = feature.odds_away if feature else None

        bet_picks = evaluate_betting_opportunity(
            prob_home, prob_draw, prob_away,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
        )

        for bp in bet_picks:
            bp["reasoning"] = _build_reasoning(bp["pick_type"], bp["pick_value"], ctx)
            all_candidates.append({
                "prediction": prediction,
                "match": match,
                "competition_id": match.competition_id,
                **bp,
            })

    # Sort by edge (value) then confidence, diversify across leagues, take top picks
    from src.config import settings
    all_candidates.sort(key=lambda x: (x.get("edge", 0), x["confidence"]), reverse=True)
    league_counts: dict[int, int] = defaultdict(int)
    selected = []
    for c in all_candidates:
        if len(selected) >= settings.max_picks_per_matchday:
            break
        comp_id = c["competition_id"]
        if league_counts[comp_id] >= 2:
            continue
        selected.append(c)
        league_counts[comp_id] += 1

    # Save picks and format output
    output = []
    for p in selected:
        pick = Pick(
            prediction_id=p["prediction"].id,
            match_id=p["match"].id,
            pick_type=p["pick_type"],
            pick_value=p["pick_value"],
            confidence=p["confidence"],
            edge=p.get("edge"),
            odds_decimal=p.get("odds_decimal"),
            reasoning=p.get("reasoning", ""),
            matchday_label=f"MD{p['match'].matchday or '?'}",
        )
        db.add(pick)
        await db.flush()
        output.append(_format_pick(pick, p["match"]))

    await db.commit()
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

    if force:
        # Delete existing picks and predictions for this date
        start, end = _date_range(parsed_date)
        match_ids_stmt = select(Match.id).where(Match.match_date >= start, Match.match_date < end)
        match_ids = (await db.execute(match_ids_stmt)).scalars().all()
        if match_ids:
            from sqlalchemy import delete
            await db.execute(delete(Pick).where(Pick.match_id.in_(match_ids)))
            await db.execute(delete(Prediction).where(Prediction.match_id.in_(match_ids)))
            await db.commit()

        # Fetch latest odds before regenerating so predictions use current market prices
        try:
            odds_collector = OddsApiCollector()
            updated = await odds_collector.enrich_odds(db)
            logger.info("Refreshed odds for %d matches before regeneration", updated)
        except Exception:
            logger.warning("Failed to refresh odds before regeneration — using cached odds")

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
