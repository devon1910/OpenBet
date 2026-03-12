"""Picks endpoints - betting recommendations by date."""

import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.api.deps import get_db
from src.engine.betting import evaluate_betting_opportunity
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.prediction import Pick, Prediction
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

    # Find matches on this date, with optional features
    stmt = (
        select(Match)
        .where(
            Match.match_date >= start,
            Match.match_date < end,
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

        # Use feature data if available, otherwise defaults
        home_attack = feature.home_attack_strength if feature and feature.home_attack_strength else 1.0
        home_defense = feature.home_defense_strength if feature and feature.home_defense_strength else 1.0
        away_attack = feature.away_attack_strength if feature and feature.away_attack_strength else 1.0
        away_defense = feature.away_defense_strength if feature and feature.away_defense_strength else 1.0

        poisson_result = match_outcome_probabilities(
            home_attack=home_attack,
            home_defense=home_defense,
            away_attack=away_attack,
            away_defense=away_defense,
        )

        prob_home = poisson_result["home_win"]
        prob_draw = poisson_result["draw"]
        prob_away = poisson_result["away_win"]

        # --- 0. Blend with XGBoost if model available ---
        if _xgb_model is not None and feature:
            try:
                feat_dict = {col: getattr(feature, col, None) for col in FEATURE_COLUMNS}
                feat_df = pd.DataFrame([feat_dict])
                feat_df = prepare_features(feat_df)
                xgb_probs = _xgb_model.predict_proba(feat_df)[0]
                # Blend: 55% Poisson + 45% XGBoost
                prob_home = 0.55 * prob_home + 0.45 * float(xgb_probs[0])
                prob_draw = 0.55 * prob_draw + 0.45 * float(xgb_probs[1])
                prob_away = 0.55 * prob_away + 0.45 * float(xgb_probs[2])
            except Exception:
                logger.debug("XGBoost prediction failed for match %s, using Poisson only", match.id)

        # --- 1. Integrate xG data when available ---
        # Use xG to adjust expected goals before blending, treated as a
        # separate signal weighted alongside Poisson.
        if (feature and feature.home_xg_avg is not None
                and feature.away_xg_avg is not None):
            # xG-based goal expectations adjust the Poisson baseline
            xg_home_goals = feature.home_xg_avg
            xg_away_goals = feature.away_xg_avg
            xg_result = match_outcome_probabilities(
                home_attack=xg_home_goals / max(home_attack, 0.01),
                home_defense=home_defense,
                away_attack=xg_away_goals / max(away_attack, 0.01),
                away_defense=away_defense,
            )
            # Blend Poisson (60%) with xG-adjusted Poisson (40%)
            prob_home = 0.6 * prob_home + 0.4 * xg_result["home_win"]
            prob_draw = 0.6 * prob_draw + 0.4 * xg_result["draw"]
            prob_away = 0.6 * prob_away + 0.4 * xg_result["away_win"]

        # --- 2. Integrate home advantage ---
        if feature and feature.home_advantage is not None:
            ha = feature.home_advantage  # float 0-1, historical home win %
            # Neutral expectation is ~0.46 (league avg home win rate).
            # Shift home prob proportionally to how much this venue exceeds avg.
            ha_boost = (ha - 0.46) * 0.15  # capped adjustment
            ha_boost = max(min(ha_boost, 0.06), -0.04)
            prob_home += ha_boost
            prob_away -= ha_boost * 0.6
            prob_draw -= ha_boost * 0.4

        # --- 3. Improved Elo blending with proper draw model ---
        if feature and feature.elo_diff is not None:
            elo_diff = feature.elo_diff
            elo_expected = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
            elo_away_exp = 1.0 - elo_expected

            # Draw probability from Elo: draws are most likely when teams
            # are evenly matched. Use a Gaussian-like model centered at 0.
            # Base draw rate ~0.26 at elo_diff=0, decreasing as diff grows.
            elo_draw = 0.26 * math.exp(-(elo_diff ** 2) / (2 * 250 ** 2))
            elo_draw = max(elo_draw, 0.08)  # floor: even mismatches have some draw chance

            elo_home = elo_expected * (1.0 - elo_draw)
            elo_away_adj = elo_away_exp * (1.0 - elo_draw)

            # Blend: 55% Poisson, 45% Elo (Elo is a strong signal)
            prob_home = 0.55 * prob_home + 0.45 * elo_home
            prob_draw = 0.55 * prob_draw + 0.45 * elo_draw
            prob_away = 0.55 * prob_away + 0.45 * elo_away_adj

        # --- 4. Blend in form if available ---
        if feature and feature.home_form is not None and feature.away_form is not None:
            form_diff = feature.home_form - feature.away_form
            # form_diff ranges roughly -15 to +15, normalize to small adjustment
            form_adj = max(min(form_diff / 150.0, 0.06), -0.06)
            prob_home += form_adj
            prob_away -= form_adj
            # When form is very close, teams are evenly matched -> boost draw
            if abs(form_diff) < 3:
                draw_boost = 0.02 * (1.0 - abs(form_diff) / 3.0)
                prob_draw += draw_boost
                prob_home -= draw_boost * 0.5
                prob_away -= draw_boost * 0.5

        # --- 5. Head-to-head adjustment ---
        if feature and feature.h2h_home_wins is not None:
            h2h_total = (feature.h2h_home_wins or 0) + (feature.h2h_draws or 0) + (feature.h2h_away_wins or 0)
            if h2h_total >= 3:
                h2h_home_rate = feature.h2h_home_wins / h2h_total
                h2h_away_rate = feature.h2h_away_wins / h2h_total
                h2h_draw_rate = feature.h2h_draws / h2h_total
                prob_home = 0.90 * prob_home + 0.10 * h2h_home_rate
                prob_draw = 0.90 * prob_draw + 0.10 * h2h_draw_rate
                prob_away = 0.90 * prob_away + 0.10 * h2h_away_rate

        # --- 6. Calibration: boost draws when home/away are close ---
        margin = abs(prob_home - prob_away)
        if margin < 0.05:
            # Very close match - increase draw probability
            cal_boost = 0.03 * (1.0 - margin / 0.05)
            prob_draw += cal_boost
            prob_home -= cal_boost * 0.5
            prob_away -= cal_boost * 0.5
        elif margin < 0.10:
            cal_boost = 0.015 * (1.0 - (margin - 0.05) / 0.05)
            prob_draw += cal_boost
            prob_home -= cal_boost * 0.5
            prob_away -= cal_boost * 0.5

        # Ensure draw probability has a realistic floor (~12%) and
        # doesn't exceed a realistic ceiling (~35%)
        prob_draw = max(prob_draw, 0.12)
        prob_draw = min(prob_draw, 0.35)

        # Normalize
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total

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

        # Evaluate betting opportunities (threshold-based)
        bet_picks = evaluate_betting_opportunity(prob_home, prob_draw, prob_away)

        if bet_picks:
            for bp in bet_picks:
                bp["reasoning"] = _build_reasoning(bp["pick_type"], bp["pick_value"], ctx)
                all_candidates.append({
                    "prediction": prediction,
                    "match": match,
                    "competition_id": match.competition_id,
                    **bp,
                })
        else:
            # No threshold met — pick the best available option
            home_not_lose = prob_home + prob_draw
            away_not_lose = prob_away + prob_draw

            # Build all candidate options for this match and pick the best
            options = [
                ("STRAIGHT_WIN", "HOME", prob_home),
                ("STRAIGHT_WIN", "AWAY", prob_away),
                ("DOUBLE_CHANCE", "1X", home_not_lose),
                ("DOUBLE_CHANCE", "X2", away_not_lose),
            ]

            # Pick selection: use the dominant probability to decide pick type.
            # Straight win when one side is clearly stronger (>40%).
            # Draw pick when draw is the highest probability.
            # Double chance only for genuinely uncertain matches.
            margin = prob_home - prob_away
            max_prob = max(prob_home, prob_draw, prob_away)

            if prob_home >= 0.40 and prob_home == max_prob:
                pick_type, pick_value, conf = "STRAIGHT_WIN", "HOME", prob_home
            elif prob_away >= 0.40 and prob_away == max_prob:
                pick_type, pick_value, conf = "STRAIGHT_WIN", "AWAY", prob_away
            elif prob_draw >= 0.30 and prob_draw == max_prob:
                # Draw is the most likely outcome — pick draw via double chance
                # on the weaker side (higher value bet)
                if prob_home >= prob_away:
                    pick_type, pick_value, conf = "DOUBLE_CHANCE", "1X", home_not_lose
                else:
                    pick_type, pick_value, conf = "DOUBLE_CHANCE", "X2", away_not_lose
            elif prob_home > prob_away and margin > 0.05:
                pick_type, pick_value, conf = "STRAIGHT_WIN", "HOME", prob_home
            elif prob_away > prob_home and margin < -0.05:
                pick_type, pick_value, conf = "STRAIGHT_WIN", "AWAY", prob_away
            elif prob_home >= prob_away:
                pick_type, pick_value, conf = "DOUBLE_CHANCE", "1X", home_not_lose
            else:
                pick_type, pick_value, conf = "DOUBLE_CHANCE", "X2", away_not_lose

            reasoning = _build_reasoning(pick_type, pick_value, ctx)
            all_candidates.append({
                "prediction": prediction,
                "match": match,
                "competition_id": match.competition_id,
                "pick_type": pick_type,
                "pick_value": pick_value,
                "confidence": conf,
                "reasoning": reasoning,
            })

    # Sort by confidence, diversify across leagues, take top 12
    all_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    league_counts: dict[int, int] = defaultdict(int)
    selected = []
    for c in all_candidates:
        if len(selected) >= 12:
            break
        comp_id = c["competition_id"]
        if league_counts[comp_id] >= 4:
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
