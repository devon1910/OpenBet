"""Final pick aggregation.

Ranks all betting opportunities across matches, selects top 10-12,
and diversifies across leagues to avoid correlated failures.
"""

import logging
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.config import settings
from src.engine.betting import evaluate_betting_opportunity
from src.engine.claude_reasoning import get_batch_reasoning
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.prediction import Pick, Prediction
from src.models.team import Competition
from src.models_ml.ensemble import ensemble_predict
from src.models_ml.xgboost_model import FEATURE_COLUMNS

import pandas as pd

logger = logging.getLogger(__name__)


async def generate_predictions_and_picks(
    session: AsyncSession,
    model_version: str = "v1",
    target_date: "date | None" = None,
) -> list[Pick]:
    """Full pipeline: features → ensemble → Claude reasoning → picks.

    Args:
        target_date: If provided, only predict matches on this specific date.
                     Otherwise, predict all upcoming scheduled matches.

    Returns list of Pick objects (already added to session).
    """
    from datetime import date as date_type, datetime, timedelta, timezone

    # Load matches with features
    stmt = (
        select(Match, MatchFeature)
        .join(MatchFeature, MatchFeature.match_id == Match.id)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.competition),
        )
    )

    if target_date is not None:
        # Predict matches on the given date that haven't kicked off yet
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        now = datetime.now(timezone.utc)
        # For today, only include future matches; for other dates use start of day
        cutoff = max(start, now) if target_date <= date_type.today() else start
        stmt = stmt.where(Match.match_date >= cutoff, Match.match_date < end)
    else:
        # Default: only upcoming scheduled matches
        stmt = stmt.where(Match.status.in_(["SCHEDULED", "TIMED"]))

    result = await session.execute(stmt)
    rows = result.unique().all()

    if not rows:
        logger.info("No upcoming matches with features found")
        return []

    # Build features DataFrame
    records = []
    match_map = {}
    for match, feature in rows:
        record = {col: getattr(feature, col) for col in FEATURE_COLUMNS}
        records.append(record)
        match_map[len(records) - 1] = (match, feature)

    features_df = pd.DataFrame(records)

    # Ensemble predictions
    ensemble_results = ensemble_predict(features_df, model_version)

    # Batch Claude reasoning — single API call for all matches
    claude_inputs = []
    for i, ens in enumerate(ensemble_results):
        match, feature = match_map[i]
        claude_inputs.append({
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
            "competition": match.competition.name,
            "model_probs": ens,
            "context": {
                "home_form_str": f"Form score: {feature.home_form:.2f}" if feature.home_form else "N/A",
                "away_form_str": f"Form score: {feature.away_form:.2f}" if feature.away_form else "N/A",
                "home_position": "N/A",
                "away_position": "N/A",
            },
        })

    try:
        reasoning_results = await get_batch_reasoning(claude_inputs)
    except Exception:
        logger.warning("Claude reasoning failed, proceeding without adjustments")
        reasoning_results = [
            {"confidence_adjustment": 0, "reasoning": "", "flags": [], "unpredictable": False}
            for _ in claude_inputs
        ]

    # Process each match with its reasoning
    all_candidates = []

    for i, ens in enumerate(ensemble_results):
        match, feature = match_map[i]
        reasoning = reasoning_results[i]

        # Apply Claude adjustment to the favored outcome
        adj = reasoning["confidence_adjustment"]
        prob_home = ens["ensemble_home"] + adj * (1 if ens["ensemble_home"] > ens["ensemble_away"] else -1)
        prob_draw = ens["ensemble_draw"]
        prob_away = ens["ensemble_away"] + adj * (-1 if ens["ensemble_home"] > ens["ensemble_away"] else 1)

        # Normalize
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total

        # Save prediction
        prediction = Prediction(
            match_id=match.id,
            model_version=model_version,
            poisson_home=ens["poisson_home"],
            poisson_draw=ens["poisson_draw"],
            poisson_away=ens["poisson_away"],
            xgb_home=ens["xgb_home"],
            xgb_draw=ens["xgb_draw"],
            xgb_away=ens["xgb_away"],
            ensemble_home=ens["ensemble_home"],
            ensemble_draw=ens["ensemble_draw"],
            ensemble_away=ens["ensemble_away"],
            claude_reasoning=reasoning.get("reasoning", ""),
            claude_confidence_adj=adj,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
        )
        session.add(prediction)
        await session.flush()

        # Evaluate betting opportunities (with market odds for value detection)
        bet_picks = evaluate_betting_opportunity(
            prob_home, prob_draw, prob_away,
            odds_home=feature.odds_home,
            odds_draw=feature.odds_draw,
            odds_away=feature.odds_away,
            unpredictable=reasoning.get("unpredictable", False),
        )

        for bp in bet_picks:
            all_candidates.append({
                "prediction": prediction,
                "match": match,
                "competition_id": match.competition_id,
                **bp,
            })

    # Select top picks with diversification
    picks = _select_top_picks(all_candidates)

    # Save picks
    pick_objects = []
    for p in picks:
        pick = Pick(
            prediction_id=p["prediction"].id,
            match_id=p["match"].id,
            pick_type=p["pick_type"],
            pick_value=p["pick_value"],
            confidence=p["confidence"],
            edge=p.get("edge"),
            odds_decimal=p.get("odds_decimal"),
            reasoning=p["prediction"].claude_reasoning,
            matchday_label=f"MD{p['match'].matchday or '?'}",
        )
        session.add(pick)
        pick_objects.append(pick)

    await session.commit()
    logger.info("Generated %d picks from %d candidates", len(pick_objects), len(all_candidates))
    return pick_objects


def _select_top_picks(
    candidates: list[dict],
    max_picks: int = None,
    max_per_league: int = 4,
) -> list[dict]:
    """Select top picks, ensuring diversification across leagues."""
    if max_picks is None:
        max_picks = settings.max_picks_per_matchday

    # Sort by edge (value) descending, then confidence as tiebreaker
    candidates.sort(key=lambda x: (x.get("edge", 0), x["confidence"]), reverse=True)

    selected = []
    league_counts = defaultdict(int)

    for c in candidates:
        if len(selected) >= max_picks:
            break
        comp_id = c["competition_id"]
        if league_counts[comp_id] >= max_per_league:
            continue
        selected.append(c)
        league_counts[comp_id] += 1

    return selected
