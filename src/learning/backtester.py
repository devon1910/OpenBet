"""Walk-forward backtester.

Simulates the prediction engine over historical data, making picks
using only information available at each point in time, then evaluates
against actual results.
"""

import logging
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.engine.betting import evaluate_betting_opportunity
from src.engine.picks import _select_top_picks
from src.learning.tracker import determine_outcome
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models_ml.ensemble import ensemble_predict
from src.models_ml.xgboost_model import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


async def backtest(
    session: AsyncSession,
    start_date: date,
    end_date: date,
    model_version: str = "v1",
) -> dict:
    """Walk-forward backtest over a date range.

    For each matchday in the range, simulates what the engine would have
    predicted using only data available at that time, then compares with
    actual results.

    Returns aggregate metrics including accuracy, ROI, and per-matchday results.
    """
    # Load all finished matches with features in the date range
    stmt = (
        select(Match, MatchFeature)
        .join(MatchFeature, MatchFeature.match_id == Match.id)
        .where(
            Match.status == "FINISHED",
            Match.home_goals.is_not(None),
            Match.match_date >= f"{start_date}T00:00:00+00:00",
            Match.match_date <= f"{end_date}T23:59:59+00:00",
        )
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.competition),
        )
        .order_by(Match.match_date)
    )
    result = await session.execute(stmt)
    rows = result.unique().all()

    if not rows:
        return {"error": "no_data", "start_date": str(start_date), "end_date": str(end_date)}

    # Group matches by date (simulate matchday batches)
    matchday_groups = defaultdict(list)
    for match, feature in rows:
        day_key = match.match_date.date() if hasattr(match.match_date, 'date') else match.match_date
        matchday_groups[day_key].append((match, feature))

    all_picks = []
    matchday_results = []

    for day_key in sorted(matchday_groups.keys()):
        day_matches = matchday_groups[day_key]

        # Build features DataFrame for this matchday
        records = []
        match_map = {}
        for match, feature in day_matches:
            record = {}
            for col in FEATURE_COLUMNS:
                val = getattr(feature, col, None)
                record[col] = val if val is not None else 0.0
            records.append(record)
            match_map[len(records) - 1] = (match, feature)

        features_df = pd.DataFrame(records)

        # Run ensemble predictions
        try:
            ensemble_results = ensemble_predict(features_df, model_version)
        except Exception:
            logger.warning("Ensemble predict failed for %s, skipping", day_key)
            continue

        # Evaluate betting opportunities for each match
        candidates = []
        for i, ens in enumerate(ensemble_results):
            match, feature = match_map[i]

            prob_home = ens["ensemble_home"]
            prob_draw = ens["ensemble_draw"]
            prob_away = ens["ensemble_away"]

            bet_picks = evaluate_betting_opportunity(
                prob_home, prob_draw, prob_away,
                odds_home=feature.odds_home,
                odds_draw=feature.odds_draw,
                odds_away=feature.odds_away,
            )

            for bp in bet_picks:
                # Determine actual outcome
                outcome = determine_outcome(
                    bp["pick_type"],
                    bp["pick_value"],
                    match.home_goals,
                    match.away_goals,
                )
                candidates.append({
                    "match": match,
                    "competition_id": match.competition_id,
                    "outcome": outcome,
                    **bp,
                })

        # Select top picks (same logic as live engine)
        selected = _select_top_picks(candidates)

        # Score this matchday
        day_total = len(selected)
        day_correct = sum(1 for p in selected if p["outcome"] == "WIN")
        day_profit = 0.0
        day_staked = 0

        for p in selected:
            odds = p.get("odds_decimal")
            if odds and odds > 0:
                day_staked += 1
                if p["outcome"] == "WIN":
                    day_profit += odds - 1.0
                else:
                    day_profit -= 1.0

        all_picks.extend(selected)

        if day_total > 0:
            matchday_results.append({
                "date": str(day_key),
                "total_picks": day_total,
                "correct_picks": day_correct,
                "accuracy": round(day_correct / day_total, 3),
                "all_correct": day_correct == day_total,
                "profit": round(day_profit, 2),
            })

    # Aggregate metrics
    total_picks = len(all_picks)
    total_correct = sum(1 for p in all_picks if p["outcome"] == "WIN")
    total_staked = 0
    total_profit = 0.0

    for p in all_picks:
        odds = p.get("odds_decimal")
        if odds and odds > 0:
            total_staked += 1
            if p["outcome"] == "WIN":
                total_profit += odds - 1.0
            else:
                total_profit -= 1.0

    # Perfect matchday rate
    perfect_days = sum(1 for mr in matchday_results if mr["all_correct"])
    total_days = len(matchday_results)

    # Breakdown by bet type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in all_picks:
        by_type[p["pick_type"]]["total"] += 1
        if p["outcome"] == "WIN":
            by_type[p["pick_type"]]["correct"] += 1

    # Average edge
    edges = [p.get("edge", 0) for p in all_picks if p.get("edge") is not None]

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "model_version": model_version,
        "total_matchdays": total_days,
        "total_picks": total_picks,
        "total_correct": total_correct,
        "accuracy": round(total_correct / total_picks, 3) if total_picks else 0,
        "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else None,
        "total_profit": round(total_profit, 2),
        "perfect_matchdays": perfect_days,
        "perfect_matchday_rate": round(perfect_days / total_days, 3) if total_days else 0,
        "avg_edge": round(float(np.mean(edges)), 4) if edges else None,
        "by_bet_type": {
            k: {"total": v["total"], "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0}
            for k, v in by_type.items()
        },
        "matchday_results": matchday_results,
    }
