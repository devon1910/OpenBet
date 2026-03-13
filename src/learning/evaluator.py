"""Model performance evaluator.

Computes accuracy, Brier score, ROI, and breakdowns by bet type and league.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.models.match import Match
from src.models.prediction import ModelPerformance, Pick, Prediction

logger = logging.getLogger(__name__)


async def evaluate_period(
    session: AsyncSession,
    period_start: date,
    period_end: date,
    model_version: str = "v1",
) -> dict:
    """Evaluate model performance over a date range.

    Returns and stores accuracy, Brier score, ROI, and breakdowns.
    """
    stmt = (
        select(Pick, Prediction, Match)
        .join(Prediction, Prediction.id == Pick.prediction_id)
        .join(Match, Match.id == Pick.match_id)
        .where(
            Pick.outcome.is_not(None),
            Match.match_date >= datetime(period_start.year, period_start.month, period_start.day, tzinfo=timezone.utc),
            Match.match_date <= datetime(period_end.year, period_end.month, period_end.day, 23, 59, 59, tzinfo=timezone.utc),
            Prediction.model_version == model_version,
        )
        .options(joinedload(Match.competition))
    )
    result = await session.execute(stmt)
    rows = result.unique().all()

    if not rows:
        return {"error": "no_data", "period_start": str(period_start), "period_end": str(period_end)}

    total = len(rows)
    correct = sum(1 for pick, _, _ in rows if pick.outcome == "WIN")
    accuracy = correct / total if total else 0

    # Brier score
    brier_scores = []
    for pick, prediction, match in rows:
        if match.home_goals is None:
            continue
        if match.home_goals > match.away_goals:
            actual = [1, 0, 0]
        elif match.home_goals == match.away_goals:
            actual = [0, 1, 0]
        else:
            actual = [0, 0, 1]

        predicted = [
            prediction.prob_home or 0,
            prediction.prob_draw or 0,
            prediction.prob_away or 0,
        ]
        brier = sum((p - a) ** 2 for p, a in zip(predicted, actual)) / 3
        brier_scores.append(brier)

    brier_score = float(np.mean(brier_scores)) if brier_scores else None

    # ROI calculation (unit stake = 1.0 per pick)
    total_staked = 0
    total_profit = 0.0
    for pick, prediction, match in rows:
        if pick.odds_decimal is not None and pick.odds_decimal > 0:
            total_staked += 1
            if pick.outcome == "WIN":
                total_profit += pick.odds_decimal - 1.0
            else:
                total_profit -= 1.0

    roi = (total_profit / total_staked * 100) if total_staked > 0 else None

    # Breakdown by bet type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for pick, _, _ in rows:
        key = pick.pick_type
        by_type[key]["total"] += 1
        if pick.outcome == "WIN":
            by_type[key]["correct"] += 1
    type_breakdown = {
        k: {"total": v["total"], "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0}
        for k, v in by_type.items()
    }

    # Breakdown by league
    by_league = defaultdict(lambda: {"total": 0, "correct": 0})
    for pick, _, match in rows:
        league = match.competition.name if match.competition else "Unknown"
        by_league[league]["total"] += 1
        if pick.outcome == "WIN":
            by_league[league]["correct"] += 1
    league_breakdown = {
        k: {"total": v["total"], "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0}
        for k, v in by_league.items()
    }

    # Average edge
    edges = [pick.edge for pick, _, _ in rows if pick.edge is not None]
    avg_edge = float(np.mean(edges)) if edges else None

    # Save performance record
    perf = ModelPerformance(
        period_start=period_start,
        period_end=period_end,
        total_picks=total,
        correct_picks=correct,
        accuracy=accuracy,
        brier_score=brier_score,
        roi=roi,
        model_version=model_version,
    )
    session.add(perf)
    await session.commit()

    metrics = {
        "period_start": str(period_start),
        "period_end": str(period_end),
        "total_picks": total,
        "correct_picks": correct,
        "accuracy": accuracy,
        "brier_score": brier_score,
        "roi": roi,
        "avg_edge": avg_edge,
        "by_bet_type": type_breakdown,
        "by_league": league_breakdown,
        "model_version": model_version,
    }
    logger.info("Performance evaluation: %s", metrics)
    return metrics


async def evaluate_last_n_weeks(
    session: AsyncSession,
    n_weeks: int = 4,
    model_version: str = "v1",
) -> dict:
    """Evaluate model performance over the last N weeks."""
    end = date.today()
    start = end - timedelta(weeks=n_weeks)
    return await evaluate_period(session, start, end, model_version)
