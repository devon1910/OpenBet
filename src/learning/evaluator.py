"""Model performance evaluator.

Computes accuracy, Brier score, and ROI over specified periods.
"""

import logging
from datetime import date, timedelta

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

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

    Returns and stores accuracy, Brier score, and ROI metrics.
    """
    stmt = (
        select(Pick, Prediction, Match)
        .join(Prediction, Prediction.id == Pick.prediction_id)
        .join(Match, Match.id == Pick.match_id)
        .where(
            Pick.outcome.is_not(None),
            Match.match_date >= f"{period_start}T00:00:00+00:00",
            Match.match_date <= f"{period_end}T23:59:59+00:00",
            Prediction.model_version == model_version,
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return {"error": "no_data", "period_start": str(period_start), "period_end": str(period_end)}

    total = len(rows)
    correct = sum(1 for pick, _, _ in rows if pick.outcome == "WIN")
    accuracy = correct / total if total else 0

    # Brier score: mean squared error of predicted probability vs actual outcome
    brier_scores = []
    for pick, prediction, match in rows:
        if match.home_goals is None:
            continue
        # Actual outcome as one-hot
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

    # Save performance record
    perf = ModelPerformance(
        period_start=period_start,
        period_end=period_end,
        total_picks=total,
        correct_picks=correct,
        accuracy=accuracy,
        brier_score=brier_score,
        roi=None,  # ROI requires odds data which we don't have yet
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
