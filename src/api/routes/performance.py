"""Performance endpoints - model accuracy, ROI, and breakdowns."""

from collections import defaultdict
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.learning.backtester import backtest
from src.learning.evaluator import evaluate_period
from src.models.prediction import ModelPerformance, Pick

router = APIRouter()


@router.get("/")
async def get_performance(db: AsyncSession = Depends(get_db)):
    """Get overall model performance metrics with ROI and breakdowns."""
    # Latest performance records
    stmt = (
        select(ModelPerformance)
        .order_by(ModelPerformance.computed_at.desc())
        .limit(5)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    # All-time stats
    total = (await db.execute(
        select(func.count(Pick.id)).where(Pick.outcome.is_not(None))
    )).scalar() or 0

    wins = (await db.execute(
        select(func.count(Pick.id)).where(Pick.outcome == "WIN")
    )).scalar() or 0

    # All-time ROI + breakdown by bet type
    all_picks_stmt = select(Pick).where(Pick.outcome.is_not(None))
    all_picks_result = await db.execute(all_picks_stmt)
    all_picks = all_picks_result.scalars().all()

    total_staked = 0
    total_profit = 0.0
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "profit": 0.0, "staked": 0})

    for p in all_picks:
        bt = by_type[p.pick_type]
        bt["total"] += 1
        if p.outcome == "WIN":
            bt["correct"] += 1

        if p.odds_decimal is not None and p.odds_decimal > 0:
            total_staked += 1
            bt["staked"] += 1
            if p.outcome == "WIN":
                profit = p.odds_decimal - 1.0
            else:
                profit = -1.0
            total_profit += profit
            bt["profit"] += profit

    type_breakdown = {
        k: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0,
            "roi": round(v["profit"] / v["staked"] * 100, 2) if v["staked"] > 0 else None,
        }
        for k, v in by_type.items()
    }

    return {
        "all_time": {
            "total_picks": total,
            "correct_picks": wins,
            "accuracy": round(wins / total, 3) if total else None,
            "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else None,
        },
        "by_bet_type": type_breakdown,
        "recent_evaluations": [
            {
                "period": f"{r.period_start} to {r.period_end}",
                "accuracy": round(r.accuracy, 3) if r.accuracy else None,
                "brier_score": round(r.brier_score, 3) if r.brier_score else None,
                "roi": round(r.roi, 2) if r.roi is not None else None,
                "total_picks": r.total_picks,
                "model_version": r.model_version,
            }
            for r in records
        ],
    }


@router.post("/evaluate")
async def run_evaluation(
    weeks: int = Query(4, ge=1, le=52),
    model_version: str = Query("v1"),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a performance evaluation for the last N weeks."""
    end = date.today()
    start = end - timedelta(weeks=weeks)
    return await evaluate_period(db, start, end, model_version)


@router.post("/backtest")
async def run_backtest(
    start_date: date = Query(...),
    end_date: date = Query(...),
    model_version: str = Query("v1"),
    db: AsyncSession = Depends(get_db),
):
    """Run a walk-forward backtest over a date range.

    Returns accuracy, ROI, perfect matchday rate, and per-day breakdowns.
    """
    return await backtest(db, start_date, end_date, model_version)
