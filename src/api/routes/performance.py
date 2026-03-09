"""Performance endpoints - model accuracy and metrics."""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.prediction import ModelPerformance, Pick

router = APIRouter()


@router.get("/")
async def get_performance(db: AsyncSession = Depends(get_db)):
    """Get overall model performance metrics."""
    # Latest performance record
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

    return {
        "all_time": {
            "total_picks": total,
            "correct_picks": wins,
            "accuracy": round(wins / total, 3) if total else None,
        },
        "recent_evaluations": [
            {
                "period": f"{r.period_start} to {r.period_end}",
                "accuracy": round(r.accuracy, 3) if r.accuracy else None,
                "brier_score": round(r.brier_score, 3) if r.brier_score else None,
                "total_picks": r.total_picks,
                "model_version": r.model_version,
            }
            for r in records
        ],
    }
