"""Predictions endpoints - detailed match predictions."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.api.deps import get_db
from src.models.match import Match
from src.models.prediction import Prediction

router = APIRouter()


@router.get("/matchday/{competition_code}/{matchday}")
async def get_matchday_predictions(
    competition_code: str,
    matchday: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all predictions for a specific matchday."""
    stmt = (
        select(Prediction)
        .join(Match, Match.id == Prediction.match_id)
        .join(Match.competition)
        .where(
            Match.competition.has(external_id=competition_code.upper()),
            Match.matchday == matchday,
        )
        .options(joinedload(Prediction.match).joinedload(Match.home_team))
    )
    result = await db.execute(stmt)
    predictions = result.scalars().unique().all()

    output = []
    for pred in predictions:
        match = pred.match
        # Load away team
        away = (await db.execute(
            select(Match).where(Match.id == match.id).options(joinedload(Match.away_team))
        )).scalar_one()

        output.append({
            "match": f"{match.home_team.name} vs {away.away_team.name}",
            "match_date": match.match_date.isoformat() if match.match_date else None,
            "probabilities": {
                "home_win": round(pred.prob_home, 3) if pred.prob_home else None,
                "draw": round(pred.prob_draw, 3) if pred.prob_draw else None,
                "away_win": round(pred.prob_away, 3) if pred.prob_away else None,
            },
            "model_breakdown": {
                "poisson": {
                    "home": round(pred.poisson_home, 3) if pred.poisson_home else None,
                    "draw": round(pred.poisson_draw, 3) if pred.poisson_draw else None,
                    "away": round(pred.poisson_away, 3) if pred.poisson_away else None,
                },
                "xgboost": {
                    "home": round(pred.xgb_home, 3) if pred.xgb_home else None,
                    "draw": round(pred.xgb_draw, 3) if pred.xgb_draw else None,
                    "away": round(pred.xgb_away, 3) if pred.xgb_away else None,
                },
            },
            "claude_reasoning": pred.claude_reasoning,
            "claude_adjustment": pred.claude_confidence_adj,
        })

    return {
        "competition": competition_code.upper(),
        "matchday": matchday,
        "predictions": output,
        "count": len(output),
    }
