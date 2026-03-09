"""Teams endpoints - team statistics and ratings."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.elo import EloRating
from src.models.team import Team

router = APIRouter()


@router.get("/{team_id}/stats")
async def get_team_stats(team_id: int, db: AsyncSession = Depends(get_db)):
    """Get team statistics including Elo rating."""
    team = (await db.execute(
        select(Team).where(Team.id == team_id)
    )).scalar_one_or_none()

    if not team:
        return {"error": "Team not found"}

    elo = (await db.execute(
        select(EloRating).where(EloRating.team_id == team_id)
    )).scalar_one_or_none()

    return {
        "id": team.id,
        "name": team.name,
        "short_name": team.short_name,
        "elo_rating": round(elo.rating, 1) if elo else None,
    }


@router.get("/")
async def list_teams(db: AsyncSession = Depends(get_db)):
    """List all teams with Elo ratings."""
    stmt = select(Team).order_by(Team.name)
    result = await db.execute(stmt)
    teams = result.scalars().all()

    output = []
    for team in teams:
        elo = (await db.execute(
            select(EloRating).where(EloRating.team_id == team.id)
        )).scalar_one_or_none()

        output.append({
            "id": team.id,
            "name": team.name,
            "short_name": team.short_name,
            "elo_rating": round(elo.rating, 1) if elo else None,
        })

    return {"teams": output, "count": len(output)}
