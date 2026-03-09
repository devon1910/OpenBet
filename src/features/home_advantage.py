"""Home advantage feature.

Computes the team's home win percentage this season.
"""

from sqlalchemy import Integer, case, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match


async def compute_home_advantage(
    session: AsyncSession,
    team_id: int,
    competition_id: int,
) -> float:
    """Return home win percentage for the team in the given competition.

    Returns value in [0, 1]. Default 0.46 (league average) if no home matches.
    """
    result = await session.execute(
        select(
            func.count(Match.id).label("total"),
            func.sum(
                case((Match.home_goals > Match.away_goals, 1), else_=0)
            ).label("wins"),
        ).where(
            Match.home_team_id == team_id,
            Match.competition_id == competition_id,
            Match.status == "FINISHED",
            Match.home_goals.is_not(None),
        )
    )
    row = result.one()
    total = row.total or 0
    wins = row.wins or 0

    if total == 0:
        return 0.46

    return wins / total
