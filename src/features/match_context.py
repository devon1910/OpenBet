"""Match context features: days rest and fixture congestion.

Captures fatigue and scheduling effects that impact performance,
especially for teams in European competitions or cup runs.
"""

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match


async def compute_days_rest(
    session: AsyncSession,
    team_id: int,
    before_match_id: int,
    ref_date=None,
) -> int:
    """Compute days since the team's last finished match.

    Returns number of days, or 7 (neutral default) if no prior match found.
    """
    if ref_date is None:
        ref_match = (await session.execute(
            select(Match).where(Match.id == before_match_id)
        )).scalar_one()
        ref_date = ref_match.match_date

    stmt = (
        select(Match.match_date)
        .where(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
            Match.status == "FINISHED",
            Match.match_date < ref_date,
        )
        .order_by(Match.match_date.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    last_date = result.scalar_one_or_none()

    if last_date is None:
        return 7

    delta = ref_date - last_date
    return max(delta.days, 0)


async def compute_fixture_congestion(
    session: AsyncSession,
    team_id: int,
    before_match_id: int,
    window_days: int = 14,
    ref_date=None,
) -> int:
    """Count matches played in the last N days before this match.

    Higher values indicate fixture congestion / fatigue risk.
    Returns 0 if no matches found in the window.
    """
    if ref_date is None:
        ref_match = (await session.execute(
            select(Match).where(Match.id == before_match_id)
        )).scalar_one()
        ref_date = ref_match.match_date

    from datetime import timedelta
    window_start = ref_date - timedelta(days=window_days)

    stmt = (
        select(func.count(Match.id))
        .where(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
            Match.status == "FINISHED",
            Match.match_date >= window_start,
            Match.match_date < ref_date,
        )
    )
    result = await session.execute(stmt)
    return result.scalar() or 0
