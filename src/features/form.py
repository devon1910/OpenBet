"""Recent form calculation.

Computes a weighted form score from the last N matches.
Win=3, Draw=1, Loss=0 with exponential recency decay.
"""

import numpy as np
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match


async def compute_form(
    session: AsyncSession,
    team_id: int,
    before_match_id: int,
    n_matches: int = 6,
    ref_date=None,
    venue_filter: str = None,
) -> float:
    """Compute weighted form score for a team from their last N finished matches.

    Args:
        venue_filter: None = all matches, "home" = home only, "away" = away only

    Returns a score in [0, 3] where 3 = all wins, 0 = all losses.
    """
    if ref_date is None:
        ref_match = (await session.execute(
            select(Match).where(Match.id == before_match_id)
        )).scalar_one()
        ref_date = ref_match.match_date

    # Fetch last N finished matches for this team, optionally filtered by venue
    if venue_filter == "home":
        venue_cond = Match.home_team_id == team_id
    elif venue_filter == "away":
        venue_cond = Match.away_team_id == team_id
    else:
        venue_cond = or_(
            Match.home_team_id == team_id,
            Match.away_team_id == team_id,
        )

    stmt = (
        select(Match)
        .where(
            venue_cond,
            Match.status == "FINISHED",
            Match.match_date < ref_date,
        )
        .order_by(Match.match_date.desc())
        .limit(n_matches)
    )
    result = await session.execute(stmt)
    matches = result.scalars().all()

    if not matches:
        return 1.5  # neutral default

    points = []
    for m in matches:
        if m.home_goals is None or m.away_goals is None:
            continue
        is_home = m.home_team_id == team_id
        if is_home:
            if m.home_goals > m.away_goals:
                points.append(3)
            elif m.home_goals == m.away_goals:
                points.append(1)
            else:
                points.append(0)
        else:
            if m.away_goals > m.home_goals:
                points.append(3)
            elif m.away_goals == m.home_goals:
                points.append(1)
            else:
                points.append(0)

    if not points:
        return 1.5

    # Exponential decay weights: most recent match has highest weight
    weights = np.array([0.9**i for i in range(len(points))])
    weights /= weights.sum()
    return float(np.dot(points, weights))
