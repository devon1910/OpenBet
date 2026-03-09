"""Expected Goals (xG) feature computation.

Computes rolling xG averages for created and conceded over last N matches.
"""

import numpy as np
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match


async def compute_xg_features(
    session: AsyncSession,
    team_id: int,
    before_match_id: int,
    n_matches: int = 7,
) -> dict:
    """Compute xG features for a team from their last N finished matches.

    Returns:
        xg_created_avg: average xG created per match
        xg_conceded_avg: average xG conceded per match
        xg_diff: xg_created_avg - xg_conceded_avg
        xg_trend: slope of xG created over the window (positive = improving)
    """
    ref_match = (await session.execute(
        select(Match).where(Match.id == before_match_id)
    )).scalar_one()

    stmt = (
        select(Match)
        .where(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
            Match.status == "FINISHED",
            Match.match_date < ref_match.match_date,
            Match.home_xg.is_not(None),
            Match.away_xg.is_not(None),
        )
        .order_by(Match.match_date.desc())
        .limit(n_matches)
    )
    result = await session.execute(stmt)
    matches = result.scalars().all()

    if not matches:
        return {
            "xg_created_avg": None,
            "xg_conceded_avg": None,
            "xg_diff": None,
            "xg_trend": None,
        }

    xg_created = []
    xg_conceded = []

    for m in matches:
        if m.home_team_id == team_id:
            xg_created.append(m.home_xg)
            xg_conceded.append(m.away_xg)
        else:
            xg_created.append(m.away_xg)
            xg_conceded.append(m.home_xg)

    created_avg = float(np.mean(xg_created))
    conceded_avg = float(np.mean(xg_conceded))

    # Trend: linear regression slope over the window (reversed to chronological)
    xg_chron = list(reversed(xg_created))
    if len(xg_chron) >= 3:
        x = np.arange(len(xg_chron))
        slope = float(np.polyfit(x, xg_chron, 1)[0])
    else:
        slope = 0.0

    return {
        "xg_created_avg": created_avg,
        "xg_conceded_avg": conceded_avg,
        "xg_diff": created_avg - conceded_avg,
        "xg_trend": slope,
    }
