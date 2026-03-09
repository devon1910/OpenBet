"""Prediction outcome tracker.

After matches finish, records whether each pick was correct.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match
from src.models.prediction import Pick

logger = logging.getLogger(__name__)


def determine_outcome(
    pick_type: str,
    pick_value: str,
    home_goals: int,
    away_goals: int,
) -> str:
    """Determine if a pick won or lost based on the match result."""
    if home_goals > away_goals:
        result = "HOME"
    elif home_goals == away_goals:
        result = "DRAW"
    else:
        result = "AWAY"

    if pick_type == "STRAIGHT_WIN":
        return "WIN" if pick_value == result else "LOSS"

    if pick_type == "DOUBLE_CHANCE":
        if pick_value == "1X" and result in ("HOME", "DRAW"):
            return "WIN"
        if pick_value == "X2" and result in ("DRAW", "AWAY"):
            return "WIN"
        if pick_value == "12" and result in ("HOME", "AWAY"):
            return "WIN"
        return "LOSS"

    return "VOID"


async def update_outcomes(session: AsyncSession) -> int:
    """Check all picks without outcomes and update if match is finished.

    Returns number of picks updated.
    """
    stmt = (
        select(Pick, Match)
        .join(Match, Match.id == Pick.match_id)
        .where(
            Pick.outcome.is_(None),
            Match.status == "FINISHED",
            Match.home_goals.is_not(None),
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    count = 0
    for pick, match in rows:
        pick.outcome = determine_outcome(
            pick.pick_type,
            pick.pick_value,
            match.home_goals,
            match.away_goals,
        )
        count += 1

    await session.commit()
    logger.info("Updated outcomes for %d picks", count)
    return count
