"""Head-to-head history between two teams.

Computes win rates from past meetings to capture matchup-specific tendencies
that form and strength metrics miss.
"""

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match


async def compute_h2h(
    session: AsyncSession,
    home_team_id: int,
    away_team_id: int,
    before_match_id: int,
    n_meetings: int = 10,
) -> dict:
    """Compute head-to-head record from the last N meetings between two teams.

    Considers matches where either team was home or away (both directions).
    Returns h2h stats from the perspective of the current home team.
    """
    # Get reference match date
    ref_match = (await session.execute(
        select(Match).where(Match.id == before_match_id)
    )).scalar_one()

    # Fetch last N finished meetings between these two teams
    stmt = (
        select(Match)
        .where(
            Match.status == "FINISHED",
            Match.match_date < ref_match.match_date,
            Match.home_goals.is_not(None),
            or_(
                and_(
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                ),
                and_(
                    Match.home_team_id == away_team_id,
                    Match.away_team_id == home_team_id,
                ),
            ),
        )
        .order_by(Match.match_date.desc())
        .limit(n_meetings)
    )
    result = await session.execute(stmt)
    matches = result.scalars().all()

    if not matches:
        return {
            "h2h_home_wins": 0,
            "h2h_draws": 0,
            "h2h_away_wins": 0,
            "h2h_home_win_rate": 0.5,  # neutral default
        }

    home_wins = 0
    draws = 0
    away_wins = 0

    for m in matches:
        if m.home_goals == m.away_goals:
            draws += 1
        elif m.home_team_id == home_team_id:
            # Match played with same home/away as current fixture
            if m.home_goals > m.away_goals:
                home_wins += 1
            else:
                away_wins += 1
        else:
            # Reversed fixture: current home team was away
            if m.away_goals > m.home_goals:
                home_wins += 1  # current home team won as away
            else:
                away_wins += 1

    total = home_wins + draws + away_wins
    home_win_rate = (home_wins + 0.5 * draws) / total if total > 0 else 0.5

    return {
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
        "h2h_home_win_rate": home_win_rate,
    }
