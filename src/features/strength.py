"""Attack and defense strength calculation.

Strength = team's goals ratio compared to league average.
attack_strength = (team goals scored / games) / (league avg goals scored / games)
defense_strength = (team goals conceded / games) / (league avg goals conceded / games)
"""

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.match import Match
from src.models.team import Team


async def compute_strength(
    session: AsyncSession,
    team_id: int,
    competition_id: int,
) -> dict:
    """Compute attack and defense strength for a team relative to league average.

    Returns:
        attack_strength: >1 means stronger attack than average
        defense_strength: <1 means better defense than average
    """
    # Get team's competition matches
    team_matches = (await session.execute(
        select(Match).where(
            Match.competition_id == competition_id,
            Match.status == "FINISHED",
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
        )
    )).scalars().all()

    if not team_matches:
        return {"attack_strength": 1.0, "defense_strength": 1.0}

    team_scored = 0
    team_conceded = 0
    for m in team_matches:
        if m.home_goals is None or m.away_goals is None:
            continue
        if m.home_team_id == team_id:
            team_scored += m.home_goals
            team_conceded += m.away_goals
        else:
            team_scored += m.away_goals
            team_conceded += m.home_goals

    n_team = len(team_matches)

    # League averages
    league_result = await session.execute(
        select(
            func.sum(Match.home_goals).label("total_home"),
            func.sum(Match.away_goals).label("total_away"),
            func.count(Match.id).label("n_matches"),
        ).where(
            Match.competition_id == competition_id,
            Match.status == "FINISHED",
            Match.home_goals.is_not(None),
        )
    )
    row = league_result.one()
    total_home = row.total_home or 0
    total_away = row.total_away or 0
    n_league = row.n_matches or 1

    league_avg_scored = (total_home + total_away) / (2 * n_league) if n_league else 1.5
    league_avg_scored = max(league_avg_scored, 0.1)  # prevent division by zero

    team_avg_scored = team_scored / n_team if n_team else league_avg_scored
    team_avg_conceded = team_conceded / n_team if n_team else league_avg_scored

    return {
        "attack_strength": team_avg_scored / league_avg_scored,
        "defense_strength": team_avg_conceded / league_avg_scored,
    }
