"""Dynamic Elo rating system.

Each match result updates team Elo ratings.
K-factor: 32 standard, 40 for high-stakes.
Home advantage: +65 Elo points for home team expected score.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from src.config import settings
from src.models.elo import EloHistory, EloRating
from src.models.match import Match


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(goals_for: int, goals_against: int) -> float:
    """Convert match result to actual score: 1.0 win, 0.5 draw, 0.0 loss."""
    if goals_for > goals_against:
        return 1.0
    elif goals_for == goals_against:
        return 0.5
    return 0.0


async def get_or_create_elo(session: AsyncSession, team_id: int) -> EloRating:
    """Get existing Elo rating or create with initial rating."""
    result = await session.execute(
        select(EloRating).where(EloRating.team_id == team_id)
    )
    elo = result.scalar_one_or_none()
    if elo is None:
        elo = EloRating(team_id=team_id, rating=settings.elo_initial_rating)
        session.add(elo)
        await session.flush()
    return elo


async def update_elo_for_match(
    session: AsyncSession,
    match: Match,
    k_factor: float | None = None,
) -> tuple[float, float]:
    """Update Elo ratings for both teams after a match.

    Returns (new_home_rating, new_away_rating).
    """
    if match.home_goals is None or match.away_goals is None:
        raise ValueError("Match has no result")

    if k_factor is None:
        k_factor = settings.elo_k_factor

    home_elo = await get_or_create_elo(session, match.home_team_id)
    away_elo = await get_or_create_elo(session, match.away_team_id)

    home_rating = home_elo.rating
    away_rating = away_elo.rating

    # Home advantage offset applied to expected score calculation
    home_expected = expected_score(
        home_rating + settings.elo_home_advantage, away_rating
    )
    away_expected = 1.0 - home_expected

    home_actual = actual_score(match.home_goals, match.away_goals)
    away_actual = 1.0 - home_actual

    new_home = home_rating + k_factor * (home_actual - home_expected)
    new_away = away_rating + k_factor * (away_actual - away_expected)

    # Record history
    session.add(EloHistory(
        team_id=match.home_team_id,
        match_id=match.id,
        rating_before=home_rating,
        rating_after=new_home,
    ))
    session.add(EloHistory(
        team_id=match.away_team_id,
        match_id=match.id,
        rating_before=away_rating,
        rating_after=new_away,
    ))

    # Update current ratings
    home_elo.rating = new_home
    home_elo.last_match_id = match.id
    away_elo.rating = new_away
    away_elo.last_match_id = match.id

    return new_home, new_away


async def process_all_matches(session: AsyncSession):
    """Process all finished matches in chronological order to build Elo ratings."""
    # Preload all already-processed match IDs in one query
    processed_result = await session.execute(
        select(EloHistory.match_id).distinct()
    )
    processed_ids = {row[0] for row in processed_result.all()}

    # Preload all Elo ratings into memory
    elo_result = await session.execute(select(EloRating))
    elo_cache = {e.team_id: e for e in elo_result.scalars().all()}

    stmt = (
        select(Match)
        .where(Match.status == "FINISHED")
        .order_by(Match.match_date.asc())
    )
    result = await session.execute(stmt)
    matches = result.scalars().all()

    count = 0
    for match in matches:
        if match.id in processed_ids:
            continue

        # Use cached Elo or create new
        if match.home_team_id not in elo_cache:
            elo = EloRating(team_id=match.home_team_id, rating=settings.elo_initial_rating)
            session.add(elo)
            elo_cache[match.home_team_id] = elo
        if match.away_team_id not in elo_cache:
            elo = EloRating(team_id=match.away_team_id, rating=settings.elo_initial_rating)
            session.add(elo)
            elo_cache[match.away_team_id] = elo

        await update_elo_for_match(session, match)
        count += 1

        if count % 200 == 0:
            await session.commit()
    await session.commit()
    logger.info("Processed Elo for %d new matches", count)
