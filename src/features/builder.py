"""Feature builder - orchestrates all feature modules and writes to match_features table."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.features.elo import get_or_create_elo
from src.features.form import compute_form
from src.features.h2h import compute_h2h
from src.features.home_advantage import compute_home_advantage
from src.features.match_context import compute_days_rest, compute_fixture_congestion
from src.features.strength import compute_strength
from src.features.xg import compute_xg_features
from src.models.feature import MatchFeature
from src.models.match import Match

logger = logging.getLogger(__name__)


async def build_features_for_match(
    session: AsyncSession,
    match: Match,
) -> MatchFeature:
    """Compute all features for a single match and return MatchFeature."""
    home_id = match.home_team_id
    away_id = match.away_team_id
    comp_id = match.competition_id
    ref_date = match.match_date

    # Form (overall + venue-specific)
    home_form = await compute_form(session, home_id, match.id, ref_date=ref_date)
    away_form = await compute_form(session, away_id, match.id, ref_date=ref_date)
    home_form_home = await compute_form(session, home_id, match.id, ref_date=ref_date, venue_filter="home")
    away_form_away = await compute_form(session, away_id, match.id, ref_date=ref_date, venue_filter="away")

    # Strength
    home_str = await compute_strength(session, home_id, comp_id)
    away_str = await compute_strength(session, away_id, comp_id)

    # Elo
    home_elo = await get_or_create_elo(session, home_id)
    away_elo = await get_or_create_elo(session, away_id)

    # xG
    home_xg = await compute_xg_features(session, home_id, match.id, ref_date=ref_date)
    away_xg = await compute_xg_features(session, away_id, match.id, ref_date=ref_date)

    # Home advantage
    home_adv = await compute_home_advantage(session, home_id, comp_id)

    # Head-to-head
    h2h = await compute_h2h(session, home_id, away_id, match.id, ref_date=ref_date)

    # Match context
    home_rest = await compute_days_rest(session, home_id, match.id, ref_date=ref_date)
    away_rest = await compute_days_rest(session, away_id, match.id, ref_date=ref_date)
    home_congestion = await compute_fixture_congestion(session, home_id, match.id, ref_date=ref_date)
    away_congestion = await compute_fixture_congestion(session, away_id, match.id, ref_date=ref_date)

    xg_diff = None
    if home_xg["xg_created_avg"] is not None and away_xg["xg_created_avg"] is not None:
        xg_diff = home_xg["xg_created_avg"] - away_xg["xg_created_avg"]

    feature = MatchFeature(
        match_id=match.id,
        home_form=home_form,
        away_form=away_form,
        home_form_home=home_form_home,
        away_form_away=away_form_away,
        home_attack_strength=home_str["attack_strength"],
        home_defense_strength=home_str["defense_strength"],
        away_attack_strength=away_str["attack_strength"],
        away_defense_strength=away_str["defense_strength"],
        elo_diff=home_elo.rating - away_elo.rating,
        home_xg_avg=home_xg["xg_created_avg"],
        away_xg_avg=away_xg["xg_created_avg"],
        home_xg_conceded_avg=home_xg["xg_conceded_avg"],
        away_xg_conceded_avg=away_xg["xg_conceded_avg"],
        xg_diff=xg_diff,
        home_advantage=home_adv,
        h2h_home_wins=h2h["h2h_home_wins"],
        h2h_draws=h2h["h2h_draws"],
        h2h_away_wins=h2h["h2h_away_wins"],
        h2h_home_win_rate=h2h["h2h_home_win_rate"],
        home_days_rest=home_rest,
        away_days_rest=away_rest,
        home_fixture_congestion=home_congestion,
        away_fixture_congestion=away_congestion,
    )
    return feature


async def build_features_for_upcoming(session: AsyncSession):
    """Build features for all upcoming scheduled matches that don't have features yet."""
    stmt = (
        select(Match)
        .where(
            Match.status.in_(["SCHEDULED", "TIMED"]),
        )
        .outerjoin(MatchFeature, MatchFeature.match_id == Match.id)
        .where(MatchFeature.id.is_(None))
    )
    result = await session.execute(stmt)
    matches = result.scalars().all()

    count = 0
    for match in matches:
        try:
            feature = await build_features_for_match(session, match)
            session.add(feature)
            count += 1
        except Exception:
            logger.exception("Failed to build features for match %s", match.id)

    await session.commit()
    logger.info("Built features for %d matches", count)
    return count
