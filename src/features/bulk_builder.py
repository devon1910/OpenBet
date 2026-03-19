"""Bulk feature builder — computes all features in-memory instead of per-match DB queries.

Loads all finished matches once, then computes form, strength, xG, h2h, etc.
in Python. Reduces ~17 DB queries per match to 2 total queries for the entire backfill.
"""

import logging
from collections import defaultdict
from datetime import timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.features.elo import get_or_create_elo
from src.models.elo import EloRating
from src.models.feature import MatchFeature
from src.models.match import Match

logger = logging.getLogger(__name__)


def _weighted_form(points: list[int]) -> float:
    """Exponential decay weighted form from most recent to oldest."""
    if not points:
        return 1.5
    weights = np.array([0.9 ** i for i in range(len(points))])
    weights /= weights.sum()
    return float(np.dot(points, weights))


def _match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0


class BulkFeatureComputer:
    """Preloads all matches and computes features in-memory."""

    def __init__(self, all_matches: list[Match], elo_cache: dict[int, float]):
        # Index matches by team for fast lookup
        self.team_matches: dict[int, list[Match]] = defaultdict(list)
        self.pair_matches: dict[tuple[int, int], list[Match]] = defaultdict(list)
        self.comp_matches: dict[int, list[Match]] = defaultdict(list)
        self.elo_cache = elo_cache  # team_id -> rating

        for m in all_matches:
            if m.status != "FINISHED" or m.home_goals is None:
                continue
            self.team_matches[m.home_team_id].append(m)
            self.team_matches[m.away_team_id].append(m)
            self.comp_matches[m.competition_id].append(m)
            # Both orderings for h2h lookup
            key = tuple(sorted([m.home_team_id, m.away_team_id]))
            self.pair_matches[key].append(m)

        # Pre-sort by date for all indexes
        for team_id in self.team_matches:
            self.team_matches[team_id].sort(key=lambda m: m.match_date)
        for key in self.pair_matches:
            self.pair_matches[key].sort(key=lambda m: m.match_date)

    def _get_prior_matches(self, team_id: int, ref_date, n: int = 6, venue: str = None) -> list[Match]:
        """Get last N matches for team before ref_date, optionally filtered by venue."""
        result = []
        for m in reversed(self.team_matches.get(team_id, [])):
            if m.match_date >= ref_date:
                continue
            if venue == "home" and m.home_team_id != team_id:
                continue
            if venue == "away" and m.away_team_id != team_id:
                continue
            result.append(m)
            if len(result) >= n:
                break
        return result

    def compute_form(self, team_id: int, ref_date, venue: str = None) -> float:
        matches = self._get_prior_matches(team_id, ref_date, n=6, venue=venue)
        points = []
        for m in matches:
            is_home = m.home_team_id == team_id
            gf = m.home_goals if is_home else m.away_goals
            ga = m.away_goals if is_home else m.home_goals
            points.append(_match_points(gf, ga))
        return _weighted_form(points)

    def compute_strength(self, team_id: int, competition_id: int, ref_date) -> dict:
        comp_matches = [m for m in self.comp_matches.get(competition_id, [])
                        if m.home_goals is not None and m.match_date < ref_date]
        if not comp_matches:
            return {"attack_strength": 1.0, "defense_strength": 1.0}

        # League averages
        total_goals = sum(m.home_goals + m.away_goals for m in comp_matches)
        n_league = len(comp_matches)
        league_avg = total_goals / (2 * n_league) if n_league else 1.5
        league_avg = max(league_avg, 0.1)

        # Team stats
        team_scored = 0
        team_conceded = 0
        n_team = 0
        for m in comp_matches:
            if m.home_team_id == team_id:
                team_scored += m.home_goals
                team_conceded += m.away_goals
                n_team += 1
            elif m.away_team_id == team_id:
                team_scored += m.away_goals
                team_conceded += m.home_goals
                n_team += 1

        if n_team == 0:
            return {"attack_strength": 1.0, "defense_strength": 1.0}

        return {
            "attack_strength": (team_scored / n_team) / league_avg,
            "defense_strength": (team_conceded / n_team) / league_avg,
        }

    def compute_xg(self, team_id: int, ref_date) -> dict:
        matches = self._get_prior_matches(team_id, ref_date, n=7)
        # Filter to matches with xG data
        matches = [m for m in matches if m.home_xg is not None and m.away_xg is not None]
        if not matches:
            return {"xg_created_avg": None, "xg_conceded_avg": None}

        created, conceded = [], []
        for m in matches:
            if m.home_team_id == team_id:
                created.append(m.home_xg)
                conceded.append(m.away_xg)
            else:
                created.append(m.away_xg)
                conceded.append(m.home_xg)

        return {
            "xg_created_avg": float(np.mean(created)),
            "xg_conceded_avg": float(np.mean(conceded)),
        }

    def compute_home_advantage(self, team_id: int, competition_id: int, ref_date) -> float:
        home_matches = [m for m in self.comp_matches.get(competition_id, [])
                        if m.home_team_id == team_id and m.home_goals is not None and m.match_date < ref_date]
        if not home_matches:
            return 0.46
        wins = sum(1 for m in home_matches if m.home_goals > m.away_goals)
        return wins / len(home_matches)

    def compute_h2h(self, home_team_id: int, away_team_id: int, ref_date) -> dict:
        key = tuple(sorted([home_team_id, away_team_id]))
        meetings = [m for m in self.pair_matches.get(key, []) if m.match_date < ref_date]
        meetings = meetings[-10:]  # last 10

        if not meetings:
            return {"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0, "h2h_home_win_rate": 0.5}

        home_wins = draws = away_wins = 0
        for m in meetings:
            if m.home_goals == m.away_goals:
                draws += 1
            elif m.home_team_id == home_team_id:
                if m.home_goals > m.away_goals:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                if m.away_goals > m.home_goals:
                    home_wins += 1
                else:
                    away_wins += 1

        total = home_wins + draws + away_wins
        return {
            "h2h_home_wins": home_wins,
            "h2h_draws": draws,
            "h2h_away_wins": away_wins,
            "h2h_home_win_rate": (home_wins + 0.5 * draws) / total if total > 0 else 0.5,
        }

    def compute_days_rest(self, team_id: int, ref_date) -> int:
        matches = self._get_prior_matches(team_id, ref_date, n=1)
        if not matches:
            return 7
        delta = ref_date - matches[0].match_date
        return max(delta.days, 0)

    def compute_fixture_congestion(self, team_id: int, ref_date, window_days: int = 14) -> int:
        window_start = ref_date - timedelta(days=window_days)
        count = 0
        for m in reversed(self.team_matches.get(team_id, [])):
            if m.match_date >= ref_date:
                continue
            if m.match_date < window_start:
                break
            count += 1
        return count

    def build_feature(self, match: Match) -> MatchFeature:
        """Compute all features for a single match using in-memory data."""
        home_id = match.home_team_id
        away_id = match.away_team_id
        comp_id = match.competition_id
        ref_date = match.match_date

        home_form = self.compute_form(home_id, ref_date)
        away_form = self.compute_form(away_id, ref_date)
        home_form_home = self.compute_form(home_id, ref_date, venue="home")
        away_form_away = self.compute_form(away_id, ref_date, venue="away")

        home_str = self.compute_strength(home_id, comp_id, ref_date)
        away_str = self.compute_strength(away_id, comp_id, ref_date)

        home_elo = self.elo_cache.get(home_id, 1500.0)
        away_elo = self.elo_cache.get(away_id, 1500.0)

        home_xg = self.compute_xg(home_id, ref_date)
        away_xg = self.compute_xg(away_id, ref_date)

        home_adv = self.compute_home_advantage(home_id, comp_id, ref_date)
        h2h = self.compute_h2h(home_id, away_id, ref_date)

        home_rest = self.compute_days_rest(home_id, ref_date)
        away_rest = self.compute_days_rest(away_id, ref_date)
        home_congestion = self.compute_fixture_congestion(home_id, ref_date)
        away_congestion = self.compute_fixture_congestion(away_id, ref_date)

        xg_diff = None
        if home_xg["xg_created_avg"] is not None and away_xg["xg_created_avg"] is not None:
            xg_diff = home_xg["xg_created_avg"] - away_xg["xg_created_avg"]

        return MatchFeature(
            match_id=match.id,
            home_form=home_form,
            away_form=away_form,
            home_form_home=home_form_home,
            away_form_away=away_form_away,
            home_attack_strength=home_str["attack_strength"],
            home_defense_strength=home_str["defense_strength"],
            away_attack_strength=away_str["attack_strength"],
            away_defense_strength=away_str["defense_strength"],
            elo_diff=home_elo - away_elo,
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


async def bulk_build_features(
    session: AsyncSession,
    matches_to_build: list[Match],
    status_callback=None,
) -> int:
    """Build features for many matches using in-memory computation.

    Args:
        session: DB session
        matches_to_build: matches that need features
        status_callback: optional async fn(message) for progress updates

    Returns:
        Number of features built
    """
    # 1. Load ALL finished matches in one query
    if status_callback:
        await status_callback("Loading all match data...")

    all_matches_result = await session.execute(
        select(Match).where(Match.status == "FINISHED").order_by(Match.match_date.asc())
    )
    all_matches = all_matches_result.scalars().all()
    logger.info("[bulk] Loaded %d finished matches", len(all_matches))

    # 2. Load Elo ratings
    elo_result = await session.execute(select(EloRating))
    elo_cache = {e.team_id: e.rating for e in elo_result.scalars().all()}

    # 3. Build the in-memory computer
    computer = BulkFeatureComputer(all_matches, elo_cache)

    # 4. Compute features in batches (commit every 100 to avoid statement timeout)
    count = 0
    errors = 0
    total = len(matches_to_build)
    batch_size = 100
    batch = []

    for match in matches_to_build:
        try:
            feature = computer.build_feature(match)
            batch.append(feature)
        except Exception:
            errors += 1
            logger.exception("[bulk] Failed to compute features for match %s", match.id)
            continue

        if len(batch) >= batch_size:
            try:
                session.add_all(batch)
                await session.commit()
                count += len(batch)
                if status_callback:
                    await status_callback(f"Built {count}/{total} features...")
                logger.info("[bulk] Committed batch %d/%d", count, total)
            except Exception:
                logger.exception("[bulk] Batch commit failed at %d", count)
                await session.rollback()
                errors += len(batch)
            batch = []

    # Final batch
    if batch:
        try:
            session.add_all(batch)
            await session.commit()
            count += len(batch)
        except Exception:
            logger.exception("[bulk] Final batch commit failed")
            await session.rollback()
            errors += len(batch)

    logger.info("[bulk] Built %d features (%d errors)", count, errors)
    return count
