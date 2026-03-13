"""Client for The Odds API (the-odds-api.com).

Fetches pre-match bookmaker odds for football matches and converts
decimal odds to implied probabilities.
"""

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collectors.base import BaseCollector
from src.config import settings
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.team import Team

logger = logging.getLogger(__name__)

# The Odds API sport keys for football leagues
SPORT_KEYS = [
    "soccer_epl",              # Premier League
    "soccer_spain_la_liga",    # La Liga
    "soccer_germany_bundesliga",  # Bundesliga
    "soccer_italy_serie_a",    # Serie A
    "soccer_france_ligue_one", # Ligue 1
]


def odds_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def normalize_probs(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Normalize implied probabilities to sum to 1 (removes overround)."""
    total = home + draw + away
    if total <= 0:
        return 0.0, 0.0, 0.0
    return home / total, draw / total, away / total


class OddsApiCollector(BaseCollector):
    """Collector for The Odds API."""

    def __init__(self):
        super().__init__(
            base_url="https://api.the-odds-api.com",
            headers={},
            calls_per_minute=10,
        )
        self.api_key = settings.odds_api_key

    async def fetch_odds_for_sport(self, sport_key: str) -> list[dict]:
        """Fetch upcoming match odds for a sport.

        Returns list of event dicts with bookmaker odds.
        """
        if not self.api_key:
            logger.warning("No odds API key configured")
            return []

        data = await self.get(f"/v4/sports/{sport_key}/odds", params={
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        })

        return data if isinstance(data, list) else []

    def _extract_best_odds(self, event: dict) -> dict | None:
        """Extract averaged odds across bookmakers for an event.

        Averages across all bookmakers for more stable implied probabilities.
        """
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []

        home_team = event.get("home_team", "")

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if "Draw" in outcomes:
                    # Find home and away by matching team names
                    for name, price in outcomes.items():
                        if name == "Draw":
                            draw_odds_list.append(price)
                        elif name == home_team:
                            home_odds_list.append(price)
                        else:
                            away_odds_list.append(price)

        if not home_odds_list or not draw_odds_list or not away_odds_list:
            return None

        # Average across bookmakers
        avg_home = sum(home_odds_list) / len(home_odds_list)
        avg_draw = sum(draw_odds_list) / len(draw_odds_list)
        avg_away = sum(away_odds_list) / len(away_odds_list)

        # Convert to implied probabilities and normalize
        imp_home = odds_to_implied_prob(avg_home)
        imp_draw = odds_to_implied_prob(avg_draw)
        imp_away = odds_to_implied_prob(avg_away)

        prob_home, prob_draw, prob_away = normalize_probs(imp_home, imp_draw, imp_away)

        return {
            "home_team": home_team,
            "away_team": event.get("away_team", ""),
            "odds_home": prob_home,
            "odds_draw": prob_draw,
            "odds_away": prob_away,
            "commence_time": event.get("commence_time", ""),
        }

    async def enrich_odds(self, session: AsyncSession):
        """Fetch odds for all supported leagues and update MatchFeature records."""
        total_updated = 0

        for sport_key in SPORT_KEYS:
            try:
                events = await self.fetch_odds_for_sport(sport_key)
            except Exception:
                logger.exception("Failed to fetch odds for %s", sport_key)
                continue

            for event in events:
                odds = self._extract_best_odds(event)
                if not odds:
                    continue

                # Find the match by team names and date
                home_name = odds["home_team"]
                away_name = odds["away_team"]

                home_team = (await session.execute(
                    select(Team).where(Team.name.ilike(f"%{home_name}%"))
                )).scalar_one_or_none()
                away_team = (await session.execute(
                    select(Team).where(Team.name.ilike(f"%{away_name}%"))
                )).scalar_one_or_none()

                if not home_team or not away_team:
                    continue

                # Find upcoming match between these teams
                match = (await session.execute(
                    select(Match).where(
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id,
                        Match.status.in_(["SCHEDULED", "TIMED"]),
                    )
                )).scalar_one_or_none()

                if not match:
                    continue

                # Update the MatchFeature record
                feature = (await session.execute(
                    select(MatchFeature).where(MatchFeature.match_id == match.id)
                )).scalar_one_or_none()

                if feature:
                    feature.odds_home = odds["odds_home"]
                    feature.odds_draw = odds["odds_draw"]
                    feature.odds_away = odds["odds_away"]
                    total_updated += 1

        await session.commit()
        logger.info("Updated odds for %d matches", total_updated)
        return total_updated
