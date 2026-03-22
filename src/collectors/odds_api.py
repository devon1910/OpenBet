"""Client for The Odds API (the-odds-api.com).

Fetches pre-match bookmaker odds for football matches and converts
decimal odds to implied probabilities.
"""

import logging
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collectors.base import BaseCollector
from src.config import settings
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.team import Team

logger = logging.getLogger(__name__)

# Known aliases: Odds API name -> football-data.org name
_TEAM_ALIASES = {
    "wolves": "wolverhampton wanderers",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "nottm forest": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "brighton": "brighton and hove albion",
    "leicester": "leicester city",
    "luton": "luton town",
    "sheffield utd": "sheffield united",
    "athletic bilbao": "club athletic",
    "atletico madrid": "club atletico de madrid",
    "atlético madrid": "club atletico de madrid",
    "celta vigo": "rc celta de vigo",
    "almeria": "ud almeria",
    "real sociedad": "real sociedad de futbol",
    "bayern munich": "fc bayern munchen",
    "bayern münchen": "fc bayern munchen",
    "bayer leverkusen": "bayer 04 leverkusen",
    "rb leipzig": "rasenballsport leipzig",
    "borussia dortmund": "borussia dortmund",
    "hertha berlin": "hertha bsc",
    "inter milan": "fc internazionale milano",
    "inter": "fc internazionale milano",
    "ac milan": "ac milan",
    "napoli": "ssc napoli",
    "roma": "as roma",
    "lazio": "ss lazio",
    "juventus": "juventus fc",
    "lyon": "olympique lyonnais",
    "marseille": "olympique de marseille",
    "psg": "paris saint-germain fc",
    "paris saint germain": "paris saint-germain fc",
    "saint-etienne": "as saint-etienne",
    "st etienne": "as saint-etienne",
    "psv": "psv eindhoven",
    "ajax": "afc ajax",
    "feyenoord": "feyenoord rotterdam",
    "benfica": "sl benfica",
    "porto": "fc porto",
    "sporting cp": "sporting clube de portugal",
    "sporting lisbon": "sporting clube de portugal",
    "galatasaray": "galatasaray sk",
    "fenerbahce": "fenerbahce sk",
    "besiktas": "besiktas jk",
}

# Suffixes to strip for normalized matching
_STRIP_SUFFIXES = re.compile(
    r"\b(fc|cf|sc|afc|ssc|as|ss|sl|rc|ud|bsc|sk|jk|se)\b", re.IGNORECASE
)


def _normalize_name(name: str) -> str:
    """Normalize a team name for matching: lowercase, strip accents, remove suffixes."""
    # Lowercase
    name = name.lower().strip()
    # Normalize unicode (é → e, ü → u, etc.)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Strip common suffixes
    name = _STRIP_SUFFIXES.sub("", name).strip()
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name

# The Odds API sport keys for football leagues
SPORT_KEYS = [
    "soccer_epl",                        # Premier League
    "soccer_spain_la_liga",              # La Liga
    "soccer_germany_bundesliga",         # Bundesliga
    "soccer_italy_serie_a",              # Serie A
    "soccer_france_ligue_one",           # Ligue 1
    "soccer_netherlands_eredivisie",     # Eredivisie
    "soccer_portugal_primeira_liga",     # Primeira Liga
    "soccer_england_championship",       # Championship
    "soccer_uefa_champs_league",         # Champions League
    "soccer_spain_segunda_division",     # La Liga 2
    "soccer_germany_bundesliga2",        # Bundesliga 2
    "soccer_turkey_super_league",        # Turkish Süper Lig
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
        }, cache_ttl=300)

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

        # Preload all teams into multiple lookup dicts for robust matching
        all_teams = (await session.execute(select(Team))).scalars().all()
        team_by_exact = {}  # exact lowercase name -> team
        team_by_normalized = {}  # normalized name -> team
        for t in all_teams:
            team_by_exact[t.name.lower()] = t
            team_by_normalized[_normalize_name(t.name)] = t
            if t.short_name:
                team_by_exact[t.short_name.lower()] = t

        # Add known aliases
        for alias, canonical in _TEAM_ALIASES.items():
            norm = _normalize_name(canonical)
            if norm in team_by_normalized:
                team_by_exact[alias] = team_by_normalized[norm]

        # Preload all scheduled matches indexed by (home_team_id, away_team_id)
        scheduled = (await session.execute(
            select(Match).where(Match.status.in_(["SCHEDULED", "TIMED"]))
        )).scalars().all()
        match_lookup = {(m.home_team_id, m.away_team_id): m for m in scheduled}

        # Preload all features indexed by match_id
        feature_ids = [m.id for m in scheduled]
        if feature_ids:
            features = (await session.execute(
                select(MatchFeature).where(MatchFeature.match_id.in_(feature_ids))
            )).scalars().all()
            feature_lookup = {f.match_id: f for f in features}
        else:
            feature_lookup = {}

        def find_team(name):
            """Match team name using exact, normalized, and partial strategies."""
            key = name.lower().strip()
            # 1. Exact match (includes aliases)
            if key in team_by_exact:
                return team_by_exact[key]
            # 2. Normalized match (strips suffixes, accents)
            norm = _normalize_name(name)
            if norm in team_by_normalized:
                return team_by_normalized[norm]
            # 3. Partial match on normalized names
            for tname, team in team_by_normalized.items():
                if norm in tname or tname in norm:
                    return team
            logger.debug("Could not match team: %s", name)
            return None

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

                home_team = find_team(odds["home_team"])
                away_team = find_team(odds["away_team"])

                if not home_team or not away_team:
                    continue

                match = match_lookup.get((home_team.id, away_team.id))
                if not match:
                    continue

                feature = feature_lookup.get(match.id)
                if feature:
                    feature.odds_home = odds["odds_home"]
                    feature.odds_draw = odds["odds_draw"]
                    feature.odds_away = odds["odds_away"]
                    total_updated += 1

        await session.commit()
        logger.info("Updated odds for %d matches", total_updated)
        return total_updated
