"""Client for Odds API (odds-api.io).

Fetches pre-match bookmaker odds for football matches and converts
decimal odds to implied probabilities.

Free tier: 5,000 requests/hour.
"""

import logging
import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collectors.base import BaseCollector
from src.config import settings
from src.models.feature import MatchFeature
from src.models.match import Match
from src.models.team import Team

logger = logging.getLogger(__name__)

# Known aliases: odds API name -> football-data.org name
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
}

# Suffixes to strip for normalized matching
_STRIP_SUFFIXES = re.compile(
    r"\b(fc|cf|sc|afc|ssc|as|ss|sl|rc|ud|bsc|sk|jk|se)\b", re.IGNORECASE
)

# odds-api.io league slugs for top 5 leagues
LEAGUE_SLUGS = [
    "england-premier-league",
    "spain-la-liga",
    "germany-bundesliga",
    "italy-serie-a",
    "france-ligue-1",
]

# Bookmakers to average odds from
BOOKMAKERS = "Bet365,Stake"


def _normalize_name(name: str) -> str:
    """Normalize a team name for matching: lowercase, strip accents, remove suffixes."""
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = _STRIP_SUFFIXES.sub("", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


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
    """Collector for odds-api.io."""

    def __init__(self):
        super().__init__(
            base_url="https://api.odds-api.io/v3",
            headers={},
            calls_per_minute=60,  # 5000/hr ≈ 83/min, stay conservative
        )
        self.api_key = settings.odds_api_key

    async def fetch_events(self, league_slug: str) -> list[dict]:
        """Fetch upcoming events for a league."""
        if not self.api_key:
            logger.warning("No odds API key configured")
            return []

        data = await self.get("/events", params={
            "apiKey": self.api_key,
            "sport": "football",
            "league": league_slug,
            "status": "pending",
        }, cache_ttl=300)

        return data if isinstance(data, list) else []

    async def fetch_odds_for_event(self, event_id: str) -> dict:
        """Fetch odds for a specific event from selected bookmakers."""
        data = await self.get("/odds", params={
            "apiKey": self.api_key,
            "eventId": event_id,
            "bookmakers": BOOKMAKERS,
        }, cache_ttl=300)

        return data if isinstance(data, dict) else {}

    def _extract_odds(self, odds_data: dict) -> dict | None:
        """Extract home/draw/away probabilities from odds-api.io response.

        Response structure:
        {
          "bookmakers": {
            "Bet365": [
              {"name": "ML", "odds": [{"home": "1.50", "draw": "3.20", "away": "4.50"}], ...}
            ],
            "Stake": [...]
          }
        }
        """
        bookmakers = odds_data.get("bookmakers", {})
        if not bookmakers:
            return None

        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []

        for bm_name, markets in bookmakers.items():
            if not isinstance(markets, list):
                continue
            # Find the ML (moneyline / match winner) market
            for market in markets:
                if market.get("name") != "ML":
                    continue
                odds_arr = market.get("odds", [])
                if not odds_arr:
                    continue
                # odds[0] contains {home, draw, away}
                odds_obj = odds_arr[0] if isinstance(odds_arr, list) else odds_arr
                try:
                    h = float(odds_obj.get("home", 0))
                    d = float(odds_obj.get("draw", 0))
                    a = float(odds_obj.get("away", 0))
                    if h > 0 and d > 0 and a > 0:
                        home_odds_list.append(h)
                        draw_odds_list.append(d)
                        away_odds_list.append(a)
                except (ValueError, TypeError, AttributeError):
                    continue

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
            "odds_home": prob_home,
            "odds_draw": prob_draw,
            "odds_away": prob_away,
        }

    async def enrich_odds(self, session: AsyncSession) -> int:
        """Fetch odds for all supported leagues and update MatchFeature records."""
        total_updated = 0

        # Preload all teams into multiple lookup dicts for robust matching
        all_teams = (await session.execute(select(Team))).scalars().all()
        team_by_exact = {}
        team_by_normalized = {}
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
            if key in team_by_exact:
                return team_by_exact[key]
            norm = _normalize_name(name)
            if norm in team_by_normalized:
                return team_by_normalized[norm]
            for tname, team in team_by_normalized.items():
                if norm in tname or tname in norm:
                    return team
            logger.debug("Could not match team: %s", name)
            return None

        for league_slug in LEAGUE_SLUGS:
            try:
                events = await self.fetch_events(league_slug)
            except Exception:
                logger.exception("Failed to fetch events for %s", league_slug)
                continue

            logger.info("Found %d events for %s", len(events), league_slug)

            for event in events:
                event_id = event.get("id")
                home_name = event.get("home", "")
                away_name = event.get("away", "")

                if not event_id or not home_name or not away_name:
                    continue

                home_team = find_team(home_name)
                away_team = find_team(away_name)

                if not home_team or not away_team:
                    logger.debug("Unmatched: %s vs %s", home_name, away_name)
                    continue

                match = match_lookup.get((home_team.id, away_team.id))
                if not match:
                    continue

                feature = feature_lookup.get(match.id)
                if not feature:
                    continue

                # Fetch odds for this event
                try:
                    odds_data = await self.fetch_odds_for_event(str(event_id))
                except Exception:
                    logger.warning("Failed to fetch odds for event %s", event_id)
                    continue

                odds = self._extract_odds(odds_data)
                if not odds:
                    continue

                feature.odds_home = odds["odds_home"]
                feature.odds_draw = odds["odds_draw"]
                feature.odds_away = odds["odds_away"]
                total_updated += 1

        await session.commit()
        logger.info("Updated odds for %d matches", total_updated)
        return total_updated
