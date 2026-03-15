"""Client for API-Football (api-sports.io) v3.

Free tier: 100 requests/day.
Used for enrichment: xG data, injuries, lineups.
"""

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collectors.base import BaseCollector
from src.config import settings
from src.models.match import Match
from src.models.team import Team

logger = logging.getLogger(__name__)

# Mapping from football-data.org league codes to API-Football league IDs
LEAGUE_MAP = {
    "PL":  39,   # Premier League
    "PD":  140,  # La Liga
    "BL1": 78,   # Bundesliga
    "SA":  135,  # Serie A
    "FL1": 61,   # Ligue 1
    "DED": 88,   # Eredivisie
    "PPL": 94,   # Primeira Liga
    "ELC": 40,   # Championship
    "CL":  2,    # Champions League
}


class ApiFootballCollector(BaseCollector):
    def __init__(self):
        super().__init__(
            base_url="https://v3.football.api-sports.io",
            headers={
                "x-apisports-key": settings.api_football_key,
            },
            calls_per_minute=10,
        )

    async def get(self, path: str, params: dict | None = None) -> dict:
        """Override to handle api-sports response wrapper."""
        data = await super().get(path, params)
        return data

    async def enrich_xg(self, session: AsyncSession, match_date: date, league_id: int):
        """Fetch xG data for fixtures on a given date and update matches."""
        data = await self.get("/fixtures", params={
            "league": league_id,
            "date": match_date.isoformat(),
            "season": match_date.year if match_date.month >= 7 else match_date.year - 1,
        })

        for fixture in data.get("response", []):
            teams_data = fixture.get("teams", {})
            stats = fixture.get("statistics", [])

            home_name = teams_data.get("home", {}).get("name", "")
            away_name = teams_data.get("away", {}).get("name", "")

            # Find matching teams in DB
            home_team = (await session.execute(
                select(Team).where(Team.name.ilike(f"%{home_name}%"))
            )).scalar_one_or_none()
            away_team = (await session.execute(
                select(Team).where(Team.name.ilike(f"%{away_name}%"))
            )).scalar_one_or_none()

            if not home_team or not away_team:
                continue

            # Find the match
            match = (await session.execute(
                select(Match).where(
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    Match.match_date >= f"{match_date}T00:00:00+00:00",
                    Match.match_date <= f"{match_date}T23:59:59+00:00",
                )
            )).scalar_one_or_none()

            if not match:
                continue

            # Extract xG from statistics
            home_xg = self._extract_stat(stats, 0, "Expected Goals")
            away_xg = self._extract_stat(stats, 1, "Expected Goals")

            if home_xg is not None:
                match.home_xg = home_xg
            if away_xg is not None:
                match.away_xg = away_xg

        await session.commit()

    @staticmethod
    def _extract_stat(stats: list, team_index: int, stat_name: str) -> float | None:
        """Extract a statistic value from API-Football statistics array."""
        if team_index >= len(stats):
            return None
        team_stats = stats[team_index].get("statistics", [])
        for s in team_stats:
            if s.get("type") == stat_name:
                val = s.get("value")
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
        return None

    async def fetch_injuries(
        self, league_id: int, season: int
    ) -> dict[str, list[dict]]:
        """Fetch current injuries for a league. Returns {team_name: [injuries]}."""
        data = await self.get("/injuries", params={
            "league": league_id,
            "season": season,
        })

        injuries: dict[str, list[dict]] = {}
        for entry in data.get("response", []):
            team_name = entry.get("team", {}).get("name", "Unknown")
            player = entry.get("player", {}).get("name", "Unknown")
            reason = entry.get("player", {}).get("reason", "")
            injuries.setdefault(team_name, []).append({
                "player": player,
                "reason": reason,
            })

        return injuries
