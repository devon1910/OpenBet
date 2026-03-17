"""Client for football-data.org API (v4).

Free tier: 10 requests/minute.
Covers: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie,
        Primeira Liga, Championship, Champions League, European Championship.
"""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collectors.base import BaseCollector
from src.config import settings
from src.models.match import Match
from src.models.team import Competition, Team

logger = logging.getLogger(__name__)

# football-data.org competition codes
# Free tier covers: PL, PD, BL1, SA, FL1, DED, PPL, ELC, CL
COMPETITIONS = {
    "PL":  "Premier League",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
    "CL":  "Champions League",
}


class FootballDataCollector(BaseCollector):
    def __init__(self):
        super().__init__(
            base_url="https://api.football-data.org/v4",
            headers={"X-Auth-Token": settings.football_data_api_key},
            calls_per_minute=10,
        )

    async def sync_competitions(self, session: AsyncSession):
        """Fetch and upsert competitions."""
        for code, name in COMPETITIONS.items():
            data = await self.get(f"/competitions/{code}")
            season = data.get("currentSeason", {})
            season_str = str(season.get("startDate", ""))[:4] if season else ""

            existing = await session.execute(
                select(Competition).where(Competition.external_id == code)
            )
            comp = existing.scalar_one_or_none()
            if comp:
                comp.season = season_str
            else:
                comp = Competition(
                    external_id=code,
                    name=name,
                    country=data.get("area", {}).get("name", ""),
                    season=season_str,
                )
                session.add(comp)
        await session.commit()

    async def sync_teams(self, session: AsyncSession, competition_code: str):
        """Fetch and upsert teams for a competition."""
        comp = await session.execute(
            select(Competition).where(Competition.external_id == competition_code)
        )
        comp = comp.scalar_one_or_none()
        if not comp:
            logger.warning("Competition %s not found in DB", competition_code)
            return

        data = await self.get(f"/competitions/{competition_code}/teams")
        for team_data in data.get("teams", []):
            ext_id = str(team_data["id"])
            existing = await session.execute(
                select(Team).where(Team.external_id == ext_id)
            )
            team = existing.scalar_one_or_none()
            if team:
                team.name = team_data.get("name", team.name)
                team.short_name = team_data.get("tla", team.short_name)
            else:
                team = Team(
                    external_id=ext_id,
                    name=team_data.get("name", ""),
                    short_name=team_data.get("tla", ""),
                    competition_id=comp.id,
                )
                session.add(team)
        await session.commit()

    async def sync_matches(
        self,
        session: AsyncSession,
        competition_code: str,
        season: str | None = None,
        matchday: int | None = None,
    ):
        """Fetch and upsert matches for a competition."""
        comp = await session.execute(
            select(Competition).where(Competition.external_id == competition_code)
        )
        comp = comp.scalar_one_or_none()
        if not comp:
            return

        # Preload all teams into a lookup dict to avoid per-match DB queries
        all_teams = (await session.execute(select(Team))).scalars().all()
        team_map = {t.external_id: t for t in all_teams}

        # Preload existing match external_ids for this competition
        existing_matches = (await session.execute(
            select(Match).where(Match.competition_id == comp.id)
        )).scalars().all()
        match_map = {m.external_id: m for m in existing_matches}

        params = {}
        if season:
            params["season"] = season
        if matchday:
            params["matchday"] = matchday

        data = await self.get(f"/competitions/{competition_code}/matches", params=params)

        for m in data.get("matches", []):
            ext_id = str(m["id"])
            home_ext = str(m["homeTeam"]["id"])
            away_ext = str(m["awayTeam"]["id"])

            home_team = team_map.get(home_ext)
            away_team = team_map.get(away_ext)

            if not home_team or not away_team:
                logger.warning("Team not found for match %s", ext_id)
                continue

            score = m.get("score", {}).get("fullTime", {})
            match_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))

            match = match_map.get(ext_id)
            if match:
                match.status = m.get("status", match.status)
                match.home_goals = score.get("home")
                match.away_goals = score.get("away")
                match.matchday = m.get("matchday")
            else:
                match = Match(
                    external_id=ext_id,
                    competition_id=comp.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    matchday=m.get("matchday"),
                    match_date=match_date,
                    status=m.get("status", "SCHEDULED"),
                    home_goals=score.get("home"),
                    away_goals=score.get("away"),
                )
                session.add(match)
                match_map[ext_id] = match

        await session.commit()
        logger.info("Synced matches for %s", competition_code)

    async def sync_all(self, session: AsyncSession):
        """Full sync: competitions → teams → matches."""
        await self.sync_competitions(session)
        for code in COMPETITIONS:
            await self.sync_teams(session, code)
            await self.sync_matches(session, code)
