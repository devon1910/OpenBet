from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.database import Base


class MatchFeature(Base):
    __tablename__ = "match_features"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True)

    # Form
    home_form = Column(Float)
    away_form = Column(Float)

    # Strength
    home_attack_strength = Column(Float)
    home_defense_strength = Column(Float)
    away_attack_strength = Column(Float)
    away_defense_strength = Column(Float)

    # Elo
    elo_diff = Column(Float)

    # xG
    home_xg_avg = Column(Float)
    away_xg_avg = Column(Float)
    home_xg_conceded_avg = Column(Float)
    away_xg_conceded_avg = Column(Float)
    xg_diff = Column(Float)

    # Home advantage
    home_advantage = Column(Float)

    # Head to head
    h2h_home_wins = Column(Integer, default=0)
    h2h_draws = Column(Integer, default=0)
    h2h_away_wins = Column(Integer, default=0)

    # Bookmaker odds (implied probabilities)
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)

    # Match context
    h2h_home_win_rate = Column(Float)
    home_days_rest = Column(Integer)
    away_days_rest = Column(Integer)
    home_fixture_congestion = Column(Integer)
    away_fixture_congestion = Column(Integer)

    # Injuries
    home_injuries_key = Column(Integer, default=0)
    away_injuries_key = Column(Integer, default=0)

    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="features")
