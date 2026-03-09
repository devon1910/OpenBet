from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.database import Base


class EloRating(Base):
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), unique=True)
    rating = Column(Float, default=1500.0)
    last_match_id = Column(Integer, ForeignKey("matches.id"), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    team = relationship("Team", back_populates="elo_rating")


class EloHistory(Base):
    __tablename__ = "elo_history"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    match_id = Column(Integer, ForeignKey("matches.id"))
    rating_before = Column(Float)
    rating_after = Column(Float)

    __table_args__ = (UniqueConstraint("team_id", "match_id"),)
