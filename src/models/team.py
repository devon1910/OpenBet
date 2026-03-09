from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from src.database import Base


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(20), unique=True, nullable=False)
    name = Column(String(100))
    country = Column(String(50))
    season = Column(String(10))

    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(20), unique=True, nullable=False)
    name = Column(String(100))
    short_name = Column(String(10))
    competition_id = Column(Integer, ForeignKey("competitions.id"))

    competition = relationship("Competition", back_populates="teams")
    elo_rating = relationship("EloRating", uselist=False, back_populates="team")
