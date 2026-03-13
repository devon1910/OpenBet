from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    model_version = Column(String(20))

    # Poisson model probabilities
    poisson_home = Column(Float)
    poisson_draw = Column(Float)
    poisson_away = Column(Float)

    # XGBoost model probabilities
    xgb_home = Column(Float)
    xgb_draw = Column(Float)
    xgb_away = Column(Float)

    # Ensemble probabilities
    ensemble_home = Column(Float)
    ensemble_draw = Column(Float)
    ensemble_away = Column(Float)

    # Claude reasoning
    claude_reasoning = Column(Text)
    claude_confidence_adj = Column(Float, default=0.0)

    # Final adjusted probabilities
    prob_home = Column(Float)
    prob_draw = Column(Float)
    prob_away = Column(Float)

    predicted_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="prediction")
    picks = relationship("Pick", back_populates="prediction")


class Pick(Base):
    __tablename__ = "picks"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    match_id = Column(Integer, ForeignKey("matches.id"))
    pick_type = Column(String(20))   # STRAIGHT_WIN, DOUBLE_CHANCE
    pick_value = Column(String(20))  # HOME, AWAY, 1X, X2
    confidence = Column(Float)
    edge = Column(Float)             # value edge: model_prob - market_prob
    odds_decimal = Column(Float)     # decimal odds at time of pick
    reasoning = Column(Text)
    outcome = Column(String(10))     # WIN, LOSS, VOID
    matchday_label = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction = relationship("Prediction", back_populates="picks")


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True)
    period_start = Column(Date)
    period_end = Column(Date)
    total_picks = Column(Integer)
    correct_picks = Column(Integer)
    accuracy = Column(Float)
    brier_score = Column(Float)
    roi = Column(Float)
    model_version = Column(String(20))
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
