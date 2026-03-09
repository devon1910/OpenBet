"""Training pipeline for the XGBoost model.

Loads historical match features, splits train/validation, trains, and evaluates.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.feature import MatchFeature
from src.models.match import Match
from src.models_ml.xgboost_model import (
    FEATURE_COLUMNS,
    encode_outcome,
    prepare_features,
    train,
)

logger = logging.getLogger(__name__)


async def load_training_data(session: AsyncSession) -> pd.DataFrame:
    """Load all finished matches with features as a DataFrame."""
    stmt = (
        select(Match, MatchFeature)
        .join(MatchFeature, MatchFeature.match_id == Match.id)
        .where(
            Match.status == "FINISHED",
            Match.home_goals.is_not(None),
        )
        .order_by(Match.match_date.asc())
    )
    result = await session.execute(stmt)
    rows = result.all()

    records = []
    for match, feature in rows:
        record = {col: getattr(feature, col) for col in FEATURE_COLUMNS}
        record["home_goals"] = match.home_goals
        record["away_goals"] = match.away_goals
        record["match_date"] = match.match_date
        record["match_id"] = match.id
        records.append(record)

    return pd.DataFrame(records)


async def train_and_evaluate(
    session: AsyncSession,
    version: str = "v1",
) -> dict:
    """Full training pipeline with time-series cross-validation.

    Returns evaluation metrics.
    """
    df = await load_training_data(session)
    if len(df) < 50:
        logger.warning("Not enough data for training: %d matches", len(df))
        return {"error": "insufficient_data", "n_matches": len(df)}

    logger.info("Training on %d matches", len(df))

    # Time-series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    X = prepare_features(df)
    y = df.apply(lambda r: encode_outcome(r["home_goals"], r["away_goals"]), axis=1)

    accuracies = []
    log_losses = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        from src.models_ml.xgboost_model import create_model
        model = create_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        log_losses.append(log_loss(y_val, y_prob))

    # Final model on all data
    final_model = train(df, version)

    metrics = {
        "n_matches": len(df),
        "cv_accuracy_mean": float(np.mean(accuracies)),
        "cv_accuracy_std": float(np.std(accuracies)),
        "cv_logloss_mean": float(np.mean(log_losses)),
        "cv_logloss_std": float(np.std(log_losses)),
        "model_version": version,
    }
    logger.info("Training complete: %s", metrics)
    return metrics
