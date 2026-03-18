"""Training pipeline for XGBoost model and stacking meta-learner.

Loads historical match features, trains base models with time-series CV,
collects out-of-fold predictions, and trains a meta-learner on top.
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.feature import MatchFeature
from src.models.match import Match
from src.models_ml.ensemble import (
    _build_meta_features,
    train_meta_learner,
)
from src.models_ml.poisson import match_outcome_probabilities
from src.models_ml.xgboost_model import (
    FEATURE_COLUMNS,
    create_model,
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


def _get_poisson_probs(df: pd.DataFrame) -> np.ndarray:
    """Compute Poisson probabilities for each row in df."""
    probs = []
    for _, row in df.iterrows():
        result = match_outcome_probabilities(
            home_attack=row.get("home_attack_strength", 1.0) or 1.0,
            home_defense=row.get("home_defense_strength", 1.0) or 1.0,
            away_attack=row.get("away_attack_strength", 1.0) or 1.0,
            away_defense=row.get("away_defense_strength", 1.0) or 1.0,
        )
        probs.append([result["home_win"], result["draw"], result["away_win"]])
    return np.array(probs)


def _get_odds_probs(df: pd.DataFrame) -> np.ndarray | None:
    """Extract odds implied probabilities from df, or None if not available."""
    if not all(col in df.columns for col in ["odds_home", "odds_draw", "odds_away"]):
        return None
    odds = df[["odds_home", "odds_draw", "odds_away"]].fillna(0).values
    if odds.sum() == 0:
        return None
    return odds


def train_from_dataframe(df: pd.DataFrame, version: str = "v1") -> dict:
    """CPU-bound training pipeline. Safe to call from a thread.

    1. Train XGBoost with TimeSeriesSplit CV
    2. Collect out-of-fold predictions from all base models
    3. Train stacking meta-learner on those predictions
    4. Train final XGBoost on all data

    Returns evaluation metrics.
    """
    if len(df) < 50:
        logger.warning("Not enough data for training: %d matches", len(df))
        return {"error": "insufficient_data", "n_matches": len(df)}

    t0 = time.time()
    logger.info("Training on %d matches", len(df))

    tscv = TimeSeriesSplit(n_splits=3)
    X = prepare_features(df)
    y = df.apply(lambda r: encode_outcome(r["home_goals"], r["away_goals"]), axis=1)
    logger.info("[train] prepare_features done in %.1fs", time.time() - t0)

    accuracies = []
    log_losses = []

    # Collect out-of-fold predictions for meta-learner training
    oof_meta_X = []
    oof_meta_y = []

    t1 = time.time()
    poisson_probs_all = _get_poisson_probs(X)
    logger.info("[train] poisson_probs done in %.1fs", time.time() - t1)

    odds_probs_all = _get_odds_probs(X)

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        t2 = time.time()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train XGBoost on this fold
        model = create_model()
        model.fit(X_train, y_train)
        logger.info("[train] fold %d XGBoost fit done in %.1fs", fold_i, time.time() - t2)

        # XGBoost validation predictions
        xgb_val_probs = model.predict_proba(X_val)
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        log_losses.append(log_loss(y_val, xgb_val_probs))

        # Collect out-of-fold base model predictions for meta-learner
        poisson_val = poisson_probs_all[val_idx]
        odds_val = odds_probs_all[val_idx] if odds_probs_all is not None else None

        meta_X_fold = _build_meta_features(poisson_val, xgb_val_probs, odds_val)
        oof_meta_X.append(meta_X_fold)
        oof_meta_y.append(y_val.values)

    # Train meta-learner on all out-of-fold predictions
    all_meta_X = np.vstack(oof_meta_X)
    all_meta_y = np.concatenate(oof_meta_y)

    t3 = time.time()
    meta_model = train_meta_learner(all_meta_X, all_meta_y, version)
    logger.info("[train] meta_learner done in %.1fs", time.time() - t3)

    # Evaluate meta-learner
    meta_pred = meta_model.predict(all_meta_X)
    meta_proba = meta_model.predict_proba(all_meta_X)
    meta_accuracy = accuracy_score(all_meta_y, meta_pred)
    meta_logloss = log_loss(all_meta_y, meta_proba)

    # Train final XGBoost on all data
    t4 = time.time()
    final_model = train(df, version)
    logger.info("[train] final XGBoost done in %.1fs", time.time() - t4)

    total_time = time.time() - t0
    metrics = {
        "n_matches": len(df),
        "xgb_cv_accuracy_mean": float(np.mean(accuracies)),
        "xgb_cv_accuracy_std": float(np.std(accuracies)),
        "xgb_cv_logloss_mean": float(np.mean(log_losses)),
        "xgb_cv_logloss_std": float(np.std(log_losses)),
        "meta_accuracy": float(meta_accuracy),
        "meta_logloss": float(meta_logloss),
        "meta_n_features": int(all_meta_X.shape[1]),
        "has_odds": odds_probs_all is not None,
        "model_version": version,
        "training_time_seconds": round(total_time, 1),
    }
    logger.info("Training complete in %.1fs: %s", total_time, metrics)
    return metrics


async def train_and_evaluate(
    session: AsyncSession,
    version: str = "v1",
) -> dict:
    """Async wrapper: loads data from DB, then runs training."""
    df = await load_training_data(session)
    return train_from_dataframe(df, version)
