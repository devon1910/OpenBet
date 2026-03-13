"""Ensemble model using a stacking meta-learner.

Replaces hardcoded blending weights with a LogisticRegression that learns
optimal combination of base model outputs (Poisson, XGBoost, and bookmaker odds).
Falls back to equal-weight averaging when the meta-learner is not available.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models_ml.poisson import match_outcome_probabilities
from src.models_ml.xgboost_model import load_model, predict, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

META_MODEL_DIR = Path("trained_models")


def _build_meta_features(
    poisson_probs: np.ndarray,
    xgb_probs: np.ndarray,
    odds_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Stack base model outputs into meta-feature matrix.

    Args:
        poisson_probs: (n, 3) array of Poisson [home, draw, away]
        xgb_probs: (n, 3) array of XGBoost [home, draw, away]
        odds_probs: (n, 3) array of bookmaker implied [home, draw, away], or None

    Returns:
        (n, 6 or 9) meta-feature matrix
    """
    parts = [poisson_probs, xgb_probs]
    if odds_probs is not None:
        parts.append(odds_probs)
    return np.hstack(parts)


def train_meta_learner(
    meta_X: np.ndarray,
    meta_y: np.ndarray,
    version: str = "v1",
) -> LogisticRegression:
    """Train the stacking meta-learner on out-of-fold base model predictions.

    Args:
        meta_X: (n, 6 or 9) stacked base model probabilities
        meta_y: (n,) outcome labels (0=Home, 1=Draw, 2=Away)
        version: model version string for saving

    Returns:
        Trained LogisticRegression meta-learner
    """
    meta_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
    )
    meta_model.fit(meta_X, meta_y)

    META_MODEL_DIR.mkdir(exist_ok=True)
    path = META_MODEL_DIR / f"meta_learner_{version}.joblib"
    joblib.dump(meta_model, path)
    logger.info("Meta-learner saved to %s", path)

    return meta_model


def load_meta_learner(version: str = "v1") -> LogisticRegression | None:
    """Load a trained meta-learner. Returns None if not found."""
    path = META_MODEL_DIR / f"meta_learner_{version}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _fallback_blend(
    poisson_probs: np.ndarray,
    xgb_probs: np.ndarray,
    odds_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Simple average when meta-learner is unavailable."""
    if odds_probs is not None:
        combined = (poisson_probs + xgb_probs + odds_probs) / 3.0
    else:
        combined = (poisson_probs + xgb_probs) / 2.0
    # Normalize rows
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return combined / row_sums


def ensemble_predict(
    features_df: pd.DataFrame,
    model_version: str = "v1",
    **kwargs,
) -> list[dict]:
    """Produce ensemble probabilities using the stacking meta-learner.

    Falls back to equal-weight averaging if meta-learner is not available.

    Args:
        features_df: DataFrame with FEATURE_COLUMNS
        model_version: model version to load

    Returns:
        List of dicts with poisson_*, xgb_*, ensemble_* probabilities.
    """
    # XGBoost predictions
    xgb_model = load_model(model_version)
    xgb_probs = predict(xgb_model, features_df)

    # Poisson predictions for each row
    poisson_list = []
    for _, row in features_df.iterrows():
        poisson_result = match_outcome_probabilities(
            home_attack=row.get("home_attack_strength", 1.0) or 1.0,
            home_defense=row.get("home_defense_strength", 1.0) or 1.0,
            away_attack=row.get("away_attack_strength", 1.0) or 1.0,
            away_defense=row.get("away_defense_strength", 1.0) or 1.0,
        )
        poisson_list.append([
            poisson_result["home_win"],
            poisson_result["draw"],
            poisson_result["away_win"],
        ])
    poisson_probs = np.array(poisson_list)

    # Odds implied probabilities (if available in features)
    odds_probs = None
    if all(col in features_df.columns for col in ["odds_home", "odds_draw", "odds_away"]):
        odds_cols = features_df[["odds_home", "odds_draw", "odds_away"]].fillna(0).values
        if odds_cols.sum() > 0:
            odds_probs = odds_cols

    # Try stacking meta-learner
    meta_model = load_meta_learner(model_version)
    if meta_model is not None:
        meta_X = _build_meta_features(poisson_probs, xgb_probs, odds_probs)
        # Handle feature count mismatch (model trained with/without odds)
        expected = meta_model.n_features_in_
        if meta_X.shape[1] == expected:
            ensemble_probs = meta_model.predict_proba(meta_X)
        else:
            logger.warning(
                "Meta-learner expects %d features, got %d. Using fallback.",
                expected, meta_X.shape[1],
            )
            ensemble_probs = _fallback_blend(poisson_probs, xgb_probs, odds_probs)
    else:
        logger.info("No meta-learner found, using equal-weight fallback")
        ensemble_probs = _fallback_blend(poisson_probs, xgb_probs, odds_probs)

    # Build results
    results = []
    for i in range(len(features_df)):
        e_home, e_draw, e_away = ensemble_probs[i]

        results.append({
            "poisson_home": float(poisson_probs[i][0]),
            "poisson_draw": float(poisson_probs[i][1]),
            "poisson_away": float(poisson_probs[i][2]),
            "xgb_home": float(xgb_probs[i][0]),
            "xgb_draw": float(xgb_probs[i][1]),
            "xgb_away": float(xgb_probs[i][2]),
            "ensemble_home": float(e_home),
            "ensemble_draw": float(e_draw),
            "ensemble_away": float(e_away),
        })

    return results
