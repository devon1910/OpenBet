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
    xgb_probs: np.ndarray | None = None,
    odds_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Simple average when meta-learner is unavailable."""
    components = [poisson_probs]
    if xgb_probs is not None:
        components.append(xgb_probs)
    if odds_probs is not None:
        components.append(odds_probs)
    combined = sum(components) / len(components)
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
    # XGBoost predictions (may not be available if model hasn't been trained yet)
    try:
        xgb_model = load_model(model_version)
        xgb_probs = predict(xgb_model, features_df)
    except FileNotFoundError:
        logger.warning("XGBoost model %s not found — using Poisson only", model_version)
        xgb_model = None
        xgb_probs = None

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
        odds_raw = features_df[["odds_home", "odds_draw", "odds_away"]].copy()
        # Check if any row has valid odds
        has_valid = (odds_raw > 0).all(axis=1).any()
        if has_valid:
            # Replace missing/invalid with neutral 1/3
            invalid = odds_raw.isna() | (odds_raw <= 0)
            odds_raw = odds_raw.fillna(1 / 3)
            odds_raw[invalid] = 1 / 3
            odds_probs = odds_raw.values

    # Try stacking meta-learner (only if XGBoost is available)
    meta_model = load_meta_learner(model_version) if xgb_probs is not None else None
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
        xgb_h = float(xgb_probs[i][0]) if xgb_probs is not None else None
        xgb_d = float(xgb_probs[i][1]) if xgb_probs is not None else None
        xgb_a = float(xgb_probs[i][2]) if xgb_probs is not None else None

        results.append({
            "poisson_home": float(poisson_probs[i][0]),
            "poisson_draw": float(poisson_probs[i][1]),
            "poisson_away": float(poisson_probs[i][2]),
            "xgb_home": xgb_h,
            "xgb_draw": xgb_d,
            "xgb_away": xgb_a,
            "ensemble_home": float(e_home),
            "ensemble_draw": float(e_draw),
            "ensemble_away": float(e_away),
        })

    return results
