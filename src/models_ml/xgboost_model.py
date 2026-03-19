"""XGBoost match outcome classifier.

Predicts Home Win / Draw / Away Win probabilities from match features.
Uses softprob objective for 3-class probability output.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "home_form",
    "away_form",
    "home_form_home",
    "away_form_away",
    "home_attack_strength",
    "home_defense_strength",
    "away_attack_strength",
    "away_defense_strength",
    "elo_diff",
    "home_xg_avg",
    "away_xg_avg",
    "home_xg_conceded_avg",
    "away_xg_conceded_avg",
    "xg_diff",
    "home_advantage",
    "h2h_home_win_rate",
    "home_days_rest",
    "away_days_rest",
    "home_fixture_congestion",
    "away_fixture_congestion",
    "odds_home",
    "odds_draw",
    "odds_away",
]

MODEL_DIR = Path("trained_models")


def create_model() -> XGBClassifier:
    """Create a new XGBoost classifier with tuned hyperparameters."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean feature columns. Handle missing values properly."""
    features = df[FEATURE_COLUMNS].copy()
    # Convert all columns to numeric (handles None/object types from DB)
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    # Odds columns: fill missing with -1 sentinel (0.0 would imply infinite probability)
    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    for col in odds_cols:
        if col in features.columns:
            features[col] = features[col].fillna(-1.0)

    # Add has_odds indicator so the model knows when odds are real vs missing
    if "odds_home" in features.columns:
        features["has_odds"] = (features["odds_home"] > 0).astype(float)

    # Fill remaining features with 0 (form, xG, etc.)
    features = features.fillna(0.0)
    return features


def encode_outcome(home_goals: int, away_goals: int) -> int:
    """Encode match outcome: 0=Home Win, 1=Draw, 2=Away Win."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    return 2


def train(df: pd.DataFrame, version: str = "v1") -> XGBClassifier:
    """Train XGBoost model on historical match data with features.

    df must contain FEATURE_COLUMNS plus 'home_goals' and 'away_goals'.
    """
    X = prepare_features(df)
    y = df.apply(lambda r: encode_outcome(r["home_goals"], r["away_goals"]), axis=1)

    model = create_model()
    # Final model trains on all data — disable early stopping
    model.set_params(early_stopping_rounds=None)
    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"xgboost_{version}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)

    return model


def load_model(version: str = "v1") -> XGBClassifier:
    """Load a trained model from disk."""
    path = MODEL_DIR / f"xgboost_{version}.joblib"
    return joblib.load(path)


def predict(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    """Predict probabilities for each match.

    Returns array of shape (n_matches, 3) with columns [home_win, draw, away_win].
    """
    X = prepare_features(features)
    return model.predict_proba(X)
