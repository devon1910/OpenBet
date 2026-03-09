"""Ensemble model combining Poisson and XGBoost predictions.

Uses a weighted average calibrated on historical validation performance.
"""

import numpy as np
import pandas as pd

from src.models_ml.poisson import match_outcome_probabilities
from src.models_ml.xgboost_model import load_model, predict, FEATURE_COLUMNS


def ensemble_predict(
    features_df: pd.DataFrame,
    model_version: str = "v1",
    poisson_weight: float = 0.4,
    xgboost_weight: float = 0.6,
) -> list[dict]:
    """Produce ensemble probabilities for each match in features_df.

    Args:
        features_df: DataFrame with FEATURE_COLUMNS
        model_version: XGBoost model version to load
        poisson_weight: weight for Poisson model (default 0.4)
        xgboost_weight: weight for XGBoost model (default 0.6)

    Returns:
        List of dicts with poisson_*, xgb_*, ensemble_* probabilities.
    """
    # XGBoost predictions
    xgb_model = load_model(model_version)
    xgb_probs = predict(xgb_model, features_df)

    results = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        # Poisson predictions
        poisson_result = match_outcome_probabilities(
            home_attack=row["home_attack_strength"],
            home_defense=row["home_defense_strength"],
            away_attack=row["away_attack_strength"],
            away_defense=row["away_defense_strength"],
        )

        p_home = poisson_result["home_win"]
        p_draw = poisson_result["draw"]
        p_away = poisson_result["away_win"]

        x_home = xgb_probs[i][0]
        x_draw = xgb_probs[i][1]
        x_away = xgb_probs[i][2]

        # Weighted ensemble
        e_home = poisson_weight * p_home + xgboost_weight * x_home
        e_draw = poisson_weight * p_draw + xgboost_weight * x_draw
        e_away = poisson_weight * p_away + xgboost_weight * x_away

        # Normalize
        total = e_home + e_draw + e_away
        if total > 0:
            e_home /= total
            e_draw /= total
            e_away /= total

        results.append({
            "poisson_home": p_home,
            "poisson_draw": p_draw,
            "poisson_away": p_away,
            "xgb_home": x_home,
            "xgb_draw": x_draw,
            "xgb_away": x_away,
            "ensemble_home": e_home,
            "ensemble_draw": e_draw,
            "ensemble_away": e_away,
        })

    return results
