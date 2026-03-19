"""Model retrainer.

Monitors performance and triggers retraining when accuracy drops below threshold.
"""

import logging
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.learning.evaluator import evaluate_last_n_weeks
from src.models.prediction import ModelPerformance
from src.models_ml.training import train_and_evaluate

logger = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.45  # Retrain if rolling accuracy drops below this
BRIER_THRESHOLD = 0.30     # Retrain if Brier score exceeds this


async def check_and_retrain(
    session: AsyncSession,
    model_version: str = "v1",
) -> dict:
    """Check recent performance and retrain if needed.

    Returns status dict with action taken.
    """
    metrics = await evaluate_last_n_weeks(session, n_weeks=4, model_version=model_version)

    if "error" in metrics:
        return {"action": "skip", "reason": metrics["error"]}

    accuracy = metrics.get("accuracy", 1.0)
    brier = metrics.get("brier_score")

    needs_retrain = False
    reasons = []

    if accuracy < ACCURACY_THRESHOLD:
        needs_retrain = True
        reasons.append(f"accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD}")

    if brier is not None and brier > BRIER_THRESHOLD:
        needs_retrain = True
        reasons.append(f"brier={brier:.3f} > {BRIER_THRESHOLD}")

    if not needs_retrain:
        logger.info("Model performance acceptable, no retraining needed")
        return {"action": "none", "metrics": metrics}

    # Bump version
    version_num = int(model_version.lstrip("v")) + 1
    new_version = f"v{version_num}"

    logger.info("Retraining triggered: %s", ", ".join(reasons))
    train_metrics = await train_and_evaluate(session, new_version)

    if "error" not in train_metrics:
        # Update the running config so the new model is actually used
        settings.model_version = new_version
        logger.info("Model version updated to %s", new_version)

        # Hot-reload the model in the picks module
        try:
            from src.api.routes import picks
            from src.models_ml.xgboost_model import load_model
            picks._xgb_model = load_model(new_version)
            logger.info("Hot-reloaded XGBoost model %s", new_version)
        except Exception:
            logger.warning("Could not hot-reload model — restart to use %s", new_version)

    return {
        "action": "retrained",
        "reasons": reasons,
        "old_version": model_version,
        "new_version": new_version,
        "train_metrics": train_metrics,
    }
