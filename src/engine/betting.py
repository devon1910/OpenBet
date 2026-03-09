"""Betting decision engine.

Applies threshold rules to model probabilities to identify betting opportunities:
- Straight Win: probability > 60%
- Double Chance: probability of not losing > 75%
"""

from src.config import settings


def evaluate_betting_opportunity(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    unpredictable: bool = False,
) -> list[dict]:
    """Evaluate a match's probabilities and return viable betting picks.

    Returns list of pick dicts, each with:
        pick_type: STRAIGHT_WIN or DOUBLE_CHANCE
        pick_value: HOME, AWAY, 1X, X2, 12
        confidence: the probability backing the pick
    """
    if unpredictable:
        return []

    picks = []

    # Straight win checks
    if prob_home > settings.straight_win_threshold:
        picks.append({
            "pick_type": "STRAIGHT_WIN",
            "pick_value": "HOME",
            "confidence": prob_home,
        })

    if prob_away > settings.straight_win_threshold:
        picks.append({
            "pick_type": "STRAIGHT_WIN",
            "pick_value": "AWAY",
            "confidence": prob_away,
        })

    # Double chance checks (only if no straight win already found for this side)
    home_not_lose = prob_home + prob_draw
    away_not_lose = prob_away + prob_draw

    if home_not_lose > settings.double_chance_threshold and prob_home <= settings.straight_win_threshold:
        picks.append({
            "pick_type": "DOUBLE_CHANCE",
            "pick_value": "1X",
            "confidence": home_not_lose,
        })

    if away_not_lose > settings.double_chance_threshold and prob_away <= settings.straight_win_threshold:
        picks.append({
            "pick_type": "DOUBLE_CHANCE",
            "pick_value": "X2",
            "confidence": away_not_lose,
        })

    return picks
