"""Betting decision engine.

Applies threshold rules AND value edge detection to model probabilities.
Only recommends picks where the model finds genuine edge over the market.
"""

from src.config import settings


def evaluate_betting_opportunity(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    odds_home: float = None,
    odds_draw: float = None,
    odds_away: float = None,
    unpredictable: bool = False,
) -> list[dict]:
    """Evaluate a match's probabilities and return viable betting picks.

    Only returns picks where:
    1. Model probability exceeds threshold (straight win or double chance)
    2. Model probability exceeds market implied probability by min_value_edge

    Returns list of pick dicts with:
        pick_type: STRAIGHT_WIN or DOUBLE_CHANCE
        pick_value: HOME, AWAY, 1X, X2
        confidence: the probability backing the pick
        edge: value edge over market (model_prob - market_prob)
        odds_decimal: decimal odds for ROI tracking
    """
    if unpredictable:
        return []

    picks = []

    # Market implied probabilities (default to model probs if no odds available)
    market_home = odds_home if odds_home is not None else prob_home
    market_draw = odds_draw if odds_draw is not None else prob_draw
    market_away = odds_away if odds_away is not None else prob_away

    # Straight win checks
    if prob_home > settings.straight_win_threshold:
        edge = prob_home - market_home
        if edge >= settings.min_value_edge or odds_home is None:
            picks.append({
                "pick_type": "STRAIGHT_WIN",
                "pick_value": "HOME",
                "confidence": prob_home,
                "edge": edge,
                "odds_decimal": 1.0 / market_home if market_home > 0 else None,
            })

    if prob_away > settings.straight_win_threshold:
        edge = prob_away - market_away
        if edge >= settings.min_value_edge or odds_away is None:
            picks.append({
                "pick_type": "STRAIGHT_WIN",
                "pick_value": "AWAY",
                "confidence": prob_away,
                "edge": edge,
                "odds_decimal": 1.0 / market_away if market_away > 0 else None,
            })

    # Double chance checks (only if no straight win already found for this side)
    home_not_lose = prob_home + prob_draw
    away_not_lose = prob_away + prob_draw
    market_home_not_lose = market_home + market_draw
    market_away_not_lose = market_away + market_draw

    if home_not_lose > settings.double_chance_threshold and prob_home <= settings.straight_win_threshold:
        edge = home_not_lose - market_home_not_lose
        if edge >= settings.min_value_edge or odds_home is None:
            picks.append({
                "pick_type": "DOUBLE_CHANCE",
                "pick_value": "1X",
                "confidence": home_not_lose,
                "edge": edge,
                "odds_decimal": 1.0 / market_home_not_lose if market_home_not_lose > 0 else None,
            })

    if away_not_lose > settings.double_chance_threshold and prob_away <= settings.straight_win_threshold:
        edge = away_not_lose - market_away_not_lose
        if edge >= settings.min_value_edge or odds_away is None:
            picks.append({
                "pick_type": "DOUBLE_CHANCE",
                "pick_value": "X2",
                "confidence": away_not_lose,
                "edge": edge,
                "odds_decimal": 1.0 / market_away_not_lose if market_away_not_lose > 0 else None,
            })

    return picks
