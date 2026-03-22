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

    has_odds = odds_home and odds_home > 0

    # odds_home/draw/away are already normalized implied probabilities from the odds
    # collector — use them directly (fall back to model probs when no odds available)
    market_home = odds_home if has_odds else prob_home
    market_draw = odds_draw if odds_draw and odds_draw > 0 else prob_draw
    market_away = odds_away if odds_away and odds_away > 0 else prob_away

    # When no odds are available, require a higher threshold since we can't
    # validate edge against the market — avoids flooding with low-quality picks
    sw_threshold = settings.straight_win_threshold if has_odds else settings.straight_win_threshold + 0.05
    dc_threshold = settings.double_chance_threshold if has_odds else settings.double_chance_threshold + 0.03

    # Straight win checks
    if prob_home > sw_threshold:
        edge = prob_home - market_home
        if edge >= settings.min_value_edge or not has_odds:
            picks.append({
                "pick_type": "STRAIGHT_WIN",
                "pick_value": "HOME",
                "confidence": prob_home,
                "edge": edge if has_odds else 0.0,
                "odds_decimal": 1.0 / market_home if market_home > 0 else None,
            })

    if prob_away > sw_threshold:
        edge = prob_away - market_away
        if edge >= settings.min_value_edge or not has_odds:
            picks.append({
                "pick_type": "STRAIGHT_WIN",
                "pick_value": "AWAY",
                "confidence": prob_away,
                "edge": edge if has_odds else 0.0,
                "odds_decimal": 1.0 / market_away if market_away > 0 else None,
            })

    # Double chance checks — evaluated independently so a match where the
    # straight win lacks edge can still surface a double chance pick.
    # When both qualify for the same side, keep only the better edge.
    home_not_lose = prob_home + prob_draw
    away_not_lose = prob_away + prob_draw
    market_home_not_lose = market_home + market_draw
    market_away_not_lose = market_away + market_draw

    if home_not_lose > dc_threshold:
        edge = home_not_lose - market_home_not_lose
        if edge >= settings.min_value_edge or not has_odds:
            picks.append({
                "pick_type": "DOUBLE_CHANCE",
                "pick_value": "1X",
                "confidence": home_not_lose,
                "edge": edge if has_odds else 0.0,
                "odds_decimal": 1.0 / market_home_not_lose if market_home_not_lose > 0 else None,
            })

    if away_not_lose > dc_threshold:
        edge = away_not_lose - market_away_not_lose
        if edge >= settings.min_value_edge or not has_odds:
            picks.append({
                "pick_type": "DOUBLE_CHANCE",
                "pick_value": "X2",
                "confidence": away_not_lose,
                "edge": edge if has_odds else 0.0,
                "odds_decimal": 1.0 / market_away_not_lose if market_away_not_lose > 0 else None,
            })

    # Deduplicate: if both straight win and double chance exist for the same
    # side, keep only the one with the better edge
    if len(picks) > 1:
        home_sw = [p for p in picks if p["pick_value"] == "HOME"]
        home_dc = [p for p in picks if p["pick_value"] == "1X"]
        away_sw = [p for p in picks if p["pick_value"] == "AWAY"]
        away_dc = [p for p in picks if p["pick_value"] == "X2"]

        drop = set()
        if home_sw and home_dc:
            loser = home_dc[0] if home_sw[0]["edge"] >= home_dc[0]["edge"] else home_sw[0]
            drop.add(id(loser))
        if away_sw and away_dc:
            loser = away_dc[0] if away_sw[0]["edge"] >= away_dc[0]["edge"] else away_sw[0]
            drop.add(id(loser))

        if drop:
            picks = [p for p in picks if id(p) not in drop]

    return picks
