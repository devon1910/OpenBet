"""Poisson goal distribution model.

Uses attack/defense strength to estimate expected goals (lambda),
then computes a goal probability grid to derive H/D/A probabilities.
"""

import numpy as np
from scipy.stats import poisson


def estimate_lambda(
    attack_strength: float,
    defense_strength_opponent: float,
    league_avg_goals: float = 1.4,
) -> float:
    """Estimate expected goals (lambda) for a team.

    lambda = attack_strength * opponent_defense_strength * league_avg_goals

    A high opponent defense_strength (>1) means opponent concedes more than average,
    so the team is expected to score more.
    """
    return max(attack_strength * defense_strength_opponent * league_avg_goals, 0.1)


def goal_probability_grid(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 6,
) -> np.ndarray:
    """Compute joint probability grid P(home=i, away=j).

    Returns array of shape (max_goals+1, max_goals+1).
    """
    home_probs = np.array([poisson.pmf(k, home_lambda) for k in range(max_goals + 1)])
    away_probs = np.array([poisson.pmf(k, away_lambda) for k in range(max_goals + 1)])
    return np.outer(home_probs, away_probs)


def match_outcome_probabilities(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_avg_goals: float = 1.4,
) -> dict:
    """Compute H/D/A probabilities using Poisson model.

    Args:
        home_attack: Home team attack strength
        home_defense: Home team defense strength
        away_attack: Away team attack strength
        away_defense: Away team defense strength
        league_avg_goals: Average goals per team per match in the league

    Returns:
        dict with keys: home_win, draw, away_win, home_lambda, away_lambda,
        and goal_probs (dict of P(exactly N goals) for each team)
    """
    # Home team expected goals = home attack * away defense * avg
    # (away defense > 1 means they concede more than average)
    home_lambda = estimate_lambda(home_attack, away_defense, league_avg_goals)
    # Away team expected goals = away attack * home defense * avg
    # No home/away penalty here — home advantage handled via Elo blending
    away_lambda = estimate_lambda(away_attack, home_defense, league_avg_goals)

    grid = goal_probability_grid(home_lambda, away_lambda)

    # Home win: sum where i > j
    home_win = float(np.sum(np.tril(grid, -1).T))
    # Draw: sum of diagonal
    draw = float(np.sum(np.diag(grid)))
    # Away win: sum where j > i
    away_win = float(np.sum(np.tril(grid, -1)))

    # Normalize to ensure they sum to 1
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    # Individual goal probabilities
    home_goal_probs = {k: float(poisson.pmf(k, home_lambda)) for k in range(5)}
    away_goal_probs = {k: float(poisson.pmf(k, away_lambda)) for k in range(5)}

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "home_goal_probs": home_goal_probs,
        "away_goal_probs": away_goal_probs,
    }
