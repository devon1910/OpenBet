"""Claude AI reasoning layer.

Uses Claude API to provide contextual analysis of match predictions,
accounting for factors statistical models cannot capture:
- Title race / relegation motivation
- Player injuries and suspensions
- Team confidence and momentum
- Derby / rivalry intensity
- Competition importance (dead rubber vs must-win)
"""

import json
import logging

import anthropic

from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional football betting analyst. You receive statistical model predictions for football matches and provide contextual reasoning.

Your role:
1. Evaluate whether the model probabilities make sense given the match context
2. Identify factors the model may have missed (motivation, fatigue, tactical changes, etc.)
3. Provide a confidence adjustment (-0.10 to +0.10) to the model's predicted probability
4. Flag matches that are too unpredictable to bet on

Always respond with valid JSON in this exact format:
{
    "confidence_adjustment": <float between -0.10 and 0.10>,
    "reasoning": "<2-3 sentence explanation>",
    "flags": ["<flag1>", "<flag2>"],
    "unpredictable": <true or false>
}

Possible flags: "title_decider", "relegation_battle", "derby", "dead_rubber",
"key_injuries", "fixture_congestion", "manager_bounce", "form_collapse"
"""


async def get_match_reasoning(
    home_team: str,
    away_team: str,
    competition: str,
    model_probs: dict,
    context: dict,
) -> dict:
    """Get Claude's contextual reasoning for a match prediction.

    Args:
        home_team: Home team name
        away_team: Away team name
        competition: Competition name
        model_probs: dict with ensemble_home, ensemble_draw, ensemble_away
        context: dict with additional context (standings, form, injuries, etc.)

    Returns:
        dict with confidence_adjustment, reasoning, flags, unpredictable
    """
    if not settings.anthropic_api_key:
        logger.warning("No Anthropic API key configured, skipping reasoning")
        return {
            "confidence_adjustment": 0.0,
            "reasoning": "AI reasoning unavailable (no API key)",
            "flags": [],
            "unpredictable": False,
        }

    user_message = f"""Match: {home_team} vs {away_team}
Competition: {competition}

Model Probabilities:
- Home Win: {model_probs.get('ensemble_home', 0):.1%}
- Draw: {model_probs.get('ensemble_draw', 0):.1%}
- Away Win: {model_probs.get('ensemble_away', 0):.1%}

Context:
- Home team league position: {context.get('home_position', 'N/A')}
- Away team league position: {context.get('away_position', 'N/A')}
- Home team recent form: {context.get('home_form_str', 'N/A')}
- Away team recent form: {context.get('away_form_str', 'N/A')}
- Home team key injuries: {context.get('home_injuries', 'None known')}
- Away team key injuries: {context.get('away_injuries', 'None known')}
- Head-to-head recent: {context.get('h2h', 'N/A')}
- Additional notes: {context.get('notes', 'None')}

Analyze this match and provide your assessment."""

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = message.content[0].text
        result = json.loads(response_text)

        # Clamp adjustment to allowed range
        adj = result.get("confidence_adjustment", 0.0)
        max_adj = settings.claude_max_adjustment
        result["confidence_adjustment"] = max(-max_adj, min(max_adj, adj))

        return result

    except json.JSONDecodeError:
        logger.error("Failed to parse Claude response as JSON")
        return {
            "confidence_adjustment": 0.0,
            "reasoning": "AI reasoning failed (invalid response)",
            "flags": [],
            "unpredictable": False,
        }
    except Exception:
        logger.exception("Claude reasoning failed")
        return {
            "confidence_adjustment": 0.0,
            "reasoning": "AI reasoning unavailable (API error)",
            "flags": [],
            "unpredictable": False,
        }
