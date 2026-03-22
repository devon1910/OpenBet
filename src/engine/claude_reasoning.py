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

You will receive MULTIPLE matches. Respond with a JSON array — one object per match, in the same order.

Each object must have this exact format:
{
    "match": "<Home Team> vs <Away Team>",
    "confidence_adjustment": <float between -0.10 and 0.10>,
    "reasoning": "<2-3 sentence explanation>",
    "flags": ["<flag1>", "<flag2>"],
    "unpredictable": <true or false>
}

Possible flags: "title_decider", "relegation_battle", "derby", "dead_rubber",
"key_injuries", "fixture_congestion", "manager_bounce", "form_collapse"
"""

_DEFAULT_REASONING = {
    "confidence_adjustment": 0.0,
    "reasoning": "",
    "flags": [],
    "unpredictable": False,
}


async def get_batch_reasoning(matches: list[dict]) -> list[dict]:
    """Get Claude's contextual reasoning for multiple matches in a single API call.

    Args:
        matches: list of dicts with keys:
            home_team, away_team, competition, model_probs, context

    Returns:
        list of dicts (same order) with confidence_adjustment, reasoning, flags, unpredictable
    """
    if not settings.anthropic_api_key:
        logger.warning("No Anthropic API key configured, skipping reasoning")
        return [dict(_DEFAULT_REASONING) for _ in matches]

    if not matches:
        return []

    # Build a single prompt with all matches
    parts = []
    for i, m in enumerate(matches, 1):
        probs = m["model_probs"]
        ctx = m["context"]
        parts.append(f"""Match {i}: {m['home_team']} vs {m['away_team']}
Competition: {m['competition']}
Model Probabilities: Home {probs.get('ensemble_home', 0):.1%}, Draw {probs.get('ensemble_draw', 0):.1%}, Away {probs.get('ensemble_away', 0):.1%}
Home form: {ctx.get('home_form_str', 'N/A')} | Away form: {ctx.get('away_form_str', 'N/A')}
Home position: {ctx.get('home_position', 'N/A')} | Away position: {ctx.get('away_position', 'N/A')}
Home injuries: {ctx.get('home_injuries', 'None known')} | Away injuries: {ctx.get('away_injuries', 'None known')}
H2H: {ctx.get('h2h', 'N/A')}""")

    user_message = "\n\n".join(parts)
    user_message += f"\n\nAnalyze all {len(matches)} matches and respond with a JSON array."

    try:
        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=120.0,
        )
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200 * len(matches),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = message.content[0].text
        results = json.loads(response_text)

        if not isinstance(results, list):
            logger.error("Claude returned non-array response")
            return [dict(_DEFAULT_REASONING) for _ in matches]

        # Clamp adjustments and pad if Claude returned fewer results
        max_adj = settings.claude_max_adjustment
        output = []
        for i in range(len(matches)):
            if i < len(results):
                r = results[i]
                adj = r.get("confidence_adjustment", 0.0)
                r["confidence_adjustment"] = max(-max_adj, min(max_adj, adj))
                output.append(r)
            else:
                output.append(dict(_DEFAULT_REASONING))

        return output

    except json.JSONDecodeError:
        logger.error("Failed to parse Claude batch response as JSON")
        return [dict(_DEFAULT_REASONING) for _ in matches]
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            logger.warning("Anthropic API credits exhausted — skipping Claude reasoning")
        else:
            logger.exception("Claude BadRequestError")
        return [dict(_DEFAULT_REASONING) for _ in matches]
    except anthropic.RateLimitError:
        logger.warning("Anthropic API rate limit reached — skipping Claude reasoning")
        return [dict(_DEFAULT_REASONING) for _ in matches]
    except Exception:
        logger.exception("Claude batch reasoning failed")
        return [dict(_DEFAULT_REASONING) for _ in matches]


async def get_match_reasoning(
    home_team: str,
    away_team: str,
    competition: str,
    model_probs: dict,
    context: dict,
) -> dict:
    """Get Claude's contextual reasoning for a single match.

    Convenience wrapper around get_batch_reasoning for single-match use.
    """
    results = await get_batch_reasoning([{
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "model_probs": model_probs,
        "context": context,
    }])
    return results[0]
