"""AI reasoning layer (Gemini primary, Claude fallback).

Uses an LLM to provide contextual analysis of match predictions,
accounting for factors statistical models cannot capture:
- Title race / relegation motivation
- Player injuries and suspensions
- Team confidence and momentum
- Derby / rivalry intensity
- Competition importance (dead rubber vs must-win)
"""

import json
import logging

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


def _build_user_message(matches: list[dict]) -> str:
    """Build the prompt from match data."""
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
    return user_message


def _parse_and_clamp(response_text: str, num_matches: int) -> list[dict] | None:
    """Parse JSON response and clamp adjustments. Returns None on failure."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    results = json.loads(text)

    if not isinstance(results, list):
        return None

    max_adj = settings.claude_max_adjustment
    output = []
    for i in range(num_matches):
        if i < len(results):
            r = results[i]
            adj = r.get("confidence_adjustment", 0.0)
            r["confidence_adjustment"] = max(-max_adj, min(max_adj, adj))
            output.append(r)
        else:
            output.append(dict(_DEFAULT_REASONING))

    return output


async def _gemini_reasoning(matches: list[dict]) -> list[dict] | None:
    """Try Gemini API. Returns None on failure."""
    if not settings.gemini_api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        user_message = _build_user_message(matches)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
            config={
                "response_mime_type": "application/json",
                "max_output_tokens": 250 * len(matches),
            },
        )

        return _parse_and_clamp(response.text, len(matches))

    except json.JSONDecodeError:
        logger.error("Failed to parse Gemini response as JSON")
        return None
    except Exception:
        logger.exception("Gemini reasoning failed")
        return None


async def _claude_reasoning(matches: list[dict]) -> list[dict] | None:
    """Try Claude API. Returns None on failure."""
    if not settings.anthropic_api_key:
        return None

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=120.0,
        )
        user_message = _build_user_message(matches)

        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200 * len(matches),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return _parse_and_clamp(message.content[0].text, len(matches))

    except json.JSONDecodeError:
        logger.error("Failed to parse Claude response as JSON")
        return None
    except Exception:
        logger.exception("Claude reasoning failed")
        return None


async def get_batch_reasoning(matches: list[dict]) -> list[dict]:
    """Get AI contextual reasoning for multiple matches.

    Tries Gemini first (free tier), falls back to Claude, then returns defaults.

    Args:
        matches: list of dicts with keys:
            home_team, away_team, competition, model_probs, context

    Returns:
        list of dicts (same order) with confidence_adjustment, reasoning, flags, unpredictable
    """
    if not matches:
        return []

    defaults = [dict(_DEFAULT_REASONING) for _ in matches]

    # 1. Try Gemini (free)
    result = await _gemini_reasoning(matches)
    if result is not None:
        logger.info("Reasoning provided by Gemini")
        return result

    # 2. Try Claude (paid fallback)
    result = await _claude_reasoning(matches)
    if result is not None:
        logger.info("Reasoning provided by Claude")
        return result

    # 3. No AI available
    logger.warning("No AI reasoning available — skipping adjustments")
    return defaults


async def get_match_reasoning(
    home_team: str,
    away_team: str,
    competition: str,
    model_probs: dict,
    context: dict,
) -> dict:
    """Get AI contextual reasoning for a single match.

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
