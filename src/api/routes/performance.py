"""Performance endpoints - model accuracy, ROI, and breakdowns."""

from collections import defaultdict
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.learning.backtester import backtest
from src.learning.evaluator import evaluate_period
from src.models.prediction import ModelPerformance, Pick

router = APIRouter()


@router.get("/")
async def get_performance(db: AsyncSession = Depends(get_db)):
    """Get overall model performance metrics with ROI and breakdowns."""
    # Latest performance records
    stmt = (
        select(ModelPerformance)
        .order_by(ModelPerformance.computed_at.desc())
        .limit(5)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    # All-time stats
    total = (await db.execute(
        select(func.count(Pick.id)).where(Pick.outcome.is_not(None))
    )).scalar() or 0

    wins = (await db.execute(
        select(func.count(Pick.id)).where(Pick.outcome == "WIN")
    )).scalar() or 0

    # All-time ROI + breakdown by bet type
    all_picks_stmt = select(Pick).where(Pick.outcome.is_not(None))
    all_picks_result = await db.execute(all_picks_stmt)
    all_picks = all_picks_result.scalars().all()

    total_staked = 0
    total_profit = 0.0
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "profit": 0.0, "staked": 0})

    for p in all_picks:
        bt = by_type[p.pick_type]
        bt["total"] += 1
        if p.outcome == "WIN":
            bt["correct"] += 1

        if p.odds_decimal is not None and p.odds_decimal > 0:
            total_staked += 1
            bt["staked"] += 1
            if p.outcome == "WIN":
                profit = p.odds_decimal - 1.0
            else:
                profit = -1.0
            total_profit += profit
            bt["profit"] += profit

    type_breakdown = {
        k: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0,
            "roi": round(v["profit"] / v["staked"] * 100, 2) if v["staked"] > 0 else None,
        }
        for k, v in by_type.items()
    }

    return {
        "all_time": {
            "total_picks": total,
            "correct_picks": wins,
            "accuracy": round(wins / total, 3) if total else None,
            "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else None,
        },
        "by_bet_type": type_breakdown,
        "recent_evaluations": [
            {
                "period": f"{r.period_start} to {r.period_end}",
                "accuracy": round(r.accuracy, 3) if r.accuracy else None,
                "brier_score": round(r.brier_score, 3) if r.brier_score else None,
                "roi": round(r.roi, 2) if r.roi is not None else None,
                "total_picks": r.total_picks,
                "model_version": r.model_version,
            }
            for r in records
        ],
    }


@router.get("/paper-trading")
async def get_paper_trading(db: AsyncSession = Depends(get_db)):
    """Paper trading simulation — flat 1-unit stakes on all picks with odds."""
    from src.models.match import Match

    # Load all picks with their match dates, ordered chronologically
    stmt = (
        select(Pick, Match.match_date, Match.home_goals, Match.away_goals)
        .join(Match, Match.id == Pick.match_id)
        .order_by(Match.match_date.asc(), Pick.id.asc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return {"error": "no_picks", "message": "No picks found yet."}

    starting_bankroll = 100.0
    bankroll = starting_bankroll
    unit_stake = 1.0

    # Build equity curve and stats
    equity_curve = []
    daily = defaultdict(lambda: {"picks": 0, "profit": 0.0})
    settled = 0
    pending = 0
    wins = 0
    losses = 0
    current_streak = 0
    best_win_streak = 0
    worst_loss_streak = 0
    by_type = defaultdict(lambda: {"total": 0, "wins": 0, "profit": 0.0, "staked": 0})

    recent_results = []

    for pick, match_date, home_goals, away_goals in rows:
        day_key = match_date.strftime("%Y-%m-%d") if hasattr(match_date, 'strftime') else str(match_date)

        if pick.outcome is None:
            pending += 1
            continue

        settled += 1
        bt = by_type[pick.pick_type]
        bt["total"] += 1

        profit = 0.0
        if pick.odds_decimal and pick.odds_decimal > 0:
            bt["staked"] += 1
            if pick.outcome == "WIN":
                profit = (pick.odds_decimal - 1.0) * unit_stake
            else:
                profit = -unit_stake
            bt["profit"] += profit

        if pick.outcome == "WIN":
            wins += 1
            bt["wins"] += 1
            current_streak = max(1, current_streak + 1) if current_streak >= 0 else 1
            best_win_streak = max(best_win_streak, current_streak)
        elif pick.outcome == "LOSS":
            losses += 1
            current_streak = min(-1, current_streak - 1) if current_streak <= 0 else -1
            worst_loss_streak = max(worst_loss_streak, abs(current_streak))

        bankroll += profit
        daily[day_key]["picks"] += 1
        daily[day_key]["profit"] += profit

        recent_results.append({
            "pick_type": pick.pick_type,
            "pick_value": pick.pick_value,
            "confidence": round(pick.confidence, 3) if pick.confidence else None,
            "edge": round(pick.edge, 3) if pick.edge else None,
            "odds": round(pick.odds_decimal, 2) if pick.odds_decimal else None,
            "outcome": pick.outcome,
            "profit": round(profit, 2),
            "date": day_key,
        })

    # Build equity curve
    running_bankroll = starting_bankroll
    for day_key in sorted(daily.keys()):
        d = daily[day_key]
        running_bankroll += d["profit"]
        equity_curve.append({
            "date": day_key,
            "bankroll": round(running_bankroll, 2),
            "picks": d["picks"],
            "profit": round(d["profit"], 2),
        })

    streak_label = f"W{current_streak}" if current_streak > 0 else f"L{abs(current_streak)}" if current_streak < 0 else "—"

    total_staked = sum(v["staked"] for v in by_type.values())
    total_profit = bankroll - starting_bankroll

    return {
        "starting_bankroll": starting_bankroll,
        "current_bankroll": round(bankroll, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else None,
        "unit_stake": unit_stake,
        "total_picks": settled + pending,
        "settled_picks": settled,
        "pending_picks": pending,
        "wins": wins,
        "losses": losses,
        "accuracy": round(wins / settled, 3) if settled > 0 else None,
        "streak": {
            "current": streak_label,
            "best_win": best_win_streak,
            "worst_loss": worst_loss_streak,
        },
        "by_type": {
            k: {
                "total": v["total"],
                "wins": v["wins"],
                "accuracy": round(v["wins"] / v["total"], 3) if v["total"] else 0,
                "roi": round(v["profit"] / v["staked"] * 100, 2) if v["staked"] > 0 else None,
            }
            for k, v in by_type.items()
        },
        "equity_curve": equity_curve,
        "recent_results": recent_results[-20:],
    }


@router.post("/evaluate")
async def run_evaluation(
    weeks: int = Query(4, ge=1, le=52),
    model_version: str = Query("v1"),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a performance evaluation for the last N weeks."""
    end = date.today()
    start = end - timedelta(weeks=weeks)
    return await evaluate_period(db, start, end, model_version)


@router.post("/backtest")
async def run_backtest(
    start_date: date = Query(...),
    end_date: date = Query(...),
    model_version: str = Query("v1"),
    db: AsyncSession = Depends(get_db),
):
    """Run a walk-forward backtest over a date range.

    Returns accuracy, ROI, perfect matchday rate, and per-day breakdowns.
    """
    return await backtest(db, start_date, end_date, model_version)
