"""Create all tables from scratch.

Revision ID: 000_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "000_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("external_id", sa.String(20), unique=True, nullable=False),
        sa.Column("name", sa.String(100)),
        sa.Column("country", sa.String(50)),
        sa.Column("season", sa.String(10)),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("external_id", sa.String(20), unique=True, nullable=False),
        sa.Column("name", sa.String(100)),
        sa.Column("short_name", sa.String(10)),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id")),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("external_id", sa.String(20), unique=True, nullable=False),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id")),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("matchday", sa.Integer()),
        sa.Column("match_date", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(20)),
        sa.Column("home_goals", sa.Integer()),
        sa.Column("away_goals", sa.Integer()),
        sa.Column("home_xg", sa.Float()),
        sa.Column("away_xg", sa.Float()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "elo_ratings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), unique=True),
        sa.Column("rating", sa.Float(), default=1500.0),
        sa.Column("last_match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "elo_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id")),
        sa.Column("rating_before", sa.Float()),
        sa.Column("rating_after", sa.Float()),
        sa.UniqueConstraint("team_id", "match_id"),
    )

    op.create_table(
        "match_features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), unique=True),
        sa.Column("home_form", sa.Float()),
        sa.Column("away_form", sa.Float()),
        sa.Column("home_attack_strength", sa.Float()),
        sa.Column("home_defense_strength", sa.Float()),
        sa.Column("away_attack_strength", sa.Float()),
        sa.Column("away_defense_strength", sa.Float()),
        sa.Column("elo_diff", sa.Float()),
        sa.Column("home_xg_avg", sa.Float()),
        sa.Column("away_xg_avg", sa.Float()),
        sa.Column("home_xg_conceded_avg", sa.Float()),
        sa.Column("away_xg_conceded_avg", sa.Float()),
        sa.Column("xg_diff", sa.Float()),
        sa.Column("home_advantage", sa.Float()),
        sa.Column("h2h_home_wins", sa.Integer(), default=0),
        sa.Column("h2h_draws", sa.Integer(), default=0),
        sa.Column("h2h_away_wins", sa.Integer(), default=0),
        # Columns from migration 001
        sa.Column("odds_home", sa.Float()),
        sa.Column("odds_draw", sa.Float()),
        sa.Column("odds_away", sa.Float()),
        # Columns from migration 002
        sa.Column("h2h_home_win_rate", sa.Float()),
        sa.Column("home_days_rest", sa.Integer()),
        sa.Column("away_days_rest", sa.Integer()),
        sa.Column("home_fixture_congestion", sa.Integer()),
        sa.Column("away_fixture_congestion", sa.Integer()),
        sa.Column("home_injuries_key", sa.Integer(), default=0),
        sa.Column("away_injuries_key", sa.Integer(), default=0),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id")),
        sa.Column("model_version", sa.String(20)),
        sa.Column("poisson_home", sa.Float()),
        sa.Column("poisson_draw", sa.Float()),
        sa.Column("poisson_away", sa.Float()),
        sa.Column("xgb_home", sa.Float()),
        sa.Column("xgb_draw", sa.Float()),
        sa.Column("xgb_away", sa.Float()),
        sa.Column("ensemble_home", sa.Float()),
        sa.Column("ensemble_draw", sa.Float()),
        sa.Column("ensemble_away", sa.Float()),
        sa.Column("claude_reasoning", sa.Text()),
        sa.Column("claude_confidence_adj", sa.Float(), default=0.0),
        sa.Column("prob_home", sa.Float()),
        sa.Column("prob_draw", sa.Float()),
        sa.Column("prob_away", sa.Float()),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "picks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), sa.ForeignKey("predictions.id")),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id")),
        sa.Column("pick_type", sa.String(20)),
        sa.Column("pick_value", sa.String(20)),
        sa.Column("confidence", sa.Float()),
        # Columns from migration 002
        sa.Column("edge", sa.Float()),
        sa.Column("odds_decimal", sa.Float()),
        sa.Column("reasoning", sa.Text()),
        sa.Column("outcome", sa.String(10)),
        sa.Column("matchday_label", sa.String(20)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_performance",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("period_start", sa.Date()),
        sa.Column("period_end", sa.Date()),
        sa.Column("total_picks", sa.Integer()),
        sa.Column("correct_picks", sa.Integer()),
        sa.Column("accuracy", sa.Float()),
        sa.Column("brier_score", sa.Float()),
        sa.Column("roi", sa.Float()),
        sa.Column("model_version", sa.String(20)),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("model_performance")
    op.drop_table("picks")
    op.drop_table("predictions")
    op.drop_table("match_features")
    op.drop_table("elo_history")
    op.drop_table("elo_ratings")
    op.drop_table("matches")
    op.drop_table("teams")
    op.drop_table("competitions")
