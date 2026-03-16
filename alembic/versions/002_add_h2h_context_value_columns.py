"""Add H2H, match context, and value betting columns.

Revision ID: 002_h2h_context_value
"""

from alembic import op
import sqlalchemy as sa

revision = "002_h2h_context_value"
down_revision = "001_add_odds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Skip if columns already exist (created by 000_initial on fresh DBs)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='match_features' AND column_name='h2h_home_win_rate'"
    ))
    if result.fetchone():
        return

    # Match context features on match_features
    op.add_column("match_features", sa.Column("h2h_home_win_rate", sa.Float(), nullable=True))
    op.add_column("match_features", sa.Column("home_days_rest", sa.Integer(), nullable=True))
    op.add_column("match_features", sa.Column("away_days_rest", sa.Integer(), nullable=True))
    op.add_column("match_features", sa.Column("home_fixture_congestion", sa.Integer(), nullable=True))
    op.add_column("match_features", sa.Column("away_fixture_congestion", sa.Integer(), nullable=True))

    # Value betting columns on picks
    op.add_column("picks", sa.Column("edge", sa.Float(), nullable=True))
    op.add_column("picks", sa.Column("odds_decimal", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("picks", "odds_decimal")
    op.drop_column("picks", "edge")
    op.drop_column("match_features", "away_fixture_congestion")
    op.drop_column("match_features", "home_fixture_congestion")
    op.drop_column("match_features", "away_days_rest")
    op.drop_column("match_features", "home_days_rest")
    op.drop_column("match_features", "h2h_home_win_rate")
