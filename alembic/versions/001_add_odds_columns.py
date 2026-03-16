"""Add bookmaker odds columns to match_features.

Revision ID: 001_add_odds
"""

from alembic import op
import sqlalchemy as sa

revision = "001_add_odds"
down_revision = "000_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Skip if columns already exist (created by 000_initial on fresh DBs)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='match_features' AND column_name='odds_home'"
    ))
    if result.fetchone():
        return
    op.add_column("match_features", sa.Column("odds_home", sa.Float(), nullable=True))
    op.add_column("match_features", sa.Column("odds_draw", sa.Float(), nullable=True))
    op.add_column("match_features", sa.Column("odds_away", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("match_features", "odds_away")
    op.drop_column("match_features", "odds_draw")
    op.drop_column("match_features", "odds_home")
