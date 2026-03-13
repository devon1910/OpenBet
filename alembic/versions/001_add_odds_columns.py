"""Add bookmaker odds columns to match_features.

Revision ID: 001_add_odds
"""

from alembic import op
import sqlalchemy as sa

revision = "001_add_odds"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("match_features", sa.Column("odds_home", sa.Float(), nullable=True))
    op.add_column("match_features", sa.Column("odds_draw", sa.Float(), nullable=True))
    op.add_column("match_features", sa.Column("odds_away", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("match_features", "odds_away")
    op.drop_column("match_features", "odds_draw")
    op.drop_column("match_features", "odds_home")
