"""Add venue-specific form columns to match_features.

Revision ID: 004_venue_form
"""

from alembic import op
import sqlalchemy as sa

revision = "004_venue_form"
down_revision = "003_job_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("match_features", sa.Column("home_form_home", sa.Float, nullable=True))
    op.add_column("match_features", sa.Column("away_form_away", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("match_features", "away_form_away")
    op.drop_column("match_features", "home_form_home")
