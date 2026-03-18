"""Add job_status table for persistent background job tracking.

Revision ID: 003_job_status
"""

from alembic import op
import sqlalchemy as sa

revision = "003_job_status"
down_revision = "002_h2h_context_value"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_status",
        sa.Column("action", sa.String(50), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="idle"),
        sa.Column("message", sa.Text, server_default=""),
        sa.Column("extra_json", sa.Text, server_default=""),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("job_status")
