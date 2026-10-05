"""add authentication and session ownership

Revision ID: fcbe3faf6ae9
Revises: ae2a099ee1d2
Create Date: 2026-10-04 22:23:40.276551
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "fcbe3faf6ae9"
down_revision: Union[str, Sequence[str], None] = "ae2a099ee1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename instead of dropping existing password hashes.
    op.alter_column(
        "users", "password_hash",
        new_column_name="hashed_password",
        existing_type=sa.VARCHAR(length=255),
        type_=sa.String(length=1024),
        existing_nullable=False,
    )
    # Database defaults also handle previously registered users.
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("users", sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("users", sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.drop_constraint("users_email_key", "users", type_="unique")
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # Requires tutor_sessions to be empty: pre-existing sessions have no known owner.
    op.add_column("tutor_sessions", sa.Column("user_id", sa.Integer(), nullable=False))
    op.create_index("ix_tutor_sessions_user_id", "tutor_sessions", ["user_id"], unique=False)
    op.create_foreign_key(
        "fk_tutor_sessions_user_id_users", "tutor_sessions", "users", ["user_id"], ["id"]
    )


def downgrade() -> None:
    op.drop_constraint("fk_tutor_sessions_user_id_users", "tutor_sessions", type_="foreignkey")
    op.drop_index("ix_tutor_sessions_user_id", table_name="tutor_sessions")
    op.drop_column("tutor_sessions", "user_id")
    op.drop_index("ix_users_email", table_name="users")
    op.create_unique_constraint("users_email_key", "users", ["email"])
    op.drop_column("users", "is_verified")
    op.drop_column("users", "is_superuser")
    op.drop_column("users", "is_active")
    op.alter_column(
        "users", "hashed_password",
        new_column_name="password_hash",
        existing_type=sa.String(length=1024),
        type_=sa.VARCHAR(length=255),
        existing_nullable=False,
    )
