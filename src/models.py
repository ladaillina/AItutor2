from sqlalchemy import String, Uuid, ForeignKey, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from uuid import UUID, uuid4
from fastapi_users.db import SQLAlchemyBaseUserTable


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)


class TutorSession(Base):
    __tablename__ = "tutor_sessions"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )


class TutorMessage(Base):
    __tablename__ = "tutor_messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    session_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tutor_sessions.id", ondelete="CASCADE"),
        index=True,
    )

    role: Mapped[str] = mapped_column(String(16))

    content: Mapped[str] = mapped_column(Text)