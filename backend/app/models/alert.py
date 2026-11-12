from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class Alert(Base):
    __tablename__ = "alerts"

    __table_args__ = (
        UniqueConstraint(
            "alert_id",
            name="uq_alerts_alert_id",
        ),
        Index("ix_alerts_transaction_id", "transaction_id"),
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_risk_level", "risk_level"),
        Index("ix_alerts_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    alert_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    transaction_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
    )

    reasons: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    reviewer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reviewed_by: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
