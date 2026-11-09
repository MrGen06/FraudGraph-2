from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class Transaction(Base):
    __tablename__ = "transactions"

    __table_args__ = (
        UniqueConstraint(
            "transaction_id",
            name="uq_transactions_transaction_id",
        ),
        Index("ix_transactions_timestamp", "timestamp"),
        Index("ix_transactions_source_account", "source_account_id"),
        Index("ix_transactions_destination_account", "destination_account_id"),
        Index("ix_transactions_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    source_account_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    destination_account_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    card_id: Mapped[str | None] = mapped_column(String(100))

    device_id: Mapped[str | None] = mapped_column(String(100))

    ip_id: Mapped[str | None] = mapped_column(String(100))

    merchant_id: Mapped[str | None] = mapped_column(String(100))

    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    country: Mapped[str | None] = mapped_column(String(2))

    region: Mapped[str | None] = mapped_column(String(100))