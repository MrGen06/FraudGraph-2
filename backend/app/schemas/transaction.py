from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TransactionLocation(BaseModel):
    country: str = Field(min_length=2, max_length=2)
    region: str | None = None


class TransactionCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    transaction_id: str = Field(min_length=1)
    timestamp: datetime

    source_account_id: str
    destination_account_id: str

    user_id: str
    card_id: str | None = None
    device_id: str | None = None
    ip_id: str | None = None
    merchant_id: str | None = None

    amount: Decimal = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)

    location: TransactionLocation | None = None


class TransactionResponse(TransactionCreate):
    id: int