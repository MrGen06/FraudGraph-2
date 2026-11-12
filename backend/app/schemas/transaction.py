from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionLocation(BaseModel):
    country: str = Field(default="US", min_length=2, max_length=2)
    region: Optional[str] = None


class TransactionCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
    )

    transaction_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    source_account_id: str = Field(default="ACC-DEFAULT-01")
    destination_account_id: str = Field(default="ACC-DEFAULT-02")

    user_id: str = Field(default="USER-DEFAULT-01")
    card_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_id: Optional[str] = None
    merchant_id: Optional[str] = None

    amount: Decimal = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)

    location: Optional[TransactionLocation] = None
    country: Optional[str] = None
    region: Optional[str] = None


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: int
    transaction_id: str
    timestamp: datetime

    source_account_id: str
    destination_account_id: str

    user_id: str
    card_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_id: Optional[str] = None
    merchant_id: Optional[str] = None

    amount: Decimal
    currency: str

    country: Optional[str] = None
    region: Optional[str] = None

    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    is_flagged: bool = False
    consensus_verdict: Optional[str] = None
    consensus_reasoning: Optional[str] = None
