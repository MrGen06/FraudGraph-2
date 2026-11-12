from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AlertResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="ignore",
    )

    id: Optional[int] = None
    alert_id: str = Field(min_length=1)
    transaction_id: str = Field(min_length=1)

    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(default="MEDIUM")

    status: str = Field(default="PENDING")
    reasons: list[str] = Field(default_factory=list)

    reviewer_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlertReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    status: Literal["RESOLVED_FRAUD", "DISMISSED_LEGITIMATE", "INVESTIGATING"]
    reviewer_notes: Optional[str] = None
    reviewed_by: Optional[str] = "analyst"
