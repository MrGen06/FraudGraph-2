from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas.alert import AlertResponse, AlertReviewRequest
from app.services.alert_service import (
    get_alert_by_id,
    get_alerts,
    review_alert,
)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


def _format_alert(alt) -> dict:
    """Format ORM Alert into AlertResponse dict."""
    reasons_list = [r.strip() for r in alt.reasons.split(";") if r.strip()] if alt.reasons else []
    return {
        "id": alt.id,
        "alert_id": alt.alert_id,
        "transaction_id": alt.transaction_id,
        "risk_score": float(alt.risk_score),
        "risk_level": alt.risk_level,
        "status": alt.status,
        "reasons": reasons_list or [alt.reasons] if alt.reasons else [],
        "reviewer_notes": alt.reviewer_notes,
        "reviewed_by": alt.reviewed_by,
        "reviewed_at": alt.reviewed_at,
        "created_at": alt.created_at,
    }


@router.get(
    "/",
    response_model=List[AlertResponse],
    status_code=status.HTTP_200_OK,
)
def list_alerts(
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[str] = None,
    risk_level: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List fraud alerts with optional status and risk tier filtering."""
    alerts = get_alerts(db, skip=skip, limit=limit, status_filter=status_filter, risk_level=risk_level)
    return [_format_alert(a) for a in alerts]


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
)
def get_alert_detail(
    alert_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve detailed alert record by ID or alert_id."""
    alt = get_alert_by_id(db, alert_id)
    if not alt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return _format_alert(alt)


@router.post(
    "/{alert_id}/review",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
)
def update_alert_review(
    alert_id: str,
    payload: AlertReviewRequest,
    db: Session = Depends(get_db),
):
    """Review and update the disposition of a fraud alert."""
    alt = review_alert(
        db,
        alert_id=alert_id,
        new_status=payload.status,
        reviewer_notes=payload.reviewer_notes,
        reviewed_by=payload.reviewed_by,
    )
    return _format_alert(alt)
