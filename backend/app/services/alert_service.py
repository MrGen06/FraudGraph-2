import logging
from datetime import datetime
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.alert import Alert

logger = logging.getLogger(__name__)


def get_alerts(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> List[Alert]:
    """Retrieve paginated fraud alerts with optional filtering."""
    query = select(Alert).order_by(Alert.id.desc())

    if status_filter and isinstance(status_filter, str):
        query = query.where(Alert.status == status_filter.upper())

    if risk_level and isinstance(risk_level, str):
        query = query.where(Alert.risk_level == risk_level.upper())

    query = query.offset(skip).limit(min(limit, 200))
    return list(db.execute(query).scalars().all())


def get_alert_by_id(db: Session, alert_id: str) -> Optional[Alert]:
    """Find alert by numerical ID or alert_id string."""
    if alert_id.isdigit():
        alt = db.execute(select(Alert).where(Alert.id == int(alert_id))).scalar_one_or_none()
        if alt:
            return alt
    return db.execute(select(Alert).where(Alert.alert_id == alert_id)).scalar_one_or_none()


def review_alert(
    db: Session,
    alert_id: str,
    new_status: str,
    reviewer_notes: Optional[str] = None,
    reviewed_by: Optional[str] = "analyst",
) -> Alert:
    """Update review verdict on a fraud alert."""
    alert = get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )

    alert.status = new_status
    alert.reviewer_notes = reviewer_notes
    alert.reviewed_by = reviewed_by
    alert.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(alert)
    logger.info(f"Alert {alert.alert_id} reviewed with status: {new_status}")
    return alert
