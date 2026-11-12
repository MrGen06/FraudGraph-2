import pytest
from app.api.v1.endpoints.alerts import (
    get_alert_detail,
    list_alerts,
    update_alert_review,
)
from app.models.alert import Alert
from app.schemas.alert import AlertReviewRequest


def test_alerts_lifecycle(db_session):
    """Verify alert listing, detail retrieval, and review workflow."""
    alert = Alert(
        alert_id="ALT-UNIT-001",
        transaction_id="TX-UNIT-001",
        risk_score=0.88,
        risk_level="CRITICAL",
        status="PENDING",
        reasons="Elevated tabular fraud probability; new device signature.",
    )
    db_session.add(alert)
    db_session.commit()

    # 1. List alerts
    alerts = list_alerts(db=db_session)
    assert len(alerts) >= 1
    target = next((a for a in alerts if a["alert_id"] == "ALT-UNIT-001"), None)
    assert target is not None
    assert target["risk_level"] == "CRITICAL"

    # 2. Get alert detail
    detail = get_alert_detail("ALT-UNIT-001", db=db_session)
    assert detail["alert_id"] == "ALT-UNIT-001"

    # 3. Review alert
    review_payload = AlertReviewRequest(
        status="RESOLVED_FRAUD",
        reviewer_notes="Confirmed stolen card pattern with bank issuer.",
        reviewed_by="lead_investigator",
    )
    updated = update_alert_review("ALT-UNIT-001", review_payload, db=db_session)
    assert updated["status"] == "RESOLVED_FRAUD"
    assert updated["reviewed_by"] == "lead_investigator"
    assert "stolen card" in updated["reviewer_notes"]
