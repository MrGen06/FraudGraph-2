import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.alert import Alert
from app.models.transaction import Transaction
from app.schemas.transaction import TransactionCreate
from app.services.graph_service import create_transaction_graph
from app.services.model_service import FraudModelService

logger = logging.getLogger(__name__)


def create_transaction(
    db: Session,
    payload: TransactionCreate,
) -> Transaction:
    """
    Ingest a financial transaction:
    1. Check for duplicates (returns 409 Conflict).
    2. Execute ML dual-model inference & risk scoring.
    3. Persist transaction and risk evaluation into PostgreSQL.
    4. Resiliently update Neo4j entity graph.
    5. Generate alert if risk threshold exceeded.
    """
    # 1. Check duplicate
    existing = db.execute(
        select(Transaction).where(Transaction.transaction_id == payload.transaction_id)
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transaction with ID '{payload.transaction_id}' already exists.",
        )

    # 2. Score transaction via Model Service (XGBoost sole automated predictor)
    model_service = FraudModelService.get_instance()
    try:
        score_res = model_service.predict_transaction({
            "transaction_id": payload.transaction_id,
            "amount": float(payload.amount),
            "card_id": payload.card_id or "13926",
            "device_id": payload.device_id or "Desktop",
            "hour": payload.timestamp.hour if hasattr(payload.timestamp, "hour") else 14,
            "day_of_week": payload.timestamp.weekday() if hasattr(payload.timestamp, "weekday") else 2,
        })
    except RuntimeError as re:
        logger.error(f"Scoring service unavailable for transaction {payload.transaction_id}: {re}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transaction scoring service unavailable (fail-closed policy enforced): {str(re)}",
        )
    except Exception as e:
        logger.error(f"Inference error scoring transaction {payload.transaction_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error scoring transaction {payload.transaction_id}: {str(e)}",
        )

    risk_score = float(score_res.get("risk_score", score_res.get("tabular_score", 0.0)))
    risk_level = str(score_res.get("risk_level", score_res.get("consensus_verdict", "LOW")))
    consensus_verdict = risk_level
    consensus_reasoning = str(score_res.get("decision_reasoning", score_res.get("consensus_reasoning", "")))
    is_flagged = bool(risk_level in ["MEDIUM", "HIGH", "CRITICAL"] or risk_score >= 0.30)

    location = payload.location
    country = payload.country or (location.country if location else None)
    region = payload.region or (location.region if location else None)

    # 3. Persist transaction in PostgreSQL
    transaction = Transaction(
        transaction_id=payload.transaction_id,
        timestamp=payload.timestamp,
        source_account_id=payload.source_account_id,
        destination_account_id=payload.destination_account_id,
        user_id=payload.user_id,
        card_id=payload.card_id,
        device_id=payload.device_id,
        ip_id=payload.ip_id,
        merchant_id=payload.merchant_id,
        amount=payload.amount,
        currency=payload.currency,
        country=country,
        region=region,
        risk_score=risk_score,
        risk_level=risk_level,
        is_flagged=is_flagged,
        consensus_verdict=consensus_verdict,
        consensus_reasoning=consensus_reasoning,
    )

    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    # 4. Asynchronously / Resiliently update Neo4j
    create_transaction_graph(payload)

    # 5. Create alert if flagged
    if is_flagged:
        alert_id = f"ALT-{payload.transaction_id}"
        alert = Alert(
            alert_id=alert_id,
            transaction_id=payload.transaction_id,
            risk_score=risk_score,
            risk_level=risk_level,
            status="PENDING",
            reasons=consensus_reasoning,
        )
        db.add(alert)
        try:
            db.commit()
            logger.info(f"Generated fraud alert {alert_id} for transaction {payload.transaction_id}")
        except Exception as e:
            db.rollback()
            logger.warning(f"Could not persist alert {alert_id}: {e}")

    return transaction


def get_transactions(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    risk_level: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Transaction]:
    """Retrieve paginated transactions with optional risk tier and text filtering."""
    query = select(Transaction).order_by(Transaction.id.desc())

    if risk_level and isinstance(risk_level, str):
        query = query.where(Transaction.risk_level == risk_level.upper())

    if search and isinstance(search, str):
        s = f"%{search}%"
        query = query.where(
            or_(
                Transaction.transaction_id.ilike(s),
                Transaction.source_account_id.ilike(s),
                Transaction.user_id.ilike(s),
                Transaction.card_id.ilike(s),
            )
        )

    query = query.offset(skip).limit(min(limit, 200))
    return list(db.execute(query).scalars().all())


def get_transaction_by_id(db: Session, identifier: str) -> Optional[Transaction]:
    """Find transaction by numerical ID or transaction_id string."""
    if identifier.isdigit():
        tx = db.execute(select(Transaction).where(Transaction.id == int(identifier))).scalar_one_or_none()
        if tx:
            return tx
    return db.execute(select(Transaction).where(Transaction.transaction_id == identifier)).scalar_one_or_none()
