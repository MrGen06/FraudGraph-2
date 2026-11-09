from sqlalchemy.orm import Session

from app.models.transaction import Transaction
from app.schemas.transaction import TransactionCreate

from app.services.graph_service import create_transaction_graph


def create_transaction(
    db: Session,
    payload: TransactionCreate,
) -> Transaction:
    location = payload.location

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
        country=location.country if location else None,
        region=location.region if location else None,
    )

    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    create_transaction_graph(payload)

    return transaction