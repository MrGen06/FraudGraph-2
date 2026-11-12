from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas.transaction import TransactionCreate, TransactionResponse
from app.services.transaction_service import (
    create_transaction,
    get_transaction_by_id,
    get_transactions,
)

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post(
    "/",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_transaction(
    payload: TransactionCreate,
    db: Session = Depends(get_db),
):
    """
    Ingest transaction into FraudGraph:
    Runs validation, checks duplicates, calculates risk score,
    persists in PostgreSQL, and synchronizes to Neo4j.
    """
    return create_transaction(db, payload)


@router.get(
    "/",
    response_model=List[TransactionResponse],
    status_code=status.HTTP_200_OK,
)
def list_transactions(
    skip: int = 0,
    limit: int = 50,
    risk_level: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List paginated transactions with optional risk tier and text filtering."""
    return get_transactions(db, skip=skip, limit=limit, risk_level=risk_level, search=search)


@router.get(
    "/{identifier}",
    response_model=TransactionResponse,
    status_code=status.HTTP_200_OK,
)
def get_transaction_detail(
    identifier: str,
    db: Session = Depends(get_db),
):
    """Retrieve detailed transaction record by ID or TransactionID."""
    tx = get_transaction_by_id(db, identifier)
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{identifier}' not found.",
        )
    return tx
