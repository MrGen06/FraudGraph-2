from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas.transaction import TransactionCreate, TransactionResponse
from app.services.transaction_service import create_transaction

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
    return create_transaction(db, payload)