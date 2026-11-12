import pytest
from app.api.v1.endpoints.graph import get_graph_by_transaction_id
from app.api.v1.endpoints.transactions import (
    get_transaction_detail,
    ingest_transaction,
    list_transactions,
)
from app.schemas.transaction import TransactionCreate
from fastapi import HTTPException


def test_transaction_ingestion_and_scoring(db_session):
    """Verify end-to-end ingestion, risk scoring, and persistence."""
    payload = TransactionCreate(
        transaction_id="API-TEST-INGEST-001",
        source_account_id="ACC-SRC-101",
        destination_account_id="ACC-DST-202",
        user_id="USER-999",
        amount=1500.00,
        currency="USD",
        card_id="13926",
        device_id="Chrome_Win11",
        country="US",
        region="California",
    )

    data = ingest_transaction(payload, db=db_session)
    assert data.transaction_id == "API-TEST-INGEST-001"
    assert data.risk_score is not None
    assert data.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data.consensus_verdict in ["LOW", "MEDIUM", "CRITICAL"]


def test_duplicate_transaction_rejected(db_session):
    """Verify duplicate transaction ID raises 409 Conflict."""
    payload = TransactionCreate(
        transaction_id="API-TEST-DUP-001",
        source_account_id="ACC-01",
        destination_account_id="ACC-02",
        user_id="USER-01",
        amount=250.00,
        currency="USD",
    )

    tx1 = ingest_transaction(payload, db=db_session)
    assert tx1.transaction_id == "API-TEST-DUP-001"

    with pytest.raises(HTTPException) as excinfo:
        ingest_transaction(payload, db=db_session)
    assert excinfo.value.status_code == 409
    assert "already exists" in excinfo.value.detail


def test_transaction_listing_and_filtering(db_session):
    """Verify paginated listing and searching."""
    for i in range(3):
        ingest_transaction(
            TransactionCreate(
                transaction_id=f"API-LIST-{i}",
                source_account_id=f"ACC-{i}",
                destination_account_id="ACC-DST",
                user_id=f"USER-{i}",
                amount=100.0 + i * 50,
                currency="USD",
            ),
            db=db_session,
        )

    items = list_transactions(skip=0, limit=10, db=db_session)
    assert len(items) >= 3


def test_transaction_detail(db_session):
    """Verify retrieving transaction detail by ID."""
    tx_id = "API-DETAIL-001"
    ingest_transaction(
        TransactionCreate(
            transaction_id=tx_id,
            source_account_id="ACC-10",
            destination_account_id="ACC-20",
            user_id="USER-10",
            amount=88.0,
            currency="USD",
        ),
        db=db_session,
    )

    tx = get_transaction_detail(tx_id, db=db_session)
    assert tx.transaction_id == tx_id


def test_graph_subgraph_endpoint(db_session):
    """Verify graph endpoint retrieves nodes and links for transaction."""
    tx_id = "API-GRAPH-001"
    ingest_transaction(
        TransactionCreate(
            transaction_id=tx_id,
            source_account_id="ACC-GRAPH",
            destination_account_id="ACC-DST",
            user_id="USER-GRAPH",
            card_id="13926_150_2_visa_111_credit",
            device_id="Mobile_Android",
            amount=340.0,
            currency="USD",
        ),
        db=db_session,
    )

    graph = get_graph_by_transaction_id(tx_id, db=db_session)
    assert "nodes" in graph
    assert "links" in graph
    assert len(graph["nodes"]) >= 2
