from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints.transactions import get_db
from app.services.transaction_service import create_transaction

client = TestClient(app)


def fake_create_transaction(db, payload):
    return {
        "id": 1,
        **payload.model_dump(),
    }


def test_transaction_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.endpoints.transactions.create_transaction",
        fake_create_transaction,
    )

    response = client.post(
        "/api/v1/transactions/",
        json={
            "transaction_id": "API-TEST-001",
            "timestamp": "2026-11-09T12:00:00Z",
            "source_account_id": "ACC-001",
            "destination_account_id": "ACC-002",
            "user_id": "USER-001",
            "amount": 2500,
            "currency": "INR",
            "location": {
                "country": "IN",
                "region": "Haryana",
            },
        },
    )

    assert response.status_code == 201
    assert response.json()["transaction_id"] == "API-TEST-001"


def test_invalid_transaction_rejected():
    response = client.post(
        "/api/v1/transactions/",
        json={
            "transaction_id": "API-TEST-002",
            "amount": -100,
        },
    )

    assert response.status_code == 422