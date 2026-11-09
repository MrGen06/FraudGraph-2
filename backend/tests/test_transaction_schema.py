import pytest
from pydantic import ValidationError

from app.schemas.transaction import TransactionCreate


def valid_payload():
    return {
        "transaction_id": "TEST-001",
        "timestamp": "2026-11-09T12:00:00Z",
        "source_account_id": "ACC-001",
        "destination_account_id": "ACC-002",
        "user_id": "USER-001",
        "card_id": "CARD-001",
        "device_id": "DEVICE-001",
        "ip_id": "IP-001",
        "merchant_id": "MERCHANT-001",
        "amount": 2500.00,
        "currency": "INR",
        "location": {
            "country": "IN",
            "region": "Haryana"
        }
    }


def test_valid_transaction():
    transaction = TransactionCreate(**valid_payload())

    assert transaction.transaction_id == "TEST-001"
    assert transaction.amount == 2500.00
    assert transaction.currency == "INR"


def test_negative_amount_rejected():
    payload = valid_payload()
    payload["amount"] = -100

    with pytest.raises(ValidationError):
        TransactionCreate(**payload)


def test_missing_transaction_id_rejected():
    payload = valid_payload()
    del payload["transaction_id"]

    with pytest.raises(ValidationError):
        TransactionCreate(**payload)


def test_invalid_country_code_rejected():
    payload = valid_payload()
    payload["location"]["country"] = "IND"

    with pytest.raises(ValidationError):
        TransactionCreate(**payload)