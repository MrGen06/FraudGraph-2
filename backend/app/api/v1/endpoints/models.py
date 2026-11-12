"""
FastAPI Endpoints for Production XGBoost Risk Inference, Neo4j Graph Evidence, & Benchmarking
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.model_service import FraudModelService

router = APIRouter(prefix="/models", tags=["Production Fraud Risk Scoring"])


class TransactionInferenceRequest(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(default=120.0, gt=0, description="Transaction monetary amount")
    card_id: str = Field(default="13926", description="Card identification number")
    card_type: str = Field(default="credit", description="Card type: credit or debit")
    card_network: str = Field(default="visa", description="Card network: visa, mastercard, discover, etc.")
    device_id: str = Field(default="Windows 10/Chrome", description="Device or browser fingerprint")
    device_type: str = Field(default="desktop", description="Device type: desktop or mobile")
    email_domain: str = Field(default="gmail.com", description="Purchaser email domain")
    hour: int = Field(default=14, ge=0, le=23, description="Hour of transaction execution (0-23)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    dist1: float = Field(default=15.0, ge=0, description="Billing to shipping distance")
    product_cd: str = Field(default="W", description="Product category code")
    custom_features: Optional[Dict[str, Any]] = None


SAMPLE_PRESETS = [
    {
        "id": "legit_standard",
        "name": "Standard Legitimate Purchase",
        "description": "Routine daytime grocery charge on established card and personal laptop.",
        "payload": {
            "transaction_id": "TX_BENIGN_001",
            "amount": 48.50,
            "card_id": "13926",
            "card_type": "debit",
            "card_network": "visa",
            "device_id": "Windows 10/Chrome",
            "device_type": "desktop",
            "email_domain": "gmail.com",
            "hour": 14,
            "day_of_week": 2,
            "dist1": 4.5,
            "product_cd": "W",
        },
    },
    {
        "id": "graph_compromised_ring",
        "name": "Card-Sharing Mule Ring (Graph Evidence)",
        "description": "Moderate amount, but card ID #13926 links to multiple entities in graph exploration.",
        "payload": {
            "transaction_id": "TX_RING_002",
            "amount": 195.00,
            "card_id": "13926",
            "card_type": "credit",
            "card_network": "discover",
            "device_id": "SAMSUNG SM-G892A",
            "device_type": "mobile",
            "email_domain": "hotmail.com",
            "hour": 19,
            "day_of_week": 5,
            "dist1": 12.0,
            "product_cd": "C",
        },
    },
    {
        "id": "tabular_amount_anomaly",
        "name": "Extreme Amount & Nocturnal Anomaly (High Tabular Risk)",
        "description": "Massive $4,850.00 charge at 03:30 AM with distant delivery and unverified device.",
        "payload": {
            "transaction_id": "TX_ANOM_003",
            "amount": 4850.00,
            "card_id": "9500",
            "card_type": "credit",
            "card_network": "visa",
            "device_id": "Unknown/TorBrowser",
            "device_type": "desktop",
            "email_domain": "protonmail.com",
            "hour": 3,
            "day_of_week": 6,
            "dist1": 450.0,
            "product_cd": "H",
        },
    },
    {
        "id": "critical_fraud_attack",
        "name": "Critical Fraud Attack (High Risk & Compromised Entities)",
        "description": "High amount ($1,420), abnormal midnight hour, and specific mobile fingerprint linked in network.",
        "payload": {
            "transaction_id": "TX_CRITICAL_004",
            "amount": 1420.00,
            "card_id": "13926",
            "card_type": "credit",
            "card_network": "mastercard",
            "device_id": "SAMSUNG SM-G892A",
            "device_type": "mobile",
            "email_domain": "anonymous-mail.org",
            "hour": 2,
            "day_of_week": 0,
            "dist1": 210.0,
            "product_cd": "C",
        },
    },
]


@router.post("/predict", status_code=status.HTTP_200_OK)
def predict_transaction(payload: TransactionInferenceRequest):
    """
    Execute automated fraud risk scoring using production XGBoost model.
    Returns automated risk score, risk level, TreeSHAP attributions, and Neo4j relational graph evidence.
    """
    service = FraudModelService.get_instance()
    if not service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud models are not loaded or currently unavailable.",
        )
    try:
        result = service.predict_transaction(payload.model_dump())
        return result
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )


@router.get("/status", status_code=status.HTTP_200_OK)
def get_models_status():
    """Retrieve operational status, loaded artifact verification, and ensemble parameters."""
    service = FraudModelService.get_instance()
    return service.get_status()


@router.get("/benchmark", status_code=status.HTTP_200_OK)
def get_benchmark_results():
    """Retrieve saved evaluation metrics, ROC/PR curves data, and model comparison table."""
    service = FraudModelService.get_instance()
    return service.get_benchmark_data()


@router.get("/sample-transactions", status_code=status.HTTP_200_OK)
def get_sample_transactions():
    """Return pre-configured test scenarios for live interactive testing."""
    return SAMPLE_PRESETS
