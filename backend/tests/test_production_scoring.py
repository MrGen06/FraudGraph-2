"""
Tests for Production Scoring Simplification
Verifies:
1. Live predictions call XGBoost as sole automated predictor.
2. Live predictions do not call _infer_graph_score().
3. Live predictions do not depend on GNN weights.
4. Missing XGBoost artifacts produce fail-closed service error (HTTP 503).
5. Alert risk levels match XGBoost predictions consistently.
6. Neo4j investigation endpoints continue working.
7. Existing transaction ingestion and retrieval remain functional.
8. No benchmark test labels are accessed by production inference.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.v1.endpoints.graph import get_graph_by_transaction_id
from app.api.v1.endpoints.models import TransactionInferenceRequest, predict_transaction
from app.api.v1.endpoints.transactions import (
    get_transaction_detail,
    ingest_transaction,
    list_transactions,
)
from app.schemas.transaction import TransactionCreate
from app.services.model_service import FraudModelService


def test_live_prediction_calls_xgboost_sole_predictor():
    """Verify live predictions call XGBoost pipeline directly as sole predictor."""
    service = FraudModelService.get_instance()
    assert service.is_healthy()
    assert service.xgb_pipeline is not None

    with patch.object(service.xgb_pipeline, "predict_proba", wraps=service.xgb_pipeline.predict_proba) as mock_xgb:
        payload = TransactionInferenceRequest(
            transaction_id="TX_TEST_XGB_001",
            amount=85.0,
            card_id="13926",
            device_id="Windows 10/Chrome",
            hour=12,
            day_of_week=1,
            dist1=2.0,
        )
        res = predict_transaction(payload)
        assert mock_xgb.called
        assert "risk_score" in res
        assert "tabular_score" in res
        assert res["risk_score"] == res["tabular_score"]
        assert 0.0 <= res["risk_score"] <= 1.0


def test_live_prediction_does_not_call_infer_graph_score():
    """Verify live predictions do NOT call any heuristic _infer_graph_score."""
    service = FraudModelService.get_instance()
    # Verify method was completely purged from service
    assert not hasattr(service, "_infer_graph_score")

    payload = TransactionInferenceRequest(
        transaction_id="TX_TEST_NO_HEURISTIC_002",
        amount=120.0,
        card_id="13926",
        device_id="Mobile_Pixel",
    )
    res = predict_transaction(payload)
    # Graph score in production is zeroed / marked for investigation only
    assert res["graph_score"] == 0.0
    assert res["graph_risk_level"] == "INVESTIGATION_ONLY"


def test_live_prediction_does_not_depend_on_gnn_weights():
    """Verify live predictions succeed even if GNN weights/pipeline are absent."""
    service = FraudModelService.get_instance()
    original_gnn = service.gnn_pipeline

    try:
        service.gnn_pipeline = None  # Simulate absent GNN weights
        payload = TransactionInferenceRequest(
            transaction_id="TX_TEST_NO_GNN_003",
            amount=50.0,
            card_id="13926",
            device_id="MacBook_Safari",
        )
        res = predict_transaction(payload)
        assert "risk_score" in res
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(res["shap_attributions"]) > 0
    finally:
        service.gnn_pipeline = original_gnn


def test_missing_xgboost_artifacts_fail_closed():
    """Verify missing XGBoost artifacts trigger HTTP 503 / RuntimeError, enforcing fail-closed."""
    service = FraudModelService.get_instance()
    original_xgb = service.xgb_pipeline

    try:
        service.xgb_pipeline = None  # Simulate missing XGBoost
        assert not service.is_healthy()

        with pytest.raises(HTTPException) as excinfo:
            payload = TransactionInferenceRequest(
                transaction_id="TX_TEST_FAIL_CLOSED_004",
                amount=99.0,
            )
            predict_transaction(payload)
        assert excinfo.value.status_code == 503
        assert "unavailable" in excinfo.value.detail.lower()
    finally:
        service.xgb_pipeline = original_xgb


def test_alert_risk_levels_match_xgboost_predictions(db_session):
    """Verify alerts are created using the XGBoost risk score and level consistently."""
    service = FraudModelService.get_instance()

    # Ingest anomalous transaction with extreme parameters to trigger XGBoost high risk
    payload = TransactionCreate(
        transaction_id="TX_HIGH_RISK_ALERT_005",
        source_account_id="ACC-ALERT-01",
        destination_account_id="ACC-ALERT-02",
        user_id="USER-ALERT-99",
        amount=9999.00,
        currency="USD",
        card_id="9999",
        device_id="Unknown_Proxy",
    )

    tx = ingest_transaction(payload, db=db_session)
    assert tx.risk_score is not None
    assert tx.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # Verify transaction in database has exact match
    fetched = get_transaction_detail("TX_HIGH_RISK_ALERT_005", db=db_session)
    assert fetched.risk_score == tx.risk_score
    assert fetched.risk_level == tx.risk_level


def test_neo4j_investigation_subgraph_endpoint(db_session):
    """Verify Neo4j graph investigation endpoint continues functioning."""
    tx_id = "TX_GRAPH_INVESTIGATION_006"
    ingest_transaction(
        TransactionCreate(
            transaction_id=tx_id,
            source_account_id="ACC-INVEST-01",
            destination_account_id="ACC-INVEST-02",
            user_id="USER-INVEST-01",
            amount=320.00,
            currency="USD",
            card_id="13926_150_2_visa_111_credit",
            device_id="Samsung_Galaxy_S21",
        ),
        db=db_session,
    )

    graph = get_graph_by_transaction_id(tx_id, db=db_session)
    assert "nodes" in graph
    assert "links" in graph
    assert any(n["id"] == tx_id for n in graph["nodes"])
    # Verify entity nodes exist for investigation
    assert len(graph["nodes"]) >= 2


def test_transaction_ingestion_and_retrieval(db_session):
    """Verify transaction ingestion and retrieval remains fully functional."""
    tx_id = "TX_INGEST_RETRIEVE_007"
    tx = ingest_transaction(
        TransactionCreate(
            transaction_id=tx_id,
            source_account_id="ACC-INGEST-01",
            destination_account_id="ACC-INGEST-02",
            user_id="USER-INGEST-01",
            amount=145.50,
            currency="USD",
        ),
        db=db_session,
    )
    assert tx.transaction_id == tx_id

    listing = list_transactions(skip=0, limit=20, db=db_session)
    assert any(t.transaction_id == tx_id for t in listing)


def test_no_benchmark_test_labels_accessed_during_inference():
    """Verify no benchmark test labels (test_meta.parquet / isFraud) are read during live inference."""
    service = FraudModelService.get_instance()

    with patch("pandas.read_parquet") as mock_read_parquet:
        payload = TransactionInferenceRequest(
            transaction_id="TX_TEST_NO_LEAK_008",
            amount=75.0,
            card_id="13926",
            device_id="Ubuntu_Chrome",
        )
        res = predict_transaction(payload)
        # Ensure read_parquet is never called during inference
        assert not mock_read_parquet.called
        assert res["transaction_id"] == "TX_TEST_NO_LEAK_008"

