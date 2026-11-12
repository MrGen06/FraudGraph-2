"""
Unit tests for Dual-Model ML Inference and Evaluation Endpoints
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api.v1.endpoints.models import (
    TransactionInferenceRequest,
    get_benchmark_results,
    get_models_status,
    get_sample_transactions,
    predict_transaction,
)


def test_models_status_endpoint():
    """Verify models health and artifact tracking endpoint."""
    status_info = get_models_status()
    assert "status" in status_info
    assert "models" in status_info
    assert "xgboost" in status_info["models"]
    assert "gnn" in status_info["models"]
    assert "ensemble_weights" in status_info
    assert status_info["models"]["xgboost"]["loaded"] is True


def test_models_benchmark_endpoint():
    """Verify benchmark metrics and comparison table endpoint."""
    data = get_benchmark_results()
    assert "metrics" in data
    assert "curves" in data
    assert "comparison_table" in data
    assert len(data["comparison_table"]) >= 2


def test_models_sample_transactions_endpoint():
    """Verify curated sample preset scenarios."""
    presets = get_sample_transactions()
    assert len(presets) >= 4
    for p in presets:
        assert "id" in p
        assert "name" in p
        assert "payload" in p
        assert "amount" in p["payload"]


def test_models_predict_endpoint_legitimate():
    """Verify live inference on legitimate transaction."""
    payload = TransactionInferenceRequest(
        transaction_id="TX_TEST_001",
        amount=35.0,
        card_id="13926",
        card_type="debit",
        card_network="visa",
        device_id="Windows 10/Chrome",
        device_type="desktop",
        email_domain="gmail.com",
        hour=14,
        day_of_week=2,
        dist1=5.0,
        product_cd="W",
    )
    res = predict_transaction(payload)
    assert "tabular_score" in res
    assert "graph_score" in res
    assert "hybrid_score" in res
    assert res["consensus_verdict"] in ["LOW", "MEDIUM", "CRITICAL"]
    assert len(res["consensus_reasoning"]) > 10
    assert "shap_attributions" in res
    assert "graph_explanation" in res
    assert len(res["graph_explanation"]["nodes"]) >= 4
