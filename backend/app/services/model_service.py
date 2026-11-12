"""
Model Inference Service for FraudGraph Backend
Production Architecture:
- XGBoost: Sole automated fraud predictor and probability source.
- Neo4j: Graph-based investigation and relationship exploration.
- GNN: Offline experimental research only (weights and heuristics excluded from production inference).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["MPLCONFIGDIR"] = str(ROOT_DIR / ".matplotlib_cache")

from ml.gnn_model import GNNFraudPipeline
from ml.graph_dataset import HeteroGraphBuilder
from ml.hybrid_scorer import HybridRiskScorer
from ml.preprocessing import IEEEFraudPreprocessor
from ml.tabular_xgboost import XGBoostFraudPipeline

logger = logging.getLogger(__name__)

GENERIC_DEVICE_OS = {
    "windows",
    "ios device",
    "macos",
    "desktop",
    "mobile",
    "linux",
    "android",
    "unknown",
    "missing",
    "nan",
    "other",
    "windows 10/chrome",
    "desktop_chrome",
}

GENERIC_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "comcast.net",
    "icloud.com",
    "msn.com",
    "outlook.com",
    "sbcglobal.net",
    "verizon.net",
    "att.net",
    "bellsouth.net",
    "ymail.com",
    "mail.com",
    "anonymous.com",
}


class FraudModelService:
    """Singleton service for production XGBoost fraud scoring and Neo4j investigation explainability."""

    _instance: Optional["FraudModelService"] = None

    def __init__(self):
        self.root_dir = ROOT_DIR
        self.preprocessor_path = self.root_dir / "data" / "processed" / "preprocessor.joblib"
        self.xgboost_dir = self.root_dir / "artifacts" / "models" / "xgboost"
        self.gnn_dir = self.root_dir / "artifacts" / "models" / "gnn"
        self.eval_results_dir = self.root_dir / "artifacts" / "eval_results"

        self.preprocessor: Optional[IEEEFraudPreprocessor] = None
        self.xgb_pipeline: Optional[XGBoostFraudPipeline] = None
        self.gnn_pipeline: Optional[GNNFraudPipeline] = None
        self.gnn_builder: Optional[HeteroGraphBuilder] = None
        # Production scoring: 100% XGBoost tabular probability
        self.hybrid_scorer = HybridRiskScorer(tabular_weight=1.0, graph_weight=0.0)
        self._benchmark_cache: Optional[Dict[str, Any]] = None

        self._load_models()

    @classmethod
    def get_instance(cls) -> "FraudModelService":
        if cls._instance is None:
            cls._instance = FraudModelService()
        return cls._instance

    def _load_models(self) -> None:
        """Load fitted production models and entity mappings from artifacts."""
        try:
            if self.preprocessor_path.exists():
                self.preprocessor = IEEEFraudPreprocessor.load(str(self.preprocessor_path))
                logger.info("Loaded IEEE-CIS preprocessor state.")

            if (self.xgboost_dir / "xgboost_model.json").exists():
                self.xgb_pipeline = XGBoostFraudPipeline.load(str(self.xgboost_dir))
                logger.info("Loaded production XGBoost model pipeline.")

            # GNN is preserved exclusively for offline research exploration
            if (self.gnn_dir / "gnn_weights.pt").exists() and (self.gnn_dir / "graph_mappings.joblib").exists():
                self.gnn_pipeline = GNNFraudPipeline.load(str(self.gnn_dir))
                self.gnn_builder = HeteroGraphBuilder.load_mappings(str(self.gnn_dir / "graph_mappings.joblib"))
                logger.info("Loaded GNN model pipeline and entity mappings (offline research mode).")

        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        """Check whether production scoring pipeline (preprocessor + XGBoost) is fully loaded."""
        return self.preprocessor is not None and self.xgb_pipeline is not None

    def get_status(self) -> Dict[str, Any]:
        """Detailed status of all ML models and inference dependencies."""
        return {
            "status": "HEALTHY" if self.is_healthy() else "DEGRADED",
            "production_architecture": {
                "primary_predictor": "XGBoost",
                "role": "SOLE_AUTOMATED_PREDICTOR",
                "graph_role": "INVESTIGATION_AND_EXPLORATION",
                "gnn_role": "OFFLINE_RESEARCH_ONLY",
            },
            "models": {
                "xgboost": {
                    "loaded": self.xgb_pipeline is not None,
                    "artifact_exists": (self.xgboost_dir / "xgboost_model.json").exists(),
                    "features_count": len(self.xgb_pipeline.feature_names) if self.xgb_pipeline else 0,
                    "role": "PRODUCTION_PRIMARY",
                },
                "gnn": {
                    "loaded": self.gnn_pipeline is not None,
                    "artifact_exists": (self.gnn_dir / "gnn_weights.pt").exists(),
                    "mappings_loaded": self.gnn_builder is not None,
                    "role": "OFFLINE_RESEARCH_ONLY",
                },
                "preprocessor": {
                    "loaded": self.preprocessor is not None,
                    "artifact_exists": self.preprocessor_path.exists(),
                },
            },
            "ensemble_weights": {
                "tabular_weight": 1.0,
                "graph_weight": 0.0,
            },
        }

    def predict_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute production fraud inference on incoming transaction payload.
        XGBoost is the sole automated predictor.
        Fails explicitly if models are not loaded (fail-closed behavior).
        """
        if not self.is_healthy():
            raise RuntimeError(
                "Model service is unavailable: preprocessor or XGBoost model pipeline is not loaded."
            )

        tx_id = str(payload.get("transaction_id") or f"TX_{np.random.randint(100000, 999999)}")
        amount = float(payload.get("amount", 100.0))
        card_id = str(payload.get("card_id", "13926"))
        device_id = str(payload.get("device_id", "Desktop_Chrome"))
        email_domain = str(payload.get("email_domain", "gmail.com"))
        hour = int(payload.get("hour", 14))
        day_of_week = int(payload.get("day_of_week", 2))
        dist1 = float(payload.get("dist1", 10.0))

        # 1. Synthesize tabular features for preprocessor
        raw_row = {
            "TransactionID": tx_id,
            "TransactionAmt": amount,
            "card1": float(card_id.split("_")[0]) if card_id.replace("_", "").isdigit() else 13926.0,
            "card4": str(payload.get("card_network", "visa")),
            "card6": str(payload.get("card_type", "credit")),
            "DeviceInfo": device_id,
            "DeviceType": str(payload.get("device_type", "desktop")),
            "P_emaildomain": email_domain,
            "TransactionDT": hour * 3600 + day_of_week * 86400,
            "dist1": dist1,
            "ProductCD": str(payload.get("product_cd", "W")),
        }

        if "custom_features" in payload and isinstance(payload["custom_features"], dict):
            raw_row.update(payload["custom_features"])

        df_single = pd.DataFrame([raw_row])

        # 2. Real tabular inference via preprocessor + XGBoost (Sole automated predictor)
        X_single, meta_single = self.preprocessor.transform(df_single)
        probs = self.xgb_pipeline.predict_proba(X_single)
        tabular_prob = float(probs[0, 1])
        shap_attributions = self.xgb_pipeline.explain_transaction(X_single.iloc[0], top_n=6)

        # 3. Extract structural graph evidence for visual investigation (not a model score)
        graph_explanation = self._extract_graph_explanation(tx_id, card_id, device_id, email_domain, amount)

        # 4. Production scoring verdict (100% XGBoost, no fake graph heuristics)
        verdict_data = self.hybrid_scorer.score_transaction(
            tabular_prob=tabular_prob,
            graph_prob=None,
            shap_attributions=shap_attributions,
            graph_explanation=graph_explanation,
            transaction_amount=amount,
            is_production=True,
        )

        return {
            "transaction_id": tx_id,
            **verdict_data,
            "shap_attributions": shap_attributions,
            "graph_explanation": graph_explanation,
        }

    def _extract_graph_explanation(
        self,
        tx_id: str,
        card_id: str,
        device_id: str,
        email_domain: str,
        amount: float,
    ) -> Dict[str, Any]:
        """
        Build relational graph explanation based on real entity mappings and topology.
        Filters out generic OS identifiers and generic email domains to prevent artificial hubs.
        """
        is_generic_device = device_id.lower().strip() in GENERIC_DEVICE_OS
        is_generic_email = email_domain.lower().strip() in GENERIC_EMAIL_DOMAINS

        card_known = False
        device_known = False
        email_known = False

        if self.gnn_builder and self.gnn_builder.mappings:
            card_known = card_id in self.gnn_builder.mappings.card_to_idx
            if not is_generic_device:
                device_known = device_id in self.gnn_builder.mappings.device_to_idx
            if not is_generic_email:
                email_known = email_domain in self.gnn_builder.mappings.email_to_idx

        nodes = [
            {"id": tx_id, "label": f"Tx {tx_id}", "type": "transaction", "is_center": True, "risk_flag": False},
        ]
        links = []

        if card_id and card_id not in ["", "None", "nan"]:
            nodes.append({
                "id": f"card_{card_id}",
                "label": f"Card: {card_id[:16]}",
                "type": "card",
                "is_center": False,
                "risk_flag": False,
            })
            links.append({"source": tx_id, "target": f"card_{card_id}", "relationship": "USES_CARD"})

        if device_id and device_id not in ["", "None", "nan"]:
            label_suffix = " (Generic OS)" if is_generic_device else ""
            nodes.append({
                "id": f"dev_{device_id}",
                "label": f"Device: {device_id[:16]}{label_suffix}",
                "type": "device",
                "is_center": False,
                "risk_flag": False,
            })
            links.append({"source": tx_id, "target": f"dev_{device_id}", "relationship": "ORIGINATES_FROM"})

        if email_domain and email_domain not in ["", "None", "nan"]:
            label_suffix = " (Public Provider)" if is_generic_email else ""
            nodes.append({
                "id": f"email_{email_domain}",
                "label": f"Domain: {email_domain}{label_suffix}",
                "type": "email",
                "is_center": False,
                "risk_flag": False,
            })
            links.append({"source": tx_id, "target": f"email_{email_domain}", "relationship": "HAS_EMAIL"})

        reasons = []
        if card_known:
            reasons.append(f"Card {card_id[:16]} identified in historical transaction network.")
        else:
            reasons.append("Card entity is novel (first appearance in topological network).")

        if device_known and not is_generic_device:
            reasons.append(f"Specific device fingerprint '{device_id[:16]}' established in network registry.")
        elif is_generic_device:
            reasons.append("Generic device/OS fingerprint detected; excluded from shared entity clustering.")

        if is_generic_email:
            reasons.append(f"Common email provider '{email_domain}' unlinked from syndicate clustering.")

        return {
            "nodes": nodes,
            "links": links,
            "card_id": str(card_id),
            "device_id": str(device_id),
            "email_id": str(email_domain),
            "shared_entity_peers_count": 1 if (card_known or (device_known and not is_generic_device)) else 0,
            "flagged_neighbor_fraud_count": 0,
            "reasoning": " ".join(reasons),
        }

    def get_benchmark_data(self) -> Dict[str, Any]:
        """Return cached evaluation metrics, curves, and comparison table."""
        if self._benchmark_cache is not None:
            return self._benchmark_cache

        res: Dict[str, Any] = {"status": "ok"}
        metrics_file = self.eval_results_dir / "metrics.json"
        curves_file = self.eval_results_dir / "curves_data.json"
        csv_file = self.eval_results_dir / "benchmark_comparison.csv"

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                res["metrics"] = json.load(f)

        if curves_file.exists():
            with open(curves_file, "r") as f:
                res["curves"] = json.load(f)

        if csv_file.exists():
            df_csv = pd.read_csv(csv_file)
            res["comparison_table"] = df_csv.to_dict(orient="records")

        self._benchmark_cache = res
        return res

