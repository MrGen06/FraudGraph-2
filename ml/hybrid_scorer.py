"""
Production Fraud Scoring Engine & Historical Ensemble Evaluator
Production Decision: XGBoost is the sole automated fraud predictor.
Neo4j provides relational investigation evidence.
GNN and hybrid ensembling are preserved for offline research and historical benchmark evaluation.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from sklearn.metrics import average_precision_score

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class HybridRiskScorer:
    """
    Risk Scorer supporting:
    1. Production Scoring: XGBoost as the sole automated fraud predictor (tabular_weight=1.0, graph_weight=0.0).
    2. Historical/Research Scoring: Ensembling for offline experiments.
    """

    def __init__(
        self,
        tabular_weight: float = 1.0,
        graph_weight: float = 0.0,
        low_threshold: float = 0.30,
        critical_threshold: float = 0.70,
    ):
        self.tabular_weight = tabular_weight
        self.graph_weight = graph_weight
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold

    def fit_weights_from_validation(
        self,
        val_tabular_probs: np.ndarray,
        val_graph_probs: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Learn the optimal ensemble weight w in [0, 1] maximizing PR-AUC strictly on validation data.
        Retained strictly for offline research benchmark evaluation.
        """
        best_w = 1.0
        best_prauc = -1.0

        for w in np.linspace(0.0, 1.0, 21):
            p_ensemble = w * val_tabular_probs + (1.0 - w) * val_graph_probs
            prauc = average_precision_score(y_val, p_ensemble)
            if prauc > best_prauc:
                best_prauc = prauc
                best_w = float(w)

        self.tabular_weight = round(best_w, 2)
        self.graph_weight = round(1.0 - best_w, 2)
        return self.tabular_weight

    def calculate_risk_level(self, score: float) -> str:
        """Categorize risk score into consistent risk tier."""
        if score < self.low_threshold:
            return "LOW"
        elif score < self.critical_threshold:
            return "MEDIUM"
        else:
            return "CRITICAL"

    def score_transaction(
        self,
        tabular_prob: float,
        graph_prob: Optional[float] = None,
        shap_attributions: Optional[List[Dict[str, Any]]] = None,
        graph_explanation: Optional[Dict[str, Any]] = None,
        transaction_amount: Optional[float] = None,
        is_production: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute transaction risk score and generate explainable reasoning.
        In production (is_production=True), XGBoost is the sole automated predictor.
        Graph topological signals are included solely as structural investigation context.
        """
        shap_attributions = shap_attributions or []
        graph_explanation = graph_explanation or {}

        # In production, automated risk probability is 100% XGBoost tabular probability
        if is_production or self.graph_weight == 0.0:
            final_risk_score = max(0.0, min(1.0, float(tabular_prob)))
            effective_graph_score = 0.0
            graph_level = "INVESTIGATION_ONLY"
        else:
            # Historical / offline experimental weighting
            effective_graph_score = float(graph_prob) if graph_prob is not None else 0.0
            final_risk_score = (self.tabular_weight * tabular_prob) + (self.graph_weight * effective_graph_score)
            final_risk_score = max(0.0, min(1.0, float(final_risk_score)))
            graph_level = self.calculate_risk_level(effective_graph_score)

        tab_level = self.calculate_risk_level(tabular_prob)
        verdict = self.calculate_risk_level(final_risk_score)

        # Extract SHAP top contributing risk features
        top_positive_features = [
            f"{item.get('description', item.get('feature'))} ({item.get('feature')})"
            for item in shap_attributions
            if item.get("direction") == "risk_increasing"
        ][:2]

        top_feature_str = (
            ", ".join(top_positive_features)
            if top_positive_features
            else "transaction velocity and amount parameters"
        )

        # Graph structural context for investigators (not a model score)
        flagged_neighbors = graph_explanation.get("flagged_neighbor_fraud_count", 0)
        card_id = graph_explanation.get("card_id", "Unknown")
        shared_peers = graph_explanation.get("shared_entity_peers_count", 0)

        reasoning_parts = []
        if verdict == "CRITICAL":
            reasoning_parts.append(
                f"CRITICAL RISK: Automated XGBoost model flagged severe fraud probability ({tabular_prob*100:.1f}%), "
                f"primarily driven by {top_feature_str}."
            )
            if flagged_neighbors > 0:
                reasoning_parts.append(
                    f"Graph investigation reveals {flagged_neighbors} historically flagged fraud links associated with Card #{str(card_id)[:16]}."
                )
            elif shared_peers > 0:
                reasoning_parts.append(
                    f"Graph investigation notes entity connectivity across {shared_peers} shared network peers."
                )
        elif verdict == "MEDIUM":
            reasoning_parts.append(
                f"MODERATE RISK: Automated XGBoost model flagged elevated risk ({tabular_prob*100:.1f}%), "
                f"influenced by {top_feature_str}. Manual analyst review recommended."
            )
            if shared_peers > 0:
                reasoning_parts.append(
                    f"Graph investigation displays {shared_peers} entity link(s) for analyst review."
                )
        else:
            reasoning_parts.append(
                f"LOW RISK (Approved): Automated XGBoost model scored risk at {tabular_prob*100:.1f}%, "
                f"within standard clearing thresholds. Transaction authorized."
            )

        return {
            "risk_score": round(float(final_risk_score), 4),
            "risk_level": verdict,
            "tabular_score": round(float(tabular_prob), 4),
            "tabular_risk_level": tab_level,
            "graph_score": round(float(effective_graph_score), 4),
            "graph_risk_level": graph_level,
            "hybrid_score": round(float(final_risk_score), 4),
            "consensus_verdict": verdict,
            "consensus_reasoning": " ".join(reasoning_parts),
            "decision_reasoning": " ".join(reasoning_parts),
        }
