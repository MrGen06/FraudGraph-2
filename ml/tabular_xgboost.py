"""
XGBoost Baseline Tabular Fraud Detection Pipeline
Features Stratified K-Fold CV, PR-AUC early stopping, and SHAP TreeExplainer integration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DESCRIPTIONS = {
    "TransactionAmt": "Transaction monetary amount",
    "log_TransactionAmt": "Log-scaled transaction amount",
    "amt_to_card_mean": "Ratio of amount to card's historical average amount",
    "card1_tx_count": "Historical transaction volume for this card",
    "device_tx_count": "Historical transaction frequency on this device",
    "email_tx_count": "Transaction frequency for this email domain",
    "is_missing_id": "Absence of digital identity verification record",
    "is_missing_dist1": "Missing billing/delivery distance signal",
    "dist1": "Distance from registered address to delivery location",
    "dt_hour": "Hour of transaction execution (0-23)",
    "dt_day_of_week": "Day of week transaction occurred",
    "ProductCD_code": "Merchant product category classification code",
    "card4_code": "Payment card network (Visa, Mastercard, etc.)",
    "card6_code": "Payment card type (credit vs. debit)",
    "P_emaildomain_code": "Purchaser email domain category",
    "DeviceInfo_code": "Hardware/browser fingerprint device identity",
    "C1": "Transaction count attribute 1 (frequency counter)",
    "D1": "Timedelta days between transaction and previous event",
    "V258": "Card authentication verification consistency score",
}


class XGBoostFraudPipeline:
    """Production-grade XGBoost tabular classifier with SHAP interpretability."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        default_params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "min_child_weight": 3,
            "tree_method": "hist",
            "eval_metric": ["aucpr", "auc"],
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = feature_names or []
        self.cv_results_: Dict[str, Any] = {}

    def fit_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        early_stopping_rounds: int = 30,
    ) -> Dict[str, Any]:
        """Train with Stratified K-Fold CV, tracking out-of-fold PR-AUC and ROC-AUC."""
        logger.info(f"Training XGBoost baseline with {n_splits}-fold Stratified CV...")
        self.feature_names = list(X.columns)

        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        scale_pos_weight = neg_count / max(1, pos_count)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.params["random_state"])
        oof_preds = np.zeros(len(y))
        fold_metrics = []

        best_score = -1.0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

            fold_params = dict(self.params)
            fold_params["scale_pos_weight"] = scale_pos_weight
            fold_params["early_stopping_rounds"] = early_stopping_rounds

            clf = xgb.XGBClassifier(**fold_params)
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_va, y_val)],
                verbose=False,
            )

            val_preds = clf.predict_proba(X_va)[:, 1]
            oof_preds[val_idx] = val_preds

            prauc = average_precision_score(y_val, val_preds)
            rocauc = roc_auc_score(y_val, val_preds)
            fold_metrics.append({"fold": fold + 1, "pr_auc": prauc, "roc_auc": rocauc})
            logger.info(f"Fold {fold + 1} - PR-AUC: {prauc:.4f}, ROC-AUC: {rocauc:.4f}")

            if prauc > best_score:
                best_score = prauc
                best_model = clf

        overall_prauc = average_precision_score(y, oof_preds)
        overall_rocauc = roc_auc_score(y, oof_preds)
        logger.info(f"XGBoost CV Complete - Mean PR-AUC: {overall_prauc:.4f}, Mean ROC-AUC: {overall_rocauc:.4f}")

        self.model = best_model
        self.cv_results_ = {
            "fold_metrics": fold_metrics,
            "mean_pr_auc": float(overall_prauc),
            "mean_roc_auc": float(overall_rocauc),
            "oof_preds": oof_preds,
            "scale_pos_weight": float(scale_pos_weight),
        }

        # Initialize SHAP explainer
        self._init_shap(X.sample(min(len(X), 1000), random_state=42))
        return self.cv_results_

    def fit_single(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 30,
    ) -> "XGBoostFraudPipeline":
        """Fit single model with early stopping on validation split."""
        self.feature_names = list(X_train.columns)
        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        scale_pos_weight = neg_count / max(1, pos_count)

        params = dict(self.params)
        params["scale_pos_weight"] = scale_pos_weight
        params["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._init_shap(X_train.sample(min(len(X_train), 1000), random_state=42))
        return self

    def _init_shap(self, background_data: pd.DataFrame) -> None:
        """Initialize SHAP TreeExplainer."""
        if self.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP TreeExplainer: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities [P(non-fraud), P(fraud)]."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        # Align features
        if list(X.columns) != self.feature_names:
            X = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict_proba(X)

    def explain_transaction(self, x_row: pd.Series, top_n: int = 8) -> List[Dict[str, Any]]:
        """
        Produce top local feature attributions using SHAP for a single transaction.
        Returns sorted list of features pushing risk up or down.
        """
        if self.explainer is None:
            if self.model is not None:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                return []

        df_single = pd.DataFrame([x_row]).reindex(columns=self.feature_names, fill_value=0.0)
        shap_values = self.explainer.shap_values(df_single)

        if isinstance(shap_values, list):
            # Binary classification list: [class_0, class_1]
            raw_shap = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            raw_shap = shap_values[0]
        else:
            raw_shap = shap_values

        attributions = []
        for feat_name, sv, val in zip(self.feature_names, raw_shap, df_single.iloc[0]):
            direction = "risk_increasing" if sv > 0 else "risk_decreasing"
            desc = FEATURE_DESCRIPTIONS.get(feat_name, f"Feature value: {val:.2f}" if isinstance(val, (int, float)) else str(val))
            attributions.append({
                "feature": feat_name,
                "value": float(val) if not np.isnan(val) else 0.0,
                "shap_value": float(sv),
                "magnitude": abs(float(sv)),
                "direction": direction,
                "description": desc,
            })

        # Sort by absolute impact
        attributions.sort(key=lambda item: item["magnitude"], reverse=True)
        return attributions[:top_n]

    def save(self, directory: str) -> None:
        """Save model and metadata."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_model(str(dir_path / "xgboost_model.json"))

        metadata = {
            "feature_names": self.feature_names,
            "params": {k: v for k, v in self.params.items() if not callable(v)},
            "cv_results": {k: v for k, v in self.cv_results_.items() if k != "oof_preds"},
        }
        with open(dir_path / "xgboost_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"XGBoost pipeline saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> "XGBoostFraudPipeline":
        """Load saved XGBoost pipeline."""
        dir_path = Path(directory)
        with open(dir_path / "xgboost_meta.json", "r") as f:
            metadata = json.load(f)

        pipeline = cls(
            params=metadata.get("params"),
            feature_names=metadata.get("feature_names"),
        )
        model = xgb.XGBClassifier()
        model.load_model(str(dir_path / "xgboost_model.json"))
        pipeline.model = model
        return pipeline

