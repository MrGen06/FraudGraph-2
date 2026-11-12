"""
Training Script for XGBoost Baseline Tabular Pipeline
Executes chronological training with early stopping on validation split (PR-AUC)
and saves model artifact with SHAP explainer.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from ml.tabular_xgboost import XGBoostFraudPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_xgboost(
    data_dir: str = "data/processed",
    output_dir: str = "artifacts/models/xgboost",
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading preprocessed features from {data_dir}...")
    X_train = pd.read_parquet(data_path / "train_features.parquet")
    meta_train = pd.read_parquet(data_path / "train_meta.parquet")
    y_train = meta_train["isFraud"]

    # Load validation split
    val_feat_path = data_path / "val_features.parquet"
    if val_feat_path.exists():
        X_val = pd.read_parquet(val_feat_path)
        meta_val = pd.read_parquet(data_path / "val_meta.parquet")
        y_val = meta_val["isFraud"]
    else:
        # Fallback split if val parquet is missing
        n_tr = int(len(X_train) * 0.8)
        X_val = X_train.iloc[n_tr:]
        y_val = y_train.iloc[n_tr:]
        X_train = X_train.iloc[:n_tr]
        y_train = y_train.iloc[:n_tr]

    X_test = pd.read_parquet(data_path / "test_features.parquet")
    meta_test = pd.read_parquet(data_path / "test_meta.parquet")
    y_test = meta_test["isFraud"]

    logger.info(f"Loaded splits: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    logger.info(
        f"Fraud rates: Train={y_train.mean()*100:.2f}%, Val={y_val.mean()*100:.2f}%, Test={y_test.mean()*100:.2f}%"
    )

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": ["aucpr", "auc"],
    }

    pipeline = XGBoostFraudPipeline(params=params)

    # 1. Train with Validation Early Stopping
    logger.info("Fitting XGBoost on chronological Train split with Validation early stopping...")
    pipeline.fit_single(X_train, y_train, X_val, y_val, early_stopping_rounds=30)

    # 2. Evaluate on Validation Set
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    logger.info(f"Chronological Validation Results - PR-AUC: {val_pr_auc:.4f}, ROC-AUC: {val_roc_auc:.4f}")

    # 3. Evaluate on Holdout Test Split
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    logger.info(f"Holdout Test Results - PR-AUC: {test_pr_auc:.4f}, ROC-AUC: {test_roc_auc:.4f}")

    # 4. Save Model & Metadata
    pipeline.save(str(out_path))

    # Save validation & test predictions for evaluation benchmark suite
    val_eval_df = pd.DataFrame({
        "TransactionID": meta_val["TransactionID"],
        "isFraud": y_val,
        "xgboost_prob": val_probs,
    })
    val_eval_df.to_parquet(out_path / "val_predictions.parquet", index=False)

    test_eval_df = pd.DataFrame({
        "TransactionID": meta_test["TransactionID"],
        "isFraud": y_test,
        "xgboost_prob": test_probs,
    })
    test_eval_df.to_parquet(out_path / "test_predictions.parquet", index=False)

    summary = {
        "model_type": "XGBoost",
        "split_strategy": "Chronological (TransactionDT)",
        "n_features": X_train.shape[1],
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
    }
    with open(out_path / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"XGBoost training completed successfully. Artifacts saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Tabular Baseline Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/xgboost")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)

    args = parser.parse_args()
    train_xgboost(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
