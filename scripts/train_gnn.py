"""
Training Script for Heterogeneous Graph Neural Network (PyG) Pipeline
Constructs heterogeneous graph from transaction entities without label leakage,
trains HeteroFraudGNN on chronological train split with validation monitoring,
and saves model weights, graph mappings, and predictions.
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
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.gnn_model import GNNFraudPipeline
from ml.graph_dataset import HeteroGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_gnn(
    data_dir: str = "data/processed",
    output_dir: str = "artifacts/models/gnn",
    epochs: int = 20,
    hidden_channels: int = 64,
    lr: float = 0.003,
) -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading preprocessed features and metadata...")
    X_train = pd.read_parquet(data_path / "train_features.parquet")
    meta_train = pd.read_parquet(data_path / "train_meta.parquet")

    X_val = pd.read_parquet(data_path / "val_features.parquet")
    meta_val = pd.read_parquet(data_path / "val_meta.parquet")

    X_test = pd.read_parquet(data_path / "test_features.parquet")
    meta_test = pd.read_parquet(data_path / "test_meta.parquet")

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    # 1. Build Unified HeteroData Graph across chronological sequence
    X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
    meta_all = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)

    logger.info(f"Building unified graph ({len(X_all)} nodes: {n_train} train, {n_val} val, {n_test} test)...")
    builder = HeteroGraphBuilder()
    full_graph = builder.build_graph(
        X_all,
        meta_all,
        is_train=True,
        train_len=n_train,
        val_len=n_val,
    )

    # Save builder mappings
    builder.save_mappings(str(out_path / "graph_mappings.joblib"))

    # 2. Train Hetero GNN with validation monitoring
    pipeline = GNNFraudPipeline(
        hidden_channels=hidden_channels,
        lr=lr,
    )
    fit_result = pipeline.fit(
        full_graph,
        epochs=epochs,
    )

    # 3. Generate probabilities across all transaction nodes
    all_probs = pipeline.predict_proba(full_graph)

    val_probs = all_probs[n_train : n_train + n_val]
    y_val = meta_val["isFraud"].values
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    logger.info(f"GNN Validation Results - PR-AUC: {val_pr_auc:.4f}, ROC-AUC: {val_roc_auc:.4f}")

    test_probs = all_probs[n_train + n_val :]
    y_test = meta_test["isFraud"].values
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    logger.info(f"GNN Holdout Test Results - PR-AUC: {test_pr_auc:.4f}, ROC-AUC: {test_roc_auc:.4f}")

    # 4. Save Model Artifacts
    pipeline.save(str(out_path))

    # Save validation & test predictions for evaluation benchmark suite
    val_eval_df = pd.DataFrame({
        "TransactionID": meta_val["TransactionID"],
        "isFraud": y_val,
        "gnn_prob": val_probs,
    })
    val_eval_df.to_parquet(out_path / "val_predictions.parquet", index=False)

    test_eval_df = pd.DataFrame({
        "TransactionID": meta_test["TransactionID"],
        "isFraud": y_test,
        "gnn_prob": test_probs,
    })
    test_eval_df.to_parquet(out_path / "test_predictions.parquet", index=False)

    summary = {
        "model_type": "HeteroFraudGNN (SAGEConv/HeteroConv)",
        "epochs": epochs,
        "hidden_channels": hidden_channels,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "loss_history": fit_result.get("history", []),
    }
    with open(out_path / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"GNN training completed successfully. Artifacts saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Hetero GNN Fraud Detection Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/gnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)

    args = parser.parse_args()
    train_gnn(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
