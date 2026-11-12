"""
Evaluation & Benchmarking Suite for FraudGraph
Compares XGBoost Tabular Baseline, Heterogeneous GNN, and Hybrid Consensus Model against Ground Truth.
Rigorously evaluates on held-out chronological test data using validation-selected thresholds and weights.
Generates publication-ready metrics, curves, and visualization artifacts.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["MPLCONFIGDIR"] = str(ROOT_DIR / ".matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.hybrid_scorer import HybridRiskScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_fpr_at_recall(y_true: np.ndarray, y_probs: np.ndarray, target_recall: float) -> float:
    """Calculate False Positive Rate at target Recall threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    idx = np.where(tpr >= target_recall)[0]
    if len(idx) > 0:
        return float(fpr[idx[0]])
    return 1.0


def select_optimal_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Select decision threshold maximizing minority F1 on validation split only."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def compute_model_metrics(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict:
    """Evaluate test set performance using a fixed pre-selected threshold."""
    pr_auc = float(average_precision_score(y_test, test_probs))
    roc_auc = float(roc_auc_score(y_test, test_probs))

    y_pred = (test_probs >= threshold).astype(int)
    p = float(precision_score(y_test, y_pred, zero_division=0))
    r = float(recall_score(y_test, y_pred, zero_division=0))
    f1_minority = float(f1_score(y_test, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    cm = confusion_matrix(y_test, y_pred).tolist()

    fpr_90 = compute_fpr_at_recall(y_test, test_probs, 0.90)
    fpr_95 = compute_fpr_at_recall(y_test, test_probs, 0.95)
    fpr_99 = compute_fpr_at_recall(y_test, test_probs, 0.99)

    return {
        "model": model_name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "optimal_threshold": round(threshold, 4),
        "f1_minority": round(f1_minority, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "fpr_at_90_recall": round(fpr_90, 4),
        "fpr_at_95_recall": round(fpr_95, 4),
        "fpr_at_99_recall": round(fpr_99, 4),
        "confusion_matrix": cm,
    }


def generate_benchmark_suite(
    data_dir: str = "data/processed",
    xgboost_dir: str = "artifacts/models/xgboost",
    gnn_dir: str = "artifacts/models/gnn",
    output_dir: str = "artifacts/eval_results",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running FraudGraph evaluation & benchmarking suite...")

    # 1. Load validation predictions to tune thresholds and weights
    val_xgb = pd.read_parquet(Path(xgboost_dir) / "val_predictions.parquet")
    val_gnn = pd.read_parquet(Path(gnn_dir) / "val_predictions.parquet")
    y_val = val_xgb["isFraud"].values
    p_val_xgb = val_xgb["xgboost_prob"].values
    p_val_gnn = val_gnn["gnn_prob"].values

    # Determine thresholds on validation split
    thresh_xgb = select_optimal_threshold(y_val, p_val_xgb)
    thresh_gnn = select_optimal_threshold(y_val, p_val_gnn)

    # Determine ensemble weight on validation split
    hybrid_scorer = HybridRiskScorer()
    best_tab_w = hybrid_scorer.fit_weights_from_validation(p_val_xgb, p_val_gnn, y_val)
    best_graph_w = round(1.0 - best_tab_w, 2)
    logger.info(f"Learned Validation Ensemble Weights: Tabular={best_tab_w:.2f}, Graph={best_graph_w:.2f}")

    p_val_hyb = best_tab_w * p_val_xgb + best_graph_w * p_val_gnn
    thresh_hyb = select_optimal_threshold(y_val, p_val_hyb)

    logger.info(
        f"Validation-Tuned Decision Thresholds: XGBoost={thresh_xgb:.4f}, GNN={thresh_gnn:.4f}, Hybrid={thresh_hyb:.4f}"
    )

    # 2. Load held-out test predictions
    test_xgb = pd.read_parquet(Path(xgboost_dir) / "test_predictions.parquet")
    test_gnn = pd.read_parquet(Path(gnn_dir) / "test_predictions.parquet")
    y_test = test_xgb["isFraud"].values
    p_test_xgb = test_xgb["xgboost_prob"].values
    p_test_gnn = test_gnn["gnn_prob"].values
    p_test_hyb = best_tab_w * p_test_xgb + best_graph_w * p_test_gnn

    # 3. Compute Metrics on Untouched Test Split
    metrics_xgb = compute_model_metrics(y_test, p_test_xgb, thresh_xgb, "XGBoost Baseline")
    metrics_gnn = compute_model_metrics(y_test, p_test_gnn, thresh_gnn, "Hetero GNN (Relational)")
    metrics_hyb = compute_model_metrics(y_test, p_test_hyb, thresh_hyb, "Hybrid Consensus (Dual-Model)")

    models_metrics = [metrics_xgb, metrics_gnn, metrics_hyb]

    # Save metrics JSON
    benchmark_data = {
        "test_records_count": int(len(y_test)),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "validation_ensemble_weights": {
            "tabular_weight": best_tab_w,
            "graph_weight": best_graph_w,
        },
        "models": models_metrics,
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # Save comparison CSV
    df_comp = pd.DataFrame(models_metrics).drop(columns=["confusion_matrix"])
    df_comp.to_csv(out_path / "benchmark_comparison.csv", index=False)

    # 4. Compute ROC and PR Curves for Frontend Charts
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_test_xgb)
    fpr_gnn, tpr_gnn, _ = roc_curve(y_test, p_test_gnn)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_test, p_test_hyb)

    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, p_test_xgb)
    prec_gnn, rec_gnn, _ = precision_recall_curve(y_test, p_test_gnn)
    prec_hyb, rec_hyb, _ = precision_recall_curve(y_test, p_test_hyb)

    # Sample points for compact JSON (100 points per curve)
    def sample_curve(x_arr, y_arr, n_pts=100):
        if len(x_arr) <= n_pts:
            return [round(float(v), 4) for v in x_arr], [round(float(v), 4) for v in y_arr]
        idx = np.linspace(0, len(x_arr) - 1, n_pts, dtype=int)
        return [round(float(x_arr[i]), 4) for i in idx], [round(float(y_arr[i]), 4) for i in idx]

    fx_xgb, ty_xgb = sample_curve(fpr_xgb, tpr_xgb)
    fx_gnn, ty_gnn = sample_curve(fpr_gnn, tpr_gnn)
    fx_hyb, ty_hyb = sample_curve(fpr_hyb, tpr_hyb)

    rx_xgb, py_xgb = sample_curve(rec_xgb, prec_xgb)
    rx_gnn, py_gnn = sample_curve(rec_gnn, prec_gnn)
    rx_hyb, py_hyb = sample_curve(rec_hyb, prec_hyb)

    curves_data = {
        "roc": {
            "fpr_xgb": fx_xgb,
            "tpr_xgb": ty_xgb,
            "fpr_gnn": fx_gnn,
            "tpr_gnn": ty_gnn,
            "fpr_hyb": fx_hyb,
            "tpr_hyb": ty_hyb,
        },
        "pr": {
            "rec_xgb": rx_xgb,
            "prec_xgb": py_xgb,
            "rec_gnn": rx_gnn,
            "prec_gnn": py_gnn,
            "rec_hyb": rx_hyb,
            "prec_hyb": py_hyb,
            "baseline": round(float(y_test.mean()), 4),
        },
    }
    with open(out_path / "curves_data.json", "w") as f:
        json.dump(curves_data, f, indent=2)

    # 5. Generate Matplotlib Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {metrics_xgb['roc_auc']:.3f})", color="#3b82f6", lw=2)
    axes[0].plot(fpr_gnn, tpr_gnn, label=f"GNN (AUC = {metrics_gnn['roc_auc']:.3f})", color="#10b981", lw=2)
    axes[0].plot(fpr_hyb, tpr_hyb, label=f"Hybrid (AUC = {metrics_hyb['roc_auc']:.3f})", color="#8b5cf6", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0].set_title("ROC Curves (Held-out Test)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    axes[1].plot(rec_xgb, prec_xgb, label=f"XGBoost (PR-AUC = {metrics_xgb['pr_auc']:.3f})", color="#3b82f6", lw=2)
    axes[1].plot(rec_gnn, prec_gnn, label=f"GNN (PR-AUC = {metrics_gnn['pr_auc']:.3f})", color="#10b981", lw=2)
    axes[1].plot(rec_hyb, prec_hyb, label=f"Hybrid (PR-AUC = {metrics_hyb['pr_auc']:.3f})", color="#8b5cf6", lw=2)
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline ({y_test.mean()*100:.1f}%)")
    axes[1].set_title("Precision-Recall Curves (Held-out Test)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "roc_pr_curves.png", dpi=200)
    plt.close()

    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = ["XGBoost Baseline", "Hetero GNN", "Hybrid Consensus"]
    colors = ["Blues", "Greens", "Purples"]

    for ax, metric, name, cmap in zip(axes, models_metrics, names, colors):
        cm = np.array(metric["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
        ax.set_title(f"{name}\n(Threshold = {metric['optimal_threshold']:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])

    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrices.png", dpi=200)
    plt.close()

    logger.info(f"Evaluation complete. Benchmark artifacts successfully generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FraudGraph Model Benchmark Suite")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--xgboost-dir", type=str, default="artifacts/models/xgboost")
    parser.add_argument("--gnn-dir", type=str, default="artifacts/models/gnn")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval_results")

    args = parser.parse_args()
    generate_benchmark_suite(
        data_dir=args.data_dir,
        xgboost_dir=args.xgboost_dir,
        gnn_dir=args.gnn_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
