"""
Automated Ingestion Script for IEEE-CIS Fraud Detection Dataset
Executes chronological sorting by TransactionDT, non-leaking preprocessing fitting,
and rigorous 70% Train / 15% Validation / 15% Test chronological partitioning.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from ml.preprocessing import IEEEFraudPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(
    transaction_path: str,
    identity_path: str,
    output_dir: str,
    sample_size: int = 50000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting chronological IEEE-CIS dataset ingestion...")
    df_raw = IEEEFraudPreprocessor.load_raw_dataset(
        transaction_path=transaction_path,
        identity_path=identity_path,
        sample_size=sample_size,
        random_state=random_state,
    )

    n_total = len(df_raw)
    n_train = int(n_total * train_ratio)
    n_val_end = int(n_total * (train_ratio + val_ratio))

    logger.info(
        f"Chronological Partitioning ({n_total} records): "
        f"Train [0:{n_train}] (70%), "
        f"Val [{n_train}:{n_val_end}] (15%), "
        f"Test [{n_val_end}:{n_total}] (15%)"
    )

    df_train_raw = df_raw.iloc[:n_train].reset_index(drop=True)
    df_val_raw = df_raw.iloc[n_train:n_val_end].reset_index(drop=True)
    df_test_raw = df_raw.iloc[n_val_end:].reset_index(drop=True)

    logger.info(f"Train count: {len(df_train_raw)}, Val count: {len(df_val_raw)}, Test count: {len(df_test_raw)}")

    # Fit preprocessor STRICTLY on train split
    preprocessor = IEEEFraudPreprocessor()
    preprocessor.fit(df_train_raw)

    X_train, meta_train = preprocessor.transform(df_train_raw)
    X_val, meta_val = preprocessor.transform(df_val_raw)
    X_test, meta_test = preprocessor.transform(df_test_raw)

    # Calculate class weighting
    n_neg = int((meta_train["isFraud"] == 0).sum())
    n_pos = int((meta_train["isFraud"] == 1).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    logger.info(
        f"Train fraud stats: {n_pos} positive, {n_neg} negative (scale_pos_weight={scale_pos_weight:.2f}, "
        f"rate={meta_train['isFraud'].mean()*100:.2f}%)"
    )
    logger.info(
        f"Val fraud stats: {int(meta_val['isFraud'].sum())} positive "
        f"(rate={meta_val['isFraud'].mean()*100:.2f}%)"
    )
    logger.info(
        f"Test fraud stats: {int(meta_test['isFraud'].sum())} positive "
        f"(rate={meta_test['isFraud'].mean()*100:.2f}%)"
    )

    # Persist processed artifacts
    logger.info("Persisting processed splits and metadata...")
    X_train.to_parquet(output_path / "train_features.parquet", index=False)
    meta_train.to_parquet(output_path / "train_meta.parquet", index=False)

    X_val.to_parquet(output_path / "val_features.parquet", index=False)
    meta_val.to_parquet(output_path / "val_meta.parquet", index=False)

    X_test.to_parquet(output_path / "test_features.parquet", index=False)
    meta_test.to_parquet(output_path / "test_meta.parquet", index=False)

    # Sample for quick inspection
    X_train.head(100).to_csv(output_path / "train_features_sample.csv", index=False)

    # Save fitted preprocessor
    preprocessor_file = output_path / "preprocessor.joblib"
    preprocessor.save(str(preprocessor_file))

    summary = {
        "split_method": "Chronological (TransactionDT)",
        "total_records": n_total,
        "train_records": len(X_train),
        "val_records": len(X_val),
        "test_records": len(X_test),
        "feature_count": X_train.shape[1],
        "train_fraud_count": n_pos,
        "train_nonfraud_count": n_neg,
        "train_fraud_rate": float(meta_train["isFraud"].mean()),
        "val_fraud_rate": float(meta_val["isFraud"].mean()),
        "test_fraud_rate": float(meta_test["isFraud"].mean()),
        "scale_pos_weight": scale_pos_weight,
        "features": list(X_train.columns),
    }

    with open(output_path / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ingestion complete. Processed datasets saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess IEEE-CIS dataset chronologically.")
    parser.add_argument(
        "--transaction-path",
        type=str,
        default="data/raw/ieee_cis/train_transaction.csv",
        help="Path to raw train_transaction.csv",
    )
    parser.add_argument(
        "--identity-path",
        type=str,
        default="data/raw/ieee_cis/train_identity.csv",
        help="Path to raw train_identity.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of chronological records to sample",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological train fraction",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Chronological validation fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    run_ingestion(
        transaction_path=args.transaction_path,
        identity_path=args.identity_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
