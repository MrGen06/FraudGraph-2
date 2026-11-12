"""
FraudGraph Preprocessing & Feature Engineering Module
Targeted for IEEE-CIS Fraud Detection Dataset
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Key numerical and categorical features selected from IEEE-CIS
NUMERICAL_COLS = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D11",
    "D15",
    "V12",
    "V14",
    "V29",
    "V35",
    "V44",
    "V47",
    "V53",
    "V62",
    "V65",
    "V70",
    "V75",
    "V76",
    "V78",
    "V82",
    "V83",
    "V87",
    "V91",
    "V94",
    "V96",
    "V102",
    "V127",
    "V130",
    "V131",
    "V139",
    "V143",
    "V160",
    "V202",
    "V203",
    "V204",
    "V212",
    "V221",
    "V258",
    "V261",
    "V264",
    "V267",
    "V274",
    "V279",
    "V280",
    "V282",
    "V283",
    "V285",
    "V287",
    "V289",
    "V291",
    "V303",
    "V307",
    "V308",
    "V310",
    "V312",
    "V313",
    "V314",
    "V317",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_13",
    "id_17",
    "id_19",
    "id_20",
]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


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
}


def construct_card_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Build composite CardFingerprint from available card1..card6 attributes."""
    c1 = df["card1"].fillna("").astype(str).str.strip().replace(["nan", "0.0", "0"], "")
    c2 = df["card2"].fillna("").astype(str).str.strip().replace(["nan", "0.0", "0"], "") if "card2" in df.columns else ""
    c3 = df["card3"].fillna("").astype(str).str.strip().replace(["nan", "0.0", "0"], "") if "card3" in df.columns else ""
    c4 = df["card4"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "") if "card4" in df.columns else ""
    c5 = df["card5"].fillna("").astype(str).str.strip().replace(["nan", "0.0", "0"], "") if "card5" in df.columns else ""
    c6 = df["card6"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "") if "card6" in df.columns else ""

    composite = (
        c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
    )
    # Mask out records where card1 is entirely absent
    is_valid = c1 != ""
    return composite.where(is_valid, "")


def extract_device_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Extract device signature from DeviceInfo, id_31, or DeviceType."""
    dev_info = df["DeviceInfo"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "") if "DeviceInfo" in df.columns else ""
    id_31 = df["id_31"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "") if "id_31" in df.columns else ""
    dev_type = df["DeviceType"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "") if "DeviceType" in df.columns else ""

    result = dev_info
    result = result.where(result != "", id_31)
    result = result.where(result != "", dev_type)
    return result


@dataclass
class PreprocessorState:
    medians: Dict[str, float] = field(default_factory=dict)
    freq_maps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    label_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    card1_amt_mean: Dict[Any, float] = field(default_factory=dict)
    card1_tx_count: Dict[Any, int] = field(default_factory=dict)
    device_tx_count: Dict[Any, int] = field(default_factory=dict)
    email_tx_count: Dict[Any, int] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    overall_mean_amt: float = 100.0


class IEEEFraudPreprocessor:
    """End-to-end preprocessing, tabular feature engineering, and graph entity extraction."""

    def __init__(self, state: Optional[PreprocessorState] = None):
        self.state = state or PreprocessorState()
        self.is_fitted = state is not None and len(state.feature_names) > 0

    @staticmethod
    def load_raw_dataset(
        transaction_path: str,
        identity_path: str,
        sample_size: Optional[int] = 50000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Load and merge IEEE-CIS transactions and identity tables with strict chronological ordering."""
        logger.info(f"Reading transaction file: {transaction_path}")
        if sample_size and sample_size > 0:
            df_tx = pd.read_csv(transaction_path, nrows=sample_size * 2)
        else:
            df_tx = pd.read_csv(transaction_path)

        logger.info(f"Reading identity file: {identity_path}")
        df_id = pd.read_csv(identity_path)

        logger.info("Merging transaction and identity on TransactionID (left join)...")
        df = df_tx.merge(df_id, on="TransactionID", how="left")

        # Chronological ordering by TransactionDT
        if "TransactionDT" in df.columns:
            logger.info("Sorting transactions chronologically by TransactionDT...")
            df = df.sort_values("TransactionDT").reset_index(drop=True)

        if sample_size and sample_size > 0 and len(df) > sample_size:
            logger.info(f"Taking first {sample_size} chronological transactions...")
            df = df.iloc[:sample_size].reset_index(drop=True)

        logger.info(
            f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. "
            f"Fraud rate: {df['isFraud'].mean()*100:.2f}%" if "isFraud" in df.columns else f"Loaded {df.shape[0]} rows."
        )
        return df

    def fit(self, df: pd.DataFrame) -> "IEEEFraudPreprocessor":
        """Compute training set statistics, medians, frequency maps, and aggregation baselines."""
        logger.info("Fitting preprocessor statistics on training data...")
        state = PreprocessorState()

        # 1. Numerical medians
        for col in NUMERICAL_COLS:
            if col in df.columns:
                val = float(df[col].median(skipna=True))
                state.medians[col] = 0.0 if np.isnan(val) else val

        # 2. Categorical frequency and label encodings
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                series = df[col].fillna("missing").astype(str)
                freq = (series.value_counts() / len(series)).to_dict()
                state.freq_maps[col] = freq
                unique_vals = sorted(list(series.unique()))
                state.label_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}

        # 3. Transactional aggregations
        if "card1" in df.columns and "TransactionAmt" in df.columns:
            card_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "count"])
            state.card1_amt_mean = card_stats["mean"].to_dict()
            state.card1_tx_count = card_stats["count"].to_dict()
            state.overall_mean_amt = float(df["TransactionAmt"].median(skipna=True))

        if "DeviceInfo" in df.columns:
            dev_series = df["DeviceInfo"].fillna("missing").astype(str)
            state.device_tx_count = dev_series.value_counts().to_dict()

        if "P_emaildomain" in df.columns:
            email_series = df["P_emaildomain"].fillna("missing").astype(str)
            state.email_tx_count = email_series.value_counts().to_dict()

        self.state = state
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform raw transaction dataframe into:
        1. X: Cleaned, feature-engineered tabular matrix
        2. meta_df: Graph entity identifiers (card, device, email, TransactionID, isFraud)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet! Call fit() first.")

        df = df.copy()
        feat_dict: Dict[str, Any] = {}

        # 1. Informative Missing Pattern Flags
        feat_dict["is_missing_id"] = (df["DeviceInfo"].isna().astype(int) if "DeviceInfo" in df.columns else np.zeros(len(df), dtype=int))
        feat_dict["is_missing_dist1"] = (df["dist1"].isna().astype(int) if "dist1" in df.columns else np.zeros(len(df), dtype=int))
        feat_dict["is_missing_email"] = (df["P_emaildomain"].isna().astype(int) if "P_emaildomain" in df.columns else np.zeros(len(df), dtype=int))
        feat_dict["is_missing_card2"] = (df["card2"].isna().astype(int) if "card2" in df.columns else np.zeros(len(df), dtype=int))

        # 2. Temporal Signals from TransactionDT
        if "TransactionDT" in df.columns:
            dt = df["TransactionDT"].fillna(0).astype(float)
            hour = (dt // 3600) % 24
            day_of_week = (dt // (3600 * 24)) % 7
            day = (dt // (3600 * 24))

            feat_dict["dt_hour"] = hour
            feat_dict["dt_day_of_week"] = day_of_week
            feat_dict["dt_day"] = day
            feat_dict["dt_sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
            feat_dict["dt_cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
            feat_dict["dt_sin_dow"] = np.sin(2 * np.pi * day_of_week / 7.0)
            feat_dict["dt_cos_dow"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # 3. Transaction Amount Features
        if "TransactionAmt" in df.columns:
            amt = df["TransactionAmt"].fillna(self.state.overall_mean_amt).astype(float)
            feat_dict["TransactionAmt"] = amt
            feat_dict["log_TransactionAmt"] = np.log1p(np.maximum(amt, 0))
            feat_dict["amt_decimal"] = (amt - np.floor(amt)).round(3)

        # 4. Numerical Imputation via Medians
        for col in NUMERICAL_COLS:
            if col in df.columns and col != "TransactionAmt":
                median_val = self.state.medians.get(col, 0.0)
                feat_dict[col] = df[col].fillna(median_val).astype(float)

        # 5. Categorical Handling: Frequency + Label Encodings
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                series = df[col].fillna("missing").astype(str)
                freq_map = self.state.freq_maps.get(col, {})
                label_map = self.state.label_maps.get(col, {})

                feat_dict[f"{col}_freq"] = series.map(freq_map).fillna(0.0).astype(float)
                feat_dict[f"{col}_code"] = series.map(label_map).fillna(-1.0).astype(float)

        # 6. Transactional Velocity & Aggregation Features
        if "card1" in df.columns:
            card_vals = df["card1"]
            card_means = card_vals.map(self.state.card1_amt_mean).fillna(self.state.overall_mean_amt)
            card_counts = card_vals.map(self.state.card1_tx_count).fillna(1.0)

            feat_dict["card1_mean_amt"] = card_means
            feat_dict["card1_tx_count"] = card_counts
            if "TransactionAmt" in df.columns:
                feat_dict["amt_to_card_mean"] = feat_dict["TransactionAmt"] / (card_means + 1e-4)

        if "DeviceInfo" in df.columns:
            dev_series = df["DeviceInfo"].fillna("missing").astype(str)
            feat_dict["device_tx_count"] = dev_series.map(self.state.device_tx_count).fillna(1.0)

        if "P_emaildomain" in df.columns:
            email_series = df["P_emaildomain"].fillna("missing").astype(str)
            feat_dict["email_tx_count"] = email_series.map(self.state.email_tx_count).fillna(1.0)

        features = pd.DataFrame(feat_dict, index=df.index)

        # Save feature names on first transformation
        if not self.state.feature_names:
            self.state.feature_names = list(features.columns)

        # Build meta dataframe for graph construction and target linking
        card_series = construct_card_fingerprint(df)
        device_series = extract_device_fingerprint(df)
        email_series = (
            df["P_emaildomain"].fillna("").astype(str).str.strip().replace(["nan", "missing"], "")
            if "P_emaildomain" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )

        meta_dict: Dict[str, Any] = {
            "TransactionID": df["TransactionID"].astype(str) if "TransactionID" in df.columns else [f"TX_{i}" for i in range(len(df))],
            "isFraud": df["isFraud"].astype(int) if "isFraud" in df.columns else np.zeros(len(df), dtype=int),
            "card_id": card_series,
            "device_id": device_series,
            "email_id": email_series,
            "is_generic_email": email_series.isin(GENERIC_EMAIL_DOMAINS),
            "amount": df["TransactionAmt"].fillna(0.0).astype(float) if "TransactionAmt" in df.columns else np.zeros(len(df), dtype=float),
            "TransactionDT": df["TransactionDT"].fillna(0).astype(float) if "TransactionDT" in df.columns else np.zeros(len(df), dtype=float),
        }
        meta_df = pd.DataFrame(meta_dict, index=df.index)

        return features, meta_df

    def save(self, filepath: str) -> None:
        """Persist fitted preprocessor state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.state, filepath)
        logger.info(f"Preprocessor state saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "IEEEFraudPreprocessor":
        """Load fitted preprocessor state."""
        state = joblib.load(filepath)
        return cls(state=state)
