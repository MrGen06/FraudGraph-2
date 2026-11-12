"""
Heterogeneous Graph Representation Builder for Fraud Detection
Constructs PyG HeteroData objects linking Transactions with Cards, Devices, and Non-Generic Email Domains.
Eliminates target label leakage and enforces chronological train/val/test masking.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ml.preprocessing import GENERIC_EMAIL_DOMAINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GraphEntityMappings:
    tx_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_tx: Dict[int, str] = field(default_factory=dict)
    card_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_card: Dict[int, str] = field(default_factory=dict)
    device_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_device: Dict[int, str] = field(default_factory=dict)
    email_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_email: Dict[int, str] = field(default_factory=dict)
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None


class HeteroGraphBuilder:
    """Transforms tabular and transaction entity metadata into a PyG HeteroData graph."""

    def __init__(self, mappings: Optional[GraphEntityMappings] = None):
        self.mappings = mappings or GraphEntityMappings()

    def build_graph(
        self,
        X_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        is_train: bool = True,
        train_len: Optional[int] = None,
        val_len: Optional[int] = None,
    ) -> HeteroData:
        """
        Construct HeteroData graph with:
        Nodes: 'transaction', 'card', 'device', 'email'
        Edges:
          ('transaction', 'uses_card', 'card') & reverse
          ('transaction', 'originates_from', 'device') & reverse
          ('transaction', 'has_email', 'email') & reverse
        Guarantees:
          1. ZERO target label leakage: no ground truth labels in node features.
          2. Disjoint chronological train / validation / test masks.
          3. Missing values do NOT form super-hubs.
          4. Generic email domains do NOT link unrelated transactions into giant hubs.
        """
        logger.info(f"Building heterogeneous graph from {len(X_df)} transactions...")
        data = HeteroData()

        # 1. Feature normalization for transaction nodes
        X_mat = X_df.values.astype(np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        if is_train or self.mappings.feature_mean is None:
            self.mappings.feature_mean = np.mean(X_mat, axis=0)
            self.mappings.feature_std = np.std(X_mat, axis=0)
            self.mappings.feature_std[self.mappings.feature_std < 1e-6] = 1.0

        X_norm = (X_mat - self.mappings.feature_mean) / self.mappings.feature_std
        data["transaction"].x = torch.from_numpy(X_norm)

        # Labels
        y_vals = meta_df["isFraud"].values.astype(np.int64)
        data["transaction"].y = torch.from_numpy(y_vals)

        n_tx = len(meta_df)
        train_mask = torch.zeros(n_tx, dtype=torch.bool)
        val_mask = torch.zeros(n_tx, dtype=torch.bool)
        test_mask = torch.zeros(n_tx, dtype=torch.bool)

        if train_len is not None and val_len is not None:
            train_mask[:train_len] = True
            val_mask[train_len : train_len + val_len] = True
            test_mask[train_len + val_len :] = True
        elif is_train:
            # Chronological default 70/15/15
            t_split = int(n_tx * 0.70)
            v_split = int(n_tx * 0.85)
            train_mask[:t_split] = True
            val_mask[t_split:v_split] = True
            test_mask[v_split:] = True
        else:
            test_mask[:] = True

        data["transaction"].train_mask = train_mask
        data["transaction"].val_mask = val_mask
        data["transaction"].test_mask = test_mask

        # 2. Extract unique entity mappings (ignoring missing / empty)
        tx_ids = [str(t) for t in meta_df["TransactionID"]]
        cards = [str(c).strip() if pd.notna(c) and str(c).strip() not in ["", "nan", "C_missing"] else "" for c in meta_df["card_id"]]
        devices = [str(d).strip() if pd.notna(d) and str(d).strip() not in ["", "nan", "D_missing"] else "" for d in meta_df["device_id"]]
        emails = [
            str(e).strip()
            if pd.notna(e) and str(e).strip() not in ["", "nan", "E_missing"] and str(e).strip().lower() not in GENERIC_EMAIL_DOMAINS
            else ""
            for e in meta_df["email_id"]
        ]

        if is_train or not self.mappings.card_to_idx:
            self.mappings.tx_to_idx = {tx: idx for idx, tx in enumerate(tx_ids)}
            self.mappings.idx_to_tx = {idx: tx for tx, idx in self.mappings.tx_to_idx.items()}

            unique_cards = sorted([c for c in set(cards) if c != ""])
            self.mappings.card_to_idx = {c: idx for idx, c in enumerate(unique_cards)}
            self.mappings.idx_to_card = {idx: c for c, idx in self.mappings.card_to_idx.items()}

            unique_devices = sorted([d for d in set(devices) if d != ""])
            self.mappings.device_to_idx = {d: idx for idx, d in enumerate(unique_devices)}
            self.mappings.idx_to_device = {idx: d for d, idx in self.mappings.device_to_idx.items()}

            unique_emails = sorted([e for e in set(emails) if e != ""])
            self.mappings.email_to_idx = {e: idx for idx, e in enumerate(unique_emails)}
            self.mappings.idx_to_email = {idx: e for e, idx in self.mappings.email_to_idx.items()}

        # 3. Create Entity Node Features (Strictly non-target attributes: degree, volume, mean amount)
        amounts = meta_df["amount"].values.astype(np.float32)

        # Card features: [degree, total_amount, mean_amount]
        n_cards = max(1, len(self.mappings.card_to_idx))
        card_features = np.zeros((n_cards, 3), dtype=np.float32)
        card_tx_edges = []
        for tx_idx, (c, amt) in enumerate(zip(cards, amounts)):
            if c != "" and c in self.mappings.card_to_idx:
                c_idx = self.mappings.card_to_idx[c]
                card_features[c_idx, 0] += 1.0  # transaction count
                card_features[c_idx, 1] += amt  # monetary volume
                card_tx_edges.append((tx_idx, c_idx))

        nonzero_cards = card_features[:, 0] > 0
        card_features[nonzero_cards, 2] = card_features[nonzero_cards, 1] / card_features[nonzero_cards, 0]
        data["card"].x = torch.from_numpy(card_features)

        # Device features: [degree, total_amount, mean_amount]
        n_devices = max(1, len(self.mappings.device_to_idx))
        device_features = np.zeros((n_devices, 3), dtype=np.float32)
        device_tx_edges = []
        for tx_idx, (d, amt) in enumerate(zip(devices, amounts)):
            if d != "" and d in self.mappings.device_to_idx:
                d_idx = self.mappings.device_to_idx[d]
                device_features[d_idx, 0] += 1.0
                device_features[d_idx, 1] += amt
                device_tx_edges.append((tx_idx, d_idx))

        nonzero_devices = device_features[:, 0] > 0
        device_features[nonzero_devices, 2] = device_features[nonzero_devices, 1] / device_features[nonzero_devices, 0]
        data["device"].x = torch.from_numpy(device_features)

        # Email domain features: [degree, total_amount, mean_amount]
        n_emails = max(1, len(self.mappings.email_to_idx))
        email_features = np.zeros((n_emails, 3), dtype=np.float32)
        email_tx_edges = []
        for tx_idx, (e, amt) in enumerate(zip(emails, amounts)):
            if e != "" and e in self.mappings.email_to_idx:
                e_idx = self.mappings.email_to_idx[e]
                email_features[e_idx, 0] += 1.0
                email_features[e_idx, 1] += amt
                email_tx_edges.append((tx_idx, e_idx))

        nonzero_emails = email_features[:, 0] > 0
        email_features[nonzero_emails, 2] = email_features[nonzero_emails, 1] / email_features[nonzero_emails, 0]
        data["email"].x = torch.from_numpy(email_features)

        # 4. Construct Bipartite Edge Indices (only valid, non-empty pairs)
        if card_tx_edges:
            edges = torch.tensor(card_tx_edges, dtype=torch.int64).t()
            data["transaction", "uses_card", "card"].edge_index = edges
            data["card", "rev_uses_card", "transaction"].edge_index = torch.stack([edges[1], edges[0]], dim=0)
        else:
            empty_edge = torch.empty((2, 0), dtype=torch.int64)
            data["transaction", "uses_card", "card"].edge_index = empty_edge
            data["card", "rev_uses_card", "transaction"].edge_index = empty_edge

        if device_tx_edges:
            edges = torch.tensor(device_tx_edges, dtype=torch.int64).t()
            data["transaction", "originates_from", "device"].edge_index = edges
            data["device", "rev_originates_from", "transaction"].edge_index = torch.stack([edges[1], edges[0]], dim=0)
        else:
            empty_edge = torch.empty((2, 0), dtype=torch.int64)
            data["transaction", "originates_from", "device"].edge_index = empty_edge
            data["device", "rev_originates_from", "transaction"].edge_index = empty_edge

        if email_tx_edges:
            edges = torch.tensor(email_tx_edges, dtype=torch.int64).t()
            data["transaction", "has_email", "email"].edge_index = edges
            data["email", "rev_has_email", "transaction"].edge_index = torch.stack([edges[1], edges[0]], dim=0)
        else:
            empty_edge = torch.empty((2, 0), dtype=torch.int64)
            data["transaction", "has_email", "email"].edge_index = empty_edge
            data["email", "rev_has_email", "transaction"].edge_index = empty_edge

        logger.info(
            f"HeteroGraph created: {data.num_nodes} total nodes "
            f"({n_tx} tx, {len(self.mappings.card_to_idx)} cards, {len(self.mappings.device_to_idx)} devices, {len(self.mappings.email_to_idx)} emails), "
            f"{data.num_edges} edges. Train={int(train_mask.sum())}, Val={int(val_mask.sum())}, Test={int(test_mask.sum())}."
        )
        return data

    def save_mappings(self, filepath: str) -> None:
        """Persist graph entity mappings and normalization parameters."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.mappings, filepath)
        logger.info(f"Graph mappings saved to {filepath}")

    @classmethod
    def load_mappings(cls, filepath: str) -> "HeteroGraphBuilder":
        """Load graph entity mappings."""
        mappings = joblib.load(filepath)
        return cls(mappings=mappings)
