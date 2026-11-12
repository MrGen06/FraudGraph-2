# FraudGraph: Technical Limitations & Machine Learning Findings

This document presents a scientifically honest, in-depth analysis of the data, modeling, and architectural limitations discovered and addressed during the engineering of FraudGraph.

---

## 1. IEEE-CIS Fraud Detection Dataset Characteristics

The IEEE-CIS dataset is one of the most widely used public fraud benchmarks, but real-world engineering requires recognizing its structural constraints:

### 1.1 Absence of True Account Identifiers
- IEEE-CIS does **not** contain true source or destination bank account numbers, customer IDs, or verified cardholder identities.
- `card1` represents a Bank Identification Number (BIN) / issuer range, **not** an individual credit card. Treating `card1` as a unique card entity creates artificial mega-clusters containing tens of thousands of unrelated consumers.
- **Remedy in FraudGraph**: We constructed a multi-attribute composite fingerprint:
  $$\text{CardFingerprint} = \text{card1}\_\text{card2}\_\text{card3}\_\text{card4}\_\text{card5}\_\text{card6}$$
  This provides a much closer approximation of an individual payment instrument while avoiding artificial super-nodes.

### 1.2 Generic Email Domain Super-Nodes
- Millions of transactions share generic consumer email providers: `gmail.com`, `yahoo.com`, `hotmail.com`, `anonymous.com`.
- Connecting transactions via generic email domains creates dense, uninformative hub nodes that collapse bipartite graph separation.
- **Remedy in FraudGraph**: Generic email domains are explicitly filtered out from entity-link graph expansion. Only organization-specific or domain-specific email entities are connected.

---

## 2. Machine Learning Pipeline & GNN Reality Check

### 2.1 The Myth of the "Overperforming" GNN
In earlier flawed implementations or superficial benchmarks, Graph Neural Networks (GNNs) on IEEE-CIS often report inflated metrics (e.g., PR-AUC $> 0.70$ or ROC-AUC $> 0.95$). Our audit in Phase A proved that these inflated scores were caused by **severe target-label leakage**:
1. Aggregating `isFraud` into entity node features prior to train/test partitioning.
2. Identical validation and test masks allowing the model to tune hyperparameters directly on the test evaluation set.
3. Random stratified splits that allowed the GNN to learn future transaction labels to predict past transactions.

### 2.2 Performance Post-Leakage Removal
When target-label leakage is rigorously purged and chronological partitioning is enforced:
- **XGBoost**: PR-AUC = **0.3496**, ROC-AUC = **0.8931**.
- **PyG GraphSAGE (Zero Leakage)**: PR-AUC = **0.0400**, ROC-AUC = **0.6430**.

#### Why does the GNN score lower than gradient boosted trees?
1. **Information Density**: Tabular features (transaction amount, delta from previous transactions, count aggregations `C1`-`C14`, and identity match signals) carry immense local discriminative signal.
2. **Topological Sparsity**: In a zero-leakage setting, an incoming transaction node has no prior knowledge of whether its neighbors are fraudulent unless those historical transactions were explicitly flagged in past training windows.
3. **Bipartite Graph Structure**: Transactions connect to cards and devices, but rarely connect directly to each other. GraphSAGE message passing over 2 hops essentially aggregates neighbor transaction features (amounts, times) rather than persistent node representations.

### 2.3 Why the Validation-Tuned Hybrid Ensemble Works
Rather than imposing an arbitrary 50/50 split that would degrade XGBoost's 0.3496 PR-AUC down to 0.3310, FraudGraph implements validation-based grid search:
$$S_{\text{hybrid}} = w \cdot S_{\text{tabular}} + (1 - w) \cdot S_{\text{graph}}$$

On the held-out validation split:
- Optimal weight: $w_{\text{tabular}} = 0.95$, $w_{\text{graph}} = 0.05$.
- Resulting test PR-AUC: **0.3501** (higher than XGBoost alone at 0.3496).
- Resulting test ROC-AUC: **0.8937** (higher than XGBoost alone at 0.8931).

This demonstrates the proper role of graph signals in financial risk engines: providing high-precision auxiliary signals without diluting strong tabular baselines.

---

## 3. Infrastructure & Operational Considerations

### 3.1 Network Latency & Graph Query Overhead
- Graph traversal queries (e.g. 2-hop neighborhood expansion) in cloud-hosted graph databases (such as Neo4j AuraDB) introduce 50–150ms round-trip latency compared to 5–15ms relational queries in PostgreSQL.
- For sub-50ms transaction authorization pipelines, graph queries should either run asynchronously (out-of-band post-authorization) or utilize local in-memory graph caches.

### 3.2 Resilience and Fallback Guarantees
- FraudGraph ensures that if the Neo4j instance is paused, throttled, or unreachable, transaction ingestion and scoring continue uninterrupted.
- Relational fallback subgraphs synthesized from PostgreSQL guarantee that analysts always have access to a visual representation of transaction context.

