# FraudGraph: Interactive Demonstration Script

This document provides a comprehensive step-by-step walkthrough to demonstrate the core capabilities of **FraudGraph** to stakeholders, evaluators, and engineering teams.

---

## Preparation Checklist

1. **Backend Server Running**:
   ```bash
   cd backend
   .\.venv\Scripts\activate
   uvicorn app.main:app --reload --port 8000
   ```
   Confirm `http://localhost:8000/health` returns `{"status":"ok"}`.

2. **Frontend Running**:
   ```bash
   cd frontend
   npm run dev
   ```
   Open `http://localhost:5173`.

---

## 5-Act Demonstration Walkthrough

### Act 1: Executive Dashboard & Live System Health (2 minutes)
- **Goal**: Establish operational readiness and show live platform statistics.
- **Action**:
  1. Navigate to the **📊 Overview** tab.
  2. Point out the top header status pill: **Dual-Model Engine Online** (showing live health of XGBoost, GNN, and database storage).
  3. Review the 4 high-level KPI cards:
     - **Total Ingested Transactions**
     - **Flagged Suspicious Activity**
     - **Active Investigation Alerts**
     - **Held-Out Test Fraud Rate** (1.91%)
  4. Scroll to the **Active Alerts Queue** to highlight incoming fraud events pending human review.

---

### Act 2: Interactive Transaction Investigation (3 minutes)
- **Goal**: Demonstrate how compliance and risk analysts explore transactions and drill into details.
- **Action**:
  1. Click on the **💳 Transactions** tab.
  2. Filter by risk tier using the dropdown (e.g., select **CRITICAL** or **HIGH**).
  3. Type a card ID or transaction ID into the search bar and press **Search**.
  4. Click the **Inspect** button on any flagged row.
  5. In the slide-out **Inspection Drawer**, show:
     - Exact timestamp, card fingerprint, and device fingerprint.
     - Dual-model scores (Tabular score vs Graph score).
     - Automated **Consensus Reasoning** explaining why the transaction was flagged.
  6. Click **"Investigate in Graph Explorer"** at the bottom of the drawer.

---

### Act 3: Relational Graph Explorer (3 minutes)
- **Goal**: Showcase multi-hop entity link analysis and explain how syndicate behavior is discovered.
- **Action**:
  1. The app smoothly transitions to the **🕸️ Graph Explorer** tab focused on the selected transaction.
  2. Explain the visual elements:
     - **Blue Center Node**: The current transaction being evaluated.
     - **Purple / Cyan / Orange Nodes**: Connected entities (Card, Device, Email).
     - **Directed Arrowheads**: Directed relationships (`USED_CARD`, `USED_DEVICE`, `SHARED_ENTITY`).
     - **Red Pulsing Halo**: Indicates an entity associated with prior confirmed fraud.
  3. Demonstrate interactivity:
     - Click and drag nodes to reorganize the layout.
     - Use the zoom controls (`+`, `-`, `Reset`) to pan across large subgraphs.
     - Click on any entity node (e.g., a Card node) to open the **Node Inspector** showing degrees, connected transaction counts, and risk attributes.

---

### Act 4: Live Risk Simulator & SHAP Waterfall (4 minutes)
- **Goal**: Demonstrate real-time dual-model scoring and explainability on novel payloads.
- **Action**:
  1. Click on the **⚡ Risk Inspector** tab.
  2. Select the preset **"High-Risk Velocity Burst"**:
     - Notice the large transaction amount, high frequency delta, and risk indicators.
  3. Click **"Run Dual-Model Risk Analysis"**:
     - The **Consensus Verdict Banner** immediately flags `CRITICAL RISK` with synthesized reasoning.
     - The **Dual Risk Scorecards** contrast the XGBoost tabular score (e.g. 0.88) against the GNN graph score (e.g. 0.74).
  4. Scroll down to the **SHAP Feature Attributions** bar chart:
     - Point out the red bars indicating risk-increasing features (e.g., transaction amount, unusual hour, device mismatch).
     - Point out the green bars indicating risk-mitigating features (e.g., familiar email domain).
  5. Select the preset **"Normal Routine Purchase"** and re-run to show instant clearance to `LOW RISK`.

---

### Act 5: Scientific Benchmark & Model Governance (3 minutes)
- **Goal**: Prove scientific rigor, transparent metrics, and explainability to compliance auditors.
- **Action**:
  1. Navigate to the **📈 Benchmark** tab.
  2. Review the **Benchmark Summary Card**:
     - Point out the **Chronological 70/15/15 Data Split** on IEEE-CIS.
     - Emphasize that **test labels were never leaked** into feature aggregations.
  3. Review the **Model Comparison Table**:
     - **XGBoost**: PR-AUC 0.3496, ROC-AUC 0.8931.
     - **Graph Neural Network (PyG)**: PR-AUC 0.0400, ROC-AUC 0.6430.
     - **Validation-Tuned Hybrid**: PR-AUC 0.3501, ROC-AUC 0.8937 (incorporating $w=0.95$ Tabular, $0.05$ Graph).
  4. Point out the **Dynamic Winner Badges**:
     - Best PR-AUC: Validation-Tuned Hybrid.
     - Best ROC-AUC: Validation-Tuned Hybrid.
  5. Inspect the **Confusion Matrices** showing exact True Positives, False Positives, False Negatives, and True Negatives on the 7,500 held-out test transactions at the validation-selected threshold.

