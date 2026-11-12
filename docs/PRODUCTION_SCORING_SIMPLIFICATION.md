# FraudGraph — Production Scoring Simplification Report

**Date:** 2026-11-12  
**Status:** Architecture Aligned, Implemented, & Verified  
**Decision Authority:** Principal Graph ML Engineer & Lead Backend Architect  

---

## 1. Executive Summary & Architectural Decision

Following the empirical findings of the GNN forensic investigation documented in [`docs/GNN_ROOT_CAUSE_ANALYSIS.md`](file:///c:/Users/gouta/Desktop/FraudGraph-2/docs/GNN_ROOT_CAUSE_ANALYSIS.md), the FraudGraph production architecture has been officially simplified and hardened.

### The Production Mandate
- **XGBoost = Sole Automated Fraud Predictor**: Gradient boosted decision trees over 161 engineered non-linear tabular features provide 100% of real-time fraud probability, risk tiering, and alert generation.
- **Neo4j = Graph-Based Investigation & Exploration**: The Neo4j property graph is utilized strictly as human-in-the-loop investigation evidence for visual 1-hop and 2-hop entity-link analysis.
- **GNN = Offline Experimental Research Only**: The PyG `HeteroFraudGNN` architecture is retained exclusively for offline topological research, completely segregated from live transaction inference.

---

## 2. Phase 1 — Comprehensive Code Audit

Before making changes, an audit of the entire codebase was conducted to identify every location where the GNN, the historical 95/5 hybrid score, or the `_infer_graph_score()` heuristic influenced production predictions, risk tiers, alerts, or UI displays.

### Audit Findings

| Component / File | Previous Behavior | Issue / Defect Identified | Action Taken |
| :--- | :--- | :--- | :--- |
| `backend/app/services/model_service.py` | Executed `_infer_graph_score()` heuristic; blended tabular & heuristic scores with 95/5 weights. | **Training/Inference Disconnect**: The trained GNN weights (`gnn_weights.pt`) were never run during live inference; an arbitrary heuristic formula was run instead, fabricating graph probabilities. | Completely purged `_infer_graph_score()`. Set `tabular_weight=1.0, graph_weight=0.0`. XGBoost is the sole probability source. |
| `backend/app/services/transaction_service.py` | Lines 52–60 caught any model exception and fell back to `{"hybrid_score": 0.05, "consensus_verdict": "LOW"}`. | **Silent Fail-Open**: If model loading failed or artifacts were missing, transactions were silently cleared as low risk rather than failing closed. | Removed silent fallback. Enforced strict fail-closed policy raising HTTP 503 Service Unavailable on model failure. |
| `backend/app/services/graph_service.py` | Merged all transactions with identical `device_id` into shared `:Device` nodes. | **Mega-Hub Collapse**: Generic operating system strings (`'Windows'`, `'MacOS'`, `'Desktop'`) created mega-hubs linking thousands of unrelated transactions. | Added generic OS and email filtering to prevent artificial mega-hubs. |
| `ml/hybrid_scorer.py` | `score_transaction()` blended scores using validation weights and generated consensus reasoning referencing dual-model consensus. | **Diluted Decisions**: Blending degraded tabular signals and created confusing consensus language when models disagreed. | Updated to evaluate 100% on XGBoost probability in production, using graph signals strictly for structural investigation context. |
| `frontend/src/components/DualRiskScorecard.tsx` | Claimed "Pipeline B: Hetero GNN / Relational Message Passing" with a live percentage score. | **Misleading UI**: Falsely claimed a live trained GNN was generating real-time predictions. | Replaced GNN scoring card with "Neo4j Investigation Evidence", showing entity links and labeling GNN as offline research only. |
| `frontend/src/components/ConsensusBanner.tsx` | Displayed "Consensus Dual-Model Decision" and "Ensemble Risk Score". | **Misleading UI**: Suggested live multi-model consensus rather than XGBoost tabular classification. | Updated to "Automated Fraud Risk Decision (XGBoost)" and "XGBoost Fraud Probability". |
| `frontend/src/components/BenchmarkDashboard.tsx` | Listed models without clear architectural designation. | **Ambiguous Status**: Did not explicitly indicate which model is active in production vs. offline research. | Added clear architectural callout banner and "Production Active" vs. "Offline Research Only" status badges. |

---

## 3. Phase 2 — Production Scoring Simplification

### Key Modifications
1. **XGBoost as Sole Source of Automated Probability**:
   - `predict_transaction()` in `model_service.py` derives the transaction risk probability exclusively from `xgb_pipeline.predict_proba(X_single)[:, 1]`.
   - `tabular_weight` is set to `1.0` and `graph_weight` to `0.0`.
   - The returned `risk_score` equals the XGBoost probability. For backwards compatibility with external clients, `tabular_score` and `hybrid_score` are returned as direct aliases of `risk_score`.
2. **Purging of Fake Heuristic Score**:
   - The private helper `_infer_graph_score()` was completely deleted from `FraudModelService`.
   - `graph_score` is set to `0.0` and `graph_risk_level` is set to `"INVESTIGATION_ONLY"`.
3. **Consistent Risk Tiering & Alert Triggers**:
   - Risk levels are computed strictly against the validated XGBoost probability:
     - `risk_score >= 0.70` $\rightarrow$ `CRITICAL`
     - `0.30 <= risk_score < 0.70` $\rightarrow$ `MEDIUM`
     - `risk_score < 0.30` $\rightarrow$ `LOW`
   - In `transaction_service.py`, alert generation consistently triggers when `risk_level in ["MEDIUM", "HIGH", "CRITICAL"]` or `risk_score >= 0.30`.
4. **Preservation of TreeSHAP Explainability**:
   - TreeSHAP local attributions (`xgb_pipeline.explain_transaction()`) remain fully active, computing feature contributions and directional impact for every transaction.
5. **Fail-Closed Security Guarantee**:
   - If the preprocessor or XGBoost model artifact is missing or fails to load, `is_healthy()` returns `False`.
   - Any transaction ingestion attempt via `POST /api/v1/transactions/` or prediction via `POST /api/v1/models/predict` immediately raises an HTTP 503 Service Unavailable error.
   - The system never silently defaults to an arbitrary low-risk score (e.g. 0.05).
6. **Zero Label Contamination**:
   - Neither `test_meta.parquet`, `isFraud`, nor any benchmark test metadata is read or referenced during inference.
7. **Offline Preservation of GNN**:
   - `ml/gnn_model.py`, `ml/graph_dataset.py`, `scripts/train_gnn.py`, and `artifacts/models/gnn/gnn_weights.pt` remain fully preserved for offline research and academic analysis.

---

## 4. Phase 3 — Neo4j Investigation Preservation & Hardening

Neo4j remains a central component of FraudGraph, repositioned strictly to its strength: **interactive human-in-the-loop investigation**.

### Pruning Unreliable Entity Relationships
The GNN root cause analysis proved that 83.95% of the graph collapsed into a single component due to generic device and email hubs. In `backend/app/services/graph_service.py`:
1. **Generic Device Filtering**: Identifiers that normalize to common operating systems (`'windows'`, `'macos'`, `'ios device'`, `'desktop'`, `'mobile'`, `'linux'`, `'android'`, etc.) are excluded from creating shared `:Device` hubs. Only specific device models (e.g. `'SAMSUNG SM-G892A'`) create shared links.
2. **Generic Email Domain Filtering**: Public consumer email domains (`'gmail.com'`, `'yahoo.com'`, `'hotmail.com'`, `'anonymous.com'`, etc.) are prevented from linking unrelated transactions into false syndicates.
3. **Card Identity Respect**: Non-unique card BINs (e.g. `'13926'`) are displayed as card/issuer links without falsely inventing individual cardholder identities or fake bank accounts that IEEE-CIS does not provide.

### Maintained Capabilities
- **Transaction Ingestion**: Legitimate entity nodes and relationships are merged into Neo4j with non-blocking resilience.
- **Two-Hop Exploration**: Subgraph traversal (`MATCH (t)-[r]-(neighbor)-[r2]-(second_hop)`) continues powering the Cytoscape/SVG interactive graph view.
- **Relational Fallback**: If Neo4j is offline, `get_transaction_subgraph` falls back to synthesizing an entity subgraph from PostgreSQL records.

---

## 5. Phase 4 — Frontend and Documentation Alignment

### User Interface Updates
1. **DualRiskScorecard (`frontend/src/components/DualRiskScorecard.tsx`)**:
   - Card A: **Production Primary** — XGBoost Classifier (Sole Automated Predictor). Displays true model probability, risk tier, top SHAP driver, and TreeSHAP explainability engine.
   - Card B: **Neo4j Investigation** — Entity Graph Topology (Relational Investigation Evidence). Displays linked peer counts, compromised links, and clarifies that GNN is offline research only.
2. **ConsensusBanner (`frontend/src/components/ConsensusBanner.tsx`)**:
   - Replaced "Consensus Dual-Model Decision" with "Automated Fraud Risk Decision (XGBoost)".
   - Replaced "Ensemble Risk Score" with "XGBoost Fraud Probability".
3. **BenchmarkDashboard (`frontend/src/components/BenchmarkDashboard.tsx`)**:
   - Added prominent architectural alert banner detailing why XGBoost is the sole production predictor and why GNN is experimental offline research.
   - Table displays explicit **Role in Architecture** and **Deployment Status** badges ("Production Choice" vs. "Offline Research Only" vs. "Historical Reference").
4. **DashboardOverview & TransactionInvestigation**:
   - Updated KPI subtext to "XGBoost (Sole Automated Predictor) + Neo4j Graph".
   - Updated table and drawer labels from "Hybrid Risk Score" to "XGBoost Risk Score" and "Decision Reasoning".
5. **TransactionSimulator**:
   - Updated action button to "Execute Automated Risk Scoring (XGBoost)".

### Documentation Updates
- Updated `README.md`
- Updated `docs/PROJECT_ARCHITECTURE.md`
- Updated `docs/TECHNICAL_LIMITATIONS.md`
- Updated `docs/IMPLEMENTATION_STATUS.md`
- Updated `docs/DEPLOYMENT.md`
- Preserved `docs/GNN_ROOT_CAUSE_ANALYSIS.md`

---

## 6. Phase 5 — Verification & Test Results

### 1. Automated Test Suite Results
Dedicated tests were implemented in [`backend/tests/test_production_scoring.py`](file:///c:/Users/gouta/Desktop/FraudGraph-2/backend/tests/test_production_scoring.py) and executed alongside the full suite:

```bash
$ backend/.venv/Scripts/python.exe -m pytest backend/tests -v
================== 22 passed, 2 skipped, 14 warnings in 1.51s ==================
```

### 2. Specific Verification Criteria

| Verification Requirement | Test / Check Implemented | Result | Status |
| :--- | :--- | :---: | :---: |
| 1. Live predictions call XGBoost | `test_live_prediction_calls_xgboost_sole_predictor`: spies on `xgb_pipeline.predict_proba` during `predict_transaction()`. | `predict_proba` invoked; returns valid probability. | **PASS** |
| 2. Live predictions do not call `_infer_graph_score()` | `test_live_prediction_does_not_call_infer_graph_score`: asserts method does not exist on service; verifies `graph_score == 0.0`. | `hasattr(service, '_infer_graph_score') == False`. | **PASS** |
| 3. Live predictions do not depend on GNN weights | `test_live_prediction_does_not_depend_on_gnn_weights`: sets `service.gnn_pipeline = None` and verifies successful scoring. | Scoring succeeds with valid SHAP values and risk tiers. | **PASS** |
| 4. Missing XGBoost artifacts produce service error | `test_missing_xgboost_artifacts_fail_closed`: sets `service.xgb_pipeline = None` and asserts HTTP 503 is raised. | HTTP 503 returned; fail-closed behavior verified. | **PASS** |
| 5. Alert risk levels match XGBoost predictions | `test_alert_risk_levels_match_xgboost_predictions`: ingests transaction and verifies PostgreSQL alert matches XGBoost risk score and tier. | Exact score and risk level persisted in database. | **PASS** |
| 6. Neo4j investigation endpoints continue working | `test_neo4j_investigation_subgraph_endpoint`: verifies `/api/v1/graph/{id}` returns nodes and links for investigation. | Nodes and links correctly returned. | **PASS** |
| 7. Transaction ingestion and retrieval functional | `test_transaction_ingestion_and_retrieval`: verifies `create_transaction` and `get_transactions` remain functional. | Transactions persisted, queried, and listed. | **PASS** |
| 8. No benchmark test labels accessed during inference | `test_no_benchmark_test_labels_accessed_during_inference`: patches `pandas.read_parquet` to verify no test files are read. | `read_parquet` never called during inference. | **PASS** |
| 9. Frontend build succeeds | `npm --prefix frontend run build` | TypeScript clean build; Vite bundles production assets in 211ms. | **PASS** |
| 10. Backend tests pass | `pytest backend/tests -v` | 22 passed, 2 skipped (optional remote DBs), 0 failures. | **PASS** |

---

## 7. Operational Summary & Production Architecture

```
Incoming Transaction Payload
         |
         v
[IEEEFraudPreprocessor] ---> Chronological feature transform (161 features)
         |
         v
[XGBoost Classifier]    ---> Automated Fraud Probability (0.0000 - 1.0000)
         |
         +-----------------> TreeSHAP Local Attributions (Top Risk Drivers)
         |
         +-----------------> Fail-Closed Risk Tiering (LOW < 0.30 <= MEDIUM < 0.70 <= CRITICAL)
         |
         v
[PostgreSQL Ledger]     ---> Immutable transaction & alert persistence
         |
         v
[Neo4j Property Graph]  ---> Filtered entity sync (pruned of generic OS & email hubs)
         |
         v
[Analyst Web App]       ---> Visual 2-hop graph investigation & SHAP explanation
```

The system is now fully aligned with defensible production machine learning standards.

