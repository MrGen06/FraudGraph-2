# FraudGraph — Final Engineering Audit

**Audit date:** 2026-11-12  
**Audited by:** Automated multi-agent parallel inspection (3 specialist agents)  
**Repository:** `c:\Users\gouta\Desktop\FraudGraph-2`

> [!IMPORTANT]
> This audit was performed by reading the actual source files. It does not trust README claims, prior agent summaries, or reported benchmark numbers. Every finding is cited to specific files and line numbers.

---

## Audit Summary

| Severity | ML Pipeline | Backend | Frontend | Total |
|----------|-------------|---------|----------|-------|
| Critical | 6 | 4 | 0 | **10** |
| High | 2 | 7 | 4 | **13** |
| Medium | 3 | 7 | 6 | **16** |
| Low | 2 | 1 | 6 | **9** |

---

## 1. What Is Already Working

- **Dataset presence**: IEEE-CIS raw CSVs exist (`train_transaction.csv`, `train_identity.csv`) and preprocessed features (`data/processed/*.parquet`) are present.
- **XGBoost model loads**: `artifacts/models/xgboost/xgboost_model.json` exists and the model service can load it.
- **Frontend runs**: Vite + React + TypeScript compiles; `npm run build` likely passes. Three API calls are wired correctly to `/api/v1/models/*`.
- **FastAPI server starts**: Backend main.py has correct lifespan configuration, CORS, and mounts routers.
- **Database schemas**: PostgreSQL ORM models and Neo4j driver are configured.
- **Benchmark artifacts**: JSON/PNG benchmark outputs exist in `artifacts/eval_results/`.
- **Vite proxy**: `vite.config.ts` correctly proxies `/api` → `localhost:8000` for local development.

---

## 2. Critical Issues

### 2.1 ML — Label Leakage in Graph Entity Features

**File:** `ml/graph_dataset.py` lines 119–146  
**Severity:** CRITICAL — Invalidates all GNN results

Fraud labels (`isFraud`) are aggregated into card/device/email node features *before* any train/test masking is applied. This encodes the ground truth directly into the entity embeddings the GNN uses at test time. The GNN's PR-AUC of 0.165 is likely near-random because the leakage is inconsistently applied, not because it inflates to unrealistic levels.

**Fix required:** Recompute node features using only training-set labels. Apply strict temporal cutoffs.

---

### 2.2 ML — GNN Test Set Is Identical to Validation Set

**File:** `ml/graph_dataset.py` line 86  
**Severity:** CRITICAL — All GNN "test" metrics are actually validation metrics

```python
test_mask = val_mask  # BUG: test == val
```

The GNN has never been evaluated on a truly held-out set. All reported GNN metrics (PR-AUC 0.165, ROC-AUC 0.6804) are validation metrics masquerading as test metrics.

---

### 2.3 ML — No Temporal Split Anywhere

**Files:** `scripts/train_xgboost.py`, `scripts/train_gnn.py`, `ml/preprocessing.py`  
**Severity:** CRITICAL — Invalidates all benchmark comparisons

`TransactionDT` is used only as a raw feature, never to partition data into train/validation/test. Random shuffled splits are used throughout. For a fraud detection system, this causes future transactions to influence historical feature computation and allows the model to "see" future patterns.

**Evidence:** XGBoost test PR-AUC (0.641) > cross-validation PR-AUC (0.544) — consistent with temporal leakage allowing test transactions to inform training features.

---

### 2.4 ML — Full Graph Forward Pass on Every Training Step

**File:** `ml/gnn_model.py` lines 177–190  
**Severity:** CRITICAL — All validation and test node neighborhoods contaminate every gradient step

`NeighborLoader` is imported but never used. The GNN runs a full graph forward pass on every step, meaning test and validation nodes' features flow through training gradients. This is neighborhood leakage.

---

### 2.5 ML — Fabricated Graph Entities in Inference

**File:** `ml/gnn_model.py` lines 352–354  
**Severity:** CRITICAL — Fraudulent demonstration output

When a transaction is not found in the prebuilt graph, the code returns hardcoded fake entities: `"Card #14285"`, `"Windows 10/Chrome"`, `"gmail.com"`. These are presented as genuine graph analysis but are entirely fictional. This is unacceptable for any honest evaluation or presentation.

---

### 2.6 Backend — Ground-Truth Test Labels Drive Live Scoring

**File:** `backend/app/services/model_service.py` line ~180  
**Severity:** CRITICAL — Production scoring uses dataset labels, not model predictions

`meta_cache` is loaded from `test_meta.parquet` and `.isFraud` ground-truth column directly drives `graph_prob` in production scoring. Every live transaction is scored against training labels, not against model inference.

---

### 2.7 Backend — No Saga Between PostgreSQL and Neo4j

**File:** `backend/app/services/transaction_service.py` line ~35  
**Severity:** CRITICAL — Data inconsistency on every Neo4j failure

PostgreSQL commits first; if Neo4j fails, the record exists in PG with no graph node. No compensation, retry, or rollback logic.

---

### 2.8 Backend — Default Fail-Open on Model Load Failure

**File:** `backend/app/services/model_service.py` lines 120–121  
**Severity:** CRITICAL — All transactions return LOW risk if models fail

```python
xgb_prob = 0.05  # fallback
graph_prob = 0.05  # fallback
```

Silent fail-open: if models fail to load, every transaction returns "LOW risk" with no error or warning.

---

### 2.9 Backend — `DEBUG=True` and Hardcoded Credentials in Config Defaults

**File:** `backend/app/core/config.py` lines 13, 16, 20  
**Severity:** CRITICAL — Credentials ship in source code

```python
postgres_url: str = "postgresql://postgres:postgres@localhost/fraudgraph"
secret_key: str = "change_me"
```

Default values with real-looking credentials, `debug=True` default, and `change_me` JWT secret.

---

## 3. High-Severity Issues

### 3.1 ML — 50/50 Hybrid Despite Near-Random GNN

**File:** `ml/hybrid_scorer.py`  
**Severity:** HIGH

Fixed 50/50 averaging between XGBoost (PR-AUC 0.641) and GNN (PR-AUC 0.165) degrades the hybrid to 0.544. The GNN contributes noise, not signal. Weights are not learned from validation data.

### 3.2 ML — `calculate_risk_level()` Inconsistency

**File:** `ml/hybrid_scorer.py` lines 26–32  
**Severity:** HIGH

`calculate_risk_level()` never returns `"CRITICAL"` but `score_transaction()` can return `verdict = "CRITICAL"`. This creates inconsistent API responses.

### 3.3 Backend — `TransactionResponse` Schema Mismatch

**File:** `backend/app/schemas/transaction.py` line 36  
**Severity:** HIGH — Runtime serialization error on every POST response

Schema expects `location: TransactionLocation` (nested object) but ORM model has flat `country`/`region` columns. Every transaction POST response will raise a Pydantic validation error.

### 3.4 Backend — `IntegrityError` on Duplicate Transaction ID

**File:** `backend/app/services/transaction_service.py`  
**Severity:** HIGH

Duplicate `transaction_id` raises uncaught `IntegrityError` → raw 500 response with no meaningful error message.

### 3.5 Backend — No Fraud Scoring on Transaction Ingest

**Severity:** HIGH — Core feature broken

`POST /transactions/` and `POST /models/predict` are completely decoupled. A submitted transaction is never automatically scored. No `isFraud`/`risk_score` column in the ORM model means decisions are never persisted.

### 3.6 Backend — 7 Frontend-Expected Endpoints Missing

| Endpoint | Status |
|----------|--------|
| `GET /api/v1/transactions/` | Missing |
| `GET /api/v1/transactions/{id}` | Missing |
| `GET /api/v1/alerts/` | Missing |
| `GET /api/v1/alerts/{id}` | Missing |
| `POST /api/v1/alerts/{id}/review` | Missing |
| `GET /api/v1/graph/{transaction_id}` | Missing |
| `GET /api/v1/models/status` | Missing |

### 3.7 Frontend — 15+ CSS Classes Referenced but Never Defined

**Files:** `BenchmarkDashboard.tsx`, `RelationalGraphView.tsx`  
**Severity:** HIGH — Components render partially unstyled

Classes including `.loading-state`, `.empty-notice`, `.benchmark-card`, `.metric-val`, `.curve-card`, `.legend-item`, `.interpretability-card` are used in JSX but absent from all stylesheets.

### 3.8 Frontend — SVG Arrows Never Render

**File:** `RelationalGraphView.tsx` lines 94–104  
**Severity:** HIGH

Arrow marker defined in `<defs>` but `markerEnd` is never applied to any `<line>` element. Edge direction indicators never appear.

### 3.9 Frontend — No Router; Only 2 Views

**Severity:** HIGH — Major navigation missing

`react-router-dom` is not installed. The app has only a tab-toggle between "Simulator" and "Benchmark" views. There is no dashboard overview, no transaction investigation page, no graph explorer page, and no URL-based navigation.

### 3.10 Frontend — No Chart Library; Hand-drawn SVG Only

**Severity:** HIGH — Precision-recall and ROC curves have no tooltips, no axes, no interactivity

All charts are hand-drawn SVG. No recharts, d3, or nivo.

---

## 4. Medium-Severity Issues

### 4.1 ML — Card Identity Uses `card1` Alone

**File:** `ml/preprocessing.py`, `ml/graph_dataset.py`  
`card1` alone is used as card identity. The dataset has `card1`–`card6`; using all six as a composite fingerprint reduces false identity merges significantly.

### 4.2 ML — Email Domain Hub Explosion

**File:** `ml/graph_dataset.py`  
Email domains like `gmail.com`, `yahoo.com` connect tens of thousands of transactions. No filtering applied. These hubs degrade GNN message passing.

### 4.3 ML — Unpinned ML Dependencies

**File:** `backend/requirements.txt` lines 91–96  
`torch`, `torch-geometric`, `shap` are unpinned while all other deps are exact-pinned.

### 4.4 Backend — Model Loading Blocks First Request

**File:** `backend/app/services/model_service.py`  
Models load synchronously on first request, not at startup. Combined with no background loading, the first prediction call will have high latency.

### 4.5 Backend — Benchmark Does File I/O on Every Call

**File:** `backend/app/services/model_service.py`  
`get_benchmark_data()` reads JSON files from disk on every API call. No in-memory caching.

### 4.6 Backend — Neo4j Driver Never Closed

**File:** `backend/app/main.py`  
No shutdown lifecycle hook closes the Neo4j driver. Connection pool leaks on restart.

### 4.7 Frontend — `isFraud` Status Pill Always Green/Static

**File:** `BenchmarkDashboard.tsx`  
"IEEE-CIS Pipeline Active" status is always hardcoded green with no backend health check.

### 4.8 Frontend — Floating-Point Bleed on Metric Display

**File:** `BenchmarkDashboard.tsx`  
`row.optimal_threshold` and `test_fraud_rate` rendered raw without `.toFixed()`. Will display `2.7000000001%`.

### 4.9 Frontend — No Production API Base URL

**File:** `vite.config.ts`, `frontend/src/`  
Vite proxy only works in dev. Production builds will make API calls to `/api/...` with no configured base URL. Requires Vercel/Netlify environment variable injection.

---

## 5. Low-Severity Issues

### 5.1 Frontend — External Fonts Never Imported

Google Fonts (`Inter`, `JetBrains Mono`) referenced in CSS variables but no `@import` or `<link>`. Silent fallback to system fonts.

### 5.2 Frontend — Hardcoded "Top Overall ROC" Winner

"Top Overall ROC" winner pill is hardcoded to the Hybrid row, not computed from actual metrics.

### 5.3 Frontend — Bleeding-Edge Dependencies

React 19.2.8, Vite 8.3.0, TypeScript ~6.0.2 (TS6 not stable) may introduce unexpected breakage.

### 5.4 Backend — No Alembic Migrations

Schema has no version control. Dropping and recreating tables is the only migration path.

### 5.5 ML — `torch`, `shap` Unpinned

Minor reproducibility risk; version drift can break GNN loading.

---

## 6. What Can Safely Be Retained

| Component | Status |
|-----------|--------|
| FastAPI app structure (`main.py`, router setup, CORS) | Retain |
| Pydantic schemas (fix mismatch, keep structure) | Retain with fixes |
| PostgreSQL ORM models (fix columns) | Retain with fixes |
| Neo4j driver setup | Retain |
| XGBoost model file | Retrain with correct split |
| Frontend component structure | Retain, expand |
| Vite + React + TypeScript stack | Retain |
| `data/raw/ieee_cis/` raw CSVs | Retain (do not modify) |
| Benchmark artifact structure | Retain, regenerate values |

---

## 7. What Must Be Fixed

| Priority | Component | Action |
|----------|-----------|--------|
| P0 | GNN label leakage | Remove `isFraud` from node features |
| P0 | GNN test==val mask | Create proper chronological test split |
| P0 | No temporal split | Implement TransactionDT-based splits |
| P0 | Fabricated graph entities | Remove hardcoded fake entities |
| P0 | Production scoring from test labels | Fix model_service.py to use model inference |
| P0 | Hybrid weights | Learn weights from validation, not 50/50 |
| P1 | Schema mismatch (TransactionResponse) | Flatten location schema |
| P1 | Missing endpoints | Add transaction list/detail, alert, graph endpoints |
| P1 | Credential defaults | Move to env-only, no defaults |
| P1 | Fail-open on model load | Raise 503, not silent LOW risk |
| P1 | No router | Add react-router-dom |
| P1 | CSS missing classes | Add all referenced classes |
| P2 | No chart library | Add recharts |
| P2 | Hand-drawn SVG graph | Add Cytoscape.js |
| P2 | Font import missing | Add Google Fonts import |
| P2 | Model loading at startup | Add startup event |

---

## 8. What Should Not Be Changed

- Raw data files in `data/raw/ieee_cis/`
- Technology stack (FastAPI, React, PostgreSQL, Neo4j, XGBoost)
- Core API endpoint paths (`/api/v1/models/predict`, `/api/v1/models/benchmark`, `/api/v1/models/sample-transactions`)
- Database hosting providers (Supabase, Neo4j AuraDB)
- Existing test file names (expand them, do not replace)

---

## 9. Implementation Priority Order

```
Phase B: ML correctness (chronological split, label leakage, hybrid)
Phase C: Backend fixes (scoring bug, schema, endpoints, error handling)
Phase D: Frontend (router, charts, Cytoscape, CSS, polish)
Phase E: Testing + validation
Phase F: Deployment docs
```

---

*This audit document was generated from live code inspection. All cited line numbers reflect the codebase as of 2026-11-12.*

