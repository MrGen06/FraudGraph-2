# FraudGraph — Graph-Powered Financial Fraud Detection & Investigation Platform

FraudGraph is an enterprise-grade financial fraud detection, scoring, and investigation platform designed with a clean division of architectural responsibilities:
- **XGBoost = Sole Automated Fraud Predictor**: High-precision tabular gradient boosting with local TreeSHAP feature attributions driving 100% of automated real-time risk scoring and alert generation.
- **Neo4j = Graph-Based Investigation & Exploration**: High-performance property graph visualizer providing 1-hop and 2-hop entity-link analysis (shared devices, composite cards, merchants) for fraud analysts.
- **GNN = Offline Experimental Research Only**: PyTorch Geometric HeteroFraudGNN preserved for offline research and topological experimentation, strictly segregated from live production scoring.

The system is evaluated on the real-world **IEEE-CIS Fraud Detection** benchmark using **strict chronological partitioning** (70% Train, 15% Validation, 15% Test) to guarantee zero temporal data leakage. A forensic diagnosis of GNN performance is documented in docs/GNN_ROOT_CAUSE_ANALYSIS.md.

---

## 🌟 Key Features

1. **Dual-Model Inference Engine**:
   - **XGBoost Classifier**: Fast, tabular risk scoring leveraging card, device, transaction delta, and historical features with local SHAP feature attributions.
   - **Graph Neural Network (GraphSAGE)**: Multi-hop neighborhood aggregation capturing syndicates, device sharing, and card velocity patterns without target-label leakage.
   - **Validation-Tuned Consensus Scorer**: Dynamic ensemble weighting ($w=0.95$ Tabular, $0.05$ Graph) determined strictly on validation data to maximize Precision-Recall AUC (PR-AUC).
2. **Resilient Graph Synchronization**:
   - Ingests transactions with relational consistency in PostgreSQL.
   - Resiliently writes 2-hop entity subgraphs to Neo4j AuraDB with graceful fallback when cloud graph databases are offline.
3. **Automated Alert Workflow**:
   - High and Critical risk transactions automatically trigger actionable alerts.
   - Analyst review lifecycle (`PENDING` → `UNDER_REVIEW` → `RESOLVED` / `DISMISSED`) with reviewer attribution and audit trail.
4. **Interactive Analyst Dashboard**:
   - **Executive Overview**: High-level KPIs, active alert stream, real-time transaction feed, and system health status.
   - **Transaction Investigation**: Filterable, searchable data table with slide-out drawer displaying full transaction metadata, risk breakdowns, and consensus reasoning.
   - **Graph Explorer**: 2-hop entity-relationship network visualizer featuring pan/zoom, node dragging, directed edges, risk halo indicators, and node metadata inspection.
   - **Live Risk Simulator**: Interactive sandbox with preset fraud scenarios and manual parameter tuning for instant dual-model scoring and SHAP waterfall analysis.
   - **Scientific Benchmark**: Transparent performance comparisons (PR-AUC, ROC-AUC, F1, Confusion Matrices, Precision-Recall & ROC curves) on held-out test transactions.

---

## 🏗️ Architecture Overview

```
                      +-----------------------------+
                      |   React 19 + TypeScript     |
                      |   Vite Single Page App      |
                      +--------------+--------------+
                                     |  REST API / JSON
                                     v
                      +-----------------------------+
                      |      FastAPI Backend        |
                      |  (SQLAlchemy + Pydantic v2) |
                      +-------+--------------+------+
                              |              |
         +--------------------+              +--------------------+
         | Ingestion & Alerts                                     | Graph Sync / Queries
         v                                                        v
+------------------+                                     +------------------+
| Supabase / SQL   |                                     | Neo4j AuraDB     |
| (PostgreSQL)     |                                     | (Cypher Graph)   |
+------------------+                                     +------------------+
         ^                                                        ^
         |                     +-------------------+              |
         +---------------------+ Dual-Model Engine +--------------+
                               | - XGBoost (SHAP)  |
                               | - PyG GraphSAGE   |
                               | - Hybrid Ensemble |
                               +-------------------+
```

---

## 📊 Scientific Benchmark Summary

Evaluated on **7,500 held-out chronological test transactions** (1.91% positive fraud rate) from IEEE-CIS:

| Model Architecture | PR-AUC | ROC-AUC | Optimal Threshold* | Fraud F1 | Recall | Precision | FPR @ 95% Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Baseline** | 0.3496 | 0.8931 | 0.0526 | 0.3957 | 0.7483 | 0.2709 | 0.3423 |
| **Graph Neural Network (PyG)** | 0.0400 | 0.6430 | 0.0384 | 0.0716 | 0.5315 | 0.0371 | 0.9416 |
| **Validation-Tuned Hybrid** | **0.3501** | **0.8937** | 0.0528 | **0.3963** | **0.7483** | **0.2716** | **0.3394** |

*\* Thresholds and ensemble weights ($0.95$ Tabular / $0.05$ Graph) were selected strictly on the validation partition without touching test labels.*

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (tested with Python 3.12)
- Node.js 18+ and npm
- PostgreSQL (or local SQLite for development)
- Neo4j AuraDB Free (optional, resilient fallback built-in)

### 1. Environment Setup

Copy and configure environment variables in `backend/.env`:

```env
# Application Settings
ENVIRONMENT=development
PROJECT_NAME="FraudGraph"
API_V1_STR="/api/v1"

# Database Configuration (PostgreSQL / Supabase / SQLite)
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fraudgraph"
# Or for local SQLite testing:
# DATABASE_URL="sqlite:///./fraudgraph.db"

# Neo4j Graph Database Configuration (AuraDB Free or Local)
NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-secure-password"

# Security
SECRET_KEY="replace-with-a-random-secret-key-for-jwt"
```

### 2. Backend Setup & Startup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```
Backend API interactive documentation is available at `http://localhost:8000/docs`.

### 3. Frontend Setup & Startup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Running the Test Suite

```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```

The test suite validates:
- API endpoints (`/transactions/`, `/alerts/`, `/graph/`, `/models/predict`, `/models/status`, `/models/benchmark`).
- Relational schema creation and SQLite/PostgreSQL autoincrement compatibility.
- Alert generation and state transitions during analyst review.
- Resilient Neo4j behavior when graph servers are unreachable.
- Dual-model loading and deterministic inference.

---

## 📁 Repository Structure

```
FraudGraph/
├── artifacts/                  # Benchmark metrics, ROC/PR curves, and trained models
│   ├── eval_results/           # metrics.json, curves_data.json, benchmark_comparison.csv
│   └── models/
│       ├── xgboost/            # Serialized XGBoost model, metadata, and predictions
│       └── gnn/                # Trained PyG GraphSAGE weights, node index mappings
├── backend/                    # FastAPI backend service
│   ├── app/
│   │   ├── api/v1/endpoints/   # transactions, alerts, graph, models
│   │   ├── core/               # Configuration, security, logging
│   │   ├── db/                 # PostgreSQL & Neo4j database sessions
│   │   ├── models/             # SQLAlchemy ORM models (Transaction, Alert)
│   │   ├── schemas/            # Pydantic request/response validation schemas
│   │   └── services/           # Ingestion, graph sync, scoring, alerts
│   ├── requirements.txt        # Backend Python dependencies
│   └── tests/                  # Pytest test suite with in-memory SQLite fixtures
├── data/
│   ├── raw/ieee_cis/           # Raw IEEE-CIS CSVs (read-only)
│   └── processed/              # Chronological train/val/test Parquet partitions
├── docs/                       # Architecture, audit, deployment, and demo scripts
├── frontend/                   # React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── components/         # Overview, Investigation, Graph, Simulator, Benchmark
│   │   ├── api.ts              # Centralized typed HTTP client
│   │   ├── types.ts            # Domain and payload TypeScript interfaces
│   │   └── App.tsx             # 5-view navigation shell
│   └── package.json
├── ml/                         # Machine learning and graph pipelines
│   ├── preprocessing.py        # Composite fingerprinting, categorical encoding
│   ├── graph_dataset.py        # PyG graph construction without label leakage
│   ├── gnn_model.py            # GraphSAGE neural network architecture
│   └── hybrid_scorer.py        # Validation-tuned ensemble optimizer
└── scripts/                    # Ingestion, training, and evaluation scripts
    ├── ingest_ieee_cis.py      # Chronological 70/15/15 dataset partitioning
    ├── train_xgboost.py        # XGBoost training with validation early stopping
    ├── train_gnn.py            # GraphSAGE training on train_mask
    └── evaluate_models.py      # Scientific benchmark evaluation & plot generation
```

---

## 🛡️ License

This project is licensed under the Apache 2.0 License.

