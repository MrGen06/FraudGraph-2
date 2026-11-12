# FraudGraph: Cloud Deployment Guide

This guide details how to deploy FraudGraph to cloud hosting platforms (**Render / Railway** for the backend, and **Vercel / Netlify** for the frontend) using existing **Supabase** (PostgreSQL) and **Neo4j AuraDB** cloud services.

---

## 1. Architecture & Cloud Services

```
+------------------------------------+      +-----------------------------------+
|          Frontend Hosting          |      |          Backend Hosting          |
|         (Vercel / Netlify)         |      |         (Render / Railway)        |
|                                    |      |                                   |
| - React 19 + TypeScript + Vite     | ---> | - Python 3.12 + FastAPI           |
| - Environment: VITE_API_URL        |      | - Uvicorn Web Server              |
+------------------------------------+      +-----------------+-----------------+
                                                              |
                                            +-----------------+-----------------+
                                            |                                   |
                                            v                                   v
                             +-----------------------------+     +-----------------------------+
                             |     Supabase PostgreSQL     |     |       Neo4j AuraDB Free     |
                             |      (Relational DB)        |     |       (Graph Database)      |
                             +-----------------------------+     +-----------------------------+
```

---

## 2. Backend Deployment (Render or Railway)

### 2.1 Render Deployment
1. **Create New Web Service**:
   - Connect your GitHub repository.
   - Select **Root Directory**: `backend`.
   - **Environment**: `Python 3`.
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
2. **Environment Variables**:
   Add the following in Render's **Environment** tab:
   ```env
   ENVIRONMENT=production
   PROJECT_NAME=FraudGraph
   API_V1_STR=/api/v1
   DATABASE_URL=postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres
   NEO4J_URI=neo4j+s://[INSTANCE_ID].databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=[NEO4J_PASSWORD]
   SECRET_KEY=[GENERATE_A_SECURE_RANDOM_STRING]
   ```

### 2.2 Railway Deployment
1. Click **New Project** → **Deploy from GitHub Repo**.
2. Set **Root Directory** to `backend`.
3. Add environment variables matching the Render configuration above.
4. Railway will automatically detect the `requirements.txt` and launch the service.

---

## 3. Frontend Deployment (Vercel or Netlify)

### 3.1 Vercel Deployment
1. Import repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. **Framework Preset**: `Vite`.
4. **Build Command**: `npm run build`
5. **Output Directory**: `dist`
6. **Environment Variables**:
   ```env
   VITE_API_URL=https://your-backend-service.onrender.com
   ```
7. Click **Deploy**.

### 3.2 Netlify Deployment
1. Create a `frontend/netlify.toml` or set via UI:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
2. In Netlify's **Site settings** → **Environment variables**:
   ```env
   VITE_API_URL=https://your-backend-service.railway.app
   ```
3. Add single-page app rewrite rule in `frontend/public/_redirects`:
   ```
   /*    /index.html   200
   ```

---

## 4. Post-Deployment Verification

1. **Verify Backend Health**:
   ```bash
   curl https://your-backend.onrender.com/health
   # Expected: {"status":"ok"}
   ```
2. **Verify ML Status**:
   ```bash
   curl https://your-backend.onrender.com/api/v1/models/status
   # Expected: {"status":"HEALTHY","production_architecture":{"primary_predictor":"XGBoost","role":"SOLE_AUTOMATED_PREDICTOR"},"models":{"xgboost":{"loaded":true,"role":"PRODUCTION_PRIMARY"}}}
   ```
3. **Verify Frontend**:
   - Open your Vercel URL (e.g. `https://fraudgraph.vercel.app`).
   - Check that the top status pill displays **XGBoost Inference Engine Online**.
   - Navigate across all 5 views (**Overview**, **Transactions**, **Graph Explorer**, **Risk Inspector**, **Benchmark**).
   - In **Risk Inspector**, trigger a simulation and confirm XGBoost prediction and SHAP waterfall attributions render within <100ms.

