import { useEffect, useState } from 'react'
import './App.css'
import {
  fetchBenchmarkData,
  fetchModelStatus,
  fetchSamplePresets,
  predictTransaction,
} from './api'
import { BenchmarkDashboard } from './components/BenchmarkDashboard'
import { ConsensusBanner } from './components/ConsensusBanner'
import { DashboardOverview } from './components/DashboardOverview'
import { DualRiskScorecard } from './components/DualRiskScorecard'
import { RelationalGraphView } from './components/RelationalGraphView'
import { ShapFeatureChart } from './components/ShapFeatureChart'
import { TransactionInvestigation } from './components/TransactionInvestigation'
import { TransactionSimulator } from './components/TransactionSimulator'
import type {
  BenchmarkData,
  ModelSystemStatus,
  PredictResponse,
  SamplePreset,
  TransactionPayload,
} from './types'

const DEFAULT_PAYLOAD: TransactionPayload = {
  transaction_id: 'TX_SIM_1001',
  amount: 148.5,
  card_id: '13926',
  card_type: 'credit',
  card_network: 'visa',
  device_id: 'Windows 10/Chrome',
  device_type: 'desktop',
  email_domain: 'gmail.com',
  hour: 14,
  day_of_week: 2,
  dist1: 15.0,
  product_cd: 'W',
}

export type TabKey = 'overview' | 'transactions' | 'graph' | 'simulator' | 'benchmark'

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [presets, setPresets] = useState<SamplePreset[]>([])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [payload, setPayload] = useState<TransactionPayload>(DEFAULT_PAYLOAD)
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelSystemStatus | null>(null)
  const [selectedTxId, setSelectedTxId] = useState<string>('')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isBenchmarkLoading, setIsBenchmarkLoading] = useState<boolean>(false)
  const [apiError, setApiError] = useState<string | null>(null)

  // 1. Initial status and presets fetch
  useEffect(() => {
    const initData = async () => {
      try {
        const [statusRes, presetsRes] = await Promise.allSettled([
          fetchModelStatus(),
          fetchSamplePresets(),
        ])

        if (statusRes.status === 'fulfilled') {
          setModelStatus(statusRes.value)
        }

        if (presetsRes.status === 'fulfilled' && presetsRes.value.length > 0) {
          const fetchedPresets = presetsRes.value
          setPresets(fetchedPresets)
          setSelectedPresetId(fetchedPresets[0].id)
          setPayload(fetchedPresets[0].payload)
          executePrediction(fetchedPresets[0].payload)
        } else {
          executePrediction(DEFAULT_PAYLOAD)
        }
      } catch (err: any) {
        console.warn('Initial system fetch warning:', err)
        executePrediction(DEFAULT_PAYLOAD)
      }
    }

    initData()
  }, [])

  // 2. Fetch benchmark data when benchmark or overview tab accessed
  useEffect(() => {
    if (!benchmarkData && (activeTab === 'benchmark' || activeTab === 'overview')) {
      const loadBenchmark = async () => {
        setIsBenchmarkLoading(true)
        try {
          const data = await fetchBenchmarkData()
          setBenchmarkData(data)
        } catch (err: any) {
          console.error('Benchmark fetch error:', err)
        } finally {
          setIsBenchmarkLoading(false)
        }
      }
      loadBenchmark()
    }
  }, [activeTab, benchmarkData])

  const executePrediction = async (currentPayload: TransactionPayload) => {
    setIsLoading(true)
    setApiError(null)
    try {
      const data = await predictTransaction(currentPayload)
      setPrediction(data)
    } catch (err: any) {
      console.error('Inference request error:', err)
      setApiError(err.message || 'Error connecting to inference backend.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectPreset = (preset: SamplePreset) => {
    setSelectedPresetId(preset.id)
    setPayload(preset.payload)
    executePrediction(preset.payload)
  }

  const handleNavigateToGraph = (txId: string) => {
    setSelectedTxId(txId)
    setActiveTab('graph')
  }

  return (
    <div className="app-shell">
      {/* Top Global Navigation Bar */}
      <header className="app-header">
        <div className="header-brand" onClick={() => setActiveTab('overview')} style={{ cursor: 'pointer' }}>
          <div className="logo-symbol">FG</div>
          <div>
            <h1 className="brand-title">FraudGraph</h1>
            <p className="brand-subtitle">Graph-Powered Financial Fraud Detection & Investigation</p>
          </div>
        </div>

        <div className="header-controls">
          {/* Live System Status Pill */}
          <div
            className={`status-pill ${
              modelStatus?.status === 'HEALTHY' ? 'status-pill-healthy' : 'status-pill-degraded'
            }`}
            title={`XGBoost: ${modelStatus?.models?.xgboost?.loaded ? 'Loaded' : 'Offline'} | GNN: ${
              modelStatus?.models?.gnn?.loaded ? 'Loaded' : 'Offline'
            }`}
          >
            <span
              className={`status-dot ${
                modelStatus?.status === 'HEALTHY' ? 'dot-healthy' : 'dot-warning'
              }`}
            />
            <span>
              {modelStatus?.status === 'HEALTHY'
                ? 'Dual-Model Engine Online'
                : 'System Active (Degraded)'}
            </span>
          </div>

          {/* 5-Tab Navigation Bar */}
          <nav className="tab-switcher" aria-label="Main Navigation">
            <button
              type="button"
              className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              📊 Overview
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === 'transactions' ? 'active' : ''}`}
              onClick={() => setActiveTab('transactions')}
            >
              💳 Transactions
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
              onClick={() => setActiveTab('graph')}
            >
              🕸️ Graph Explorer
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
              onClick={() => setActiveTab('simulator')}
            >
              ⚡ Risk Inspector
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
              onClick={() => setActiveTab('benchmark')}
            >
              📈 Benchmark
            </button>
          </nav>
        </div>
      </header>

      {/* Main View Container */}
      <main className="app-main">
        {apiError && (
          <div className="alert-banner">
            <span>⚠️ {apiError}</span>
            <button type="button" onClick={() => setApiError(null)}>
              ✕
            </button>
          </div>
        )}

        {/* Tab 1: Executive Overview */}
        {activeTab === 'overview' && (
          <DashboardOverview
            onSelectTransaction={handleNavigateToGraph}
            onNavigateTab={setActiveTab}
          />
        )}

        {/* Tab 2: Transaction Investigation */}
        {activeTab === 'transactions' && (
          <TransactionInvestigation onInvestigateGraph={handleNavigateToGraph} />
        )}

        {/* Tab 3: Interactive Graph Explorer */}
        {activeTab === 'graph' && (
          <div className="graph-explorer-wrapper">
            <RelationalGraphView
              focusedTransactionId={selectedTxId}
              initialExplanation={prediction?.graph_explanation}
            />
          </div>
        )}

        {/* Tab 4: Live Transaction Simulator & Dual Risk Inspector */}
        {activeTab === 'simulator' && (
          <div className="inspector-layout">
            <aside className="inspector-sidebar">
              <TransactionSimulator
                presets={presets}
                selectedPresetId={selectedPresetId}
                payload={payload}
                onChange={setPayload}
                onSelectPreset={handleSelectPreset}
                onSubmit={() => executePrediction(payload)}
                isLoading={isLoading}
              />
            </aside>

            <section className="inspector-results">
              {prediction ? (
                <>
                  <ConsensusBanner
                    verdict={prediction.consensus_verdict}
                    reasoning={prediction.consensus_reasoning}
                    hybridScore={prediction.hybrid_score}
                  />

                  <DualRiskScorecard
                    tabularScore={prediction.tabular_score}
                    tabularLevel={prediction.tabular_risk_level}
                    graphScore={prediction.graph_score}
                    graphLevel={prediction.graph_risk_level}
                    topTabularFeature={prediction.shap_attributions[0]?.description}
                    graphHopCount={prediction.graph_explanation.shared_entity_peers_count ?? 0}
                    flaggedNeighbors={prediction.graph_explanation.flagged_neighbor_fraud_count ?? 0}
                  />

                  <div className="interpretability-grid">
                    <ShapFeatureChart attributions={prediction.shap_attributions} />
                    <RelationalGraphView initialExplanation={prediction.graph_explanation} />
                  </div>
                </>
              ) : (
                <div className="empty-results-placeholder">
                  <div className="spinner" />
                  <p>Awaiting transaction payload analysis...</p>
                </div>
              )}
            </section>
          </div>
        )}

        {/* Tab 5: Benchmark & Scientific Evaluation */}
        {activeTab === 'benchmark' && (
          <BenchmarkDashboard data={benchmarkData} isLoading={isBenchmarkLoading} />
        )}
      </main>
    </div>
  )
}

export default App
