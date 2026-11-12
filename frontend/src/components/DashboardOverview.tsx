import React, { useEffect, useState } from 'react'
import { fetchAlerts, fetchBenchmarkData, fetchModelStatus, fetchTransactions } from '../api'
import type { AlertRecord, BenchmarkData, ModelSystemStatus, TransactionRecord } from '../types'

interface Props {
  onSelectTransaction: (txId: string) => void
  onNavigateTab: (tab: 'transactions' | 'graph' | 'simulator' | 'benchmark') => void
}

export const DashboardOverview: React.FC<Props> = ({ onSelectTransaction, onNavigateTab }) => {
  const [transactions, setTransactions] = useState<TransactionRecord[]>([])
  const [alerts, setAlerts] = useState<AlertRecord[]>([])
  const [modelStatus, setModelStatus] = useState<ModelSystemStatus | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)

  useEffect(() => {
    const loadOverviewData = async () => {
      setIsLoading(true)
      try {
        const [txs, alts, status, bench] = await Promise.allSettled([
          fetchTransactions({ limit: 10 }),
          fetchAlerts({ limit: 6 }),
          fetchModelStatus(),
          fetchBenchmarkData(),
        ])

        if (txs.status === 'fulfilled') setTransactions(txs.value)
        if (alts.status === 'fulfilled') setAlerts(alts.value)
        if (status.status === 'fulfilled') setModelStatus(status.value)
        if (bench.status === 'fulfilled') setBenchmark(bench.value)
      } catch (err) {
        console.error('Error loading dashboard overview:', err)
      } finally {
        setIsLoading(false)
      }
    }

    loadOverviewData()
  }, [])

  const totalFlagged = transactions.filter((t) => t.is_flagged || t.risk_level === 'CRITICAL').length
  const pendingAlerts = alerts.filter((a) => a.status === 'PENDING').length
  const testFraudRate = benchmark?.metrics.test_fraud_rate
    ? (benchmark.metrics.test_fraud_rate * 100).toFixed(2)
    : '1.91'

  return (
    <div className="overview-container">
      {/* KPI Cards Row */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">INGESTED TRANSACTIONS</div>
          <div className="kpi-value">{transactions.length > 0 ? `${transactions.length}+` : '0'}</div>
          <div className="kpi-subtext">Real-time ledger stream</div>
        </div>

        <div className="kpi-card border-danger">
          <div className="kpi-label">FLAGGED HIGH RISK</div>
          <div className="kpi-value text-danger">{totalFlagged}</div>
          <div className="kpi-subtext">Requires analyst review</div>
        </div>

        <div className="kpi-card border-warning">
          <div className="kpi-label">ACTIVE FRAUD ALERTS</div>
          <div className="kpi-value text-warning">{pendingAlerts}</div>
          <div className="kpi-subtext">Pending disposition</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">EMPIRICAL FRAUD RATE</div>
          <div className="kpi-value text-cyan">{testFraudRate}%</div>
          <div className="kpi-subtext">IEEE-CIS held-out benchmark</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">ML INFERENCE ENGINE</div>
          <div className="kpi-value">
            <span
              className={`status-dot ${modelStatus?.status === 'HEALTHY' ? 'dot-healthy' : 'dot-warning'}`}
            />
            {modelStatus?.status || 'ONLINE'}
          </div>
          <div className="kpi-subtext">
            XGBoost (Sole Automated Predictor) + Neo4j Graph
          </div>
        </div>
      </div>

      {/* Main Grid: Alerts Stream & Recent Transactions */}
      <div className="overview-two-col">
        {/* Left Column: Alerts Stream */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>🚨 Active Fraud Alerts</h3>
              <p className="subtitle">Real-time alerts triggered by XGBoost fraud risk scoring</p>
            </div>
            <button
              type="button"
              className="action-link-btn"
              onClick={() => onNavigateTab('transactions')}
            >
              View All &rarr;
            </button>
          </div>

          {alerts.length === 0 ? (
            <div className="empty-notice">
              {isLoading ? 'Loading active alerts...' : 'No active alerts in queue. All clear.'}
            </div>
          ) : (
            <div className="alert-list">
              {alerts.map((alt) => (
                <div key={alt.alert_id} className={`alert-item alert-${alt.risk_level.toLowerCase()}`}>
                  <div className="alert-header">
                    <span className={`risk-badge badge-${alt.risk_level.toLowerCase()}`}>
                      {alt.risk_level} ({(alt.risk_score * 100).toFixed(1)}%)
                    </span>
                    <span className="alert-id">{alt.alert_id}</span>
                    <span className={`status-pill status-${alt.status.toLowerCase()}`}>
                      {alt.status}
                    </span>
                  </div>
                  <p className="alert-reason">{alt.reasons[0] || 'XGBoost fraud anomaly'}</p>
                  <div className="alert-actions">
                    <button
                      type="button"
                      className="btn-tiny"
                      onClick={() => onSelectTransaction(alt.transaction_id)}
                    >
                      Investigate Graph &rarr;
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column: Recent Activity */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>💳 Recent Ledger Activity</h3>
              <p className="subtitle">Latest financial transactions scored by FraudGraph</p>
            </div>
            <button
              type="button"
              className="btn-primary-sm"
              onClick={() => onNavigateTab('simulator')}
            >
              + Simulate New Tx
            </button>
          </div>

          {transactions.length === 0 ? (
            <div className="empty-notice">
              {isLoading ? 'Connecting to ledger stream...' : 'No transactions recorded yet.'}
            </div>
          ) : (
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Tx ID</th>
                    <th>Amount</th>
                    <th>User / Account</th>
                    <th>Risk</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.slice(0, 8).map((tx) => (
                    <tr key={tx.transaction_id}>
                      <td>
                        <strong>{tx.transaction_id}</strong>
                      </td>
                      <td>
                        ${Number(tx.amount).toFixed(2)} {tx.currency}
                      </td>
                      <td>
                        <span className="mono-sub">{tx.user_id}</span>
                      </td>
                      <td>
                        <span
                          className={`risk-badge badge-${(tx.risk_level || 'low').toLowerCase()}`}
                        >
                          {tx.risk_level || 'LOW'}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn-table-action"
                          onClick={() => onSelectTransaction(tx.transaction_id)}
                        >
                          Graph &rarr;
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

