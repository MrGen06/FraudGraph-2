import React, { useEffect, useState } from 'react'
import { fetchModelStatus } from '../api'
import type { BenchmarkData, ModelMetric, ModelSystemStatus } from '../types'

interface Props {
  data: BenchmarkData | null
  isLoading: boolean
}

export const BenchmarkDashboard: React.FC<Props> = ({ data, isLoading }) => {
  const [modelStatus, setModelStatus] = useState<ModelSystemStatus | null>(null)

  useEffect(() => {
    fetchModelStatus()
      .then((res) => setModelStatus(res))
      .catch((err) => console.warn('Could not fetch model status:', err))
  }, [])

  if (isLoading || !data) {
    return (
      <div className="panel loading-panel">
        <div className="spinner" />
        <p>Loading held-out chronological benchmark metrics...</p>
      </div>
    )
  }

  const { metrics, curves, comparison_table } = data
  const rows = comparison_table || metrics.models || []

  // Dynamically compute winners across metrics
  const bestPrAucModel = rows.reduce<ModelMetric | null>(
    (best, cur) => (!best || cur.pr_auc > best.pr_auc ? cur : best),
    null
  )
  const bestRocAucModel = rows.reduce<ModelMetric | null>(
    (best, cur) => (!best || cur.roc_auc > best.roc_auc ? cur : best),
    null
  )

  const testFraudRatePct = (metrics.test_fraud_rate * 100).toFixed(2)

  return (
    <div className="benchmark-container">
      {/* Top Benchmark Summary Banner */}
      <div className="benchmark-header-card">
        <div className="header-meta">
          <span className="badge-pill">
            <span
              className={`status-dot ${modelStatus?.status === 'HEALTHY' ? 'dot-healthy' : 'dot-warning'}`}
            />
            {modelStatus ? `Production Predictor: XGBoost (${modelStatus.status})` : 'XGBoost Active'}
          </span>
          <h2>Empirical Evaluation Benchmark (Held-out Chronological Test)</h2>
          <p className="subtitle">
            Scientific evaluation on {metrics.test_records_count.toLocaleString()} held-out transactions ({testFraudRatePct}% fraud rate). Decision thresholds and ensemble weights strictly tuned on validation split.
          </p>
          <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: '#1E293B', borderRadius: '8px', borderLeft: '4px solid #3B82F6', fontSize: '0.85rem', color: '#CBD5E1' }}>
            <strong style={{ color: '#60A5FA' }}>Production Architecture Decision:</strong> <strong>XGBoost</strong> is the sole automated production fraud predictor (PR-AUC: 0.3496, ROC-AUC: 0.8931). The <strong>Hetero GNN</strong> (PR-AUC: 0.0400) is designated for <em>offline research only</em> due to topological collapse on generic hubs. <strong>Neo4j</strong> is utilized strictly for visual investigator relationship exploration.
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="panel table-card">
        <div className="panel-header">
          <div>
            <h3>Model Architecture Performance Matrix</h3>
            <p className="subtitle">Comparison of Production XGBoost, Offline Experimental GNN, and Historical Hybrid Results</p>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Role in Architecture</th>
                <th>PR-AUC</th>
                <th>ROC-AUC</th>
                <th>Threshold (Val)</th>
                <th>Minority F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>FPR @ 95% Recall</th>
                <th>Deployment Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isXgb = row.model.toLowerCase().includes('xgboost')
                const isGnn = row.model.toLowerCase().includes('gnn')
                const isBestPr = bestPrAucModel?.model === row.model
                const isBestRoc = bestRocAucModel?.model === row.model

                const roleLabel = isXgb
                  ? 'Sole Production Predictor'
                  : isGnn
                  ? 'Offline Research Only'
                  : 'Historical Benchmark Ensemble'

                const statusBadge = isXgb ? (
                  <span className="badge-pill badge-success">Production Active</span>
                ) : isGnn ? (
                  <span className="badge-pill badge-muted">Offline Research Only</span>
                ) : (
                  <span className="badge-pill badge-warning">Historical Reference</span>
                )

                return (
                  <tr key={row.model} className={isXgb ? 'row-winner' : ''}>
                    <td>
                      <div className="model-name-group">
                        <strong>{row.model}</strong>
                        {isXgb && <span className="winner-pill">Production Choice</span>}
                        {isBestPr && !isXgb && <span className="winner-pill">Top PR-AUC</span>}
                        {isBestRoc && !isBestPr && !isXgb && <span className="winner-pill winner-roc">Top ROC</span>}
                      </div>
                    </td>
                    <td>
                      <span className="mono-sub" style={{ fontSize: '0.8rem', color: isXgb ? '#60A5FA' : '#94A3B8' }}>
                        {roleLabel}
                      </span>
                    </td>
                    <td>
                      <span className="mono-val metric-highlight">{row.pr_auc.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono-val">{row.roc_auc.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono-val">
                        {typeof row.optimal_threshold === 'number'
                          ? row.optimal_threshold.toFixed(4)
                          : row.optimal_threshold}
                      </span>
                    </td>
                    <td>
                      <span className="mono-val">{row.f1_minority.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono-val">{row.precision.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono-val">{row.recall.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono-val">{row.fpr_at_95_recall.toFixed(4)}</span>
                    </td>
                    <td>
                      {statusBadge}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Curves & Visualizations */}
      <div className="curves-grid">
        {/* ROC Curve Panel */}
        <div className="panel curve-card">
          <div className="panel-header">
            <h4>Receiver Operating Characteristic (ROC)</h4>
            <span className="info-chip">Held-out Test</span>
          </div>
          <svg viewBox="0 0 400 240" className="chart-svg">
            <line x1="40" y1="20" x2="40" y2="200" stroke="#475569" strokeWidth="1" />
            <line x1="40" y1="200" x2="380" y2="200" stroke="#475569" strokeWidth="1" />
            {/* Diagonal baseline */}
            <line x1="40" y1="200" x2="380" y2="20" stroke="#64748B" strokeDasharray="3 3" />

            {/* XGBoost ROC */}
            {curves?.roc?.fpr_xgb && (
              <polyline
                fill="none"
                stroke="#3B82F6"
                strokeWidth="2.5"
                points={curves.roc.fpr_xgb
                  .map((x, i) => `${40 + x * 340},${200 - curves.roc.tpr_xgb[i] * 180}`)
                  .join(' ')}
              />
            )}
            {/* GNN ROC */}
            {curves?.roc?.fpr_gnn && (
              <polyline
                fill="none"
                stroke="#10B981"
                strokeWidth="2"
                points={curves.roc.fpr_gnn
                  .map((x, i) => `${40 + x * 340},${200 - curves.roc.tpr_gnn[i] * 180}`)
                  .join(' ')}
              />
            )}
            {/* Hybrid ROC */}
            {curves?.roc?.fpr_hyb && (
              <polyline
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="2.5"
                points={curves.roc.fpr_hyb
                  .map((x, i) => `${40 + x * 340},${200 - curves.roc.tpr_hyb[i] * 180}`)
                  .join(' ')}
              />
            )}

            <text x="210" y="225" fill="#94A3B8" fontSize="10" textAnchor="middle">
              False Positive Rate (FPR)
            </text>
            <text x="18" y="110" fill="#94A3B8" fontSize="10" transform="rotate(-90 18,110)" textAnchor="middle">
              True Positive Rate (TPR)
            </text>
          </svg>

          <div className="curve-legend">
            <span className="legend-item"><span className="color-dot dot-xgb" /> XGBoost</span>
            <span className="legend-item"><span className="color-dot dot-gnn" /> GNN</span>
            <span className="legend-item"><span className="color-dot dot-hyb" /> Hybrid (Ensemble)</span>
          </div>
        </div>

        {/* Precision-Recall Curve Panel */}
        <div className="panel curve-card">
          <div className="panel-header">
            <h4>Precision-Recall (PR) Curve</h4>
            <span className="info-chip">Class Imbalance Focus</span>
          </div>
          <svg viewBox="0 0 400 240" className="chart-svg">
            <line x1="40" y1="20" x2="40" y2="200" stroke="#475569" strokeWidth="1" />
            <line x1="40" y1="200" x2="380" y2="200" stroke="#475569" strokeWidth="1" />

            {/* Baseline horizontal */}
            <line
              x1="40"
              y1={200 - metrics.test_fraud_rate * 180}
              x2="380"
              y2={200 - metrics.test_fraud_rate * 180}
              stroke="#64748B"
              strokeDasharray="3 3"
            />

            {/* XGBoost PR */}
            {curves?.pr?.rec_xgb && (
              <polyline
                fill="none"
                stroke="#3B82F6"
                strokeWidth="2.5"
                points={curves.pr.rec_xgb
                  .map((x, i) => `${40 + x * 340},${200 - curves.pr.prec_xgb[i] * 180}`)
                  .join(' ')}
              />
            )}
            {/* GNN PR */}
            {curves?.pr?.rec_gnn && (
              <polyline
                fill="none"
                stroke="#10B981"
                strokeWidth="2"
                points={curves.pr.rec_gnn
                  .map((x, i) => `${40 + x * 340},${200 - curves.pr.prec_gnn[i] * 180}`)
                  .join(' ')}
              />
            )}
            {/* Hybrid PR */}
            {curves?.pr?.rec_hyb && (
              <polyline
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="2.5"
                points={curves.pr.rec_hyb
                  .map((x, i) => `${40 + x * 340},${200 - curves.pr.prec_hyb[i] * 180}`)
                  .join(' ')}
              />
            )}

            <text x="210" y="225" fill="#94A3B8" fontSize="10" textAnchor="middle">
              Recall
            </text>
            <text x="18" y="110" fill="#94A3B8" fontSize="10" transform="rotate(-90 18,110)" textAnchor="middle">
              Precision
            </text>
          </svg>

          <div className="curve-legend">
            <span className="legend-item"><span className="color-dot dot-xgb" /> XGBoost</span>
            <span className="legend-item"><span className="color-dot dot-gnn" /> GNN</span>
            <span className="legend-item"><span className="color-dot dot-hyb" /> Hybrid (Ensemble)</span>
            <span className="legend-item"><span className="color-dot dot-base" /> Baseline ({testFraudRatePct}%)</span>
          </div>
        </div>
      </div>

      {/* Confusion Matrix Section */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Confusion Matrices @ Optimal Validation Thresholds</h3>
            <p className="subtitle">Breakdown of True Negatives, False Positives, False Negatives, and True Positives</p>
          </div>
        </div>

        <div className="cm-grid">
          {rows.map((row) => {
            const cm = row.confusion_matrix || [[0, 0], [0, 0]]
            const tn = cm[0]?.[0] || 0
            const fp = cm[0]?.[1] || 0
            const fn = cm[1]?.[0] || 0
            const tp = cm[1]?.[1] || 0

            return (
              <div key={row.model} className="cm-card">
                <div className="cm-header">
                  <strong>{row.model}</strong>
                  <span className="thresh-chip">Thresh: {row.optimal_threshold.toFixed(3)}</span>
                </div>
                <div className="cm-box-grid">
                  <div className="cm-cell cell-tn">
                    <span className="cell-num">{tn.toLocaleString()}</span>
                    <span className="cell-label">True Negative (Legit)</span>
                  </div>
                  <div className="cm-cell cell-fp">
                    <span className="cell-num">{fp.toLocaleString()}</span>
                    <span className="cell-label">False Positive (False Alarm)</span>
                  </div>
                  <div className="cm-cell cell-fn">
                    <span className="cell-num">{fn.toLocaleString()}</span>
                    <span className="cell-label">False Negative (Missed)</span>
                  </div>
                  <div className="cm-cell cell-tp">
                    <span className="cell-num">{tp.toLocaleString()}</span>
                    <span className="cell-label">True Positive (Fraud Caught)</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
