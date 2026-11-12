import React from 'react'
import type { ShapAttribution } from '../types'

interface Props {
  attributions: ShapAttribution[]
}

export const ShapFeatureChart: React.FC<Props> = ({ attributions }) => {
  if (!attributions || attributions.length === 0) {
    return (
      <div className="panel interpretability-card">
        <h3>Tabular Interpretability (SHAP)</h3>
        <p className="empty-notice">No feature attributions available for this transaction.</p>
      </div>
    )
  }

  const maxMagnitude = Math.max(...attributions.map((a) => a.magnitude), 0.1)

  return (
    <div className="panel interpretability-card">
      <div className="panel-header">
        <div className="header-title-group">
          <span className="icon-badge">📊</span>
          <div>
            <h3>Tabular Risk Drivers (SHAP TreeExplainer)</h3>
            <p className="subtitle">
              Local feature attributions pushing fraud risk up (red) or down (green)
            </p>
          </div>
        </div>
      </div>

      <div className="shap-bars-container">
        {attributions.map((attr, idx) => {
          const isRiskUp = attr.direction === 'risk_increasing'
          const barWidthPercent = (attr.magnitude / maxMagnitude) * 100

          return (
            <div key={`${attr.feature}-${idx}`} className="shap-item">
              <div className="shap-meta-row">
                <span className="shap-feature-name" title={attr.feature}>
                  {attr.description}
                </span>
                <span className="shap-feature-value">
                  val: {typeof attr.value === 'number' ? attr.value.toFixed(2) : attr.value}
                </span>
                <span className={`shap-val-badge ${isRiskUp ? 'risk-up' : 'risk-down'}`}>
                  {isRiskUp ? '+' : ''}
                  {attr.shap_value.toFixed(3)}
                </span>
              </div>

              <div className="shap-bar-track">
                <div
                  className={`shap-bar-fill ${isRiskUp ? 'fill-red' : 'fill-green'}`}
                  style={{ width: `${Math.max(6, barWidthPercent)}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="shap-legend">
        <span className="legend-chip chip-red">■ Increases Fraud Probability</span>
        <span className="legend-chip chip-green">■ Lowers Fraud Probability</span>
      </div>
    </div>
  )
}
