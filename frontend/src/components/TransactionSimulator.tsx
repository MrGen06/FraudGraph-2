import React from 'react'
import type { SamplePreset, TransactionPayload } from '../types'

interface Props {
  presets: SamplePreset[]
  selectedPresetId: string
  payload: TransactionPayload
  onChange: (updated: TransactionPayload) => void
  onSelectPreset: (preset: SamplePreset) => void
  onSubmit: () => void
  isLoading: boolean
}

export const TransactionSimulator: React.FC<Props> = ({
  presets,
  selectedPresetId,
  payload,
  onChange,
  onSelectPreset,
  onSubmit,
  isLoading,
}) => {
  const handleChange = (field: keyof TransactionPayload, val: any) => {
    onChange({
      ...payload,
      [field]: val,
    })
  }

  return (
    <div className="panel simulator-card">
      <div className="panel-header">
        <div className="header-title-group">
          <span className="icon-badge">⚡</span>
          <div>
            <h3>Live Transaction Simulator</h3>
            <p className="subtitle">Select a scenario preset or manually customize parameters</p>
          </div>
        </div>
      </div>

      {/* Preset Scenario Selector */}
      <div className="presets-container">
        <label className="field-label">Scenario Presets</label>
        <div className="preset-buttons-grid">
          {presets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`preset-btn ${selectedPresetId === preset.id ? 'active' : ''}`}
              onClick={() => onSelectPreset(preset)}
            >
              <div className="preset-btn-title">{preset.name}</div>
              <div className="preset-btn-desc">{preset.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Form Fields */}
      <div className="form-grid">
        <div className="form-group">
          <label>Transaction Amount ($)</label>
          <input
            type="number"
            step="0.01"
            min="0.01"
            value={payload.amount}
            onChange={(e) => handleChange('amount', parseFloat(e.target.value) || 0)}
          />
        </div>

        <div className="form-group">
          <label>Card ID</label>
          <input
            type="text"
            value={payload.card_id}
            onChange={(e) => handleChange('card_id', e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Card Network</label>
          <select
            value={payload.card_network}
            onChange={(e) => handleChange('card_network', e.target.value)}
          >
            <option value="visa">Visa</option>
            <option value="mastercard">Mastercard</option>
            <option value="discover">Discover</option>
            <option value="american express">American Express</option>
          </select>
        </div>

        <div className="form-group">
          <label>Card Type</label>
          <select
            value={payload.card_type}
            onChange={(e) => handleChange('card_type', e.target.value)}
          >
            <option value="credit">Credit</option>
            <option value="debit">Debit</option>
          </select>
        </div>

        <div className="form-group">
          <label>Device Fingerprint</label>
          <input
            type="text"
            value={payload.device_id}
            onChange={(e) => handleChange('device_id', e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Device Type</label>
          <select
            value={payload.device_type}
            onChange={(e) => handleChange('device_type', e.target.value)}
          >
            <option value="desktop">Desktop</option>
            <option value="mobile">Mobile</option>
          </select>
        </div>

        <div className="form-group">
          <label>Purchaser Email Domain</label>
          <input
            type="text"
            value={payload.email_domain}
            onChange={(e) => handleChange('email_domain', e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Hour of Day (0 - 23)</label>
          <input
            type="number"
            min="0"
            max="23"
            value={payload.hour}
            onChange={(e) => handleChange('hour', parseInt(e.target.value, 10) || 0)}
          />
        </div>

        <div className="form-group">
          <label>Billing-to-Delivery Distance (miles)</label>
          <input
            type="number"
            step="0.5"
            min="0"
            value={payload.dist1}
            onChange={(e) => handleChange('dist1', parseFloat(e.target.value) || 0)}
          />
        </div>

        <div className="form-group">
          <label>Product Category Code</label>
          <select
            value={payload.product_cd}
            onChange={(e) => handleChange('product_cd', e.target.value)}
          >
            <option value="W">W (Web Retail)</option>
            <option value="C">C (Credit / Cash Advance)</option>
            <option value="H">H (High-Risk Merchant)</option>
            <option value="R">R (Restricted)</option>
            <option value="S">S (Service)</option>
          </select>
        </div>
      </div>

      <div className="simulator-actions">
        <button
          type="button"
          className="run-analysis-btn"
          disabled={isLoading}
          onClick={onSubmit}
        >
          {isLoading ? (
            <span className="btn-content">
              <span className="spinner"></span> Running Automated XGBoost Scoring...
            </span>
          ) : (
            <span className="btn-content">
              <span>Execute Automated Risk Scoring (XGBoost)</span>
              <span className="btn-arrow">→</span>
            </span>
          )}
        </button>
      </div>
    </div>
  )
}
