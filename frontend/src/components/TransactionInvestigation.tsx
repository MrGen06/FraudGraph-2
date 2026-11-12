import React, { useEffect, useState } from 'react'
import { fetchTransactions } from '../api'
import type { TransactionRecord } from '../types'

interface Props {
  onInvestigateGraph: (txId: string) => void
}

export const TransactionInvestigation: React.FC<Props> = ({ onInvestigateGraph }) => {
  const [transactions, setTransactions] = useState<TransactionRecord[]>([])
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [riskFilter, setRiskFilter] = useState<string>('ALL')
  const [page, setPage] = useState<number>(0)
  const [selectedTx, setSelectedTx] = useState<TransactionRecord | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const pageSize = 15

  const loadTransactions = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await fetchTransactions({
        skip: page * pageSize,
        limit: pageSize,
        risk_level: riskFilter !== 'ALL' ? riskFilter : undefined,
        search: searchTerm.trim() ? searchTerm.trim() : undefined,
      })
      setTransactions(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load transactions')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadTransactions()
  }, [page, riskFilter])

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setPage(0)
    loadTransactions()
  }

  return (
    <div className="investigation-container">
      <div className="panel filter-panel">
        <div className="filter-row">
          <form onSubmit={handleSearchSubmit} className="search-form">
            <input
              type="text"
              placeholder="Search by Transaction ID, User ID, Card..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            <button type="submit" className="btn-search">
              Search
            </button>
          </form>

          <div className="filter-pills">
            <span className="filter-label">Filter Risk:</span>
            {['ALL', 'LOW', 'MEDIUM', 'CRITICAL'].map((tier) => (
              <button
                key={tier}
                type="button"
                className={`pill-btn ${riskFilter === tier ? 'active' : ''}`}
                onClick={() => {
                  setRiskFilter(tier)
                  setPage(0)
                }}
              >
                {tier}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="panel table-card">
        <div className="panel-header">
          <div>
            <h3>Ledger Investigation Records</h3>
            <p className="subtitle">Inspect individual transactions, contributing signals, and entity links</p>
          </div>
          <div className="pagination-controls">
            <button
              type="button"
              className="btn-tiny"
              disabled={page === 0 || isLoading}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              &larr; Prev
            </button>
            <span className="page-indicator">Page {page + 1}</span>
            <button
              type="button"
              className="btn-tiny"
              disabled={transactions.length < pageSize || isLoading}
              onClick={() => setPage((p) => p + 1)}
            >
              Next &rarr;
            </button>
          </div>
        </div>

        {error && <div className="error-notice">{error}</div>}

        {isLoading ? (
          <div className="loading-state">Querying database ledger...</div>
        ) : transactions.length === 0 ? (
          <div className="empty-notice">No transactions match your search criteria.</div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Amount</th>
                  <th>User ID</th>
                  <th>Card Fingerprint</th>
                  <th>Device</th>
                  <th>Risk Tier</th>
                  <th>Verdict</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx) => (
                  <tr
                    key={tx.transaction_id}
                    className={selectedTx?.transaction_id === tx.transaction_id ? 'row-selected' : ''}
                    onClick={() => setSelectedTx(tx)}
                  >
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
                      <span className="mono-sub">
                        {tx.card_id ? `Card #${tx.card_id.slice(0, 16)}` : 'N/A'}
                      </span>
                    </td>
                    <td>{tx.device_id ? tx.device_id.slice(0, 16) : 'Unspecified'}</td>
                    <td>
                      <span
                        className={`risk-badge badge-${(tx.risk_level || 'low').toLowerCase()}`}
                      >
                        {tx.risk_level || 'LOW'} ({(Number(tx.risk_score || 0.05) * 100).toFixed(1)}%)
                      </span>
                    </td>
                    <td>
                      <span className="verdict-label">
                        {tx.consensus_verdict || 'LOW'}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn-table-action"
                        onClick={(e) => {
                          e.stopPropagation()
                          onInvestigateGraph(tx.transaction_id)
                        }}
                      >
                        Graph Canvas &rarr;
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Slide-out detail drawer */}
      {selectedTx && (
        <div className="detail-drawer">
          <div className="drawer-header">
            <h4>Transaction Inspection: {selectedTx.transaction_id}</h4>
            <button
              type="button"
              className="close-detail-btn"
              onClick={() => setSelectedTx(null)}
            >
              ✕
            </button>
          </div>
          <div className="drawer-body">
            <div className="detail-row">
              <span>Amount:</span>
              <strong>${Number(selectedTx.amount).toFixed(2)} {selectedTx.currency}</strong>
            </div>
            <div className="detail-row">
              <span>User ID:</span>
              <strong>{selectedTx.user_id}</strong>
            </div>
            <div className="detail-row">
              <span>Source Account:</span>
              <strong>{selectedTx.source_account_id}</strong>
            </div>
            <div className="detail-row">
              <span>Destination Account:</span>
              <strong>{selectedTx.destination_account_id}</strong>
            </div>
            <div className="detail-row">
              <span>Card ID:</span>
              <strong>{selectedTx.card_id || 'Not Provided'}</strong>
            </div>
            <div className="detail-row">
              <span>Device Fingerprint:</span>
              <strong>{selectedTx.device_id || 'Not Provided'}</strong>
            </div>
            <div className="detail-row">
              <span>Location:</span>
              <strong>
                {selectedTx.country || 'US'} {selectedTx.region ? `(${selectedTx.region})` : ''}
              </strong>
            </div>
            <div className="detail-row">
              <span>XGBoost Risk Score:</span>
              <span className={`risk-badge badge-${(selectedTx.risk_level || 'low').toLowerCase()}`}>
                {(Number(selectedTx.risk_score || 0.05) * 100).toFixed(1)}% ({selectedTx.risk_level})
              </span>
            </div>
            {selectedTx.consensus_reasoning && (
              <div className="reasoning-box">
                <span className="reasoning-title">Decision Reasoning:</span>
                <p>{selectedTx.consensus_reasoning}</p>
              </div>
            )}
            <div className="drawer-actions">
              <button
                type="button"
                className="btn-primary"
                onClick={() => onInvestigateGraph(selectedTx.transaction_id)}
              >
                🕸️ Open in Graph Explorer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

