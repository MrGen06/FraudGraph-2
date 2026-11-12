import type {
  AlertRecord,
  BenchmarkData,
  GraphExplanation,
  ModelSystemStatus,
  PredictResponse,
  SamplePreset,
  TransactionPayload,
  TransactionRecord,
} from './types'

const BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const API_PREFIX = `${BASE_URL}/api/v1`

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function fetchModelStatus(): Promise<ModelSystemStatus> {
  const res = await fetch(`${API_PREFIX}/models/status`)
  if (!res.ok) throw new Error('Failed to fetch model status')
  return res.json()
}

export async function fetchBenchmarkData(): Promise<BenchmarkData> {
  const res = await fetch(`${API_PREFIX}/models/benchmark`)
  if (!res.ok) throw new Error('Failed to fetch benchmark data')
  return res.json()
}

export async function fetchSamplePresets(): Promise<SamplePreset[]> {
  const res = await fetch(`${API_PREFIX}/models/sample-transactions`)
  if (!res.ok) throw new Error('Failed to fetch sample presets')
  return res.json()
}

export async function predictTransaction(payload: TransactionPayload): Promise<PredictResponse> {
  const res = await fetch(`${API_PREFIX}/models/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const errorText = await res.text()
    throw new Error(`Inference error (${res.status}): ${errorText}`)
  }
  return res.json()
}

export async function fetchTransactions(params?: {
  skip?: number
  limit?: number
  risk_level?: string
  search?: string
}): Promise<TransactionRecord[]> {
  const query = new URLSearchParams()
  if (params?.skip) query.set('skip', String(params.skip))
  if (params?.limit) query.set('limit', String(params.limit))
  if (params?.risk_level && params.risk_level !== 'ALL') query.set('risk_level', params.risk_level)
  if (params?.search) query.set('search', params.search)

  const url = `${API_PREFIX}/transactions/?${query.toString()}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch transactions')
  return res.json()
}

export async function fetchTransactionDetail(id: string): Promise<TransactionRecord> {
  const res = await fetch(`${API_PREFIX}/transactions/${id}`)
  if (!res.ok) throw new Error(`Failed to fetch transaction ${id}`)
  return res.json()
}

export async function fetchAlerts(params?: {
  skip?: number
  limit?: number
  status?: string
  risk_level?: string
}): Promise<AlertRecord[]> {
  const query = new URLSearchParams()
  if (params?.skip) query.set('skip', String(params.skip))
  if (params?.limit) query.set('limit', String(params.limit))
  if (params?.status && params.status !== 'ALL') query.set('status', params.status)
  if (params?.risk_level && params.risk_level !== 'ALL') query.set('risk_level', params.risk_level)

  const res = await fetch(`${API_PREFIX}/alerts/?${query.toString()}`)
  if (!res.ok) throw new Error('Failed to fetch alerts')
  return res.json()
}

export async function reviewAlert(
  alertId: string,
  payload: { status: string; reviewer_notes?: string; reviewed_by?: string }
): Promise<AlertRecord> {
  const res = await fetch(`${API_PREFIX}/alerts/${alertId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Failed to review alert ${alertId}`)
  return res.json()
}

export async function fetchGraphSubgraph(transactionId: string): Promise<GraphExplanation> {
  const res = await fetch(`${API_PREFIX}/graph/${transactionId}`)
  if (!res.ok) throw new Error(`Failed to fetch graph for transaction ${transactionId}`)
  return res.json()
}

