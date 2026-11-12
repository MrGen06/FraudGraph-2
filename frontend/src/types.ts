export interface TransactionPayload {
  transaction_id?: string
  amount: number
  card_id: string
  card_type: string
  card_network: string
  device_id: string
  device_type: string
  email_domain: string
  hour: number
  day_of_week: number
  dist1: number
  product_cd: string
}

export interface SamplePreset {
  id: string
  name: string
  description: string
  payload: TransactionPayload
}

export interface ShapAttribution {
  feature: string
  value: number
  shap_value: number
  magnitude: number
  direction: 'risk_increasing' | 'risk_decreasing'
  description: string
}

export interface GraphNode {
  id: string
  label: string
  type: 'transaction' | 'card' | 'device' | 'email' | 'account' | 'merchant' | 'neighbor_tx'
  is_center: boolean
  risk_flag: boolean
}

export interface GraphLink {
  source: string
  target: string
  relationship: string
}

export interface GraphExplanation {
  transaction_id?: string
  nodes: GraphNode[]
  links: GraphLink[]
  card_id?: string
  device_id?: string
  email_id?: string
  shared_entity_peers_count?: number
  flagged_neighbor_fraud_count?: number
  reasoning?: string
  source?: string
}

export interface PredictResponse {
  transaction_id: string
  risk_score?: number
  risk_level?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  tabular_score: number
  tabular_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  graph_score?: number
  graph_risk_level?: string
  hybrid_score: number
  consensus_verdict: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  consensus_reasoning: string
  decision_reasoning?: string
  shap_attributions: ShapAttribution[]
  graph_explanation: GraphExplanation
}

export interface ModelMetric {
  model: string
  pr_auc: number
  roc_auc: number
  optimal_threshold: number
  f1_minority: number
  macro_f1: number
  precision: number
  recall: number
  fpr_at_90_recall: number
  fpr_at_95_recall: number
  fpr_at_99_recall: number
  confusion_matrix?: number[][]
}

export interface BenchmarkData {
  metrics: {
    test_records_count: number
    test_fraud_count: number
    test_fraud_rate: number
    validation_ensemble_weights?: {
      tabular_weight: number
      graph_weight: number
    }
    models: ModelMetric[]
  }
  curves: {
    roc: {
      fpr_xgb: number[]
      tpr_xgb: number[]
      fpr_gnn: number[]
      tpr_gnn: number[]
      fpr_hyb: number[]
      tpr_hyb: number[]
    }
    pr: {
      rec_xgb: number[]
      prec_xgb: number[]
      rec_gnn: number[]
      prec_gnn: number[]
      rec_hyb: number[]
      prec_hyb: number[]
      baseline: number
    }
  }
  comparison_table: ModelMetric[]
}

export interface TransactionRecord {
  id: number
  transaction_id: string
  timestamp: string
  source_account_id: string
  destination_account_id: string
  user_id: string
  card_id?: string | null
  device_id?: string | null
  ip_id?: string | null
  merchant_id?: string | null
  amount: number
  currency: string
  country?: string | null
  region?: string | null
  risk_score?: number | null
  risk_level?: string | null
  is_flagged: boolean
  consensus_verdict?: string | null
  consensus_reasoning?: string | null
}

export interface AlertRecord {
  id?: number
  alert_id: string
  transaction_id: string
  risk_score: number
  risk_level: string
  status: string
  reasons: string[]
  reviewer_notes?: string | null
  reviewed_by?: string | null
  reviewed_at?: string | null
  created_at: string
}

export interface ModelSystemStatus {
  status: string
  models: {
    xgboost: {
      loaded: boolean
      artifact_exists: boolean
      features_count: number
    }
    gnn: {
      loaded: boolean
      artifact_exists: boolean
      mappings_loaded: boolean
    }
    preprocessor: {
      loaded: boolean
      artifact_exists: boolean
    }
  }
  ensemble_weights: {
    tabular_weight: number
    graph_weight: number
  }
}
