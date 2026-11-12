import React from 'react'

interface Props {
  tabularScore: number
  tabularLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  graphScore?: number
  graphLevel?: string
  topTabularFeature?: string
  graphHopCount: number
  flaggedNeighbors: number
}

export const DualRiskScorecard: React.FC<Props> = ({
  tabularScore,
  tabularLevel,
  topTabularFeature,
  graphHopCount,
  flaggedNeighbors,
}) => {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'CRITICAL':
      case 'HIGH':
        return '#EF4444'
      case 'MEDIUM':
        return '#F59E0B'
      case 'LOW':
      default:
        return '#10B981'
    }
  }

  return (
    <div className="dual-scorecard-grid">
      {/* Primary: XGBoost Production Predictor */}
      <div className="model-card model-tabular">
        <div className="model-card-header">
          <div className="model-badge">Production Primary</div>
          <h4>XGBoost Classifier</h4>
          <span className="model-type-tag">Sole Automated Predictor</span>
        </div>

        <div className="score-display">
          <div
            className="score-circle"
            style={{
              borderColor: getLevelColor(tabularLevel),
              boxShadow: `0 0 16px ${getLevelColor(tabularLevel)}33`,
            }}
          >
            <span className="score-num">{(tabularScore * 100).toFixed(1)}%</span>
            <span className="score-unit">Risk</span>
          </div>

          <div className="score-meta">
            <div
              className="risk-level-badge"
              style={{
                backgroundColor: `${getLevelColor(tabularLevel)}20`,
                color: getLevelColor(tabularLevel),
              }}
            >
              Risk Level: {tabularLevel}
            </div>
            <p className="score-desc">
              Production decision engine evaluated on 161 engineered behavioral and chronological features.
            </p>
          </div>
        </div>

        <div className="model-insights">
          <div className="insight-row">
            <span className="insight-key">Key Risk Driver:</span>
            <span className="insight-val">{topTabularFeature || 'Transaction amount & velocity'}</span>
          </div>
          <div className="insight-row">
            <span className="insight-key">Explainability Engine:</span>
            <span className="insight-val">TreeSHAP Local Attributions</span>
          </div>
        </div>
      </div>

      {/* Secondary: Neo4j Graph Investigation Evidence */}
      <div className="model-card model-graph">
        <div className="model-card-header">
          <div className="model-badge badge-purple">Neo4j Investigation</div>
          <h4>Entity Graph Topology</h4>
          <span className="model-type-tag">Relational Investigation Evidence</span>
        </div>

        <div className="score-display">
          <div
            className="score-circle"
            style={{
              borderColor: flaggedNeighbors > 0 ? '#EF4444' : '#8B5CF6',
              boxShadow: `0 0 16px ${flaggedNeighbors > 0 ? '#EF444433' : '#8B5CF633'}`,
            }}
          >
            <span className="score-num">{graphHopCount}</span>
            <span className="score-unit">Peers</span>
          </div>

          <div className="score-meta">
            <div
              className="risk-level-badge"
              style={{
                backgroundColor: flaggedNeighbors > 0 ? '#EF444420' : '#8B5CF620',
                color: flaggedNeighbors > 0 ? '#EF4444' : '#A78BFA',
              }}
            >
              {flaggedNeighbors > 0 ? `${flaggedNeighbors} Compromised Links` : 'Structural Evidence'}
            </div>
            <p className="score-desc">
              Multi-hop entity relationships across cards, devices, and merchants for analyst exploration.
            </p>
          </div>
        </div>

        <div className="model-insights">
          <div className="insight-row">
            <span className="insight-key">Graph Investigation:</span>
            <span className="insight-val">{graphHopCount} linked entities explored</span>
          </div>
          <div className="insight-row">
            <span className="insight-key">GNN Research Status:</span>
            <span className="insight-val text-muted">
              Offline Research Only (PR-AUC 0.0400)
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}


