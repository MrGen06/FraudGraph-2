import React from 'react'

interface Props {
  verdict: 'LOW' | 'MEDIUM' | 'CRITICAL' | 'HIGH'
  reasoning: string
  hybridScore: number
}

export const ConsensusBanner: React.FC<Props> = ({
  verdict,
  reasoning,
  hybridScore,
}) => {
  const normVerdict = verdict === 'HIGH' ? 'CRITICAL' : verdict
  const getVerdictTheme = () => {
    switch (normVerdict) {
      case 'CRITICAL':
        return {
          badgeClass: 'badge-critical',
          bannerClass: 'banner-critical',
          label: 'CRITICAL FRAUD ALERT',
          icon: '🚨',
          action: 'IMMEDIATE ACTION: Transaction blocked. Flagged for fraud ring containment and card freeze.',
        }
      case 'MEDIUM':
        return {
          badgeClass: 'badge-medium',
          bannerClass: 'banner-medium',
          label: 'ELEVATED RISK REVIEW',
          icon: '⚠️',
          action: 'STEP-UP REQUIRED: Request biometric/2FA re-verification or assign for analyst triage.',
        }
      case 'LOW':
      default:
        return {
          badgeClass: 'badge-low',
          bannerClass: 'banner-low',
          label: 'LEGITIMATE VERDICT',
          icon: '🛡️',
          action: 'APPROVED: Standard clearing with normal fraud velocity parameters.',
        }
    }
  }

  const theme = getVerdictTheme()

  return (
    <div className={`consensus-banner ${theme.bannerClass}`}>
      <div className="consensus-top-row">
        <div className="verdict-identity">
          <span className="verdict-icon">{theme.icon}</span>
          <div>
            <div className="verdict-tagline">Automated Fraud Risk Decision (XGBoost)</div>
            <div className="verdict-heading">
              <span className={`verdict-pill ${theme.badgeClass}`}>{normVerdict}</span>
              <span className="verdict-subtext">{theme.label}</span>
            </div>
          </div>
        </div>

        <div className="hybrid-score-box">
          <div className="hybrid-label">XGBoost Fraud Probability</div>
          <div className="hybrid-numeric">{(hybridScore * 100).toFixed(1)}%</div>
          <div className="hybrid-bar-track">
            <div
              className={`hybrid-bar-fill fill-${normVerdict.toLowerCase()}`}
              style={{ width: `${Math.min(100, Math.max(4, hybridScore * 100))}%` }}
            />
          </div>
        </div>
      </div>

      <div className="consensus-reasoning-box">
        <div className="reasoning-label">Automated Decision Intelligence & Risk Drivers</div>
        <p className="reasoning-text">{reasoning}</p>
      </div>

      <div className="action-directive">
        <span className="directive-tag">Recommended Workflow Directive:</span> {theme.action}
      </div>
    </div>
  )
}


