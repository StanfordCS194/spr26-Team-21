import { useState } from 'react';
import type { DetectionReport, DiscriminatorResult } from '../../api/client';

interface Props {
  detection: DetectionReport | null | undefined;
}

type Tone = 'pass' | 'warn' | 'fail' | 'info';

// AUC ≈ 0.5 = indistinguishable (good). Closer to 1.0 = trivially separable (bad).
function aucTone(auc: number): Tone {
  if (auc < 0.6) return 'pass';
  if (auc < 0.75) return 'warn';
  return 'fail';
}

// Lower ECE = better-calibrated probabilities.
function eceTone(ece: number): Tone {
  if (ece < 0.05) return 'pass';
  if (ece < 0.15) return 'warn';
  return 'fail';
}

function verdictTone(verdict: string): Tone {
  const v = verdict.toLowerCase();
  if (v.includes('indistinguishable') || v.includes('near chance')) return 'pass';
  if (v.includes('trivially separable') || v.includes('low fidelity')) return 'fail';
  if (v.includes('leakage') || v.includes('separates')) return 'warn';
  return 'info';
}

function DiscriminatorCard({
  result,
  fallbackLabel,
}: {
  result: DiscriminatorResult | null | undefined;
  fallbackLabel: string;
}) {
  if (!result) {
    return (
      <div className="dp2-disc-card dp2-disc-unavailable">
        <div className="dp2-disc-name">{fallbackLabel}</div>
        <div className="dp2-disc-unavailable-text">Could not train</div>
      </div>
    );
  }
  const aTone = aucTone(result.auc);
  const eTone = eceTone(result.ece);
  return (
    <div className={`dp2-disc-card dp2-disc-${aTone}`}>
      <div className="dp2-disc-name">{result.model}</div>
      <div className="dp2-disc-rows">
        <div className="dp2-disc-row">
          <span className="dp2-disc-label">AUC</span>
          <span className={`dp2-disc-value dp2-value-${aTone}`}>{result.auc.toFixed(3)}</span>
        </div>
        <div className="dp2-disc-row">
          <span className="dp2-disc-label">ECE</span>
          <span className={`dp2-disc-value dp2-value-${eTone}`}>{result.ece.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

export default function DetectionPanel({ detection }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (!detection || !detection.available) {
    return (
      <div className="dp2-card">
        <div className="dp2-header">
          <span className="dp2-title">Detection (Indistinguishability)</span>
          <span className="dp2-badge dp2-badge-info">Unavailable</span>
        </div>
        <div className="dp2-empty">
          Requires real source data and a synthetic dataset of at least 50 rows each. Discriminator
          AUC measures whether a classifier can tell real from synthetic — closer to 0.5 is better.
        </div>
      </div>
    );
  }

  const verdict = detection.verdict ?? '';
  const tone = verdictTone(verdict);
  const xgb = detection.xgboost ?? null;
  const lr = detection.logreg ?? null;

  return (
    <div className="dp2-card">
      <button
        type="button"
        className="dp2-header dp2-header-button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span className="dp2-title">Detection (Indistinguishability)</span>
        <span className="dp2-meta">
          {detection.n_real?.toLocaleString() ?? '?'} real ·{' '}
          {detection.n_synth?.toLocaleString() ?? '?'} synth · {detection.n_features ?? '?'} features
        </span>
        <span className={`dp2-chevron${expanded ? ' dp2-chevron-open' : ''}`}>▾</span>
      </button>

      {expanded && (
        <div className="dp2-body">
          {verdict && (
            <div className={`dp2-verdict dp2-verdict-${tone}`}>
              <span className={`dp2-badge dp2-badge-${tone}`}>Verdict</span>
              <span className="dp2-verdict-text">{verdict}</span>
            </div>
          )}

          <div className="dp2-discs">
            <DiscriminatorCard result={xgb} fallbackLabel="XGBoost (non-linear)" />
            <DiscriminatorCard result={lr} fallbackLabel="Logistic Regression (linear)" />
          </div>

          {detection.agreement && (
            <div className="dp2-agreement">
              <div className="dp2-section-label">Cross-model agreement</div>
              <div className="dp2-agreement-text">{detection.agreement}</div>
            </div>
          )}

          <div className="dp2-legend">
            <span className="dp2-legend-item">
              <span className="dp2-legend-swatch dp2-legend-pass" /> AUC &lt; 0.6 ≈ indistinguishable
            </span>
            <span className="dp2-legend-item">
              <span className="dp2-legend-swatch dp2-legend-warn" /> 0.6–0.75 ≈ separable structure
            </span>
            <span className="dp2-legend-item">
              <span className="dp2-legend-swatch dp2-legend-fail" /> &gt; 0.75 ≈ trivially separable
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
