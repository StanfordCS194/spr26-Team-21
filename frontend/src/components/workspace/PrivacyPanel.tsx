import { useState } from 'react';
import type {
  PrivacyReport,
  MembershipInference,
} from '../../api/client';

interface Props {
  privacy: PrivacyReport | null | undefined;
}

type Tone = 'pass' | 'warn' | 'fail' | 'info';

function scoreTone(score: number): Tone {
  if (score >= 0.8) return 'pass';
  if (score >= 0.5) return 'warn';
  return 'fail';
}

function dupTone(pct: number, exactCount: number): Tone {
  if (exactCount > 0) return 'fail';
  if (pct >= 5) return 'fail';
  if (pct >= 1) return 'warn';
  return 'pass';
}

function mioTone(auc: number): Tone {
  if (auc < 0.55) return 'pass';
  if (auc < 0.65) return 'warn';
  return 'fail';
}

function verdictTone(verdict: string): Tone {
  const v = verdict.toLowerCase();
  if (v.includes('strong privacy') || v.includes('as far from real as random')) return 'pass';
  if (v.includes('leak') || v.includes('do not share') || v.includes('exact cop')) return 'fail';
  if (v.includes('mild') || v.includes('acceptable')) return 'warn';
  return 'info';
}

function MetricCard({
  label,
  primary,
  secondary,
  tone,
  caption,
}: {
  label: string;
  primary: string;
  secondary?: string;
  tone: Tone;
  caption?: string;
}) {
  return (
    <div className={`pp-metric-card pp-metric-${tone}`}>
      <div className="pp-metric-label">{label}</div>
      <div className="pp-metric-primary">{primary}</div>
      {secondary && <div className="pp-metric-secondary">{secondary}</div>}
      {caption && <div className="pp-metric-caption">{caption}</div>}
    </div>
  );
}

function MIASection({ mia }: { mia: MembershipInference | null | undefined }) {
  if (!mia || !mia.available) {
    return (
      <div className="pp-mia-card pp-mia-unavailable">
        <div className="pp-mia-header">
          <span className="pp-section-label">Membership Inference Attack</span>
          <span className="pp-badge pp-badge-info">Unavailable</span>
        </div>
        <div className="pp-mia-explainer">
          Requires a held-out set of real records the synthesizer never saw. Without it, the
          gold-standard privacy test can't be run — distance-based metrics above are still valid.
        </div>
      </div>
    );
  }

  const tone = mioTone(mia.roc_auc);
  return (
    <div className={`pp-mia-card pp-mia-${tone}`}>
      <div className="pp-mia-header">
        <span className="pp-section-label">Membership Inference Attack</span>
        <span className={`pp-badge pp-badge-${tone}`}>AUC {mia.roc_auc.toFixed(3)}</span>
      </div>
      <div className="pp-mia-grid">
        <div className="pp-mia-cell">
          <div className="pp-mia-cell-label">TPR @ 1% FPR</div>
          <div className="pp-mia-cell-value">{(mia.tpr_at_1pct_fpr * 100).toFixed(1)}%</div>
        </div>
        <div className="pp-mia-cell">
          <div className="pp-mia-cell-label">Members</div>
          <div className="pp-mia-cell-value">{mia.n_members.toLocaleString()}</div>
        </div>
        <div className="pp-mia-cell">
          <div className="pp-mia-cell-label">Non-members</div>
          <div className="pp-mia-cell-value">{mia.n_nonmembers.toLocaleString()}</div>
        </div>
      </div>
      <div className="pp-mia-interpretation">{mia.interpretation}</div>
    </div>
  );
}

export default function PrivacyPanel({ privacy }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (!privacy || !privacy.available) {
    return (
      <div className="pp-card">
        <div className="pp-header">
          <span className="pp-title">Privacy</span>
          <span className="pp-badge pp-badge-info">Unavailable</span>
        </div>
        <div className="pp-empty">
          Requires real source data uploaded for grounding. Privacy metrics (DCR, NNDR,
          baseline-protection, and membership-inference) are computed against the records the
          synthesizer was fit on.
        </div>
      </div>
    );
  }

  const dcr = privacy.dcr;
  const nndr = privacy.nndr;
  const baseline = privacy.baseline_protection;
  const mia = privacy.membership_inference ?? null;
  const verdict = privacy.verdict ?? '';
  const tone = verdictTone(verdict);

  return (
    <div className="pp-card">
      <button
        type="button"
        className="pp-header pp-header-button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span className="pp-title">Privacy</span>
        <span className="pp-meta">
          {privacy.n_real?.toLocaleString() ?? '?'} real ·{' '}
          {privacy.n_synth?.toLocaleString() ?? '?'} synth · {privacy.n_features ?? '?'} features
        </span>
        <span className={`pp-chevron${expanded ? ' pp-chevron-open' : ''}`}>▾</span>
      </button>

      {expanded && (
        <div className="pp-body">
          {verdict && (
            <div className={`pp-verdict pp-verdict-${tone}`}>
              <span className={`pp-badge pp-badge-${tone}`}>Verdict</span>
              <span className="pp-verdict-text">{verdict}</span>
            </div>
          )}

          <div className="pp-metric-grid">
            {dcr && (
              <MetricCard
                label="DCR (median distance)"
                primary={dcr.median.toFixed(3)}
                secondary={`p5 = ${dcr.p5.toFixed(3)}`}
                tone={dupTone(dcr.near_duplicate_pct, dcr.n_exact_matches)}
                caption={
                  dcr.n_exact_matches > 0
                    ? `${dcr.n_exact_matches} exact match${dcr.n_exact_matches === 1 ? '' : 'es'}`
                    : `${dcr.n_near_duplicates.toLocaleString()} near-dups (${dcr.near_duplicate_pct.toFixed(2)}%)`
                }
              />
            )}
            {nndr && (
              <MetricCard
                label="NNDR (nearest / 2nd-nearest)"
                primary={nndr.median.toFixed(3)}
                secondary={`p5 = ${nndr.p5.toFixed(3)}`}
                tone={nndr.median >= 0.7 ? 'pass' : nndr.median >= 0.5 ? 'warn' : 'fail'}
                caption="High ratio = synthetic sits between real records (good)"
              />
            )}
            {baseline && (
              <MetricCard
                label="Baseline protection"
                primary={baseline.score.toFixed(3)}
                secondary={`synth ${baseline.synth_dcr_median.toFixed(3)} / random ${baseline.random_dcr_median.toFixed(3)}`}
                tone={scoreTone(baseline.score)}
                caption="1.0 = as far from real as random noise"
              />
            )}
          </div>

          <MIASection mia={mia} />
        </div>
      )}
    </div>
  );
}
