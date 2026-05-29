import { useEffect, useState } from 'react';
import type {
  DiagnosticsReport,
  ConfusionMatrix,
  FeatureAblationEntry,
} from '../../api/client';

interface Props {
  diagnostics: DiagnosticsReport | null | undefined;
}

type Tone = 'pass' | 'warn' | 'fail' | 'info';

function pct(n: number, total: number): string {
  if (total <= 0) return '—';
  return `${Math.round((n / total) * 1000) / 10}%`;
}

function interpretationTone(interpretation: FeatureAblationEntry['interpretation']): Tone {
  if (interpretation === 'high reliance') return 'pass';
  if (interpretation === 'useful') return 'warn';
  if (interpretation === 'marginal') return 'info';
  return 'fail'; // harmful when present
}

function ConfusionGrid({
  label,
  abbrev,
  matrix,
}: {
  label: string;
  abbrev: string;
  matrix: ConfusionMatrix;
}) {
  const total = matrix.tn + matrix.fp + matrix.fn + matrix.tp;
  return (
    <div className="dp-conf-matrix">
      <div className="dp-conf-label">
        {label} <span className="dp-muted">({abbrev})</span>
      </div>
      <table className="dp-conf-grid">
        <thead>
          <tr>
            <th></th>
            <th className="dp-conf-head">Pred 0</th>
            <th className="dp-conf-head">Pred 1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className="dp-conf-head">True 0</th>
            <td className="dp-conf-cell dp-conf-correct">
              <div className="dp-conf-n">{matrix.tn.toLocaleString()}</div>
              <div className="dp-conf-pct">{pct(matrix.tn, total)}</div>
              <div className="dp-conf-tag">TN</div>
            </td>
            <td className="dp-conf-cell dp-conf-fp">
              <div className="dp-conf-n">{matrix.fp.toLocaleString()}</div>
              <div className="dp-conf-pct">{pct(matrix.fp, total)}</div>
              <div className="dp-conf-tag">FP</div>
            </td>
          </tr>
          <tr>
            <th className="dp-conf-head">True 1</th>
            <td className="dp-conf-cell dp-conf-fn">
              <div className="dp-conf-n">{matrix.fn.toLocaleString()}</div>
              <div className="dp-conf-pct">{pct(matrix.fn, total)}</div>
              <div className="dp-conf-tag">FN</div>
            </td>
            <td className="dp-conf-cell dp-conf-correct">
              <div className="dp-conf-n">{matrix.tp.toLocaleString()}</div>
              <div className="dp-conf-pct">{pct(matrix.tp, total)}</div>
              <div className="dp-conf-tag">TP</div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function AblationBars({
  ablation,
  animated,
}: {
  ablation: FeatureAblationEntry[];
  animated: boolean;
}) {
  const maxAbs = Math.max(1, ...ablation.map((a) => Math.abs(a.recall_delta_pct)));
  return (
    <div className="dp-abl-list">
      {ablation.map((a) => {
        const tone = interpretationTone(a.interpretation);
        const width = Math.min(100, (Math.abs(a.recall_delta_pct) / maxAbs) * 100);
        const sign = a.recall_delta_pct >= 0 ? '+' : '';
        return (
          <div key={a.feature} className="dp-abl-row">
            <div className="dp-abl-name dp-mono" title={a.feature}>
              {a.feature}
            </div>
            <div className="dp-abl-bar-track">
              <div
                className={`dp-abl-bar dp-bar-${tone}`}
                style={{ width: animated ? `${width}%` : '0%' }}
              />
            </div>
            <div className="dp-abl-delta dp-mono">
              {sign}
              {a.recall_delta_pct}pt
            </div>
            <div className="dp-abl-tag dp-muted">{a.interpretation}</div>
          </div>
        );
      })}
    </div>
  );
}

function MisclassCards({
  overlap,
}: {
  overlap: NonNullable<DiagnosticsReport['misclassification_overlap']>;
}) {
  const c = overlap.counts;
  const buckets: Array<{ name: string; count: number; tone: Tone; desc: string }> = [
    {
      name: 'Synthetic helps',
      count: c.trtr_only_wrong,
      tone: 'pass',
      desc: 'Real-only got these wrong; synthetic-only got them right',
    },
    {
      name: 'Synthetic hurts',
      count: c.tstr_only_wrong,
      tone: 'fail',
      desc: 'Real-only got these right; synthetic-only got them wrong',
    },
    {
      name: 'Both wrong',
      count: c.both_wrong,
      tone: 'warn',
      desc: 'Genuinely hard rows neither regime classifies correctly',
    },
    {
      name: 'Augmentation saves',
      count: c.augmentation_saves,
      tone: 'pass',
      desc: 'Augmented model correct where at least one base regime failed',
    },
  ];

  return (
    <div className="dp-mc-grid">
      {buckets.map((b) => (
        <div key={b.name} className={`dp-mc-card dp-mc-card-${b.tone}`}>
          <div className="dp-mc-name">{b.name}</div>
          <div className="dp-mc-count">{b.count.toLocaleString()}</div>
          <div className="dp-mc-desc dp-muted">{b.desc}</div>
        </div>
      ))}
    </div>
  );
}

export default function DiagnosticsPanel({ diagnostics }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = window.setTimeout(() => setAnimated(true), 60);
    return () => window.clearTimeout(t);
  }, []);

  if (!diagnostics || !diagnostics.available) {
    return (
      <div className="dp-card">
        <div className="dp-header">
          <span className="dp-title">Experiment Diagnostics</span>
          <span className="dp-badge dp-badge-info">Unavailable</span>
        </div>
        <div className="dp-empty">
          Requires a real source dataset with a label column. Upload data via the integrations
          panel and re-generate to enable confusion matrices, feature ablation, and row-level
          misclassification overlap.
        </div>
      </div>
    );
  }

  const target = diagnostics.target ?? '?';
  const nTest = diagnostics.n_test ?? 0;
  const conf = diagnostics.confusion_matrices;
  const ablation = diagnostics.feature_ablation ?? [];
  const overlap = diagnostics.misclassification_overlap;
  const observations = diagnostics.observations ?? [];
  const recommendations = diagnostics.recommendations ?? [];

  return (
    <div className="dp-card">
      <button
        type="button"
        className="dp-header dp-header-button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span className="dp-title">Experiment Diagnostics</span>
        <span className="dp-meta">
          Target: <span className="dp-mono">{target}</span> · {nTest.toLocaleString()} held-out
          test rows
        </span>
        <span className={`dp-chevron${expanded ? ' dp-chevron-open' : ''}`}>▾</span>
      </button>

      {expanded && (
        <div className="dp-body">
          {conf && (
            <div className="dp-section">
              <div className="dp-section-label">Confusion Matrices</div>
              <div className="dp-conf-matrices">
                <ConfusionGrid label="Real only" abbrev="TRTR" matrix={conf.trtr} />
                <ConfusionGrid label="Synthetic only" abbrev="TSTR" matrix={conf.tstr} />
                <ConfusionGrid label="Real + Synthetic" abbrev="TR+STR" matrix={conf.augmented} />
              </div>
            </div>
          )}

          {ablation.length > 0 && (
            <div className="dp-section">
              <div className="dp-section-label">Feature Ablation (Augmented Model)</div>
              <div className="dp-section-hint">
                Recall drop when each top-importance feature is permuted in the test set. Larger
                drops = model relies on that feature.
              </div>
              <AblationBars ablation={ablation} animated={animated} />
            </div>
          )}

          {overlap && overlap.counts && (
            <div className="dp-section">
              <div className="dp-section-label">Row-Level Misclassification Overlap</div>
              <div className="dp-section-hint">{overlap.summary}</div>
              <MisclassCards overlap={overlap} />
            </div>
          )}

          {observations.length > 0 && (
            <div className="dp-section">
              <div className="dp-section-label">Observations</div>
              <ul className="dp-list">
                {observations.map((o, i) => (
                  <li key={i}>{o}</li>
                ))}
              </ul>
            </div>
          )}

          {recommendations.length > 0 && (
            <div className="dp-section">
              <div className="dp-section-label">Recommendations</div>
              <ol className="dp-list-num">
                {recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
