import { useEffect, useState } from 'react';
import { Check, ChevronDown } from '../icons/Icons';
import type {
  ValidationReport as ValidationReportType,
  ValidationStatus,
  DistributionDistance,
  CorrelationDrift,
  KAnonymity,
} from '../../constants/mockWorkspace';

interface Props {
  report: ValidationReportType;
}

function WarnIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function StatusBadge({ status }: { status: ValidationStatus }) {
  return (
    <span className={`ws-validation-badge ws-validation-badge-${status}`}>
      {status === 'pass' ? 'Pass' : status === 'warn' ? 'Warn' : 'Fail'}
    </span>
  );
}

function DistributionDistanceSection({ data, animated }: { data: DistributionDistance; animated: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const allCols = [...data.numericColumns, ...data.categoricalColumns];
  const flagged = allCols.filter((c) => c.status !== 'pass');

  return (
    <div className="ws-validation-subsection">
      <button
        className="ws-validation-section-btn"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span>
          Distribution Distance
          {flagged.length > 0 && (
            <span className="ws-validation-section-badge ws-validation-badge-warn">
              {flagged.length} flagged
            </span>
          )}
        </span>
        <span className={`ws-chevron${expanded ? ' ws-chevron-open' : ''}`}>
          <ChevronDown size={12} strokeWidth={2.5} />
        </span>
      </button>
      {expanded && (
        <div className="ws-validation-cols">
          {data.numericColumns.map((col) => (
            <div key={col.column} className="ws-validation-col-row">
              <span className="ws-validation-col-name">{col.column}</span>
              <div className="ws-validation-mini-track">
                <div
                  className={`ws-validation-mini-bar ws-validation-bar-${col.status}`}
                  style={{ width: animated ? `${Math.round((1 - col.jsDivergence) * 100)}%` : '0%' }}
                />
              </div>
              <span className={`ws-validation-col-pct ws-validation-score-${col.status}`}>
                JS {col.jsDivergence.toFixed(3)}
              </span>
              <StatusBadge status={col.status} />
            </div>
          ))}
          {data.categoricalColumns.map((col) => (
            <div key={col.column} className="ws-validation-col-row">
              <span className="ws-validation-col-name">{col.column}</span>
              <div className="ws-validation-mini-track">
                <div
                  className={`ws-validation-mini-bar ws-validation-bar-${col.status}`}
                  style={{ width: animated ? (col.status === 'pass' ? '90%' : col.status === 'warn' ? '55%' : '20%') : '0%' }}
                />
              </div>
              <span className={`ws-validation-col-pct ws-validation-score-${col.status}`}>
                p={col.pValue < 0.001 ? '<0.001' : col.pValue.toFixed(3)}
              </span>
              <StatusBadge status={col.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CorrelationDriftSection({ data }: { data: CorrelationDrift }) {
  const [expanded, setExpanded] = useState(false);
  if (data.driftedPairs.length === 0) return null;

  return (
    <div className="ws-validation-subsection">
      <button
        className="ws-validation-section-btn"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span>
          Correlation Drift
          <span className={`ws-validation-section-badge ws-validation-badge-${data.status}`}>
            {data.driftedPairs.length} pair{data.driftedPairs.length !== 1 ? 's' : ''}
          </span>
        </span>
        <span className={`ws-chevron${expanded ? ' ws-chevron-open' : ''}`}>
          <ChevronDown size={12} strokeWidth={2.5} />
        </span>
      </button>
      {expanded && (
        <div className="ws-validation-cols">
          {data.driftedPairs.map((pair, i) => (
            <div key={i} className="ws-validation-col-row">
              <span className="ws-validation-col-name" style={{ flex: '0 0 auto', maxWidth: '55%' }}>
                {pair.columns[0]} × {pair.columns[1]}
              </span>
              <span className="ws-validation-col-note" style={{ marginLeft: 'auto' }}>
                {pair.sourceCorr >= 0 ? '+' : ''}{pair.sourceCorr.toFixed(2)} → {pair.syntheticCorr >= 0 ? '+' : ''}{pair.syntheticCorr.toFixed(2)}
              </span>
              <StatusBadge status={pair.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function KAnonymityRow({ data }: { data: KAnonymity }) {
  return (
    <div className={`ws-sanity-row ws-sanity-${data.status}`}>
      <span className="ws-sanity-name">k-Anonymity</span>
      <span className="ws-sanity-value">
        min k={data.minK} over [{data.quasiIdentifiers.join(', ')}]
        {data.groupsBelowThreshold > 0 && ` · ${data.groupsBelowThreshold} group(s) below k≥${data.kThreshold}`}
      </span>
      <span className={`ws-validation-badge ws-validation-badge-${data.status}`}>
        {data.status === 'pass' ? `k≥${data.kThreshold}` : data.status === 'warn' ? `k=${data.minK}` : `k=${data.minK} Risk`}
      </span>
    </div>
  );
}

export default function ValidationReport({ report }: Props) {
  const [colsExpanded, setColsExpanded] = useState(true);
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = window.setTimeout(() => setAnimated(true), 60);
    return () => window.clearTimeout(t);
  }, []);

  return (
    <div className="ws-validation-card">
      <div className="ws-validation-header">
        <span className="ws-validation-title">Validation Report</span>
        <span className={`ws-validation-verdict ws-validation-verdict-${report.verdictStatus}`}>
          {report.verdictStatus === 'pass' && <Check size={11} strokeWidth={2.5} />}
          {report.verdictStatus === 'warn' && <WarnIcon />}
          {report.verdict}
        </span>
      </div>

      {report.edgeCaseCoverage && (
        <div className="ws-coverage-banner">
          <div className="ws-coverage-top">
            <span className="ws-coverage-label">Edge Case Coverage</span>
            <span className="ws-coverage-pct">{report.edgeCaseCoverage.coveragePct}%</span>
          </div>
          <div className="ws-coverage-desc">{report.edgeCaseCoverage.description}</div>
          <div className="ws-coverage-bar-track">
            <div
              className="ws-coverage-bar"
              style={{ width: animated ? `${report.edgeCaseCoverage.coveragePct}%` : '0%' }}
            />
          </div>
          <div className="ws-coverage-count">
            {report.edgeCaseCoverage.generated.toLocaleString()} of{' '}
            {report.edgeCaseCoverage.requested.toLocaleString()} requested cases generated
          </div>
        </div>
      )}

      <div className="ws-validation-metrics">
        {report.metrics.map((m) => (
          <div key={m.label} className="ws-validation-metric">
            <span className="ws-validation-metric-label">{m.label}</span>
            <div className="ws-validation-bar-track">
              <div
                className={`ws-validation-bar ws-validation-bar-${m.status}`}
                style={{ width: animated ? `${m.score}%` : '0%' }}
              />
            </div>
            <div className="ws-validation-metric-footer">
              <span className={`ws-validation-score ws-validation-score-${m.status}`}>
                {m.score}
              </span>
              <span className={`ws-validation-badge ws-validation-badge-${m.status}`}>
                {m.status === 'pass' ? 'Pass' : m.status === 'warn' ? 'Warn' : 'Fail'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {report.distributionDistance && (
        <DistributionDistanceSection data={report.distributionDistance} animated={animated} />
      )}

      <button
        className="ws-validation-section-btn"
        onClick={() => setColsExpanded((e) => !e)}
        aria-expanded={colsExpanded}
      >
        <span>Column Fidelity</span>
        <span className={`ws-chevron${colsExpanded ? ' ws-chevron-open' : ''}`}>
          <ChevronDown size={12} strokeWidth={2.5} />
        </span>
      </button>

      {colsExpanded && (
        <div className="ws-validation-cols">
          {report.columns.map((col) => (
            <div key={col.column} className="ws-validation-col-row">
              <span className="ws-validation-col-name">{col.column}</span>
              <div className="ws-validation-mini-track">
                <div
                  className={`ws-validation-mini-bar ws-validation-bar-${col.status}`}
                  style={{ width: animated ? `${col.fidelity}%` : '0%' }}
                />
              </div>
              <span className={`ws-validation-col-pct ws-validation-score-${col.status}`}>
                {col.fidelity}%
              </span>
              <div className={`ws-validation-col-icon ws-validation-col-icon-${col.status}`}>
                {col.status === 'pass' && <Check size={10} strokeWidth={2.5} />}
                {col.status === 'warn' && <WarnIcon />}
              </div>
              {col.note && (
                <span className="ws-validation-col-note">{col.note}</span>
              )}
              {col.skewnessDrift !== undefined && col.skewnessDrift > 0.5 && (
                <span className="ws-validation-col-note ws-validation-col-note-warn">
                  skew drift {col.skewnessDrift.toFixed(2)} (src {col.sourceSkewness?.toFixed(2)} → syn {col.syntheticSkewness?.toFixed(2)})
                </span>
              )}
              {col.boundaryViolations !== undefined && col.boundaryViolations > 0 && (
                <span className="ws-validation-col-note ws-validation-col-note-warn">
                  {col.boundaryViolations.toLocaleString()} out-of-range value{col.boundaryViolations !== 1 ? 's' : ''}
                </span>
              )}
              {col.cardinalityScore !== undefined && col.cardinalityScore < 80 && (
                <span className="ws-validation-col-note ws-validation-col-note-warn">
                  cardinality {col.syntheticCardinality}/{col.sourceCardinality} categories ({col.cardinalityScore}%)
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {report.correlationDrift && (
        <CorrelationDriftSection data={report.correlationDrift} />
      )}

      {report.edgeCases && report.edgeCases.length > 0 && (
        <div className="ws-edge-cases">
          <div className="ws-edge-cases-label">Edge Case Enforcement</div>
          {report.edgeCases.map((ec, i) => {
            const status: ValidationStatus = !ec.parsed
              ? 'warn'
              : ec.satisfied
                ? 'pass'
                : 'fail';
            const pct = ec.actualPct ?? 0;
            const target = ec.targetPct;
            return (
              <div key={i} className="ws-edge-case-row">
                <div className="ws-edge-case-top">
                  <span className="ws-edge-case-desc">{ec.description}</span>
                  <span className={`ws-validation-badge ws-validation-badge-${status}`}>
                    {!ec.parsed ? 'Unparsed' : ec.satisfied ? 'Met' : 'Short'}
                  </span>
                </div>
                {ec.parsed ? (
                  <>
                    <div className="ws-edge-case-bar-track">
                      <div
                        className={`ws-edge-case-bar ws-validation-bar-${status}`}
                        style={{
                          width: animated ? `${Math.min(100, (pct / Math.max(target, 0.01)) * 100)}%` : '0%',
                        }}
                      />
                      <div
                        className="ws-edge-case-target-marker"
                        style={{ left: '100%' }}
                      />
                    </div>
                    <div className="ws-edge-case-count">
                      {ec.actualCount?.toLocaleString() ?? '—'} of{' '}
                      {ec.targetCount?.toLocaleString() ?? '—'} target rows ({pct}% / {target}%)
                    </div>
                  </>
                ) : (
                  <div className="ws-edge-case-error">
                    {ec.error ?? 'Could not parse — consider rephrasing with column names'}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {(report.duplicates || (report.diversityIssues && report.diversityIssues.length > 0) || report.kAnonymity) && (
        <div className="ws-sanity-section">
          <div className="ws-sanity-label">Distribution Sanity</div>
          {report.duplicates && (
            <div className={`ws-sanity-row ws-sanity-${report.duplicates.status}`}>
              <span className="ws-sanity-name">Duplicate rows</span>
              <span className="ws-sanity-value">
                {report.duplicates.count.toLocaleString()} ({report.duplicates.pct}%)
              </span>
              <span className={`ws-validation-badge ws-validation-badge-${report.duplicates.status}`}>
                {report.duplicates.status === 'pass'
                  ? 'OK'
                  : report.duplicates.status === 'warn'
                    ? 'Elevated'
                    : 'High'}
              </span>
            </div>
          )}
          {report.diversityIssues?.map((issue, i) => (
            <div key={i} className={`ws-sanity-row ws-sanity-${issue.status}`}>
              <span className="ws-sanity-name">{issue.column}</span>
              <span className="ws-sanity-value">{issue.detail}</span>
              <span className={`ws-validation-badge ws-validation-badge-${issue.status}`}>
                {issue.issue === 'constant' ? 'Constant' : 'Mode-dominant'}
              </span>
            </div>
          ))}
          {report.kAnonymity && <KAnonymityRow data={report.kAnonymity} />}
        </div>
      )}

      <div className="ws-validation-insights">
        <div className="ws-validation-insights-label">Key Insights</div>
        {report.insights.map((insight, i) => (
          <div key={i} className="ws-validation-insight">
            <span className="ws-validation-insight-dot" />
            <span>{insight}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
