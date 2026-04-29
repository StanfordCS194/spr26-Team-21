import { useState } from 'react';
import { Check, ChevronDown } from '../icons/Icons';
import type { ValidationReport as ValidationReportType } from '../../constants/mockWorkspace';

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

export default function ValidationReport({ report }: Props) {
  const [colsExpanded, setColsExpanded] = useState(true);

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
              style={{ width: `${report.edgeCaseCoverage.coveragePct}%` }}
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
                style={{ width: `${m.score}%` }}
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
                  style={{ width: `${col.fidelity}%` }}
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
            </div>
          ))}
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
