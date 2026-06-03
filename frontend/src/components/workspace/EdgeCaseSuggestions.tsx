import { useEffect, useState } from 'react';
import { Check, Plus } from '../icons/Icons';
import {
  discoverEdgeCases,
  type EdgeCaseSeverity,
  type EdgeCaseSuggestion,
  type SchemaColumn,
  type SourceStats,
} from '../../api/client';

interface Props {
  schemaColumns: SchemaColumn[];
  sourceStats: SourceStats;
  sourceId?: string | null;
  approvedConditions: string[];
  onApprovedChange: (next: string[]) => void;
}

type Status = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

function pctText(pct: number | null): string {
  if (pct === null) return '—';
  if (pct < 0.1) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(1)}%`;
}

function severityLabel(s: EdgeCaseSeverity): string {
  return s === 'high' ? 'High priority' : s === 'medium' ? 'Medium' : 'Low';
}

function detectorLabel(d: string): string {
  if (d === 'sparse_categorical') return 'Rare category';
  if (d === 'tail_outlier') return 'Distribution tail';
  if (d === 'class_imbalance') return 'Class imbalance';
  if (d === 'sparse_conjunction') return 'Rare combination';
  if (d === 'domain_llm') return 'Domain expert';
  return d;
}

export default function EdgeCaseSuggestions({
  schemaColumns,
  sourceStats,
  sourceId,
  approvedConditions,
  onApprovedChange,
}: Props) {
  const [status, setStatus] = useState<Status>('idle');
  const [suggestions, setSuggestions] = useState<EdgeCaseSuggestion[]>([]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fire discovery once we have schema + stats. Re-fires only if columns change.
  const schemaKey = schemaColumns.map((c) => c.column).join('|');
  const hasStats = Object.keys(sourceStats).length > 0;

  useEffect(() => {
    if (!schemaColumns.length || !hasStats) return;
    let cancelled = false;
    setStatus('loading');
    setError(null);
    discoverEdgeCases(schemaColumns, sourceStats, sourceId)
      .then((resp) => {
        if (cancelled) return;
        if (!resp.suggestions || resp.suggestions.length === 0) {
          setStatus('empty');
          setSuggestions([]);
          return;
        }
        setSuggestions(resp.suggestions);
        setStatus('ready');
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus('error');
        setError(err instanceof Error ? err.message : 'Discovery failed');
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemaKey, sourceId]);

  if (status === 'idle' || status === 'empty') return null;

  const approvedSet = new Set(approvedConditions);

  const toggleApprove = (condition: string) => {
    const next = approvedSet.has(condition)
      ? approvedConditions.filter((c) => c !== condition)
      : [...approvedConditions, condition];
    onApprovedChange(next);
  };

  return (
    <div className="ws-eds-card">
      <div className="ws-eds-header">
        <div className="ws-eds-title">Discovered Edge Cases</div>
        <div className="ws-eds-subtitle">
          {status === 'loading'
            ? 'Scanning source data for under-represented patterns…'
            : status === 'error'
              ? error ?? 'Discovery failed'
              : `${suggestions.length} pattern${suggestions.length === 1 ? '' : 's'} found · click to enforce`}
        </div>
      </div>

      {status === 'loading' && (
        <div className="ws-eds-loading">
          <div className="ws-typing">
            <span className="ws-typing-dot" />
            <span className="ws-typing-dot" />
            <span className="ws-typing-dot" />
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="ws-eds-empty">
          Edge-case discovery unavailable on this run. Generation will proceed without suggested
          patterns.
        </div>
      )}

      {status === 'ready' && (
        <div className="ws-eds-list">
          {suggestions.map((s, i) => {
            const approved = approvedSet.has(s.condition_text);
            const expanded = expandedIdx === i;
            return (
              <div
                key={`${s.condition_text}-${i}`}
                className={`ws-eds-chip ws-eds-chip-${s.severity}${
                  approved ? ' ws-eds-chip-approved' : ''
                }`}
              >
                <button
                  type="button"
                  className="ws-eds-chip-main"
                  onClick={() => toggleApprove(s.condition_text)}
                  aria-pressed={approved}
                >
                  <span className="ws-eds-chip-icon">
                    {approved ? <Check size={11} strokeWidth={2.5} /> : <Plus size={11} strokeWidth={2.5} />}
                  </span>
                  <span className="ws-eds-chip-condition">{s.condition_text}</span>
                  <span className="ws-eds-chip-meta">
                    <span className="ws-eds-chip-source">source {pctText(s.source_pct)}</span>
                    <span className="ws-eds-chip-sep">·</span>
                    <span className={`ws-eds-chip-severity ws-eds-sev-${s.severity}`}>
                      {severityLabel(s.severity)}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  className="ws-eds-chip-why"
                  onClick={() => setExpandedIdx(expanded ? null : i)}
                  aria-expanded={expanded}
                >
                  {expanded ? 'Hide' : 'Why?'}
                </button>
                {expanded && (
                  <div className="ws-eds-chip-reason">
                    <div className="ws-eds-chip-reason-tag">
                      {detectorLabel(s.detector)} · target {s.suggested_target_pct}%
                    </div>
                    <div>{s.reason}</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {status === 'ready' && approvedConditions.length > 0 && (
        <div className="ws-eds-footer">
          {approvedConditions.length} edge case{approvedConditions.length === 1 ? '' : 's'} approved
          · will be enforced on next generation
        </div>
      )}
    </div>
  );
}
