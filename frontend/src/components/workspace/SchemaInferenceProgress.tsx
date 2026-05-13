import type { SchemaInferenceState } from '../../constants/mockWorkspace';

interface Props {
  state: SchemaInferenceState;
}

export default function SchemaInferenceProgress({ state }: Props) {
  if (state.phase === 'idle') return null;

  const total = state.phase === 'scanning' || state.phase === 'fitting' || state.phase === 'done' ? state.total : 0;
  const idx = state.phase === 'scanning' ? state.idx : total;
  const pct = total > 0 ? Math.min(100, Math.round((idx / total) * 100)) : 0;
  const sourceRows =
    state.phase === 'scanning' || state.phase === 'fitting' || state.phase === 'done'
      ? state.sourceRows
      : 0;

  const headline =
    state.phase === 'scanning'
      ? `Profiling column ${idx} of ${total}${state.latest ? ` · ${state.latest}` : ''}`
      : state.phase === 'fitting'
        ? `Fitting GaussianCopula synthesizer · ${total} cols × ${sourceRows.toLocaleString()} rows`
        : `Schema inference complete · ${total} columns`;

  return (
    <div className={`schema-progress phase-${state.phase}`}>
      <div className="schema-progress-head">
        <span className="schema-progress-tag">
          {state.phase === 'fitting' ? 'fitting' : state.phase === 'done' ? 'done' : 'scanning'}
        </span>
        <span className="schema-progress-headline">{headline}</span>
        {state.phase !== 'done' && <span className="schema-progress-dot pulse" />}
      </div>
      <div className="schema-progress-bar">
        <div
          className="schema-progress-bar-fill"
          style={{ width: state.phase === 'fitting' || state.phase === 'done' ? '100%' : `${pct}%` }}
        />
      </div>
    </div>
  );
}
