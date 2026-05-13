import type { AgentTurn } from '../../constants/mockWorkspace';

interface AgentTraceProps {
  turns: AgentTurn[];
}

export default function AgentTrace({ turns }: AgentTraceProps) {
  if (!turns.length) return null;

  return (
    <div className="agent-trace">
      <div className="agent-trace-header">
        <span className="agent-trace-title">Sourcing Agent</span>
        <span className="agent-trace-meta">{turns.length} step{turns.length === 1 ? '' : 's'}</span>
      </div>
      <ol className="agent-trace-list">
        {turns.map((t) => (
          <li key={t.turn} className={`agent-trace-turn status-${t.status}`}>
            <div className="agent-trace-turn-head">
              <span className="agent-trace-turn-num">Turn {t.turn}</span>
              <span className="agent-trace-tool">{t.tool}</span>
              <span className="agent-trace-status">
                {t.status === 'running' && <span className="agent-trace-dot pulse" />}
                {t.status === 'done' && (
                  <>
                    <span className="agent-trace-check">✓</span>
                    {typeof t.durationMs === 'number' && (
                      <span className="agent-trace-duration">{(t.durationMs / 1000).toFixed(1)}s</span>
                    )}
                  </>
                )}
                {t.status === 'error' && <span className="agent-trace-err">!</span>}
              </span>
            </div>
            <div className="agent-trace-turn-body">
              <div className="agent-trace-input">{t.inputSummary}</div>
              {t.resultSummary && (
                <div className="agent-trace-result">→ {t.resultSummary}</div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
