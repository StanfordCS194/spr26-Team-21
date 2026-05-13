import type { GroundingStrategyData } from '../../constants/mockWorkspace';

interface GroundingStrategyProps {
  strategy: GroundingStrategyData;
}

export default function GroundingStrategy({ strategy }: GroundingStrategyProps) {
  return (
    <div className="grounding-strategy">
      <div className="grounding-strategy-header">
        <span className="grounding-strategy-tag">Strategy</span>
        {typeof strategy.totalRows === 'number' && (
          <span className="grounding-strategy-rows">
            {strategy.totalRows.toLocaleString()} rows grounded
          </span>
        )}
      </div>
      <p className="grounding-strategy-rationale">{strategy.rationale}</p>
      {strategy.queries.length > 0 && (
        <ul className="grounding-strategy-queries">
          {strategy.queries.map((q, idx) => (
            <li key={idx} className="grounding-strategy-query">
              <span className="grounding-strategy-query-label">{q.label || `slice ${idx + 1}`}</span>
              <span className="grounding-strategy-query-rows">
                {q.rows.toLocaleString()} rows
              </span>
              <code className="grounding-strategy-query-filter">{JSON.stringify(q.filter)}</code>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
