import type { SchemaRow } from '../../constants/mockWorkspace';

interface SchemaCardProps {
  rows: SchemaRow[];
}

export default function SchemaCard({ rows }: SchemaCardProps) {
  return (
    <div className="ws-schema-card">
      <div className="ws-schema-header">
        <span className="ws-schema-title">Inferred schema · {rows.length} columns</span>
      </div>
      <div className="ws-schema-table">
        <div className="ws-schema-row ws-schema-row-head">
          <span>Column</span>
          <span>Type</span>
          <span>Distribution</span>
          <span>Sample</span>
        </div>
        {rows.map((r) => (
          <div key={r.column} className="ws-schema-row">
            <span className="ws-schema-col-name">{r.column}</span>
            <span className="ws-schema-col-type">{r.type}</span>
            <span className="ws-schema-col-dist">{r.distribution}</span>
            <span className="ws-schema-col-sample">{r.sample}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
