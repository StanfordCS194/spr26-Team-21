interface Props {
  rows: Record<string, unknown>[];
}

export default function PreviewTable({ rows }: Props) {
  if (rows.length === 0) return null;

  const columns = Object.keys(rows[0]);

  return (
    <div className="ws-preview-card">
      <div className="ws-preview-header">
        <span className="ws-preview-title">Sample preview · {rows.length} rows</span>
        <span className="ws-preview-badge">Synthetic</span>
      </div>
      <div className="ws-preview-scroll">
        <table className="ws-preview-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => {
                  const val = String(row[col] ?? '');
                  return (
                    <td key={col} title={val}>
                      {val.length > 18 ? val.slice(0, 16) + '…' : val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
