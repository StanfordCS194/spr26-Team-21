interface Props {
  rows: Record<string, unknown>[];
}

function isNumericCol(rows: Record<string, unknown>[], col: string): boolean {
  const nonNull = rows.slice(0, 10).map((r) => r[col]).filter((v) => v != null);
  return nonNull.length > 0 && nonNull.every((v) => typeof v === 'number');
}

export default function PreviewTable({ rows }: Props) {
  if (rows.length === 0) return null;

  const columns = Object.keys(rows[0]);
  const numericCols = new Set(columns.filter((c) => isNumericCol(rows, c)));

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
                <th key={col} className={numericCols.has(col) ? 'ws-preview-th-num' : ''}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => {
                  const raw = row[col];
                  const isNull = raw == null;
                  const val = isNull ? '' : String(raw);
                  const display = val.length > 28 ? val.slice(0, 26) + '…' : val;
                  return (
                    <td
                      key={col}
                      title={isNull ? undefined : val.length > 28 ? val : undefined}
                      className={[
                        isNull ? 'ws-preview-null' : '',
                        numericCols.has(col) ? 'ws-preview-td-num' : '',
                      ]
                        .filter(Boolean)
                        .join(' ') || undefined}
                    >
                      {isNull ? 'null' : display}
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
