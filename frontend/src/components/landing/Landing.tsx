import type { Dispatch, SetStateAction } from 'react';
import Logo from '../Logo';
import PromptBox from './PromptBox';
import type { Profile } from '../../constants/integrations';
import type { SchemaColumn } from '../../api/client';

interface LandingProps {
  prompt: string;
  setPrompt: (value: string) => void;
  onSubmit: () => void;
  submitting: boolean;
  profiles: Profile[];
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  selectedId: string;
  setSelectedId: Dispatch<SetStateAction<string>>;
  groundingFiles?: File[];
  onGroundingFilesChange?: (files: File[]) => void;
  inferredSchema?: SchemaColumn[] | null;
  schemaInferring?: boolean;
  schemaError?: string | null;
  profileSummary?: { columns: number; sourceRows: number } | null;
  rowCount: number;
  setRowCount: (n: number) => void;
  format: 'csv' | 'jsonl' | 'parquet';
  setFormat: (f: 'csv' | 'jsonl' | 'parquet') => void;
}

const OUTPUT_CHIPS = [
  '↓ CSV · JSONL · Parquet',
  'Validation report',
  'k-Anonymity checked',
];

const MOCK_ROWS = [
  { claim_id: 'CLM-00192', date: '2024-03-12', peril: 'Collision',  loss_amount: 8420,  at_fault: true,  fraud_flag: false },
  { claim_id: 'CLM-00193', date: '2024-03-14', peril: 'Theft',      loss_amount: 22100, at_fault: false, fraud_flag: true  },
  { claim_id: 'CLM-00194', date: '2024-03-15', peril: 'Vandalism',  loss_amount: 1850,  at_fault: false, fraud_flag: false },
  { claim_id: 'CLM-00195', date: '2024-03-18', peril: 'Collision',  loss_amount: 5670,  at_fault: true,  fraud_flag: false },
  { claim_id: 'CLM-00196', date: '2024-03-19', peril: 'Fire',       loss_amount: 41300, at_fault: false, fraud_flag: true  },
];
const MOCK_NUMERIC_COLS = new Set(['loss_amount']);

const SAMPLE_PROMPTS = [
  'Generate 10k auto fraud claims with realistic patterns — amplify the rare cases so models can actually learn',
  'Suspicious-claim cohort: policyholder at fault, recent address change, no witnesses or police report',
  'Balanced training set focused on young drivers and All Perils coverage where fraud rates run highest',
  'Edge cases for fraud detection — rural accidents, utility vehicles, and brand-new policyholders filing right away',
];

function buildSteps(fileNames: string[]) {
  const fileRef =
    fileNames.length === 0
      ? 'your connected sources'
      : fileNames.length === 1
        ? fileNames[0]
        : `${fileNames.slice(0, 2).join(', ')}${fileNames.length > 2 ? ` +${fileNames.length - 2} more` : ''}`;

  return [
    {
      num: '01',
      title: 'Schema inference',
      desc: `Scans ${fileRef} to extract claim signatures, infer types, and model severity, frequency, and report-lag distributions.`,
    },
    {
      num: '02',
      title: 'Sample synthesis',
      desc: 'Generates loss-ratio-faithful rows with Poisson frequency, lognormal severity, and preserved peril–geo correlations.',
    },
    {
      num: '03',
      title: 'Fidelity validation',
      desc: 'Cross-validates synthesized distributions against source bordereaux and issues an actuarial fidelity report before delivery.',
    },
  ];
}

export default function Landing(props: LandingProps) {
  const {
    groundingFiles = [],
    onGroundingFilesChange,
    inferredSchema,
    schemaInferring,
    schemaError,
    profileSummary,
    rowCount,
    setRowCount,
    format,
    setFormat,
    ...promptProps
  } = props;

  const steps = buildSteps(groundingFiles.map((f) => f.name));
  const schemaRows = inferredSchema ? inferredSchema.slice(0, 4) : null;
  const extraCols = inferredSchema && inferredSchema.length > 4 ? inferredSchema.length - 4 : 0;
  const mockCols = Object.keys(MOCK_ROWS[0]);

  return (
    <div className="landing">
      <h1 className="landing-title">
        <Logo className="landing-title-logo" />
        What claims data should we synthesize?
      </h1>

      <PromptBox
        {...promptProps}
        onGroundingFilesChange={onGroundingFilesChange}
        profileSummary={profileSummary}
        rowCount={rowCount}
        setRowCount={setRowCount}
        format={format}
        setFormat={setFormat}
      />
      {schemaError && (
        <p className="landing-schema-error" role="alert">{schemaError}</p>
      )}

      <div className="landing-output-chips" aria-label="Output artifacts">
        {OUTPUT_CHIPS.map((chip) => (
          <span key={chip} className="landing-output-chip">{chip}</span>
        ))}
      </div>

      <div className="landing-samples" role="list" aria-label="Sample prompts">
        {SAMPLE_PROMPTS.map((text) => (
          <button
            key={text}
            type="button"
            role="listitem"
            className="landing-sample-chip"
            onClick={() => promptProps.setPrompt(text)}
          >
            {text}
          </button>
        ))}
      </div>

      <div className="landing-steps">
        {steps.map((s) => (
          <div key={s.num} className="landing-step">
            <span className="landing-step-num">{s.num}</span>
            <span className="landing-step-title">{s.title}</span>
            <p className="landing-step-desc">{s.desc}</p>
          </div>
        ))}
      </div>

      <div className="ws-preview-card landing-mock-table">
        <div className="ws-preview-header">
          <span className="ws-preview-title">Sample output · {MOCK_ROWS.length} rows</span>
          <span className="ws-preview-badge">Synthetic</span>
        </div>
        <div className="ws-preview-scroll">
          <table className="ws-preview-table">
            <thead>
              <tr>
                {mockCols.map((col) => (
                  <th key={col} className={MOCK_NUMERIC_COLS.has(col) ? 'ws-preview-th-num' : ''}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MOCK_ROWS.map((row, i) => (
                <tr key={i}>
                  {mockCols.map((col) => {
                    const val = row[col as keyof typeof row];
                    const isNum = MOCK_NUMERIC_COLS.has(col);
                    return (
                      <td
                        key={col}
                        className={isNum ? 'ws-preview-td-num' : ''}
                      >
                        {isNum ? (val as number).toLocaleString() : String(val)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {schemaRows && (
        <div className="landing-schema-card">
          <span className="landing-output-label">
            Inferred schema
            {schemaInferring && <span className="landing-output-inferring">inferring…</span>}
          </span>
          <div className="landing-schema">
            <div className="landing-schema-header">
              <span>Column</span>
              <span>Type</span>
              <span>Sample</span>
            </div>
            {schemaRows.map((row) => (
              <div key={row.column} className="landing-schema-row">
                <span className="landing-schema-name">{row.column}</span>
                <span className="landing-schema-type">{row.type}</span>
                <span className="landing-schema-sample">{row.sample || '—'}</span>
              </div>
            ))}
            {extraCols > 0 && (
              <div className="landing-schema-extra">+{extraCols} more column{extraCols > 1 ? 's' : ''}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
