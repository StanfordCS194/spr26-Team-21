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
  profileSummary?: { columns: number; sourceRows: number } | null;
}

const VALIDATION_LABELS = ['Realism', 'Diversity', 'Safety / PII'];

const PLACEHOLDER_ROWS = [null, null, null] as const;

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
    profileSummary,
    ...promptProps
  } = props;

  const steps = buildSteps(groundingFiles.map((f) => f.name));
  const schemaRows = inferredSchema ? inferredSchema.slice(0, 4) : null;
  const extraCols = inferredSchema && inferredSchema.length > 4 ? inferredSchema.length - 4 : 0;

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
      />

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

      <div className="landing-output">
        <div className="landing-output-col">
          <span className="landing-output-label">
            Schema
            {schemaInferring && <span className="landing-output-inferring">inferring…</span>}
          </span>
          <div className="landing-schema">
            <div className="landing-schema-header">
              <span>Column</span>
              <span>Type</span>
              <span>Sample</span>
            </div>
            {schemaRows
              ? schemaRows.map((row) => (
                  <div key={row.column} className="landing-schema-row">
                    <span className="landing-schema-name">{row.column}</span>
                    <span className="landing-schema-type">{row.type}</span>
                    <span className="landing-schema-sample">{row.sample || '—'}</span>
                  </div>
                ))
              : PLACEHOLDER_ROWS.map((_, i) => (
                  <div key={i} className="landing-schema-row landing-schema-row-empty">
                    <span className="landing-schema-name">—</span>
                    <span className="landing-schema-type">—</span>
                    <span className="landing-schema-sample">—</span>
                  </div>
                ))}
            {extraCols > 0 && (
              <div className="landing-schema-extra">+{extraCols} more column{extraCols > 1 ? 's' : ''}</div>
            )}
          </div>
        </div>

        <div className="landing-output-col">
          <span className="landing-output-label">Validation report</span>
          <div className="landing-vp-metrics">
            {VALIDATION_LABELS.map((label) => (
              <div key={label} className="landing-vp-row">
                <span className="landing-vp-row-label">{label}</span>
                <div className="landing-vp-track" />
                <span className="landing-vp-score landing-vp-score-empty">—</span>
              </div>
            ))}
          </div>
          <span className="landing-vp-hint">Scores appear after generation</span>
        </div>
      </div>
    </div>
  );
}
