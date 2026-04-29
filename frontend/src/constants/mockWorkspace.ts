export type ValidationStatus = 'pass' | 'warn' | 'fail';

export interface ValidationMetric {
  label: string;
  score: number;
  status: ValidationStatus;
}

export interface ValidationColumnResult {
  column: string;
  fidelity: number;
  status: ValidationStatus;
  note?: string;
}

export interface EdgeCaseCoverage {
  requested: number;
  generated: number;
  description: string;
  coveragePct: number;
}

export interface ValidationReport {
  verdict: string;
  verdictStatus: ValidationStatus;
  edgeCaseCoverage?: EdgeCaseCoverage;
  metrics: ValidationMetric[];
  columns: ValidationColumnResult[];
  insights: string[];
}

export const VALIDATION_REPORT: ValidationReport = {
  verdict: 'Ready for fine-tuning',
  verdictStatus: 'pass',
  edgeCaseCoverage: {
    requested: 500,
    generated: 482,
    description: 'HbA1c > 12 + 3+ comorbidities',
    coveragePct: 96.4,
  },
  metrics: [
    { label: 'Realism', score: 94, status: 'pass' },
    { label: 'Diversity', score: 87, status: 'pass' },
    { label: 'Safety / PII', score: 100, status: 'pass' },
  ],
  columns: [
    { column: 'patient_id', fidelity: 100, status: 'pass' },
    { column: 'age', fidelity: 96, status: 'pass' },
    { column: 'sex', fidelity: 99, status: 'pass' },
    { column: 'hba1c', fidelity: 83, status: 'warn', note: 'Mild tail compression vs. source' },
    { column: 'comorbidities', fidelity: 91, status: 'pass' },
    { column: 'last_visit', fidelity: 88, status: 'pass' },
  ],
  insights: [
    'hba1c shows mild tail compression — consider increasing variance in generation params',
    'No PII detected across all 10,000 rows',
    'Inter-column correlations preserved within 4% of source statistics',
  ],
};

export interface SchemaRow {
  column: string;
  type: string;
  distribution: string;
  sample: string;
}

export type AgentStatus = 'queued' | 'running' | 'done';

export interface AgentTask {
  id: string;
  name: string;
  detail: string;
  status: AgentStatus;
}

export interface PlanStep {
  title: string;
  description: string;
}

export interface ExecutionPlan {
  intro: string;
  steps: PlanStep[];
}

export type WorkspaceMessage =
  | { id: string; role: 'user'; text: string }
  | {
      id: string;
      role: 'assistant';
      loading: boolean;
      plan?: ExecutionPlan;
      planText?: string;
      agents?: AgentTask[];
      schema?: SchemaRow[];
    };

export const SCHEMA_TAG = 'APERTURE V1 · CLINICAL';

export const SCHEMA_ROWS: SchemaRow[] = [
  { column: 'patient_id', type: 'uuid', distribution: 'unique', sample: 'pt_8f2…a4' },
  { column: 'age', type: 'int', distribution: 'Gaussian (μ=52, σ=8)', sample: '54' },
  { column: 'sex', type: 'enum', distribution: '{F:0.52, M:0.48}', sample: 'F' },
  { column: 'hba1c', type: 'float', distribution: 'LogNormal (μ=1.9)', sample: '7.2' },
  { column: 'comorbidities', type: 'array<str>', distribution: 'Multi-label', sample: '[HTN, CKD]' },
  { column: 'last_visit', type: 'date', distribution: 'Uniform 2023-2025', sample: '2024-08-12' },
];

function sourceRef(names: string[]): string {
  if (names.length === 0) return 'connected sources';
  if (names.length === 1) return names[0];
  const shown = names.slice(0, 2).join(', ');
  return names.length > 2 ? `${shown} +${names.length - 2} more` : shown;
}

export function buildExecutionPlan(sourceNames: string[]): ExecutionPlan {
  const ref = sourceRef(sourceNames);
  return {
    intro: "Here's my execution plan. Agents run sequentially — each step resolves before the next starts.",
    steps: [
      {
        title: 'Schema Inference',
        description: `Recursively scan ${ref} to extract column signatures, infer data types, and model statistical distributions across all sources.`,
      },
      {
        title: 'Sample Synthesis',
        description: 'Generate 10,000 rows that are statistically faithful to the inferred schema, respecting inter-column correlations and your profile constraints.',
      },
      {
        title: 'Fidelity Validation',
        description: 'Cross-validate synthesized distributions against source samples. Flag drift, check coverage, and confirm quality thresholds before delivery.',
      },
    ],
  };
}

export function buildAgents(sourceNames: string[]): AgentTask[] {
  const ref = sourceRef(sourceNames);
  return [
    {
      id: 'a1',
      name: 'Schema Inference',
      detail: `Recursively sampling ${ref} to extract column signatures and type distributions`,
      status: 'queued',
    },
    {
      id: 'a2',
      name: 'Sample Synthesizer',
      detail: 'Generating 10,000 statistically faithful rows from the inferred schema and profile constraints',
      status: 'queued',
    },
    {
      id: 'a3',
      name: 'Fidelity Check',
      detail: 'Validating sample distributions match source statistics and flagging drift',
      status: 'queued',
    },
  ];
}
