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

export interface EdgeCaseResult {
  description: string;
  parsed: boolean;
  targetPct: number;
  actualPct: number | null;
  targetCount?: number;
  actualCount?: number;
  satisfied: boolean;
  error?: string;
}

export interface DuplicatesCheck {
  count: number;
  pct: number;
  status: ValidationStatus;
}

export interface DiversityIssue {
  column: string;
  issue: 'constant' | 'mode_dominance';
  detail: string;
  status: ValidationStatus;
}

export interface ValidationReport {
  verdict: string;
  verdictStatus: ValidationStatus;
  edgeCaseCoverage?: EdgeCaseCoverage;
  edgeCases?: EdgeCaseResult[];
  duplicates?: DuplicatesCheck;
  diversityIssues?: DiversityIssue[];
  metrics: ValidationMetric[];
  columns: ValidationColumnResult[];
  insights: string[];
}

export const VALIDATION_REPORT: ValidationReport = {
  verdict: 'Ready for fraud model training',
  verdictStatus: 'pass',
  edgeCaseCoverage: {
    requested: 500,
    generated: 482,
    description: 'Staged-accident patterns with 3+ provider-billing anomalies',
    coveragePct: 96.4,
  },
  metrics: [
    { label: 'Actuarial fidelity', score: 94, status: 'pass' },
    { label: 'Tail coverage', score: 87, status: 'pass' },
    { label: 'PII / GLBA safety', score: 100, status: 'pass' },
  ],
  columns: [
    { column: 'claim_id', fidelity: 100, status: 'pass' },
    { column: 'policy_id', fidelity: 99, status: 'pass' },
    { column: 'peril', fidelity: 96, status: 'pass' },
    { column: 'claim_amount', fidelity: 83, status: 'warn', note: 'Mild tail compression on severity > $250k' },
    { column: 'FNOL_date', fidelity: 91, status: 'pass' },
    { column: 'report_lag_days', fidelity: 88, status: 'pass' },
  ],
  insights: [
    'Severity tail compressed above $250k — consider raising lognormal σ in generation params',
    'No PII detected across all 10,000 rows — GLBA Safeguards Rule compliant',
    'Loss ratio preserved within 1.3% of source; peril–geo copula structure intact',
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

export type ApprovalState = 'idle' | 'generating' | 'validating' | 'complete';

export interface ClarifyingQuestion {
  id: string;
  question: string;
  answer?: string;
}

export interface GenerationSpec {
  row_count: number;
  format: 'csv' | 'jsonl' | 'parquet';
  labels: string[];
  edge_cases: string[];
  constraints: string[];
}

export type AgentTurnStatus = 'running' | 'done' | 'error';

export interface AgentTurn {
  turn: number;
  tool: string;
  inputSummary: string;
  resultSummary?: string;
  durationMs?: number;
  status: AgentTurnStatus;
}

export interface GroundingStrategyData {
  rationale: string;
  totalRows?: number;
  queries: Array<{ label: string; rows: number; filter: Record<string, unknown> }>;
}

export type SchemaInferenceState =
  | { phase: 'idle' }
  | { phase: 'scanning'; idx: number; total: number; latest?: string; sourceRows: number }
  | { phase: 'fitting'; total: number; sourceRows: number }
  | { phase: 'done'; total: number; sourceRows: number };

export type WorkspaceMessage =
  | { id: string; role: 'user'; text: string }
  | {
      id: string;
      role: 'assistant';
      loading: boolean;
      plan?: ExecutionPlan;
      planText?: string;
      agents?: AgentTask[];
      agentTurns?: AgentTurn[];
      groundingStrategy?: GroundingStrategyData;
      schemaInference?: SchemaInferenceState;
      schema?: SchemaRow[];
      initialApproval?: ApprovalState;
      sourceStats?: Record<string, Record<string, unknown>>;
      originalPrompt?: string;
      clarifyingQuestions?: ClarifyingQuestion[];
      generationSpec?: GenerationSpec;
      schemaSource?: 'llm' | 'upload' | 'agent';
      modelId?: string | null;
      host?: string;
    };

export const SCHEMA_TAG = 'APERTURE V1 · P&C CLAIMS';

export const SCHEMA_ROWS: SchemaRow[] = [
  { column: 'claim_id', type: 'uuid', distribution: 'unique', sample: 'clm_8f2…a4' },
  { column: 'policy_id', type: 'uuid', distribution: 'unique (1:N claims)', sample: 'pol_b71…e9' },
  { column: 'peril', type: 'enum', distribution: '{collision:0.41, theft:0.18, liab:0.22, comp:0.19}', sample: 'collision' },
  { column: 'claim_amount', type: 'float', distribution: 'LogNormal (μ=8.4, σ=1.3)', sample: '4,820.50' },
  { column: 'FNOL_date', type: 'date', distribution: 'Uniform 2023-2025', sample: '2024-08-12' },
  { column: 'report_lag_days', type: 'int', distribution: 'Weibull (k=1.4, λ=6)', sample: '4' },
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
        description: `Recursively scan ${ref} to extract claim signatures, infer data types, and model severity, frequency, and report-lag distributions across all sources.`,
      },
      {
        title: 'Sample Synthesis',
        description: 'Generate 10,000 loss-ratio-faithful claim records with Poisson frequency, lognormal severity, and preserved peril–geo correlations.',
      },
      {
        title: 'Fidelity Validation',
        description: 'Cross-validate synthesized distributions against source bordereaux. Check loss-ratio drift, severity tail coverage, and GLBA-aligned PII safety before delivery.',
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
      detail: `Recursively sampling ${ref} to extract claim signatures and severity/frequency distributions`,
      status: 'queued',
    },
    {
      id: 'a2',
      name: 'Sample Synthesizer',
      detail: 'Generating 10,000 loss-ratio-faithful claim rows from the inferred schema and profile constraints',
      status: 'queued',
    },
    {
      id: 'a3',
      name: 'Fidelity Check',
      detail: 'Validating loss ratio, severity tails, and peril–geo copula structure against source bordereaux',
      status: 'queued',
    },
  ];
}

const DEMO_SOURCES = ['Amazon S3', 'PostgreSQL', 'MongoDB'];

export const DEMO_MESSAGES: WorkspaceMessage[] = [
  {
    id: 'demo-user-1',
    role: 'user',
    text: 'Generate 10,000 auto bodily-injury claim records grounded in our claims warehouse — include edge cases for staged-accident fraud with 3+ provider-billing anomalies',
  },
  {
    id: 'demo-assistant-1',
    role: 'assistant',
    loading: false,
    plan: buildExecutionPlan(DEMO_SOURCES),
    agents: buildAgents(DEMO_SOURCES).map((a) => ({ ...a, status: 'done' as AgentStatus })),
    schema: SCHEMA_ROWS,
    initialApproval: 'complete',
  },
];
