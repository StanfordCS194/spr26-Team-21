export interface SchemaColumn {
  column: string;
  type: string;
  distribution: string;
  sample: string;
  null_pct?: number;
}

export type SourceStats = Record<string, Record<string, unknown>>;

export interface InferSchemaResponse {
  columns: SchemaColumn[];
  stats: SourceStats;
  source_rows: number;
  model_id?: string | null;
  source_id?: string;
  error?: string;
}

export interface ValidationMetric {
  label: string;
  score: number;
  status: 'pass' | 'warn' | 'fail';
}

export interface ValidationColumnResult {
  column: string;
  fidelity: number;
  status: 'pass' | 'warn' | 'fail';
  note?: string;
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
  status: 'pass' | 'warn' | 'fail';
}

export interface DiversityIssue {
  column: string;
  issue: 'constant' | 'mode_dominance';
  detail: string;
  status: 'pass' | 'warn' | 'fail';
}

export interface ValidationReportData {
  verdict: string;
  verdictStatus: 'pass' | 'warn' | 'fail';
  metrics: ValidationMetric[];
  columns: ValidationColumnResult[];
  insights: string[];
  edgeCases?: EdgeCaseResult[];
  duplicates?: DuplicatesCheck;
  diversityIssues?: DiversityIssue[];
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface FeatureAblationEntry {
  feature: string;
  importance: number;
  recall_delta_pct: number;
  baseline_recall: number;
  permuted_recall: number;
  interpretation: 'high reliance' | 'useful' | 'marginal' | 'harmful when present';
}

export interface MisclassificationOverlap {
  tstr_only_wrong: number[];
  trtr_only_wrong: number[];
  both_wrong: number[];
  augmentation_saves: number[];
  counts: {
    tstr_only_wrong: number;
    trtr_only_wrong: number;
    both_wrong: number;
    augmentation_saves: number;
    total_test_rows: number;
  };
  summary: string;
}

export interface DiagnosticsReport {
  available: boolean;
  target?: string;
  n_test?: number;
  confusion_matrices?: {
    trtr: ConfusionMatrix;
    tstr: ConfusionMatrix;
    augmented: ConfusionMatrix;
  };
  feature_ablation?: FeatureAblationEntry[];
  misclassification_overlap?: MisclassificationOverlap;
  observations?: string[];
  recommendations?: string[];
}

export interface GenerateResponse {
  session_id: string;
  row_count: number;
  file_size_kb: number;
  format?: string;
  filename?: string;
  validation: ValidationReportData;
  diagnostics?: DiagnosticsReport | null;
}

export interface ClarifyingQuestion {
  id: string;
  question: string;
}

export interface GenerationSpec {
  row_count: number;
  format: 'csv' | 'jsonl' | 'parquet';
  labels: string[];
  edge_cases: string[];
  constraints: string[];
}

export interface PlanResponse {
  schema: SchemaColumn[];
  generation_spec: GenerationSpec;
  generation_plan: {
    intro: string;
    steps: Array<{ title: string; description: string }>;
  };
  clarifying_questions: ClarifyingQuestion[];
}

const BASE = '/api';

export async function inferSchema(files: File[]): Promise<InferSchemaResponse> {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  const res = await fetch(`${BASE}/infer-schema`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Schema inference failed: ${res.statusText}`);
  return res.json();
}

export async function planFromPrompt(prompt: string): Promise<PlanResponse> {
  const res = await fetch(`${BASE}/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`Plan generation failed: ${res.statusText}`);
  return res.json();
}

export async function previewDataset(
  schemaColumns: SchemaColumn[],
  sourceStats: SourceStats,
  modelId?: string | null,
  edgeCases: string[] = [],
): Promise<{ rows: Record<string, unknown>[] }> {
  const res = await fetch(`${BASE}/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema_columns: schemaColumns,
      source_stats: sourceStats,
      row_count: 10,
      prompt: '',
      model_id: modelId ?? null,
      edge_cases: edgeCases,
    }),
  });
  if (!res.ok) throw new Error(`Preview failed: ${res.statusText}`);
  return res.json();
}

export async function generate(
  schemaColumns: SchemaColumn[],
  sourceStats: SourceStats,
  rowCount: number,
  prompt: string,
  format: 'csv' | 'jsonl' | 'parquet' = 'csv',
  edgeCases: string[] = [],
  modelId?: string | null,
  sourceId?: string | null,
  discoveredSuggestions: EdgeCaseSuggestion[] = [],
): Promise<GenerateResponse> {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema_columns: schemaColumns,
      source_stats: sourceStats,
      row_count: rowCount,
      prompt,
      format,
      edge_cases: edgeCases,
      model_id: modelId ?? null,
      source_id: sourceId ?? null,
      discovered_suggestions: discoveredSuggestions,
    }),
  });
  if (!res.ok) throw new Error(`Generation failed: ${res.statusText}`);
  return res.json();
}

export function downloadUrl(sessionId: string): string {
  return `${BASE}/download/${sessionId}`;
}

export function reportUrl(sessionId: string): string {
  return `${BASE}/report/${sessionId}`;
}

// ── Edge-case discovery ──────────────────────────────────────────────────

export type EdgeCaseSeverity = 'high' | 'medium' | 'low';

export interface EdgeCaseSuggestion {
  description: string;
  condition_text: string;
  source_pct: number | null;
  suggested_target_pct: number;
  reason: string;
  severity: EdgeCaseSeverity;
  detector: string;
}

export interface DiscoverEdgeCasesResponse {
  suggestions: EdgeCaseSuggestion[];
}

export async function discoverEdgeCases(
  schemaColumns: SchemaColumn[],
  sourceStats: SourceStats,
  sourceId?: string | null,
): Promise<DiscoverEdgeCasesResponse> {
  const res = await fetch(`${BASE}/discover-edge-cases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema_columns: schemaColumns,
      source_stats: sourceStats,
      source_id: sourceId ?? null,
    }),
  });
  if (!res.ok) throw new Error(`Edge-case discovery failed: ${res.statusText}`);
  return res.json();
}

export interface MongoTestResponse {
  ok: boolean;
  host?: string;
  databases?: Array<{ name: string }>;
  error?: string;
}

export interface MongoCollectionsResponse {
  ok: boolean;
  collections?: Array<{ name: string; count: number | null }>;
  error?: string;
}

export interface MongoInferResponse extends InferSchemaResponse {
  host?: string;
}

export async function mongoTest(uri: string): Promise<MongoTestResponse> {
  const res = await fetch(`${BASE}/sources/mongo/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uri }),
  });
  if (!res.ok) throw new Error(`Mongo test failed: ${res.statusText}`);
  return res.json();
}

export async function mongoListCollections(
  uri: string,
  db: string,
): Promise<MongoCollectionsResponse> {
  const res = await fetch(`${BASE}/sources/mongo/collections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uri, db }),
  });
  if (!res.ok) throw new Error(`Mongo collections failed: ${res.statusText}`);
  return res.json();
}

export async function mongoInferSchema(
  uri: string,
  db: string,
  collection: string,
): Promise<MongoInferResponse> {
  const res = await fetch(`${BASE}/sources/mongo/infer-schema`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uri, db, collection }),
  });
  if (!res.ok) throw new Error(`Mongo schema inference failed: ${res.statusText}`);
  return res.json();
}

// ── Sourcing agent (SSE stream) ──────────────────────────────────────────

export type AgentEvent =
  | { type: 'step_start'; turn: number; tool: string; input_summary: string }
  | { type: 'step_complete'; turn: number; result_summary: string; duration_ms: number }
  | { type: 'rationale'; text: string; queries: Array<Record<string, unknown>> }
  | { type: 'schema_start'; total_columns: number; source_rows: number }
  | { type: 'schema_column'; idx: number; total: number; column: SchemaColumn }
  | { type: 'fitting_model'; total_columns: number; source_rows: number }
  | ({ type: 'final' } & MongoInferResponse & {
        grounding_strategy?: {
          rationale: string;
          queries: Array<{ label: string; rows: number; filter: Record<string, unknown> }>;
          total_rows: number;
        };
      })
  | { type: 'error'; message: string };

async function pumpAgentStream(
  res: Response,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  if (!res.body) throw new Error('No response body from auto-infer');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = raw.trim();
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        onEvent(JSON.parse(payload) as AgentEvent);
      } catch {
        // ignore malformed event
      }
    }
  }
}

export async function mongoAutoInferStream(
  uri: string,
  db: string,
  collection: string | null,
  prompt: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const res = await fetch(`${BASE}/sources/mongo/auto-infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uri, db, collection, prompt }),
  });
  await pumpAgentStream(res, onEvent);
}

// ── Amazon S3 source ─────────────────────────────────────────────────────

export interface S3Creds {
  access_key_id: string;
  secret_access_key: string;
  session_token?: string;
  region?: string;
}

export interface S3TestResponse {
  ok: boolean;
  buckets?: Array<{ name: string }>;
  error?: string;
}

export interface S3ObjectsResponse {
  ok: boolean;
  host?: string;
  objects?: Array<{ key: string; size: number | null }>;
  error?: string;
}

export async function s3Test(creds: S3Creds): Promise<S3TestResponse> {
  const res = await fetch(`${BASE}/sources/s3/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ creds }),
  });
  if (!res.ok) throw new Error(`S3 test failed: ${res.statusText}`);
  return res.json();
}

export async function s3ListObjects(
  creds: S3Creds,
  bucket: string,
  prefix = '',
): Promise<S3ObjectsResponse> {
  const res = await fetch(`${BASE}/sources/s3/objects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ creds, bucket, prefix }),
  });
  if (!res.ok) throw new Error(`S3 list objects failed: ${res.statusText}`);
  return res.json();
}

export async function s3InferSchema(
  creds: S3Creds,
  bucket: string,
  key: string,
): Promise<MongoInferResponse> {
  const res = await fetch(`${BASE}/sources/s3/infer-schema`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ creds, bucket, key }),
  });
  if (!res.ok) throw new Error(`S3 schema inference failed: ${res.statusText}`);
  return res.json();
}

export async function s3AutoInferStream(
  creds: S3Creds,
  bucket: string,
  prefix: string | null,
  key: string | null,
  prompt: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const res = await fetch(`${BASE}/sources/s3/auto-infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ creds, bucket, prefix, key, prompt }),
  });
  await pumpAgentStream(res, onEvent);
}
