import { useEffect, useState } from 'react';
import './App.css';
import './styles/landing.css';
import './styles/attachment.css';
import './styles/profile.css';
import './styles/modal.css';
import './styles/workspace.css';
import Logo from './components/Logo';
import Landing from './components/landing/Landing';
import Workspace from './components/workspace/Workspace';
import { INITIAL_PROFILES, type Profile } from './constants/integrations';
import {
  buildAgents,
  buildExecutionPlan,
  SCHEMA_ROWS,
  type AgentStatus,
  type AgentTurn,
  type GroundingStrategyData,
  type SchemaInferenceState,
  type SchemaRow,
  type WorkspaceMessage,
  type ClarifyingQuestion,
  type GenerationSpec,
} from './constants/mockWorkspace';
import {
  inferSchema,
  mongoAutoInferStream,
  mongoInferSchema,
  planFromPrompt,
  s3AutoInferStream,
  s3InferSchema,
  type AgentEvent,
  type S3Creds,
  type SchemaColumn,
  type SourceStats,
} from './api/client';

const PROFILES_STORAGE_KEY = 'aperture:profiles:v2';

// The integrations we actually support, in display order.
const CANONICAL_INTEGRATIONS = INITIAL_PROFILES[0].integrations;

// Reconcile a stored profile against the canonical integration list: drop
// integrations we no longer support and add newly-introduced ones, while
// preserving any connection (enabled + config) the user already set up. This
// migrates browsers that cached the old, longer integration list.
function reconcileProfile(profile: Profile): Profile {
  const stored = new Map((profile.integrations ?? []).map((i) => [i.slug, i]));
  return {
    ...profile,
    integrations: CANONICAL_INTEGRATIONS.map((base) => {
      const prev = stored.get(base.slug);
      return prev ? { ...base, enabled: prev.enabled, config: prev.config } : { ...base };
    }),
  };
}

function loadProfiles(): Profile[] {
  try {
    const raw = localStorage.getItem(PROFILES_STORAGE_KEY);
    if (!raw) return INITIAL_PROFILES;
    const parsed = JSON.parse(raw) as Profile[];
    if (!Array.isArray(parsed) || parsed.length === 0) return INITIAL_PROFILES;
    return parsed.map(reconcileProfile);
  } catch {
    return INITIAL_PROFILES;
  }
}

const wait = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

function App() {
  const [view, setView] = useState<'landing' | 'workspace'>('landing');
  const [prompt, setPrompt] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [profiles, setProfiles] = useState<Profile[]>(loadProfiles);

  useEffect(() => {
    try {
      localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));
    } catch {
      // out of quota or in private mode — silently ignore
    }
  }, [profiles]);
  const [selectedId, setSelectedId] = useState('default');
  const [landingRowCount, setLandingRowCount] = useState(10_000);
  const [landingFormat, setLandingFormat] = useState<'csv' | 'jsonl' | 'parquet'>('csv');
  const [messages, setMessages] = useState<WorkspaceMessage[]>([]);
  const [groundingFiles, setGroundingFiles] = useState<File[]>([]);
  const [landingSchema, setLandingSchema] = useState<SchemaColumn[] | null>(null);
  const [landingStats, setLandingStats] = useState<SourceStats | null>(null);
  const [landingSourceRows, setLandingSourceRows] = useState<number>(0);
  const [landingModelId, setLandingModelId] = useState<string | null>(null);
  const [landingSourceId, setLandingSourceId] = useState<string | null>(null);
  const [schemaInferring, setSchemaInferring] = useState(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);

  const activeProfile = profiles.find((p) => p.id === selectedId) ?? profiles[0];

  const handleGroundingFilesChange = async (files: File[]) => {
    setGroundingFiles(files);
    if (files.length === 0) {
      setLandingSchema(null);
      setLandingStats(null);
      setLandingSourceRows(0);
      setLandingModelId(null);
      setLandingSourceId(null);
      return;
    }
    setSchemaInferring(true);
    setSchemaError(null);
    try {
      const result = await inferSchema(files);
      if (!result.error) {
        setLandingSchema(result.columns);
        setLandingStats(result.stats);
        setLandingSourceRows(result.source_rows);
        setLandingModelId(result.model_id ?? null);
        setLandingSourceId(result.source_id ?? null);
      } else {
        setLandingSchema(null);
        setLandingStats(null);
        setLandingSourceRows(0);
        setLandingModelId(null);
        setLandingSourceId(null);
        setSchemaError(result.error ?? 'Could not read the file — check the format and try again.');
      }
    } catch {
      setLandingSchema(null);
      setLandingStats(null);
      setLandingModelId(null);
      setLandingSourceId(null);
      setSchemaError('Could not reach the server — check your connection and try again.');
    } finally {
      setSchemaInferring(false);
    }
  };

  const handleLandingSubmit = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || submitting) return;
    setSubmitting(true);

    const sourceNames = activeProfile.integrations
      .filter((i) => i.enabled)
      .map((i) => i.name);
    const msgId = 'm-assistant-1';
    const initialAgents = buildAgents(sourceNames);

    const patch = (p: Partial<Extract<WorkspaceMessage, { role: 'assistant' }>>) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === msgId && m.role === 'assistant' ? { ...m, ...p } : m)),
      );

    setMessages([
      { id: 'm-user-1', role: 'user', text: trimmed },
      { id: msgId, role: 'assistant', loading: true },
    ]);
    setPrompt('');
    setView('workspace');
    setSubmitting(false);

    // Fetch plan / schema from the appropriate source
    let schema: SchemaColumn[] | null = null;
    let stats: SourceStats | null = null;
    let clarifyingQuestions: ClarifyingQuestion[] = [];
    let generationSpec: GenerationSpec | undefined;
    let schemaSource: 'llm' | 'upload' | undefined;
    let modelId: string | null = null;
    let sourceId: string | null = null;
    let planData = buildExecutionPlan(sourceNames);

    const mongoIntegration = activeProfile.integrations.find(
      (i) => i.slug === 'mongodb' && i.enabled && i.config?.kind === 'mongo',
    );
    const mongoConfig =
      mongoIntegration?.config?.kind === 'mongo' ? mongoIntegration.config.mongo : null;

    // S3 is a secondary grounding source; MongoDB takes precedence if both are connected.
    const s3Integration = activeProfile.integrations.find(
      (i) => i.slug === 'amazons3' && i.enabled && i.config?.kind === 's3',
    );
    const s3Config =
      !mongoConfig && s3Integration?.config?.kind === 's3' ? s3Integration.config.s3 : null;
    const s3Creds: S3Creds | null = s3Config
      ? {
          access_key_id: s3Config.accessKeyId,
          secret_access_key: s3Config.secretAccessKey,
          session_token: s3Config.sessionToken,
          region: s3Config.region,
        }
      : null;

    const mongoAuto = mongoConfig?.collection === '__auto__';
    const s3Auto = s3Config?.key === '__auto__';

    // ── Sourcing agent path (auto-select, Mongo or S3) ─────────────────
    if ((mongoAuto || s3Auto) && groundingFiles.length === 0) {
      const turns: AgentTurn[] = [];
      let strategy: GroundingStrategyData | null = null;
      let agentSchema: SchemaColumn[] | null = null;
      let agentStats: SourceStats | null = null;
      let agentModelId: string | null = null;
      let agentHost: string | undefined;
      const liveSchema: SchemaRow[] = [];
      let schemaState: SchemaInferenceState = { phase: 'idle' };

      patch({ loading: false, plan: planData, agentTurns: [], schemaInference: schemaState });

      try {
        const handleAgentEvent = (event: AgentEvent) => {
            if (event.type === 'step_start') {
              turns.push({
                turn: event.turn,
                tool: event.tool,
                inputSummary: event.input_summary,
                status: 'running',
              });
              patch({ agentTurns: [...turns] });
            } else if (event.type === 'step_complete') {
              const last = turns.find(
                (t) => t.turn === event.turn && t.status === 'running',
              );
              if (last) {
                last.status = 'done';
                last.resultSummary = event.result_summary;
                last.durationMs = event.duration_ms;
              }
              patch({ agentTurns: [...turns] });
            } else if (event.type === 'rationale') {
              strategy = {
                rationale: event.text,
                queries: [],
              };
              patch({ groundingStrategy: strategy });
            } else if (event.type === 'schema_start') {
              schemaState = {
                phase: 'scanning',
                idx: 0,
                total: event.total_columns,
                sourceRows: event.source_rows,
              };
              patch({ schemaInference: schemaState, schema: [] });
            } else if (event.type === 'schema_column') {
              liveSchema.push({
                column: event.column.column,
                type: event.column.type,
                distribution: event.column.distribution,
                sample: event.column.sample,
              });
              schemaState = {
                phase: 'scanning',
                idx: event.idx,
                total: event.total,
                latest: event.column.column,
                sourceRows:
                  schemaState.phase === 'scanning' ? schemaState.sourceRows : 0,
              };
              patch({ schemaInference: schemaState, schema: [...liveSchema] });
            } else if (event.type === 'fitting_model') {
              schemaState = {
                phase: 'fitting',
                total: event.total_columns,
                sourceRows: event.source_rows,
              };
              patch({ schemaInference: schemaState });
            } else if (event.type === 'final') {
              agentSchema = event.columns;
              agentStats = event.stats;
              agentModelId = event.model_id ?? null;
              agentHost = event.host;
              schemaState = {
                phase: 'done',
                total: event.columns.length,
                sourceRows: event.source_rows ?? 0,
              };
              if (event.grounding_strategy) {
                strategy = {
                  rationale: event.grounding_strategy.rationale,
                  totalRows: event.grounding_strategy.total_rows,
                  queries: event.grounding_strategy.queries,
                };
                patch({ groundingStrategy: strategy, schemaInference: schemaState });
              } else {
                patch({ schemaInference: schemaState });
              }
            } else if (event.type === 'error') {
              const lastRunning = turns.find((t) => t.status === 'running');
              if (lastRunning) {
                lastRunning.status = 'error';
                lastRunning.resultSummary = event.message;
              }
              patch({ agentTurns: [...turns] });
            }
        };

        if (mongoAuto && mongoConfig) {
          await mongoAutoInferStream(mongoConfig.uri, mongoConfig.db, null, trimmed, handleAgentEvent);
        } else if (s3Config && s3Creds) {
          await s3AutoInferStream(
            s3Creds,
            s3Config.bucket,
            s3Config.prefix ?? null,
            null,
            trimmed,
            handleAgentEvent,
          );
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Unknown error';
        const sourceLabel = mongoAuto ? 'MongoDB' : 'Amazon S3';
        patch({
          loading: false,
          planText: `Connection to ${sourceLabel} failed: ${msg}. Check your credentials in the profile settings and try again.`,
        });
        return;
      }

      patch({
        schema: agentSchema ?? (liveSchema.length ? liveSchema : SCHEMA_ROWS),
        sourceStats: agentStats ?? undefined,
        modelId: agentModelId,
        host: agentHost,
        originalPrompt: trimmed,
        schemaSource: 'agent',
        generationSpec: {
          row_count: landingRowCount,
          format: landingFormat,
          labels: [],
          edge_cases: [],
          constraints: [],
        },
      });
      return;
    }

    if (groundingFiles.length > 0) {
      // Use already-inferred schema if available, otherwise infer now
      const result =
        landingSchema && landingStats
          ? {
              columns: landingSchema,
              stats: landingStats,
              source_rows: landingSourceRows,
              model_id: landingModelId,
              source_id: landingSourceId ?? undefined,
            }
          : await inferSchema(groundingFiles).catch(() => null);

      if (result && !result.error) {
        schema = result.columns;
        stats = result.stats;
        modelId = result.model_id ?? null;
        sourceId = result.source_id ?? null;
        schemaSource = 'upload';
        generationSpec = {
          row_count: landingRowCount,
          format: landingFormat,
          labels: [],
          edge_cases: [],
          constraints: [],
        };
      } else {
        patch({
          loading: false,
          planText: result?.error
            ? `File upload failed: ${result.error}`
            : 'Could not read the uploaded file — check the format (CSV, JSON, or Parquet) and try again.',
        });
        return;
      }
    } else if (mongoConfig) {
      // MongoDB is the active grounding source — pull the collection through the backend
      const result = await mongoInferSchema(
        mongoConfig.uri,
        mongoConfig.db,
        mongoConfig.collection,
      ).catch(() => null);

      if (result && !result.error) {
        schema = result.columns;
        stats = result.stats;
        modelId = result.model_id ?? null;
        sourceId = result.source_id ?? null;
        schemaSource = 'upload';
        generationSpec = {
          row_count: landingRowCount,
          format: landingFormat,
          labels: [],
          edge_cases: [],
          constraints: [],
        };
      } else {
        patch({
          loading: false,
          planText: result?.error
            ? `MongoDB connection failed: ${result.error}`
            : 'Could not connect to MongoDB — check your URI and database name in the profile settings.',
        });
        return;
      }
    } else if (s3Config && s3Creds) {
      // S3 is the active grounding source — pull the object through the backend
      const result = await s3InferSchema(s3Creds, s3Config.bucket, s3Config.key).catch(() => null);

      if (result && !result.error) {
        schema = result.columns;
        stats = result.stats;
        modelId = result.model_id ?? null;
        sourceId = result.source_id ?? null;
        schemaSource = 'upload';
        generationSpec = {
          row_count: landingRowCount,
          format: landingFormat,
          labels: [],
          edge_cases: [],
          constraints: [],
        };
      } else {
        patch({
          loading: false,
          planText: result?.error
            ? `S3 connection failed: ${result.error}`
            : 'Could not read the S3 object — check your bucket and key in the profile settings.',
        });
        return;
      }
    } else {
      // No grounding files — use the Prompt-to-Action agent
      const result = await planFromPrompt(trimmed).catch(() => null);
      if (result) {
        schema = result.schema;
        clarifyingQuestions = result.clarifying_questions ?? [];
        generationSpec = result.generation_spec;
        schemaSource = 'llm';
        planData = {
          intro: result.generation_plan.intro,
          steps: result.generation_plan.steps,
        };
      } else {
        patch({
          loading: false,
          planText: 'Could not reach the planning service — check your connection and try again.',
        });
        return;
      }
    }

    // Legacy non-agent path: schema is done as soon as the API returns.
    // Sample synthesis + fidelity validation stay queued until the user clicks Generate
    // (AssistantMessage flips them locally when the SDV generate call fires).
    const sequencedAgents = initialAgents.map((a) =>
      a.id === 'a1' ? { ...a, status: 'done' as AgentStatus } : a,
    );

    patch({
      loading: false,
      plan: planData,
      agents: sequencedAgents,
      schema: schema ?? SCHEMA_ROWS,
      sourceStats: stats ?? undefined,
      originalPrompt: trimmed,
      clarifyingQuestions: clarifyingQuestions.length > 0 ? clarifyingQuestions : undefined,
      generationSpec,
      schemaSource,
      modelId,
      sourceId,
    });
  };

  const handleFollowUp = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const userId = `m-user-${Date.now()}`;
    const assistantId = `m-assistant-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userId, role: 'user', text: trimmed },
      { id: assistantId, role: 'assistant', loading: true },
    ]);

    await wait(1200);

    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId && m.role === 'assistant'
          ? { ...m, loading: false, planText: 'Updated. I applied your refinements to the schema.' }
          : m,
      ),
    );
  };

  const handleNewProject = () => {
    setMessages([]);
    setPrompt('');
    setGroundingFiles([]);
    setLandingSchema(null);
    setLandingStats(null);
    setLandingSourceRows(0);
    setLandingModelId(null);
    setLandingSourceId(null);
    setView('landing');
  };

  const profileSummary =
    landingSchema && landingSchema.length > 0
      ? { columns: landingSchema.length, sourceRows: landingSourceRows }
      : null;

  return (
    <div className="app">
      {view === 'landing' && (
        <header className="header">
          <div className="header-left" aria-hidden="true">
            <Logo className="header-logo" />
            <span>Aperture</span>
          </div>
        </header>
      )}

      {view === 'landing' ? (
        <Landing
          prompt={prompt}
          setPrompt={setPrompt}
          onSubmit={handleLandingSubmit}
          submitting={submitting}
          profiles={profiles}
          setProfiles={setProfiles}
          selectedId={selectedId}
          setSelectedId={setSelectedId}
          groundingFiles={groundingFiles}
          onGroundingFilesChange={handleGroundingFilesChange}
          inferredSchema={landingSchema}
          schemaInferring={schemaInferring}
          schemaError={schemaError}
          profileSummary={profileSummary}
          rowCount={landingRowCount}
          setRowCount={setLandingRowCount}
          format={landingFormat}
          setFormat={setLandingFormat}
        />
      ) : (
        <Workspace
          activeProfile={activeProfile}
          setProfiles={setProfiles}
          messages={messages}
          onFollowUp={handleFollowUp}
          onNewProject={handleNewProject}
        />
      )}
    </div>
  );
}

export default App;
