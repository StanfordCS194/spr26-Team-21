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
  type WorkspaceMessage,
  type ClarifyingQuestion,
  type GenerationSpec,
} from './constants/mockWorkspace';
import {
  inferSchema,
  mongoInferSchema,
  planFromPrompt,
  type SchemaColumn,
  type SourceStats,
} from './api/client';

const PROFILES_STORAGE_KEY = 'aperture:profiles:v2';

function loadProfiles(): Profile[] {
  try {
    const raw = localStorage.getItem(PROFILES_STORAGE_KEY);
    if (!raw) return INITIAL_PROFILES;
    const parsed = JSON.parse(raw) as Profile[];
    if (!Array.isArray(parsed) || parsed.length === 0) return INITIAL_PROFILES;
    return parsed;
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
  const [messages, setMessages] = useState<WorkspaceMessage[]>([]);
  const [groundingFiles, setGroundingFiles] = useState<File[]>([]);
  const [landingSchema, setLandingSchema] = useState<SchemaColumn[] | null>(null);
  const [landingStats, setLandingStats] = useState<SourceStats | null>(null);
  const [landingSourceRows, setLandingSourceRows] = useState<number>(0);
  const [landingModelId, setLandingModelId] = useState<string | null>(null);
  const [schemaInferring, setSchemaInferring] = useState(false);

  const activeProfile = profiles.find((p) => p.id === selectedId) ?? profiles[0];

  const handleGroundingFilesChange = async (files: File[]) => {
    setGroundingFiles(files);
    if (files.length === 0) {
      setLandingSchema(null);
      setLandingStats(null);
      setLandingSourceRows(0);
      setLandingModelId(null);
      return;
    }
    setSchemaInferring(true);
    try {
      const result = await inferSchema(files);
      if (!result.error) {
        setLandingSchema(result.columns);
        setLandingStats(result.stats);
        setLandingSourceRows(result.source_rows);
        setLandingModelId(result.model_id ?? null);
      }
    } catch {
      setLandingSchema(null);
      setLandingStats(null);
      setLandingModelId(null);
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

    const setAgentStatus = (id: string, status: AgentStatus) =>
      patch({
        agents: initialAgents.map((a) => {
          if (a.id === id) return { ...a, status };
          if (status === 'running' && a.id < id) return { ...a, status: 'done' };
          return a;
        }),
      });

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
    let planData = buildExecutionPlan(sourceNames);

    const mongoIntegration = activeProfile.integrations.find(
      (i) => i.slug === 'mongodb' && i.enabled && i.config?.kind === 'mongo',
    );
    const mongoConfig =
      mongoIntegration?.config?.kind === 'mongo' ? mongoIntegration.config.mongo : null;

    if (groundingFiles.length > 0) {
      // Use already-inferred schema if available, otherwise infer now
      const result =
        landingSchema && landingStats
          ? { columns: landingSchema, stats: landingStats, source_rows: landingSourceRows, model_id: landingModelId }
          : await inferSchema(groundingFiles).catch(() => null);

      if (result) {
        schema = result.columns;
        stats = result.stats;
        modelId = result.model_id ?? null;
        schemaSource = 'upload';
        generationSpec = {
          row_count: 10_000,
          format: 'csv',
          labels: [],
          edge_cases: [],
          constraints: [],
        };
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
        schemaSource = 'upload';
        generationSpec = {
          row_count: 10_000,
          format: 'csv',
          labels: [],
          edge_cases: [],
          constraints: [],
        };
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
      }
    }

    patch({ loading: false, plan: planData, agents: initialAgents });

    await wait(500);
    setAgentStatus('a1', 'running');
    await wait(1100);
    setAgentStatus('a2', 'running');
    await wait(1100);
    setAgentStatus('a3', 'running');
    await wait(1100);

    patch({
      agents: initialAgents.map((a) => ({ ...a, status: 'done' })),
      schema: schema ?? SCHEMA_ROWS,
      sourceStats: stats ?? undefined,
      originalPrompt: trimmed,
      clarifyingQuestions: clarifyingQuestions.length > 0 ? clarifyingQuestions : undefined,
      generationSpec,
      schemaSource,
      modelId,
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
          profileSummary={profileSummary}
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
