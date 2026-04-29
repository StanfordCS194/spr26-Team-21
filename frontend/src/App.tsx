import { useState } from 'react';
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
} from './constants/mockWorkspace';

const wait = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

function App() {
  const [view, setView] = useState<'landing' | 'workspace'>('landing');
  const [prompt, setPrompt] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [profiles, setProfiles] = useState<Profile[]>(INITIAL_PROFILES);
  const [selectedId, setSelectedId] = useState('default');
  const [messages, setMessages] = useState<WorkspaceMessage[]>([]);

  const activeProfile = profiles.find((p) => p.id === selectedId) ?? profiles[0];

  const handleLandingSubmit = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || submitting) return;

    setSubmitting(true);
    await wait(800);

    const sourceNames = activeProfile.integrations
      .filter((i) => i.enabled)
      .map((i) => i.name);
    const msgId = 'm-assistant-1';
    const initialAgents = buildAgents(sourceNames);
    const plan = buildExecutionPlan(sourceNames);

    const patch = (p: Partial<Extract<WorkspaceMessage, { role: 'assistant' }>>) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === msgId && m.role === 'assistant' ? { ...m, ...p } : m))
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

    await wait(700);
    patch({ loading: false, plan, agents: initialAgents });

    await wait(500);
    setAgentStatus('a1', 'running');
    await wait(1100);
    setAgentStatus('a2', 'running');
    await wait(1100);
    setAgentStatus('a3', 'running');
    await wait(1100);
    patch({
      agents: initialAgents.map((a) => ({ ...a, status: 'done' })),
      schema: SCHEMA_ROWS,
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
          : m
      )
    );
  };

  const handleNewProject = () => {
    setMessages([]);
    setPrompt('');
    setView('landing');
  };

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
