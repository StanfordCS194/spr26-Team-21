import { useState } from 'react';
import { Check, ChevronDown } from '../icons/Icons';
import type { AgentTask } from '../../constants/mockWorkspace';

interface AgentTaskListProps {
  agents: AgentTask[];
}

export default function AgentTaskList({ agents }: AgentTaskListProps) {
  const [expanded, setExpanded] = useState(true);
  const allDone = agents.every((a) => a.status === 'done');
  const runningAgent = agents.find((a) => a.status === 'running');

  return (
    <div className="ws-agent-card">
      <button
        className="ws-agent-card-header"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span className="ws-agent-card-title">Spawning agents · {agents.length} tasks</span>
        <span className="ws-agent-card-right">
          {allDone ? (
            <span className="ws-agent-status-done">Complete</span>
          ) : runningAgent ? (
            <span className="ws-agent-status-running">{runningAgent.name}</span>
          ) : null}
          <span className={`ws-chevron${expanded ? ' ws-chevron-open' : ''}`}>
            <ChevronDown size={12} strokeWidth={2.5} />
          </span>
        </span>
      </button>

      {expanded && agents.map((agent) => (
        <div key={agent.id} className={`ws-agent-row ws-agent-row-${agent.status}`}>
          <div className="ws-agent-icon">
            {agent.status === 'done' && <Check size={11} strokeWidth={2.5} />}
            {agent.status === 'running' && <span className="spinner ws-agent-spinner" />}
            {agent.status === 'queued' && <span className="ws-agent-queued-dot" />}
          </div>
          <div className="ws-agent-info">
            <span className="ws-agent-name">{agent.name}</span>
            <span className="ws-agent-detail">{agent.detail}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
