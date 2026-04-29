import { useState } from 'react';
import Logo from '../Logo';
import AgentTaskList from './AgentTaskList';
import SchemaCard from './SchemaCard';
import ValidationReport from './ValidationReport';
import { Download } from '../icons/Icons';
import type { WorkspaceMessage } from '../../constants/mockWorkspace';
import { VALIDATION_REPORT } from '../../constants/mockWorkspace';

type Props = Extract<WorkspaceMessage, { role: 'assistant' }>;
type ApprovalState = 'idle' | 'generating' | 'validating' | 'complete';

const wait = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

export default function AssistantMessage({ loading, plan, planText, agents, schema }: Props) {
  const [approval, setApproval] = useState<ApprovalState>('idle');

  const handleApprove = async () => {
    setApproval('generating');
    await wait(2200);
    setApproval('validating');
    await wait(1600);
    setApproval('complete');
  };

  return (
    <div className="ws-msg ws-msg-assistant">
      <div className="ws-msg-avatar ws-avatar-assistant" aria-hidden="true">
        <Logo />
      </div>
      <div className="ws-msg-body">
        {loading ? (
          <div className="ws-typing" aria-label="Thinking">
            <span className="ws-typing-dot" />
            <span className="ws-typing-dot" />
            <span className="ws-typing-dot" />
          </div>
        ) : (
          <>
            {plan && (
              <>
                <p className="ws-msg-text">{plan.intro}</p>
                <div className="ws-plan-steps">
                  {plan.steps.map((step, i) => (
                    <div key={i} className="ws-plan-step">
                      <span className="ws-plan-step-num">{i + 1}</span>
                      <div className="ws-plan-step-body">
                        <span className="ws-plan-step-title">{step.title}</span>
                        <span className="ws-plan-step-desc">{step.description}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
            {planText && <p className="ws-msg-text">{planText}</p>}
            {agents && <AgentTaskList agents={agents} />}
            {schema && <SchemaCard rows={schema} />}
            {schema && (
              <div className="ws-approve-row">
                {approval === 'idle' && (
                  <button className="ws-approve-btn" onClick={handleApprove}>
                    Approve & run
                  </button>
                )}
                {approval === 'generating' && (
                  <div className="ws-generating">
                    <span className="spinner" />
                    Generating dataset…
                  </div>
                )}
                {approval === 'validating' && (
                  <div className="ws-generating">
                    <span className="spinner" />
                    Running validation checks…
                  </div>
                )}
                {approval === 'complete' && (
                  <>
                    <div className="ws-download-card">
                      <div className="ws-download-info">
                        <span className="ws-download-filename">diabetic_cohort.csv</span>
                        <span className="ws-download-meta">10,000 rows · CSV · 2.4 MB</span>
                      </div>
                      <button className="ws-download-btn">
                        <Download size={13} />
                        Download
                      </button>
                    </div>
                    <ValidationReport report={VALIDATION_REPORT} />
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
