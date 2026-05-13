import { useState } from 'react';
import Logo from '../Logo';
import AgentTaskList from './AgentTaskList';
import AgentTrace from './AgentTrace';
import GroundingStrategy from './GroundingStrategy';
import SchemaCard from './SchemaCard';
import ValidationReport from './ValidationReport';
import ClarifyingQuestions from './ClarifyingQuestions';
import PreviewTable from './PreviewTable';
import { Download } from '../icons/Icons';
import type { WorkspaceMessage, ApprovalState } from '../../constants/mockWorkspace';
import { VALIDATION_REPORT } from '../../constants/mockWorkspace';
import {
  generate,
  previewDataset,
  downloadUrl,
  type GenerateResponse,
  type SourceStats,
} from '../../api/client';

type Props = Extract<WorkspaceMessage, { role: 'assistant' }>;

const FORMAT_OPTIONS = ['csv', 'jsonl', 'parquet'] as const;
type Format = (typeof FORMAT_OPTIONS)[number];

export default function AssistantMessage({
  loading,
  plan,
  planText,
  agents,
  agentTurns,
  groundingStrategy,
  schema,
  sourceStats,
  originalPrompt,
  initialApproval,
  clarifyingQuestions,
  generationSpec,
  schemaSource,
  modelId,
}: Props) {
  const [approval, setApproval] = useState<ApprovalState>(initialApproval ?? 'idle');
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);
  const [genError, setGenError] = useState<string | null>(null);

  const [rowCount, setRowCount] = useState(generationSpec?.row_count ?? 10_000);
  const [format, setFormat] = useState<Format>(generationSpec?.format ?? 'csv');

  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const [clarifyDismissed, setClarifyDismissed] = useState(false);

  const hasClarifying = (clarifyingQuestions?.length ?? 0) > 0 && !clarifyDismissed;

  const handlePreview = async () => {
    if (!schema || schema.length === 0) return;
    setPreviewLoading(true);
    try {
      const result = await previewDataset(
        schema,
        (sourceStats as SourceStats) ?? {},
        modelId,
      );
      setPreviewRows(result.rows);
    } catch {
      // non-fatal
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleApprove = async () => {
    setApproval('generating');
    setGenError(null);

    try {
      const result = await generate(
        schema ?? [],
        (sourceStats as SourceStats) ?? {},
        rowCount,
        originalPrompt ?? '',
        format,
        generationSpec?.edge_cases ?? [],
        modelId,
      );
      setApproval('validating');
      await new Promise<void>((r) => window.setTimeout(r, 800));
      setGenResult(result);
      setApproval('complete');
    } catch (err) {
      setGenError(err instanceof Error ? err.message : 'Generation failed');
      setApproval('idle');
    }
  };

  const report = genResult?.validation ?? VALIDATION_REPORT;
  const rowCountActual = genResult?.row_count ?? rowCount;
  const fileSizeKb = genResult?.file_size_kb ?? 0;
  const fileSizeFmt =
    fileSizeKb >= 1024
      ? `${(fileSizeKb / 1024).toFixed(1)} MB`
      : `${fileSizeKb} KB`;
  const filename = genResult?.filename ?? `aperture_output.${format}`;

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
            {agentTurns && agentTurns.length > 0 ? (
              <>
                <AgentTrace turns={agentTurns} />
                {groundingStrategy && <GroundingStrategy strategy={groundingStrategy} />}
              </>
            ) : (
              agents && <AgentTaskList agents={agents} />
            )}
            {schema && (
              <>
                <SchemaCard rows={schema} source={schemaSource} />

                {hasClarifying && clarifyingQuestions && (
                  <ClarifyingQuestions
                    questions={clarifyingQuestions}
                    onDismiss={() => setClarifyDismissed(true)}
                  />
                )}

                <div className="ws-approve-row">
                  {approval === 'idle' && (
                    <>
                      {/* Generation controls */}
                      <div className="ws-gen-controls">
                        <div className="ws-gen-control-group">
                          <label className="ws-gen-control-label">Rows</label>
                          <input
                            className="ws-rowcount-input"
                            type="number"
                            min={100}
                            max={100_000}
                            step={1_000}
                            value={rowCount}
                            onChange={(e) =>
                              setRowCount(
                                Math.max(100, Math.min(100_000, Number(e.target.value) || 100)),
                              )
                            }
                          />
                        </div>
                        <div className="ws-gen-control-group">
                          <label className="ws-gen-control-label">Format</label>
                          <div className="ws-format-opts">
                            {FORMAT_OPTIONS.map((f) => (
                              <button
                                key={f}
                                className={`ws-format-opt${format === f ? ' ws-format-opt-active' : ''}`}
                                onClick={() => setFormat(f)}
                              >
                                {f.toUpperCase()}
                              </button>
                            ))}
                          </div>
                        </div>
                        <span className={`ws-synth-badge${modelId ? ' ws-synth-badge-sdv' : ''}`}>
                          {modelId ? 'SDV · correlation-preserving' : 'Statistical sampler'}
                        </span>
                        <button
                          className="ws-preview-btn"
                          onClick={handlePreview}
                          disabled={previewLoading}
                        >
                          {previewLoading ? <span className="spinner ws-spinner-sm" /> : null}
                          Preview 10 rows
                        </button>
                      </div>

                      {previewRows && <PreviewTable rows={previewRows} />}

                      <div className="ws-approve-action-row">
                        <button className="ws-approve-btn" onClick={handleApprove}>
                          Approve &amp; generate
                        </button>
                        {genError && (
                          <span className="ws-gen-error">{genError}</span>
                        )}
                      </div>
                    </>
                  )}

                  {approval === 'generating' && (
                    <div className="ws-generating">
                      <span className="spinner" />
                      Generating {rowCount.toLocaleString()} rows…
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
                          <span className="ws-download-filename">{filename}</span>
                          <span className="ws-download-meta">
                            {rowCountActual.toLocaleString()} rows · {format.toUpperCase()} · {fileSizeFmt}
                          </span>
                        </div>
                        {genResult ? (
                          <a
                            className="ws-download-btn"
                            href={downloadUrl(genResult.session_id)}
                            download
                          >
                            <Download size={13} />
                            Download
                          </a>
                        ) : (
                          <button className="ws-download-btn" disabled>
                            <Download size={13} />
                            Download
                          </button>
                        )}
                      </div>
                      <ValidationReport report={report} />
                    </>
                  )}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
