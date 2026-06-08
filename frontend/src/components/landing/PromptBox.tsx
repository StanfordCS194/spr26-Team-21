import type { Dispatch, KeyboardEvent as ReactKeyboardEvent, SetStateAction } from 'react';
import AttachmentMenu from '../attachment/AttachmentMenu';
import ProfileSelector from '../profile/ProfileSelector';
import { ArrowRight } from '../icons/Icons';
import type { Profile } from '../../constants/integrations';

const FORMAT_OPTIONS = ['csv', 'jsonl', 'parquet'] as const;

interface PromptBoxProps {
  prompt: string;
  setPrompt: (value: string) => void;
  onSubmit: () => void;
  submitting: boolean;
  profiles: Profile[];
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  selectedId: string;
  setSelectedId: Dispatch<SetStateAction<string>>;
  onGroundingFilesChange?: (files: File[]) => void;
  profileSummary?: { columns: number; sourceRows: number } | null;
  rowCount: number;
  setRowCount: (n: number) => void;
  format: 'csv' | 'jsonl' | 'parquet';
  setFormat: (f: 'csv' | 'jsonl' | 'parquet') => void;
}

export default function PromptBox({
  prompt,
  setPrompt,
  onSubmit,
  submitting,
  profiles,
  setProfiles,
  selectedId,
  setSelectedId,
  onGroundingFilesChange,
  profileSummary,
  rowCount,
  setRowCount,
  format,
  setFormat,
}: PromptBoxProps) {
  const handleKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!submitting) onSubmit();
    }
  };

  return (
    <div className="prompt-box">
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="e.g. 10k auto fraud claims with realistic patterns — amplify the rare cases so models can learn..."
        aria-label="Describe the insurance dataset you want to synthesize"
      />
      <div className="prompt-box-footer">
        <div className="prompt-meta">
          <AttachmentMenu
            onGroundingChange={onGroundingFilesChange}
            profileSummary={profileSummary}
          />
          <ProfileSelector
            profiles={profiles}
            setProfiles={setProfiles}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
          />
        </div>
        <div className="prompt-gen-controls">
          <input
            className="prompt-rowcount"
            type="number"
            min={100}
            max={100_000}
            step={1_000}
            value={rowCount}
            onChange={(e) =>
              setRowCount(Math.max(100, Math.min(100_000, Number(e.target.value) || 100)))
            }
            aria-label="Row count"
          />
          <span className="prompt-rowcount-label">rows</span>
          <div className="prompt-format-opts" role="group" aria-label="Output format">
            {FORMAT_OPTIONS.map((f) => (
              <button
                key={f}
                type="button"
                className={`prompt-format-opt${format === f ? ' prompt-format-opt-active' : ''}`}
                onClick={() => setFormat(f)}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <button
          className="prompt-submit"
          onClick={onSubmit}
          aria-label="Submit"
          disabled={submitting}
        >
          {submitting ? <span className="spinner" /> : <ArrowRight />}
        </button>
      </div>
    </div>
  );
}
