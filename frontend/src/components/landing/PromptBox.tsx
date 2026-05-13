import type { Dispatch, KeyboardEvent as ReactKeyboardEvent, SetStateAction } from 'react';
import AttachmentMenu from '../attachment/AttachmentMenu';
import ProfileSelector from '../profile/ProfileSelector';
import { ArrowRight } from '../icons/Icons';
import type { Profile } from '../../constants/integrations';

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
        placeholder="e.g. 10,000 auto BI claims with staged-accident fraud patterns and provider-billing anomalies..."
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
