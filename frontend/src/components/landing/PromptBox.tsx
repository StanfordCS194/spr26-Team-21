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
        placeholder="Describe the dataset you need..."
        aria-label="Describe the dataset you want to synthesize"
      />
      <div className="prompt-box-footer">
        <div className="prompt-meta">
          <AttachmentMenu />
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
