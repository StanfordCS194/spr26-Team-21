import type { Dispatch, SetStateAction } from 'react';
import Logo from '../Logo';
import PromptBox from './PromptBox';
import type { Profile } from '../../constants/integrations';

interface LandingProps {
  prompt: string;
  setPrompt: (value: string) => void;
  onSubmit: () => void;
  submitting: boolean;
  profiles: Profile[];
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  selectedId: string;
  setSelectedId: Dispatch<SetStateAction<string>>;
}

export default function Landing(props: LandingProps) {
  return (
    <div className="landing">
      <h1 className="landing-title">
        <Logo className="landing-title-logo" />
        What should we synthesize?
      </h1>

      <PromptBox {...props} />
    </div>
  );
}
