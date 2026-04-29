import type { Dispatch, SetStateAction } from 'react';
import Sidebar from './Sidebar';
import Conversation from './Conversation';
import type { Profile } from '../../constants/integrations';
import type { WorkspaceMessage } from '../../constants/mockWorkspace';

interface WorkspaceProps {
  activeProfile: Profile;
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  messages: WorkspaceMessage[];
  onFollowUp: (text: string) => void;
  onNewProject: () => void;
}

export default function Workspace({
  activeProfile,
  setProfiles,
  messages,
  onFollowUp,
  onNewProject,
}: WorkspaceProps) {
  return (
    <div className="workspace">
      <Sidebar
        activeProfile={activeProfile}
        setProfiles={setProfiles}
        onNewProject={onNewProject}
      />
      <Conversation messages={messages} onFollowUp={onFollowUp} />
    </div>
  );
}
