import UserMessage from './UserMessage';
import AssistantMessage from './AssistantMessage';
import FollowUpInput from './FollowUpInput';
import type { WorkspaceMessage } from '../../constants/mockWorkspace';

interface ConversationProps {
  messages: WorkspaceMessage[];
  onFollowUp: (text: string) => void;
}

export default function Conversation({ messages, onFollowUp }: ConversationProps) {
  return (
    <section className="ws-main">
      <div className="ws-transcript">
        {messages.map((m) =>
          m.role === 'user' ? (
            <UserMessage key={m.id} text={m.text} />
          ) : (
            <AssistantMessage key={m.id} {...m} />
          )
        )}
      </div>

      <FollowUpInput onSubmit={onFollowUp} />
    </section>
  );
}
