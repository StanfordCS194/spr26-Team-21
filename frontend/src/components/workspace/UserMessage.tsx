interface UserMessageProps {
  text: string;
}

export default function UserMessage({ text }: UserMessageProps) {
  return (
    <div className="ws-msg ws-msg-user">
      <div className="ws-msg-avatar ws-avatar-user" aria-hidden="true">You</div>
      <div className="ws-msg-body">{text}</div>
    </div>
  );
}
