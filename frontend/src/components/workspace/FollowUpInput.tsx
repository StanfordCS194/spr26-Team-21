import { useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import { ArrowRight, Plus } from '../icons/Icons';

interface FollowUpInputProps {
  onSubmit: (text: string) => void;
}

export default function FollowUpInput({ onSubmit }: FollowUpInputProps) {
  const [value, setValue] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submit = () => {
    if (!value.trim() || submitting) return;
    onSubmit(value);
    setValue('');
    setSubmitting(true);
    window.setTimeout(() => setSubmitting(false), 1200);
  };

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="ws-followup">
      <button className="ws-followup-add" aria-label="Add">
        <Plus size={16} strokeWidth={2} />
      </button>
      <textarea
        className="ws-followup-input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Refine the schema, add constraints, or request edge cases…"
        rows={1}
        disabled={submitting}
      />
      <button
        className="ws-followup-submit"
        onClick={submit}
        aria-label="Send"
        disabled={submitting}
      >
        {submitting ? <span className="spinner" /> : <ArrowRight />}
      </button>
    </div>
  );
}
