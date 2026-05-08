import { useState } from 'react';
import type { ClarifyingQuestion } from '../../constants/mockWorkspace';

interface Props {
  questions: ClarifyingQuestion[];
  onDismiss: (answers: Record<string, string>) => void;
}

export default function ClarifyingQuestions({ questions, onDismiss }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const allAnswered = questions.every((q) => (answers[q.id] ?? '').trim().length > 0);

  return (
    <div className="ws-clarify-card">
      <div className="ws-clarify-header">
        <span className="ws-clarify-title">A few questions to refine your dataset</span>
        <span className="ws-clarify-hint">Optional — skip to use the inferred schema as-is</span>
      </div>
      <div className="ws-clarify-body">
        {questions.map((q, i) => (
          <div key={q.id} className="ws-clarify-row">
            <label className="ws-clarify-label">
              <span className="ws-clarify-num">{i + 1}</span>
              {q.question}
            </label>
            <input
              className="ws-clarify-input"
              type="text"
              placeholder="Your answer…"
              value={answers[q.id] ?? ''}
              onChange={(e) =>
                setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))
              }
            />
          </div>
        ))}
      </div>
      <div className="ws-clarify-footer">
        <button
          className="ws-approve-btn"
          disabled={!allAnswered}
          onClick={() => onDismiss(answers)}
        >
          Refine schema
        </button>
        <button className="ws-clarify-skip" onClick={() => onDismiss({})}>
          Skip, use defaults
        </button>
      </div>
    </div>
  );
}
