import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Close } from '../icons/Icons';
import type { MongoConfig } from '../../constants/integrations';
import {
  mongoListCollections,
  mongoTest,
  type MongoCollectionsResponse,
  type MongoTestResponse,
} from '../../api/client';

interface MongoConnectModalProps {
  initialConfig?: MongoConfig;
  onClose: () => void;
  onConnect: (config: MongoConfig) => void;
}

type Step = 'uri' | 'db' | 'collection';

export default function MongoConnectModal({
  initialConfig,
  onClose,
  onConnect,
}: MongoConnectModalProps) {
  const [step, setStep] = useState<Step>('uri');
  const [uri, setUri] = useState(initialConfig?.uri ?? '');
  const [host, setHost] = useState(initialConfig?.host ?? '');
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDb, setSelectedDb] = useState(initialConfig?.db ?? '');
  const [collections, setCollections] = useState<
    Array<{ name: string; count: number | null }>
  >([]);
  const [selectedCollection, setSelectedCollection] = useState(
    initialConfig?.collection ?? '',
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const handleTest = async () => {
    if (!uri.trim()) {
      setError('Connection string is required');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res: MongoTestResponse = await mongoTest(uri.trim());
      if (!res.ok) {
        setError(res.error || 'Could not connect');
        return;
      }
      setHost(res.host || '');
      setDatabases((res.databases || []).map((d) => d.name));
      setStep('db');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePickDb = async (db: string) => {
    setSelectedDb(db);
    setError(null);
    setLoading(true);
    try {
      const res: MongoCollectionsResponse = await mongoListCollections(uri.trim(), db);
      if (!res.ok) {
        setError(res.error || 'Could not list collections');
        return;
      }
      setCollections(res.collections || []);
      setStep('collection');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list collections');
    } finally {
      setLoading(false);
    }
  };

  const handlePickCollection = (name: string, count: number | null) => {
    setSelectedCollection(name);
    onConnect({
      uri: uri.trim(),
      db: selectedDb,
      collection: name,
      host,
      rowCount: count ?? undefined,
    });
  };

  return createPortal(
    <div className="modal-backdrop mongo-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal mongo-modal"
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="mongo-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title" id="mongo-modal-title">
            Connect MongoDB
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <Close />
          </button>
        </div>

        <div className="mongo-steps">
          <span className={`mongo-step ${step === 'uri' ? 'active' : 'done'}`}>1 · URI</span>
          <span
            className={`mongo-step ${step === 'db' ? 'active' : step === 'collection' ? 'done' : ''}`}
          >
            2 · Database
          </span>
          <span className={`mongo-step ${step === 'collection' ? 'active' : ''}`}>
            3 · Collection
          </span>
        </div>

        <div className="modal-body">
          {step === 'uri' && (
            <>
              <label className="field-label" htmlFor="mongo-uri">
                Connection string
              </label>
              <input
                id="mongo-uri"
                className="field-input"
                type="password"
                value={uri}
                onChange={(e) => setUri(e.target.value)}
                placeholder="mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/"
                autoFocus
                spellCheck={false}
                autoComplete="off"
              />
              <p className="mongo-hint">
                Pasted here only — stored in this browser's localStorage, never sent to any
                server except your own backend.
              </p>
            </>
          )}

          {step === 'db' && (
            <>
              <div className="mongo-meta">
                Connected to <code>{host || 'cluster'}</code>
              </div>
              <label className="field-label">Pick a database</label>
              {databases.length === 0 ? (
                <div className="modal-empty">No databases visible to this user.</div>
              ) : (
                <ul className="mongo-list">
                  {databases.map((d) => (
                    <li key={d}>
                      <button
                        type="button"
                        className="mongo-list-item"
                        onClick={() => handlePickDb(d)}
                        disabled={loading}
                      >
                        {d}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}

          {step === 'collection' && (
            <>
              <div className="mongo-meta">
                <code>
                  {host}/{selectedDb}
                </code>
              </div>
              <label className="field-label">How should Aperture pick a collection?</label>
              <ul className="mongo-list">
                <li>
                  <button
                    type="button"
                    className={`mongo-list-item mongo-auto-option ${
                      selectedCollection === '__auto__' ? 'selected' : ''
                    }`}
                    onClick={() => handlePickCollection('__auto__', null)}
                  >
                    <span>
                      <span className="mongo-auto-title">Auto-select (recommended)</span>
                      <span className="mongo-auto-sub">
                        Sourcing agent picks the best fit for each prompt
                      </span>
                    </span>
                    <span className="mongo-list-meta">agent</span>
                  </button>
                </li>
              </ul>
              <label className="field-label" style={{ marginTop: 12 }}>
                Or pick a specific collection
              </label>
              {collections.length === 0 ? (
                <div className="modal-empty">No collections in this database.</div>
              ) : (
                <ul className="mongo-list">
                  {collections.map((c) => (
                    <li key={c.name}>
                      <button
                        type="button"
                        className={`mongo-list-item ${
                          c.name === selectedCollection ? 'selected' : ''
                        }`}
                        onClick={() => handlePickCollection(c.name, c.count)}
                      >
                        <span>{c.name}</span>
                        {c.count !== null && c.count !== undefined && (
                          <span className="mongo-list-meta">
                            ~{c.count.toLocaleString()} docs
                          </span>
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}

          {error && <div className="mongo-error">{error}</div>}
        </div>

        <div className="modal-footer">
          {step !== 'uri' && (
            <button
              className="modal-btn"
              onClick={() => setStep(step === 'collection' ? 'db' : 'uri')}
              disabled={loading}
            >
              Back
            </button>
          )}
          <button className="modal-btn" onClick={onClose}>
            Cancel
          </button>
          {step === 'uri' && (
            <button
              className="modal-btn primary"
              onClick={handleTest}
              disabled={loading || !uri.trim()}
            >
              {loading ? 'Testing…' : 'Test connection'}
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
