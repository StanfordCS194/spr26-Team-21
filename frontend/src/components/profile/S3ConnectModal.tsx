import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Close } from '../icons/Icons';
import type { S3Config } from '../../constants/integrations';
import {
  s3ListObjects,
  s3Test,
  type S3ObjectsResponse,
  type S3TestResponse,
} from '../../api/client';

interface S3ConnectModalProps {
  initialConfig?: S3Config;
  onClose: () => void;
  onConnect: (config: S3Config) => void;
}

type Step = 'creds' | 'bucket' | 'object';

export default function S3ConnectModal({
  initialConfig,
  onClose,
  onConnect,
}: S3ConnectModalProps) {
  const [step, setStep] = useState<Step>('creds');
  const [accessKeyId, setAccessKeyId] = useState(initialConfig?.accessKeyId ?? '');
  const [secretAccessKey, setSecretAccessKey] = useState(initialConfig?.secretAccessKey ?? '');
  const [region, setRegion] = useState(initialConfig?.region ?? '');
  const [buckets, setBuckets] = useState<string[]>([]);
  const [selectedBucket, setSelectedBucket] = useState(initialConfig?.bucket ?? '');
  const [prefix, setPrefix] = useState(initialConfig?.prefix ?? '');
  const [objects, setObjects] = useState<Array<{ key: string; size: number | null }>>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const creds = () => ({
    access_key_id: accessKeyId.trim(),
    secret_access_key: secretAccessKey.trim(),
    region: region.trim() || undefined,
  });

  const handleTest = async () => {
    if (!accessKeyId.trim() || !secretAccessKey.trim()) {
      setError('Access key and secret are required');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res: S3TestResponse = await s3Test(creds());
      if (!res.ok) {
        setError(res.error || 'Could not connect');
        return;
      }
      setBuckets((res.buckets || []).map((b) => b.name));
      setStep('bucket');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePickBucket = async (bucket: string, nextPrefix = prefix) => {
    setSelectedBucket(bucket);
    setError(null);
    setLoading(true);
    try {
      const res: S3ObjectsResponse = await s3ListObjects(creds(), bucket, nextPrefix.trim());
      if (!res.ok) {
        setError(res.error || 'Could not list objects');
        return;
      }
      setObjects(res.objects || []);
      setStep('object');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list objects');
    } finally {
      setLoading(false);
    }
  };

  const finish = (key: string, rowCount?: number) => {
    onConnect({
      accessKeyId: accessKeyId.trim(),
      secretAccessKey: secretAccessKey.trim(),
      region: region.trim() || undefined,
      bucket: selectedBucket,
      prefix: prefix.trim() || undefined,
      key,
      host: selectedBucket,
      rowCount,
    });
  };

  return createPortal(
    <div className="modal-backdrop mongo-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal mongo-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="s3-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title" id="s3-modal-title">
            Connect Amazon S3
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <Close />
          </button>
        </div>

        <div className="mongo-steps">
          <span className={`mongo-step ${step === 'creds' ? 'active' : 'done'}`}>1 · Credentials</span>
          <span
            className={`mongo-step ${step === 'bucket' ? 'active' : step === 'object' ? 'done' : ''}`}
          >
            2 · Bucket
          </span>
          <span className={`mongo-step ${step === 'object' ? 'active' : ''}`}>3 · Object</span>
        </div>

        <div className="modal-body">
          {step === 'creds' && (
            <>
              <label className="field-label" htmlFor="s3-access-key">
                Access key ID
              </label>
              <input
                id="s3-access-key"
                className="field-input"
                value={accessKeyId}
                onChange={(e) => setAccessKeyId(e.target.value)}
                placeholder="AKIA…"
                autoFocus
                spellCheck={false}
                autoComplete="off"
              />
              <label className="field-label" htmlFor="s3-secret" style={{ marginTop: 12 }}>
                Secret access key
              </label>
              <input
                id="s3-secret"
                className="field-input"
                type="password"
                value={secretAccessKey}
                onChange={(e) => setSecretAccessKey(e.target.value)}
                placeholder="••••••••••••••••"
                spellCheck={false}
                autoComplete="off"
              />
              <label className="field-label" htmlFor="s3-region" style={{ marginTop: 12 }}>
                Region (optional)
              </label>
              <input
                id="s3-region"
                className="field-input"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                placeholder="us-east-1"
                spellCheck={false}
                autoComplete="off"
              />
              <p className="mongo-hint">
                Stored only in this browser's localStorage, never sent to any server except your
                own backend.
              </p>
            </>
          )}

          {step === 'bucket' && (
            <>
              <label className="field-label">Pick a bucket</label>
              {buckets.length === 0 ? (
                <div className="modal-empty">No buckets visible to these credentials.</div>
              ) : (
                <ul className="mongo-list">
                  {buckets.map((b) => (
                    <li key={b}>
                      <button
                        type="button"
                        className="mongo-list-item"
                        onClick={() => handlePickBucket(b)}
                        disabled={loading}
                      >
                        {b}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <label className="field-label" htmlFor="s3-prefix" style={{ marginTop: 12 }}>
                Optional key prefix
              </label>
              <input
                id="s3-prefix"
                className="field-input"
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="data/2026/"
                spellCheck={false}
                autoComplete="off"
              />
            </>
          )}

          {step === 'object' && (
            <>
              <div className="mongo-meta">
                <code>
                  {selectedBucket}/{prefix}
                </code>
              </div>
              <label className="field-label">How should Aperture pick an object?</label>
              <ul className="mongo-list">
                <li>
                  <button
                    type="button"
                    className="mongo-list-item mongo-auto-option"
                    onClick={() => finish('__auto__')}
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
                Or pick a specific object
              </label>
              {objects.length === 0 ? (
                <div className="modal-empty">No data objects under this prefix.</div>
              ) : (
                <ul className="mongo-list">
                  {objects.map((o) => (
                    <li key={o.key}>
                      <button
                        type="button"
                        className="mongo-list-item"
                        onClick={() => finish(o.key)}
                      >
                        <span>{o.key}</span>
                        {o.size !== null && o.size !== undefined && (
                          <span className="mongo-list-meta">
                            {(o.size / 1e6).toFixed(1)} MB
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
          {step !== 'creds' && (
            <button
              className="modal-btn"
              onClick={() => setStep(step === 'object' ? 'bucket' : 'creds')}
              disabled={loading}
            >
              Back
            </button>
          )}
          <button className="modal-btn" onClick={onClose}>
            Cancel
          </button>
          {step === 'creds' && (
            <button
              className="modal-btn primary"
              onClick={handleTest}
              disabled={loading || !accessKeyId.trim() || !secretAccessKey.trim()}
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
