import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Integration, IntegrationWithOrder, Profile } from '../../constants/integrations';
import { Close, Search } from '../icons/Icons';
import IntegrationGrid from './IntegrationGrid';

interface ProfileModalProps {
  mode: 'view' | 'add';
  profile?: Profile;
  onClose: () => void;
  onSave: (name: string) => void;
  onUpdate?: (integrations: Integration[]) => void;
}

export default function ProfileModal({ mode, profile, onClose, onSave, onUpdate }: ProfileModalProps) {
  const [name, setName] = useState('');
  const [search, setSearch] = useState('');
  const modalRef = useRef<HTMLDivElement>(null);

  const initialIntegrations = profile?.integrations ?? [];
  const nextOrder = initialIntegrations.filter((i) => i.enabled).length;
  const [integrations, setIntegrations] = useState<IntegrationWithOrder[]>(() =>
    initialIntegrations.map((i, idx) => ({
      ...i,
      connectedOrder: i.enabled ? idx : null,
    }))
  );
  const [orderCounter, setOrderCounter] = useState(nextOrder);

  const stripOrder = (items: IntegrationWithOrder[]): Integration[] =>
    items.map(({ connectedOrder: _, ...rest }) => rest);

  const connectIntegration = useCallback((slug: string) => {
    setIntegrations((prev) => {
      const next = prev.map((i) =>
        i.slug === slug ? { ...i, enabled: true, connectedOrder: orderCounter } : i
      );
      onUpdate?.(stripOrder(next));
      return next;
    });
    setOrderCounter((c) => c + 1);
  }, [orderCounter, onUpdate]);

  const toggleIntegration = useCallback((slug: string) => {
    setIntegrations((prev) => {
      const next = prev.map((i) => {
        if (i.slug !== slug) return i;
        if (i.enabled) return { ...i, enabled: false, connectedOrder: null };
        return { ...i, enabled: true, connectedOrder: orderCounter };
      });
      onUpdate?.(stripOrder(next));
      return next;
    });
    setOrderCounter((c) => c + 1);
  }, [orderCounter, onUpdate]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return;

    const focusable = modal.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length) focusable[0].focus();

    const trapFocus = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', trapFocus);
    return () => document.removeEventListener('keydown', trapFocus);
  }, []);

  return createPortal(
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal"
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title" id="modal-title">
            {mode === 'add' ? 'New profile' : profile?.name}
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <Close />
          </button>
        </div>

        {mode === 'add' ? (
          <div className="modal-body">
            <label className="field-label" htmlFor="profile-name">Profile name</label>
            <input
              id="profile-name"
              className="field-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Clinical Trials"
              autoFocus
            />
            <div className="modal-section-label">Integrations</div>
            <div className="modal-empty">
              No integrations yet. Add data sources after creating the profile.
            </div>
          </div>
        ) : (
          <div className="modal-body">
            <div className="modal-section-label">Integrations</div>
            <div className="integration-search-wrap">
              <Search />
              <input
                className="integration-search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search integrations..."
                aria-label="Search integrations"
              />
            </div>
            <IntegrationGrid
              integrations={integrations}
              search={search}
              onConnect={connectIntegration}
              onToggle={toggleIntegration}
            />
          </div>
        )}

        <div className="modal-footer">
          <button className="modal-btn" onClick={onClose}>Cancel</button>
          {mode === 'add' && (
            <button
              className="modal-btn primary"
              onClick={() => onSave(name)}
              disabled={!name.trim()}
            >
              Create profile
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}
