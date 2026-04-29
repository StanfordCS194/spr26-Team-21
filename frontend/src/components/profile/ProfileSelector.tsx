import { useCallback, useRef, useState, type Dispatch, type KeyboardEvent as ReactKeyboardEvent, type SetStateAction } from 'react';
import useClickOutside from '../../hooks/useClickOutside';
import { getLogoSrc, type Integration, type Profile } from '../../constants/integrations';
import { ChevronDown, Check, Plus, Settings } from '../icons/Icons';
import ProfileModal from './ProfileModal';

const MAX_VISIBLE_ICONS = 5;

interface ModalState {
  mode: 'view' | 'add';
  profile?: Profile;
}

interface ProfileSelectorProps {
  profiles: Profile[];
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  selectedId: string;
  setSelectedId: Dispatch<SetStateAction<string>>;
}

export default function ProfileSelector({ profiles, setProfiles, selectedId, setSelectedId }: ProfileSelectorProps) {
  const [open, setOpen] = useState(false);
  const [modal, setModal] = useState<ModalState | null>(null);
  const [menuIndex, setMenuIndex] = useState(-1);
  const ref = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);
  useClickOutside(ref, close);

  const selected = profiles.find((p) => p.id === selectedId) ?? profiles[0];

  const openSettings = (e: React.MouseEvent, p: Profile) => {
    e.stopPropagation();
    setOpen(false);
    setModal({ mode: 'view', profile: p });
  };

  const openAdd = () => {
    setOpen(false);
    setModal({ mode: 'add' });
  };

  const saveNewProfile = (name: string) => {
    const base = name.toLowerCase().replace(/\s+/g, '-') || 'profile';
    const id = `${base}-${Date.now()}`;
    const next: Profile = { id, name: name || 'Untitled', integrations: [] };
    setProfiles([...profiles, next]);
    setSelectedId(id);
    setModal(null);
  };

  const updateProfileIntegrations = (profileId: string, integrations: Integration[]) => {
    setProfiles((prev) =>
      prev.map((p) => (p.id === profileId ? { ...p, integrations } : p))
    );
  };

  const enabledSources = selected.integrations.filter((i) => i.enabled);
  const shownIcons = enabledSources.slice(0, MAX_VISIBLE_ICONS);
  const remainder = enabledSources.length - shownIcons.length;

  const onMenuKeyDown = (e: ReactKeyboardEvent) => {
    if (!open) return;
    const items = profiles.length;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setMenuIndex((i) => (i + 1) % items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setMenuIndex((i) => (i - 1 + items) % items);
    } else if (e.key === 'Enter' && menuIndex >= 0) {
      e.preventDefault();
      setSelectedId(profiles[menuIndex].id);
      setOpen(false);
      setMenuIndex(-1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      setMenuIndex(-1);
    }
  };

  return (
    <>
      <div className="profile-wrap" ref={ref}>
        <button
          className="profile-trigger"
          onClick={() => { setOpen((o) => !o); setMenuIndex(-1); }}
          onKeyDown={onMenuKeyDown}
          aria-haspopup="listbox"
          aria-expanded={open}
        >
          <span>{selected.name}</span>
          {shownIcons.length > 0 && (
            <span className="profile-icons">
              {shownIcons.map((i) => (
                <img
                  key={i.slug}
                  className="profile-icon"
                  src={getLogoSrc(i.slug)}
                  alt={i.name}
                />
              ))}
              {remainder > 0 && (
                <span className="profile-icon-more">+{remainder}</span>
              )}
            </span>
          )}
          <ChevronDown />
        </button>

        {open && (
          <div
            className="profile-menu"
            role="listbox"
            aria-label="Profiles"
            onKeyDown={onMenuKeyDown}
          >
            <div className="profile-menu-label">Profiles</div>
            {profiles.map((p, idx) => (
              <div
                key={p.id}
                className={`profile-item ${p.id === selectedId ? 'active' : ''} ${idx === menuIndex ? 'focused' : ''}`}
                role="option"
                aria-selected={p.id === selectedId}
                tabIndex={0}
                onClick={() => {
                  setSelectedId(p.id);
                  setOpen(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelectedId(p.id);
                    setOpen(false);
                  }
                }}
              >
                <span>{p.name}</span>
                <span className="profile-item-actions">
                  <button
                    className="profile-settings-btn"
                    onClick={(e) => openSettings(e, p)}
                    aria-label={`${p.name} settings`}
                    tabIndex={0}
                  >
                    <Settings />
                  </button>
                  {p.id === selectedId && (
                    <Check />
                  )}
                </span>
              </div>
            ))}
            <div className="profile-menu-divider" />
            <button className="profile-item add" onClick={openAdd}>
              <Plus />
              Add Profile
            </button>
          </div>
        )}
      </div>

      {modal && (
        <ProfileModal
          mode={modal.mode}
          profile={modal.profile}
          onClose={() => setModal(null)}
          onSave={saveNewProfile}
          onUpdate={(integrations) =>
            modal.profile && updateProfileIntegrations(modal.profile.id, integrations)
          }
        />
      )}
    </>
  );
}
