import { useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import Logo from '../Logo';
import { Plus } from '../icons/Icons';
import { getLogoSrc, type Integration, type Profile } from '../../constants/integrations';
import ProfileModal from '../profile/ProfileModal';

interface SidebarProps {
  activeProfile: Profile;
  setProfiles: Dispatch<SetStateAction<Profile[]>>;
  onNewProject: () => void;
}

export default function Sidebar({ activeProfile, setProfiles, onNewProject }: SidebarProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const sources = activeProfile.integrations.filter((i) => i.enabled);

  const updateIntegrations = (integrations: Integration[]) => {
    setProfiles((prev) =>
      prev.map((p) => (p.id === activeProfile.id ? { ...p, integrations } : p))
    );
  };

  return (
    <>
      <aside className="ws-sidebar">
        <div className="ws-sidebar-brand" aria-hidden="true">
          <Logo className="header-logo" />
          <span>Aperture</span>
        </div>

        <button className="ws-new-project" onClick={onNewProject}>
          <Plus />
          New project
        </button>

        <div className="ws-sidebar-section">
          <div className="ws-sidebar-label">Connected sources</div>
          <ul className="ws-source-list">
            {sources.map((s) => (
              <li key={s.slug} className="ws-source-item">
                <img className="ws-source-icon" src={getLogoSrc(s.slug)} alt="" />
                <span className="ws-source-name">{s.name}</span>
                <span className="ws-source-dot" aria-label="Connected" />
              </li>
            ))}
          </ul>
          <button className="ws-add-source" onClick={() => setModalOpen(true)}>
            <Plus />
            Add source
          </button>
        </div>
      </aside>

      {modalOpen && (
        <ProfileModal
          mode="view"
          profile={activeProfile}
          onClose={() => setModalOpen(false)}
          onSave={() => {}}
          onUpdate={updateIntegrations}
        />
      )}
    </>
  );
}
