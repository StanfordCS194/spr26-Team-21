import { getLogoSrc, type IntegrationWithOrder } from '../../constants/integrations';

interface IntegrationCardProps {
  integration: IntegrationWithOrder;
  onConnect: (slug: string) => void;
  onToggle: (slug: string) => void;
}

export default function IntegrationCard({ integration, onConnect, onToggle }: IntegrationCardProps) {
  const { slug, name, enabled } = integration;

  return (
    <div className={`integration-card ${enabled ? 'enabled' : ''}`}>
      <img
        className="integration-logo"
        src={getLogoSrc(slug)}
        alt=""
        loading="lazy"
      />
      <span className="integration-name">{name}</span>
      {enabled ? (
        <button
          className="integration-toggle on"
          onClick={() => onToggle(slug)}
          aria-label={`Disable ${name}`}
        >
          <span className="integration-toggle-dot" />
        </button>
      ) : (
        <button
          className="integration-connect-btn"
          onClick={() => onConnect(slug)}
        >
          Connect
        </button>
      )}
    </div>
  );
}
