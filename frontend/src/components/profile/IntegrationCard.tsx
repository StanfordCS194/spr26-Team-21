import { getLogoSrc, type IntegrationWithOrder } from '../../constants/integrations';

interface IntegrationCardProps {
  integration: IntegrationWithOrder;
  onConnect: (slug: string) => void;
  onToggle: (slug: string) => void;
}

export default function IntegrationCard({ integration, onConnect, onToggle }: IntegrationCardProps) {
  const { slug, name, enabled, config } = integration;
  const mongo = config?.kind === 'mongo' ? config.mongo : null;
  const hasConfig = Boolean(mongo);

  return (
    <div className={`integration-card ${enabled ? 'enabled' : ''} ${hasConfig ? 'has-config' : ''}`}>
      <img
        className="integration-logo"
        src={getLogoSrc(slug)}
        alt=""
        loading="lazy"
      />
      <span className="integration-name">{name}</span>
      {mongo && (
        <span className="integration-config-detail" title={`${mongo.host} · ${mongo.db}.${mongo.collection}`}>
          {mongo.db}.{mongo.collection}
        </span>
      )}
      {enabled ? (
        <button
          className="integration-toggle on"
          onClick={() => onToggle(slug)}
          aria-label={`Disable ${name}`}
        >
          <span className="integration-toggle-dot" />
        </button>
      ) : hasConfig ? (
        <button
          className="integration-toggle"
          onClick={() => onToggle(slug)}
          aria-label={`Enable ${name}`}
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
