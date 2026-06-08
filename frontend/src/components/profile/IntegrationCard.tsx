import {
  getLogoSrc,
  type IntegrationConfig,
  type IntegrationWithOrder,
} from '../../constants/integrations';

interface IntegrationCardProps {
  integration: IntegrationWithOrder;
  onConnect: (slug: string) => void;
  onToggle: (slug: string) => void;
}

function configDetail(config?: IntegrationConfig): { detail: string; title: string } | null {
  if (config?.kind === 'mongo') {
    const m = config.mongo;
    return { detail: `${m.db}.${m.collection}`, title: `${m.host} · ${m.db}.${m.collection}` };
  }
  if (config?.kind === 's3') {
    const s = config.s3;
    const target = s.key === '__auto__' ? `${s.prefix ?? ''}* (auto)` : s.key;
    return { detail: `${s.bucket}/${target}`, title: `${s.bucket}/${target}` };
  }
  return null;
}

export default function IntegrationCard({ integration, onConnect, onToggle }: IntegrationCardProps) {
  const { slug, name, enabled, config } = integration;
  const detail = configDetail(config);
  const hasConfig = Boolean(detail);

  return (
    <div className={`integration-card ${enabled ? 'enabled' : ''} ${hasConfig ? 'has-config' : ''}`}>
      <img
        className="integration-logo"
        src={getLogoSrc(slug)}
        alt=""
        loading="lazy"
      />
      <span className="integration-name">{name}</span>
      {detail && (
        <span className="integration-config-detail" title={detail.title}>
          {detail.detail}
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
