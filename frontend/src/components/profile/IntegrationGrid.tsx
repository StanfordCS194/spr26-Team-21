import { useMemo } from 'react';
import type { IntegrationWithOrder } from '../../constants/integrations';
import IntegrationCard from './IntegrationCard';

interface IntegrationGridProps {
  integrations: IntegrationWithOrder[];
  search: string;
  onConnect: (slug: string) => void;
  onToggle: (slug: string) => void;
}

export default function IntegrationGrid({ integrations, search, onConnect, onToggle }: IntegrationGridProps) {
  const filtered = useMemo(() =>
    integrations
      .filter((i) => i.name.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => {
        if (a.enabled && !b.enabled) return -1;
        if (!a.enabled && b.enabled) return 1;
        if (a.enabled && b.enabled) return (a.connectedOrder ?? 0) - (b.connectedOrder ?? 0);
        return 0;
      }),
    [integrations, search]
  );

  return (
    <div className="integrations">
      {filtered.length === 0 && (
        <div className="modal-empty integration-no-results">No integrations found.</div>
      )}
      {filtered.map((i) => (
        <IntegrationCard
          key={i.slug}
          integration={i}
          onConnect={onConnect}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}
