const ICON_CDN_BASE = 'https://cdn.simpleicons.org';

const LOCAL_LOGOS: Record<string, string> = {
  amazons3: '/icons/aws.svg',
};

export function getLogoSrc(slug: string): string {
  return LOCAL_LOGOS[slug] || `${ICON_CDN_BASE}/${slug}/white`;
}

export interface MongoConfig {
  uri: string;
  db: string;
  collection: string;
  host: string;
  rowCount?: number;
}

export type IntegrationConfig = { kind: 'mongo'; mongo: MongoConfig };

export interface Integration {
  name: string;
  slug: string;
  enabled: boolean;
  config?: IntegrationConfig;
}

export interface IntegrationWithOrder extends Integration {
  connectedOrder: number | null;
}

export interface Profile {
  id: string;
  name: string;
  integrations: Integration[];
}

export const INITIAL_PROFILES: Profile[] = [
  {
    id: 'default',
    name: 'Default',
    integrations: [
      { name: 'Amazon S3', slug: 'amazons3', enabled: false },
      { name: 'MongoDB', slug: 'mongodb', enabled: false },
    ],
  },
];
