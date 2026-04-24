const ICON_CDN_BASE = 'https://cdn.simpleicons.org';

const LOCAL_LOGOS: Record<string, string> = {
  amazons3: '/icons/aws.svg',
  slack: '/icons/slack.svg',
  salesforce: '/icons/salesforce.svg',
};

export function getLogoSrc(slug: string): string {
  return LOCAL_LOGOS[slug] || `${ICON_CDN_BASE}/${slug}/white`;
}

export interface Integration {
  name: string;
  slug: string;
  enabled: boolean;
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
      { name: 'Amazon S3', slug: 'amazons3', enabled: true },
      { name: 'PostgreSQL', slug: 'postgresql', enabled: true },
      { name: 'Snowflake', slug: 'snowflake', enabled: false },
      { name: 'Google BigQuery', slug: 'googlebigquery', enabled: false },
      { name: 'MongoDB', slug: 'mongodb', enabled: true },
      { name: 'Redis', slug: 'redis', enabled: false },
      { name: 'Elasticsearch', slug: 'elasticsearch', enabled: false },
      { name: 'Apache Kafka', slug: 'apachekafka', enabled: true },
      { name: 'MySQL', slug: 'mysql', enabled: false },
      { name: 'Slack', slug: 'slack', enabled: true },
      { name: 'GitHub', slug: 'github', enabled: false },
      { name: 'Jira', slug: 'jira', enabled: false },
      { name: 'Confluence', slug: 'confluence', enabled: true },
      { name: 'Notion', slug: 'notion', enabled: false },
      { name: 'Databricks', slug: 'databricks', enabled: false },
      { name: 'Google Drive', slug: 'googledrive', enabled: true },
      { name: 'Stripe', slug: 'stripe', enabled: false },
      { name: 'Salesforce', slug: 'salesforce', enabled: false },
    ],
  },
];
