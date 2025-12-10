import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A 13-Week Hands-On Course: From ROS 2 to VLA-Powered Humanoids',
  favicon: 'img/favicon.ico',

  url: 'https://kelashkumar12345.github.io',
  baseUrl: '/physical-ai-humanoid-robotics/',

  organizationName: 'kelashkumar12345',
  projectName: 'physical-ai-humanoid-robotics',

  onBrokenLinks: 'throw',

  markdown: {
    format: 'mdx',
    mdx1Compat: {
      comments: false,
      admonitions: false,
      headingIds: false,
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/kelashkumar12345/physical-ai-humanoid-robotics/tree/main/',
          showLastUpdateTime: false,
          showLastUpdateAuthor: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'courseSidebar',
          position: 'left',
          label: 'Course',
        },
        {
          to: '/appendices/glossary',
          label: 'Glossary',
          position: 'left',
        },
        {
          to: '/appendices/references',
          label: 'References',
          position: 'left',
        },
        {
          href: 'https://github.com/kelashkumar12345/physical-ai-humanoid-robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course',
          items: [
            {
              label: 'Getting Started',
              to: '/',
            },
            {
              label: 'Week 1: ROS 2 Architecture',
              to: '/module-1/week-1/introduction',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'ROS 2 Documentation',
              href: 'https://docs.ros.org/en/humble/',
            },
            {
              label: 'Gazebo Documentation',
              href: 'https://gazebosim.org/docs/harmonic/',
            },
            {
              label: 'Isaac Sim Documentation',
              href: 'https://docs.omniverse.nvidia.com/isaacsim/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/kelashkumar12345/physical-ai-humanoid-robotics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml', 'markup', 'cmake'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
