import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Physical AI & Humanoid Robotics Course Sidebar Configuration
 * 13-week curriculum structure with progressive complexity
 *
 * Note: Only Week 1 content is currently implemented. Additional weeks
 * will be added as content is developed (marked as TODO placeholders).
 */
const sidebars: SidebarsConfig = {
  courseSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Course Home',
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/prerequisites',
        'getting-started/environment-setup',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Week 1: ROS 2 Architecture',
          collapsed: false,
          items: [
            'module-1/week-1/introduction',
            'module-1/week-1/nodes-and-topics',
            'module-1/week-1/services-and-actions',
            'module-1/week-1/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 2: TF2 & Navigation',
          collapsed: true,
          items: [
            'module-1/week-2/tf2-transforms',
            'module-1/week-2/urdf-basics',
            'module-1/week-2/nav2-introduction',
            'module-1/week-2/exercises',
          ],
        },
        // Week 3 content to be added
      ],
    },
    // Module 2-5 content to be added
    {
      type: 'category',
      label: 'Appendices',
      collapsed: true,
      items: [
        'appendices/troubleshooting',
        'appendices/glossary',
        'appendices/references',
        'appendices/hardware-specs',
      ],
    },
  ],
};

export default sidebars;
