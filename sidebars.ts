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
        {
          type: 'category',
          label: 'Week 3: Perception Pipeline',
          collapsed: true,
          items: [
            'module-1/week-3/sensor-integration',
            'module-1/week-3/image-processing',
            'module-1/week-3/point-cloud-processing',
            'module-1/week-3/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo Simulation',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Week 4: Gazebo Basics',
          collapsed: true,
          items: [
            'module-2/week-4/gazebo-introduction',
            'module-2/week-4/world-building',
            'module-2/week-4/sensor-simulation',
            'module-2/week-4/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 5: Robot Models',
          collapsed: true,
          items: [
            'module-2/week-5/robot-description',
            'module-2/week-5/gazebo-plugins',
            'module-2/week-5/turtlebot3-simulation',
            'module-2/week-5/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 6: Advanced Simulation',
          collapsed: true,
          items: [
            'module-2/week-6/physics-tuning',
            'module-2/week-6/domain-randomization',
            'module-2/week-6/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac Sim & RL',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Week 7: Isaac Sim Basics',
          collapsed: true,
          items: [
            'module-3/week-7/isaac-sim-introduction',
            'module-3/week-7/isaac-sim-sensors',
            'module-3/week-7/isaac-ros-bridge',
            'module-3/week-7/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 8: Humanoid Robots',
          collapsed: true,
          items: [
            'module-3/week-8/humanoid-models',
            'module-3/week-8/humanoid-locomotion',
            'module-3/week-8/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 9: Reinforcement Learning',
          collapsed: true,
          items: [
            'module-3/week-9/isaac-gym-basics',
            'module-3/week-9/rl-policy-training',
            'module-3/week-9/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Integration',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Week 10: VLA Architecture',
          collapsed: true,
          items: [
            'module-4/week-10/vla-architecture',
            'module-4/week-10/ollama-setup',
            'module-4/week-10/vision-language-pipeline',
            'module-4/week-10/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 11: VLA Applications',
          collapsed: true,
          items: [
            'module-4/week-11/whisper-integration',
            'module-4/week-11/action-generation',
            'module-4/week-11/safety-constraints',
            'module-4/week-11/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 5: System Integration & Deployment',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Week 12: System Integration',
          collapsed: true,
          items: [
            'module-5/week-12/system-architecture',
            'module-5/week-12/integration-patterns',
            'module-5/week-12/testing-strategies',
            'module-5/week-12/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 13: Deployment & Final Assessment',
          collapsed: true,
          items: [
            'module-5/week-13/deployment-checklist',
            'module-5/week-13/final-assessment',
          ],
        },
      ],
    },
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
