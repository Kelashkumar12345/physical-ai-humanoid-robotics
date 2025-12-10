import MDXComponents from '@theme-original/MDXComponents';
import {
  ValidatedCodeBlock,
  TerminalOutput,
  Exercise,
  Hint,
  Solution,
  LearningObjectives,
  Prerequisites,
  ROSTopic,
  ROSService,
  ROSNode,
  TopicTable,
  ServiceTable,
  WeekHeader,
  ModuleOverview,
  DiagramEmbed,
  ArchitectureDiagram,
  MermaidDiagram,
} from '@site/src/components';

/**
 * Extend MDX components to include course-specific components.
 * These components are available in all MDX files without explicit imports.
 */
export default {
  ...MDXComponents,
  // Code display
  ValidatedCodeBlock,
  TerminalOutput,
  // Exercise components
  Exercise,
  Hint,
  Solution,
  // Learning structure
  LearningObjectives,
  Prerequisites,
  // ROS 2 components
  ROSTopic,
  ROSService,
  ROSNode,
  TopicTable,
  ServiceTable,
  // Week/Module structure
  WeekHeader,
  ModuleOverview,
  // Diagrams
  DiagramEmbed,
  ArchitectureDiagram,
  MermaidDiagram,
};
