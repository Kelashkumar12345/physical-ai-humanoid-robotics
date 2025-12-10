/**
 * Physical AI & Humanoid Robotics Course Components
 *
 * This module exports all reusable MDX components for the course.
 * Import components from this index for consistent usage across docs.
 */

// Code display components
export { ValidatedCodeBlock, TerminalOutput } from './CodeBlock';

// Exercise components
export { Exercise, Hint, Solution } from './Exercise';

// Learning structure components
export { LearningObjectives, Prerequisites } from './LearningObjectives';

// ROS 2 specific components
export {
  ROSTopic,
  ROSService,
  ROSNode,
  TopicTable,
  ServiceTable
} from './ROSComponents';

// Week/Module structure components
export { WeekHeader, ModuleOverview } from './WeekHeader';

// Diagram components
export { DiagramEmbed, ArchitectureDiagram, MermaidDiagram } from './DiagramEmbed';
