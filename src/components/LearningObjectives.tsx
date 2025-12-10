import React from 'react';

interface LearningObjectivesProps {
  objectives: string[];
  week?: number;
}

/**
 * LearningObjectives - Displays learning objectives for a lesson or week.
 * Used at the beginning of each chapter to set expectations.
 */
export function LearningObjectives({
  objectives,
  week,
}: LearningObjectivesProps): JSX.Element {
  return (
    <div className="learning-objectives">
      <h4>
        {week ? `Week ${week} ` : ''}Learning Objectives
      </h4>
      <p>By the end of this {week ? 'week' : 'section'}, you will be able to:</p>
      <ul>
        {objectives.map((objective, index) => (
          <li key={index}>{objective}</li>
        ))}
      </ul>
    </div>
  );
}

interface PrerequisitesProps {
  items: string[];
  optional?: string[];
}

/**
 * Prerequisites - Displays required and optional prerequisites.
 */
export function Prerequisites({
  items,
  optional,
}: PrerequisitesProps): JSX.Element {
  return (
    <div className="prerequisites-checklist">
      <h4>Prerequisites</h4>
      <p>Before starting this section, ensure you have:</p>
      <ul>
        {items.map((item, index) => (
          <li key={index}>
            <input type="checkbox" id={`prereq-${index}`} />
            <label htmlFor={`prereq-${index}`}> {item}</label>
          </li>
        ))}
      </ul>
      {optional && optional.length > 0 && (
        <>
          <p><strong>Optional but recommended:</strong></p>
          <ul>
            {optional.map((item, index) => (
              <li key={`opt-${index}`}>{item}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default LearningObjectives;
