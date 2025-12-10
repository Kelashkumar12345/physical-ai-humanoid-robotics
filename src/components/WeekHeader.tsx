import React from 'react';

interface WeekHeaderProps {
  week: number;
  title: string;
  module: number;
  estimatedHours?: number;
  skills?: string[];
}

/**
 * WeekHeader - Header component for each week's content
 * showing week number, title, and key skills covered.
 */
export function WeekHeader({
  week,
  title,
  module,
  estimatedHours,
  skills,
}: WeekHeaderProps): JSX.Element {
  return (
    <div className="week-header">
      <div className="week-header__badges">
        <span className="badge badge--week">Week {week}</span>
        <span className="badge" style={{ backgroundColor: '#22a7f0', color: 'white' }}>
          Module {module}
        </span>
        {estimatedHours && (
          <span className="badge" style={{ backgroundColor: '#6c757d', color: 'white' }}>
            ~{estimatedHours} hours
          </span>
        )}
      </div>
      <h1 className="week-header__title">{title}</h1>
      {skills && skills.length > 0 && (
        <div className="week-header__skills">
          <strong>Skills:</strong>{' '}
          {skills.map((skill, index) => (
            <span key={index} className="skill-tag">
              {skill}
              {index < skills.length - 1 ? ', ' : ''}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface ModuleOverviewProps {
  module: number;
  title: string;
  weeks: string;
  description: string;
  outcomes: string[];
}

/**
 * ModuleOverview - Overview card for each course module.
 */
export function ModuleOverview({
  module,
  title,
  weeks,
  description,
  outcomes,
}: ModuleOverviewProps): JSX.Element {
  return (
    <div className="module-overview">
      <div className="module-overview__header">
        <span className="badge" style={{ backgroundColor: '#76b900', color: 'white', fontSize: '1rem' }}>
          Module {module}
        </span>
        <span style={{ color: '#666', fontSize: '0.9rem' }}>Weeks {weeks}</span>
      </div>
      <h2>{title}</h2>
      <p>{description}</p>
      <div className="module-overview__outcomes">
        <h4>Key Outcomes</h4>
        <ul>
          {outcomes.map((outcome, index) => (
            <li key={index}>{outcome}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default WeekHeader;
