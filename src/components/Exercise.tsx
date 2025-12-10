import React, { useState } from 'react';

type Difficulty = 'beginner' | 'intermediate' | 'advanced';

interface ExerciseProps {
  title: string;
  difficulty: Difficulty;
  estimatedTime?: string;
  children: React.ReactNode;
}

const difficultyConfig: Record<Difficulty, { label: string; className: string; icon: string }> = {
  beginner: { label: 'Beginner', className: 'badge--difficulty-beginner', icon: '🟢' },
  intermediate: { label: 'Intermediate', className: 'badge--difficulty-intermediate', icon: '🟡' },
  advanced: { label: 'Advanced', className: 'badge--difficulty-advanced', icon: '🔴' },
};

/**
 * Exercise - A card component for hands-on exercises
 * with difficulty indicators and collapsible hints/solutions.
 */
export function Exercise({
  title,
  difficulty,
  estimatedTime,
  children,
}: ExerciseProps): JSX.Element {
  const config = difficultyConfig[difficulty];

  return (
    <div className="exercise-card">
      <div className="exercise-card__header">
        <span>{config.icon}</span>
        <span>{title}</span>
        <span className={`badge ${config.className}`}>{config.label}</span>
        {estimatedTime && (
          <span style={{ marginLeft: 'auto', fontSize: '0.875rem', color: '#666' }}>
            ⏱️ {estimatedTime}
          </span>
        )}
      </div>
      <div className="exercise-card__content">{children}</div>
    </div>
  );
}

interface HintProps {
  children: React.ReactNode;
}

/**
 * Hint - A collapsible hint section for exercises.
 */
export function Hint({ children }: HintProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <details className="hint-block" open={isOpen} onToggle={() => setIsOpen(!isOpen)}>
      <summary>💡 Hint</summary>
      <div className="hint-content">{children}</div>
    </details>
  );
}

interface SolutionProps {
  children: React.ReactNode;
}

/**
 * Solution - A collapsible solution section for exercises.
 */
export function Solution({ children }: SolutionProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <details className="solution-block" open={isOpen} onToggle={() => setIsOpen(!isOpen)}>
      <summary>✅ Solution</summary>
      <div className="solution-content">{children}</div>
    </details>
  );
}

export default Exercise;
