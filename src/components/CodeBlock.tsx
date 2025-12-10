import React from 'react';
import CodeBlock from '@theme/CodeBlock';

interface ValidatedCodeBlockProps {
  language: string;
  title?: string;
  children: string;
  showLineNumbers?: boolean;
  metastring?: string;
}

/**
 * ValidatedCodeBlock - A wrapper around Docusaurus CodeBlock
 * that displays code with syntax highlighting and optional line numbers.
 *
 * Used for all code examples in the course to ensure consistent styling
 * and provide copy functionality.
 */
export function ValidatedCodeBlock({
  language,
  title,
  children,
  showLineNumbers = false,
  metastring,
}: ValidatedCodeBlockProps): JSX.Element {
  return (
    <CodeBlock
      language={language}
      title={title}
      showLineNumbers={showLineNumbers}
      metastring={metastring}
    >
      {children.trim()}
    </CodeBlock>
  );
}

interface TerminalOutputProps {
  prompt?: string;
  children: string;
}

/**
 * TerminalOutput - Displays terminal/console output
 * with proper styling for command prompts and output.
 */
export function TerminalOutput({
  prompt = '$',
  children,
}: TerminalOutputProps): JSX.Element {
  const lines = children.trim().split('\n');

  return (
    <div className="terminal-output">
      {lines.map((line, index) => {
        const isCommand = line.startsWith(prompt) || line.startsWith('#');
        return (
          <div key={index}>
            {isCommand ? (
              <>
                <span className="prompt">{prompt} </span>
                <span className="command">{line.replace(/^[$#]\s*/, '')}</span>
              </>
            ) : (
              <span>{line}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default ValidatedCodeBlock;
