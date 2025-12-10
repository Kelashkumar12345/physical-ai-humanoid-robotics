import React from 'react';

interface DiagramEmbedProps {
  src: string;
  alt: string;
  caption?: string;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
}

/**
 * DiagramEmbed - Component for embedding diagrams and images
 * with proper accessibility attributes and captions.
 */
export function DiagramEmbed({
  src,
  alt,
  caption,
  width = '100%',
  align = 'center',
}: DiagramEmbedProps): JSX.Element {
  const alignStyle = {
    left: { marginRight: 'auto' },
    center: { margin: '0 auto' },
    right: { marginLeft: 'auto' },
  };

  return (
    <figure style={{ ...alignStyle[align], maxWidth: width, textAlign: align }}>
      <img
        src={src}
        alt={alt}
        style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }}
        loading="lazy"
      />
      {caption && (
        <figcaption style={{
          fontSize: '0.875rem',
          color: '#666',
          marginTop: '0.5rem',
          fontStyle: 'italic'
        }}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

interface ArchitectureDiagramProps {
  title: string;
  children: React.ReactNode;
  description?: string;
}

/**
 * ArchitectureDiagram - Wrapper for ASCII/text-based architecture diagrams.
 */
export function ArchitectureDiagram({
  title,
  children,
  description,
}: ArchitectureDiagramProps): JSX.Element {
  return (
    <div className="architecture-diagram">
      <div className="architecture-diagram__title">{title}</div>
      <pre className="architecture-diagram__content">
        {children}
      </pre>
      {description && (
        <p className="architecture-diagram__description">{description}</p>
      )}
    </div>
  );
}

interface MermaidDiagramProps {
  chart: string;
  caption?: string;
}

/**
 * MermaidDiagram - Placeholder for Mermaid diagram integration.
 * Note: Requires @docusaurus/theme-mermaid plugin.
 */
export function MermaidDiagram({ chart, caption }: MermaidDiagramProps): JSX.Element {
  return (
    <figure>
      <div className="mermaid">{chart}</div>
      {caption && (
        <figcaption style={{
          fontSize: '0.875rem',
          color: '#666',
          marginTop: '0.5rem',
          fontStyle: 'italic',
          textAlign: 'center'
        }}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

export default DiagramEmbed;
