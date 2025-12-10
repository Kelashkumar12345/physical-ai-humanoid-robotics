import React from 'react';

interface ROSTopicProps {
  name: string;
  type?: string;
  description?: string;
}

/**
 * ROSTopic - Inline display of a ROS 2 topic with styling.
 */
export function ROSTopic({ name, type, description }: ROSTopicProps): JSX.Element {
  return (
    <span className="ros-topic" title={description || `ROS 2 Topic: ${name}`}>
      {name}
      {type && <span style={{ opacity: 0.7 }}> ({type})</span>}
    </span>
  );
}

interface ROSServiceProps {
  name: string;
  type?: string;
  description?: string;
}

/**
 * ROSService - Inline display of a ROS 2 service with styling.
 */
export function ROSService({ name, type, description }: ROSServiceProps): JSX.Element {
  return (
    <span className="ros-service" title={description || `ROS 2 Service: ${name}`}>
      {name}
      {type && <span style={{ opacity: 0.7 }}> ({type})</span>}
    </span>
  );
}

interface ROSNodeProps {
  name: string;
  package?: string;
}

/**
 * ROSNode - Display of a ROS 2 node with optional package info.
 */
export function ROSNode({ name, package: pkg }: ROSNodeProps): JSX.Element {
  return (
    <code className="ros-node">
      {pkg && <span style={{ opacity: 0.7 }}>{pkg}/</span>}
      {name}
    </code>
  );
}

interface TopicTableProps {
  topics: Array<{
    name: string;
    type: string;
    description: string;
    direction?: 'publish' | 'subscribe';
  }>;
}

/**
 * TopicTable - A table displaying multiple ROS 2 topics.
 */
export function TopicTable({ topics }: TopicTableProps): JSX.Element {
  return (
    <table>
      <thead>
        <tr>
          <th>Topic</th>
          <th>Type</th>
          <th>Direction</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {topics.map((topic, index) => (
          <tr key={index}>
            <td>
              <ROSTopic name={topic.name} />
            </td>
            <td><code>{topic.type}</code></td>
            <td>
              {topic.direction === 'publish' ? '📤 Publish' :
               topic.direction === 'subscribe' ? '📥 Subscribe' : '—'}
            </td>
            <td>{topic.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface ServiceTableProps {
  services: Array<{
    name: string;
    type: string;
    description: string;
  }>;
}

/**
 * ServiceTable - A table displaying multiple ROS 2 services.
 */
export function ServiceTable({ services }: ServiceTableProps): JSX.Element {
  return (
    <table>
      <thead>
        <tr>
          <th>Service</th>
          <th>Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {services.map((service, index) => (
          <tr key={index}>
            <td>
              <ROSService name={service.name} />
            </td>
            <td><code>{service.type}</code></td>
            <td>{service.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default { ROSTopic, ROSService, ROSNode, TopicTable, ServiceTable };
