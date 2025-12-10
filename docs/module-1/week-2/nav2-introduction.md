---
sidebar_position: 3
title: Nav2 Introduction
description: Getting started with ROS 2 Navigation Stack
---

# Nav2 Introduction

The **Navigation2 (Nav2)** stack provides autonomous navigation capabilities for mobile robots. This lesson introduces Nav2 architecture and basic usage.

<LearningObjectives
  objectives={[
    "Understand Nav2 architecture and components",
    "Configure and launch Nav2 with TurtleBot3",
    "Create and use costmaps for obstacle avoidance",
    "Send navigation goals programmatically",
    "Understand behavior trees in Nav2"
  ]}
/>

## Nav2 Architecture

Nav2 is a modular navigation system with pluggable components:

<ArchitectureDiagram title="Nav2 System Architecture">
{`
┌─────────────────────────────────────────────────────────────────┐
│                        BT Navigator                              │
│              (Behavior Tree Task Orchestration)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Planner    │    │  Controller   │    │   Recovery    │
│    Server     │    │    Server     │    │    Server     │
│  (Global Path)│    │ (Local Control)│    │  (Stuck Help) │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │    Costmap    │
                     │    Server     │
                     │ (Obstacle Map)│
                     └───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │  AMCL   │          │  SLAM   │          │ Sensors │
   │(Localize)│          │(Mapping)│          │(LiDAR,..)│
   └─────────┘          └─────────┘          └─────────┘
`}
</ArchitectureDiagram>

### Core Components

| Component | Purpose |
|-----------|---------|
| **BT Navigator** | Coordinates navigation using behavior trees |
| **Planner Server** | Computes global path from A to B |
| **Controller Server** | Follows path, avoids local obstacles |
| **Recovery Server** | Handles stuck situations |
| **Costmap Server** | Maintains obstacle representation |
| **AMCL** | Localizes robot on known map |
| **SLAM Toolbox** | Builds map while navigating |

## Setting Up Nav2 with TurtleBot3

### Install Nav2 Packages

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

### Launch Navigation

**Terminal 1: Gazebo Simulation**
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

**Terminal 2: Nav2 Stack**
```bash
ros2 launch nav2_bringup bringup_launch.py \
  use_sim_time:=True \
  map:=/path/to/map.yaml
```

**Terminal 3: RViz2**
```bash
ros2 launch nav2_bringup rviz_launch.py
```

## Costmaps

Costmaps represent obstacles and free space as a 2D grid:

### Costmap Layers

| Layer | Purpose | Source |
|-------|---------|--------|
| **Static** | Known obstacles from map | Map file |
| **Obstacle** | Detected obstacles | LiDAR, depth cameras |
| **Inflation** | Safety buffer around obstacles | Computed |
| **Voxel** | 3D obstacles (optional) | 3D sensors |

### Costmap Values

| Value | Meaning |
|-------|---------|
| 0 | Free space |
| 1-252 | Varying cost (prefer lower) |
| 253 | Inscribed (robot touches) |
| 254 | Lethal (collision) |
| 255 | Unknown |

### Configuration Example

```yaml
# nav2_params.yaml
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      robot_radius: 0.22
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

## Sending Navigation Goals

### Using RViz2

1. Click **2D Pose Estimate** to set initial pose
2. Click **Nav2 Goal** to set destination
3. Watch robot navigate!

### Programmatic Goals

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self.navigator = BasicNavigator()

    def go_to_pose(self, x, y, yaw):
        """Navigate to a specific pose."""
        # Wait for Nav2 to be active
        self.navigator.waitUntilNav2Active()

        # Create goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y

        import math
        goal_pose.pose.orientation.z = math.sin(yaw / 2)
        goal_pose.pose.orientation.w = math.cos(yaw / 2)

        # Send goal
        self.navigator.goToPose(goal_pose)

        # Wait for completion
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                distance = feedback.distance_remaining
                self.get_logger().info(f'Distance remaining: {distance:.2f}m')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Goal reached!')
        else:
            self.get_logger().error(f'Navigation failed: {result}')

def main():
    rclpy.init()
    nav_client = NavigationClient()

    # Navigate to position (2.0, 1.0) facing 0 degrees
    nav_client.go_to_pose(2.0, 1.0, 0.0)

    rclpy.shutdown()
```

### Waypoint Following

```python
def follow_waypoints(self, waypoints):
    """Navigate through a list of waypoints."""
    goal_poses = []

    for wp in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = wp['x']
        pose.pose.position.y = wp['y']
        pose.pose.orientation.w = 1.0
        goal_poses.append(pose)

    self.navigator.followWaypoints(goal_poses)

    while not self.navigator.isTaskComplete():
        feedback = self.navigator.getFeedback()
        current_wp = feedback.current_waypoint
        self.get_logger().info(f'Executing waypoint {current_wp}')

# Usage
waypoints = [
    {'x': 1.0, 'y': 0.0},
    {'x': 2.0, 'y': 1.0},
    {'x': 0.0, 'y': 2.0},
    {'x': 0.0, 'y': 0.0},  # Return home
]
nav_client.follow_waypoints(waypoints)
```

## Behavior Trees

Nav2 uses **Behavior Trees (BT)** to orchestrate navigation:

### Basic BT Structure

```xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <!-- Compute path -->
      <RateController hz="1.0">
        <ComputePathToPose goal="{goal}" path="{path}"/>
      </RateController>

      <!-- Follow path -->
      <FollowPath path="{path}"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

### Common BT Nodes

| Node Type | Examples |
|-----------|----------|
| **Action** | ComputePathToPose, FollowPath, Spin, Wait |
| **Condition** | IsStuck, GoalReached, IsBatteryLow |
| **Control** | Sequence, Fallback, Parallel |
| **Decorator** | RateController, Repeat, Timeout |

## Nav2 Topics and Services

<TopicTable
  topics={[
    { name: "/goal_pose", type: "geometry_msgs/PoseStamped", direction: "subscribe", description: "Navigation goal input" },
    { name: "/plan", type: "nav_msgs/Path", direction: "publish", description: "Computed global path" },
    { name: "/local_plan", type: "nav_msgs/Path", direction: "publish", description: "Local trajectory" },
    { name: "/cmd_vel", type: "geometry_msgs/Twist", direction: "publish", description: "Velocity commands" },
    { name: "/global_costmap/costmap", type: "nav_msgs/OccupancyGrid", direction: "publish", description: "Global costmap" }
  ]}
/>

## Exercises

<Exercise title="Navigate to Multiple Goals" difficulty="beginner" estimatedTime="20 min">

Using TurtleBot3 in Gazebo:
1. Launch Nav2 with the provided map
2. Use RViz2 to send 3 sequential navigation goals
3. Observe the global and local costmaps

<Hint>
After clicking "Nav2 Goal", wait for the robot to reach the destination before sending the next goal.
</Hint>

</Exercise>

<Exercise title="Programmatic Patrol" difficulty="intermediate" estimatedTime="30 min">

Create a node that:
1. Defines 4 waypoints forming a square patrol route
2. Continuously navigates through the waypoints
3. Logs position and battery level (simulated) at each waypoint

<Hint>
Use `navigator.followWaypoints()` in a loop.
Add a delay at each waypoint using `time.sleep()`.
</Hint>

</Exercise>

## Troubleshooting

:::tip Robot Doesn't Move
1. Check initial pose is set (2D Pose Estimate)
2. Verify `/cmd_vel` is being published
3. Check TF tree is complete (`ros2 run tf2_tools view_frames`)
:::

:::tip Path Not Found
1. Check costmap shows the goal as reachable
2. Verify inflation radius isn't blocking narrow passages
3. Increase planner tolerance
:::

:::tip Robot Oscillates
1. Reduce controller gains
2. Increase goal tolerance
3. Check sensor data for noise
:::

## Summary

Key concepts covered:

- ✅ Nav2 architecture with pluggable servers
- ✅ Costmaps represent obstacles and free space
- ✅ Sending goals via RViz2 or programmatically
- ✅ BasicNavigator for Python navigation clients
- ✅ Behavior trees orchestrate navigation tasks

## Next Steps

Complete the [Week 2 Exercises](/module-1/week-2/exercises) to practice TF2, URDF, and Nav2 concepts.
