---
sidebar_position: 1
title: TF2 Transforms
description: Understanding coordinate frames and transformations in ROS 2
---

# Week 2: TF2 Transforms

<WeekHeader
  week={2}
  title="TF2 & Navigation"
  module={1}
  estimatedHours={8}
  skills={["TF2", "Coordinate Frames", "URDF", "RViz2"]}
/>

<LearningObjectives
  week={2}
  objectives={[
    "Understand coordinate frames and their relationships",
    "Use TF2 to track transformations over time",
    "Create and visualize URDF robot descriptions",
    "Debug transform issues with RViz2 and CLI tools",
    "Implement static and dynamic transform broadcasters"
  ]}
/>

## Understanding Coordinate Frames

In robotics, we work with multiple **coordinate frames** (also called reference frames). Each sensor, joint, and link on a robot has its own frame.

### Why Frames Matter

Consider a mobile robot with a camera:

```
World Frame (map)
    │
    └── Robot Base Frame (base_link)
            │
            ├── Wheel Frames (wheel_left, wheel_right)
            │
            └── Camera Frame (camera_link)
                    │
                    └── Camera Optical Frame (camera_optical)
```

When the camera detects an object at position (1.0, 0.5, 0.3) in camera coordinates, we need to transform this to:
- Robot base frame (for local planning)
- World frame (for global mapping)

### Frame Naming Conventions

| Frame Name | Description |
|------------|-------------|
| `map` | Global fixed frame, origin at map start |
| `odom` | Odometry frame, drifts over time |
| `base_link` | Robot body center, usually at ground level |
| `base_footprint` | Projection of base_link on ground |
| `*_link` | Physical component frames |
| `*_optical` | Camera optical frames (Z forward) |

## TF2 Library

**TF2** (Transform Library 2) manages coordinate frame relationships in ROS 2.

### Core Concepts

<ArchitectureDiagram title="TF2 Transform Tree">
{`
                    map
                     │
                     │ (localization)
                     ▼
                    odom
                     │
                     │ (odometry)
                     ▼
                 base_link
                /    │    \\
               /     │     \\
              ▼      ▼      ▼
        wheel_l  camera  wheel_r
                   │
                   ▼
             camera_optical
`}
</ArchitectureDiagram>

**Key Properties:**
- Transforms form a **tree** (no cycles, single root)
- Each transform has a **parent** and **child** frame
- Transforms are **timestamped** (track history)
- Can query transforms at any time point

### Transform Types

| Type | Lifetime | Use Case |
|------|----------|----------|
| **Static** | Forever | Fixed joints (camera mount) |
| **Dynamic** | Time-limited | Moving joints, odometry |

## Working with TF2

### Listening to Transforms

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class TransformListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to check transforms
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            # Get transform from camera to base_link
            transform = self.tf_buffer.lookup_transform(
                'base_link',      # Target frame
                'camera_link',    # Source frame
                rclpy.time.Time() # Latest available
            )

            self.get_logger().info(
                f'Camera position relative to base: '
                f'x={transform.transform.translation.x:.3f}, '
                f'y={transform.transform.translation.y:.3f}, '
                f'z={transform.transform.translation.z:.3f}'
            )

        except Exception as e:
            self.get_logger().warn(f'Transform not available: {e}')

    def transform_point(self, point_in_camera):
        """Transform a point from camera frame to base frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rclpy.time.Time()
            )
            return do_transform_point(point_in_camera, transform)
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
```

### Broadcasting Transforms

**Static Transform** (fixed relationship):

```python
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class StaticTFBroadcaster(Node):
    def __init__(self):
        super().__init__('static_tf_broadcaster')

        self.broadcaster = StaticTransformBroadcaster(self)

        # Camera is 0.2m forward, 0.1m up from base_link
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'camera_link'

        transform.transform.translation.x = 0.2
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.1

        # No rotation (identity quaternion)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.broadcaster.sendTransform(transform)
        self.get_logger().info('Published static transform')
```

**Dynamic Transform** (changes over time):

```python
from tf2_ros import TransformBroadcaster

class DynamicTFBroadcaster(Node):
    def __init__(self):
        super().__init__('dynamic_tf_broadcaster')

        self.broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_transform)
        self.angle = 0.0

    def broadcast_transform(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'rotating_sensor'

        # Rotate around Z axis
        transform.transform.translation.x = 0.3
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.15

        # Quaternion for Z rotation
        import math
        transform.transform.rotation.z = math.sin(self.angle / 2)
        transform.transform.rotation.w = math.cos(self.angle / 2)

        self.broadcaster.sendTransform(transform)
        self.angle += 0.1
```

## TF2 CLI Tools

### View Transform Tree

```bash
# View complete TF tree as PDF
ros2 run tf2_tools view_frames

# This creates frames.pdf in current directory
```

### Query Transforms

```bash
# Echo transform between frames
ros2 run tf2_ros tf2_echo base_link camera_link

# Monitor all transforms
ros2 run tf2_ros tf2_monitor
```

### Static Transform Publisher

```bash
# Publish static transform from command line
ros2 run tf2_ros static_transform_publisher \
  --x 0.2 --y 0.0 --z 0.1 \
  --qx 0.0 --qy 0.0 --qz 0.0 --qw 1.0 \
  --frame-id base_link \
  --child-frame-id camera_link
```

## Quaternions and Rotations

ROS 2 uses **quaternions** for rotations (avoiding gimbal lock).

### Quaternion Basics

```python
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math

# Convert Euler angles (roll, pitch, yaw) to quaternion
roll = 0.0
pitch = 0.0
yaw = math.pi / 4  # 45 degrees

q = quaternion_from_euler(roll, pitch, yaw)
# Returns (x, y, z, w)

# Convert back to Euler
r, p, y = euler_from_quaternion(q)
```

### Common Rotations

| Rotation | Quaternion (x, y, z, w) |
|----------|------------------------|
| Identity (no rotation) | (0, 0, 0, 1) |
| 90° around Z | (0, 0, 0.707, 0.707) |
| 180° around Z | (0, 0, 1, 0) |
| 90° around X | (0.707, 0, 0, 0.707) |

## Debugging Transforms

### Common Issues

:::tip Frame Not Found
Check that all transforms in the chain are being published:
```bash
ros2 run tf2_tools view_frames
```
Look for disconnected nodes in the tree.
:::

:::tip Transform Timeout
If transforms are too old:
```python
# Use time tolerance
transform = self.tf_buffer.lookup_transform(
    'base_link', 'camera_link',
    rclpy.time.Time(),
    timeout=rclpy.duration.Duration(seconds=1.0)
)
```
:::

:::tip Wrong Transform Direction
Remember: `lookup_transform(target, source)` gives you `source → target`.
To transform a point FROM camera TO base, use:
```python
lookup_transform('base_link', 'camera_link', ...)
```
:::

### RViz2 Visualization

1. Add **TF** display to see all frames
2. Set **Fixed Frame** to your reference (usually `map` or `odom`)
3. Enable **Show Names** and **Show Axes**
4. Use **Show Arrows** for transform directions

## Exercises

<Exercise title="Camera Transform Chain" difficulty="beginner" estimatedTime="20 min">

Create a node that:
1. Broadcasts a static transform from `base_link` to `camera_mount` (0.15m forward, 0.3m up)
2. Broadcasts a static transform from `camera_mount` to `camera_optical` (rotated -90° around Z, then -90° around X for optical convention)
3. Visualize in RViz2

<Hint>
The camera optical frame convention has Z pointing forward (into the scene), X pointing right, Y pointing down.
</Hint>

<Solution>
```python
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class CameraFrames(Node):
    def __init__(self):
        super().__init__('camera_frames')
        self.broadcaster = StaticTransformBroadcaster(self)

        transforms = []

        # base_link -> camera_mount
        t1 = TransformStamped()
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'camera_mount'
        t1.transform.translation.x = 0.15
        t1.transform.translation.z = 0.3
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        # camera_mount -> camera_optical
        # Rotate to optical frame convention
        t2 = TransformStamped()
        t2.header.frame_id = 'camera_mount'
        t2.child_frame_id = 'camera_optical'
        # 90° rotation: Z forward, Y down
        t2.transform.rotation.x = -0.5
        t2.transform.rotation.y = 0.5
        t2.transform.rotation.z = -0.5
        t2.transform.rotation.w = 0.5
        transforms.append(t2)

        self.broadcaster.sendTransform(transforms)

def main():
    rclpy.init()
    rclpy.spin(CameraFrames())
    rclpy.shutdown()
```
</Solution>

</Exercise>

<Exercise title="Object Tracker" difficulty="intermediate" estimatedTime="30 min">

Create a node that:
1. Subscribes to a simulated object detection topic (`/detected_object` with `geometry_msgs/PointStamped`)
2. Transforms detected points from `camera_optical` frame to `map` frame
3. Publishes transformed positions to `/object_in_map`

Test with TurtleBot3 in Gazebo.

<Hint>
Use `tf2_geometry_msgs.do_transform_point()` for the transformation.
Make sure to handle `LookupException` and `ExtrapolationException`.
</Hint>

</Exercise>

## Summary

Key concepts from this lesson:

- ✅ Coordinate frames represent positions and orientations
- ✅ TF2 maintains a tree of frame relationships
- ✅ Static transforms for fixed relationships
- ✅ Dynamic transforms for moving components
- ✅ Quaternions represent rotations without gimbal lock
- ✅ CLI tools help debug transform issues

## Next Steps

Continue to [URDF Basics](/module-1/week-2/urdf-basics) to learn how to describe your robot's structure for simulation and visualization.
