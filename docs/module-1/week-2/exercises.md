---
sidebar_position: 4
title: Week 2 Exercises
description: Hands-on exercises for TF2, URDF, and Nav2
---

# Week 2 Exercises

Complete these exercises to reinforce your understanding of coordinate transforms, robot descriptions, and navigation.

<Prerequisites
  items={[
    "Completed all Week 2 lessons",
    "TurtleBot3 simulation working in Gazebo",
    "Nav2 packages installed",
    "Understanding of TF2 transform concepts"
  ]}
/>

## Exercise 1: Transform Chain

<Exercise title="Multi-Sensor Robot Transforms" difficulty="beginner" estimatedTime="25 min">

Create a node that broadcasts a complete transform tree for a robot with:

- `base_link` - Robot center
- `lidar_link` - LiDAR sensor (0.0, 0.0, 0.2) from base
- `camera_link` - RGB camera (0.15, 0.0, 0.1) from base
- `imu_link` - IMU sensor (0.0, 0.0, 0.05) from base

**Requirements:**
1. All transforms should be static
2. Verify with `ros2 run tf2_tools view_frames`
3. Echo transforms using `ros2 run tf2_ros tf2_echo`

**Acceptance Criteria:**
- [ ] All 4 frames visible in TF tree
- [ ] Transform values are correct
- [ ] Tree has `base_link` as root

<Hint>
Use `StaticTransformBroadcaster` and send all transforms in a list:
```python
self.broadcaster.sendTransform([t1, t2, t3, t4])
```
</Hint>

<Solution>
```python
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class SensorFrames(Node):
    def __init__(self):
        super().__init__('sensor_frames')
        self.broadcaster = StaticTransformBroadcaster(self)

        transforms = []

        # LiDAR
        t_lidar = TransformStamped()
        t_lidar.header.frame_id = 'base_link'
        t_lidar.child_frame_id = 'lidar_link'
        t_lidar.transform.translation.z = 0.2
        t_lidar.transform.rotation.w = 1.0
        transforms.append(t_lidar)

        # Camera
        t_camera = TransformStamped()
        t_camera.header.frame_id = 'base_link'
        t_camera.child_frame_id = 'camera_link'
        t_camera.transform.translation.x = 0.15
        t_camera.transform.translation.z = 0.1
        t_camera.transform.rotation.w = 1.0
        transforms.append(t_camera)

        # IMU
        t_imu = TransformStamped()
        t_imu.header.frame_id = 'base_link'
        t_imu.child_frame_id = 'imu_link'
        t_imu.transform.translation.z = 0.05
        t_imu.transform.rotation.w = 1.0
        transforms.append(t_imu)

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info('Published sensor transforms')

def main():
    rclpy.init()
    node = SensorFrames()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
</Solution>

</Exercise>

## Exercise 2: Custom URDF

<Exercise title="Differential Drive Robot URDF" difficulty="intermediate" estimatedTime="35 min">

Create a complete URDF/Xacro for a differential drive robot:

**Specifications:**
- Chassis: Box 0.3m x 0.2m x 0.08m
- Two drive wheels: radius 0.04m, width 0.02m, positioned at rear
- Caster wheel: sphere radius 0.015m at front
- LiDAR mount: cylinder on top (radius 0.03m, height 0.02m)

**Requirements:**
1. Use Xacro with properties for dimensions
2. Create a wheel macro for code reuse
3. Include proper inertial values
4. Visualize with joint_state_publisher_gui

**Acceptance Criteria:**
- [ ] Robot displays correctly in RViz2
- [ ] Wheels can be moved with GUI sliders
- [ ] Colors distinguish different parts
- [ ] check_urdf reports no errors

<Hint>
Structure your Xacro file:
1. Define properties at top
2. Create wheel macro with prefix parameter
3. Define chassis link first
4. Add wheels using the macro
5. Add caster and sensors
</Hint>

</Exercise>

## Exercise 3: Point Transformer

<Exercise title="Object Position Transformer" difficulty="intermediate" estimatedTime="30 min">

Create a node that:

1. Subscribes to `/detected_point` (geometry_msgs/PointStamped in camera_link frame)
2. Transforms points to `map` frame
3. Publishes transformed points to `/point_in_map`
4. Handles transform exceptions gracefully

**Test Setup:**
- Run TurtleBot3 in Gazebo with Nav2
- Publish test points: `ros2 topic pub /detected_point geometry_msgs/msg/PointStamped ...`

**Acceptance Criteria:**
- [ ] Points are correctly transformed
- [ ] Handles missing transforms without crashing
- [ ] Logs transformation status

<Solution>
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ExtrapolationException
import tf2_geometry_msgs

class PointTransformer(Node):
    def __init__(self):
        super().__init__('point_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointStamped,
            'detected_point',
            self.point_callback,
            10
        )

        self.pub = self.create_publisher(PointStamped, 'point_in_map', 10)
        self.get_logger().info('Point transformer ready')

    def point_callback(self, msg):
        try:
            # Transform to map frame
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            transformed = tf2_geometry_msgs.do_transform_point(msg, transform)
            transformed.header.frame_id = 'map'
            self.pub.publish(transformed)

            self.get_logger().info(
                f'Transformed: ({msg.point.x:.2f}, {msg.point.y:.2f}) -> '
                f'({transformed.point.x:.2f}, {transformed.point.y:.2f})'
            )

        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f'Transform failed: {e}')

def main():
    rclpy.init()
    rclpy.spin(PointTransformer())
    rclpy.shutdown()
```
</Solution>

</Exercise>

## Exercise 4: Nav2 Patrol Mission

<Exercise title="Autonomous Patrol Robot" difficulty="advanced" estimatedTime="45 min">

Create a patrol node that:

1. Loads waypoints from a YAML file
2. Navigates to each waypoint in sequence
3. Waits 5 seconds at each waypoint
4. Logs arrival time and position
5. Repeats patrol indefinitely until shutdown
6. Handles navigation failures (retry once, then skip)

**Waypoint File Format (patrol_waypoints.yaml):**
```yaml
waypoints:
  - name: "location_a"
    x: 1.0
    y: 0.5
    yaw: 0.0
  - name: "location_b"
    x: 2.0
    y: 1.5
    yaw: 1.57
  - name: "location_c"
    x: 0.5
    y: 2.0
    yaw: 3.14
```

**Acceptance Criteria:**
- [ ] Loads waypoints from YAML
- [ ] Navigates to all waypoints
- [ ] Waits at each waypoint
- [ ] Handles failures gracefully
- [ ] Logs mission progress

<Hint>
Use the `nav2_simple_commander` package:
```python
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
```

Load YAML with:
```python
import yaml
with open(waypoint_file, 'r') as f:
    data = yaml.safe_load(f)
```
</Hint>

</Exercise>

## Exercise 5: Integration Challenge

<Exercise title="Sensor-Aware Navigation" difficulty="advanced" estimatedTime="60 min">

Create a comprehensive system that:

1. **Transform Node**: Broadcasts sensor frames for LiDAR and camera
2. **Detection Simulator**: Publishes fake obstacle detections in camera frame
3. **Mapper Node**: Transforms detections to map frame and publishes markers
4. **Navigator Node**: Avoids areas near detected obstacles

**System Architecture:**
```
┌─────────────────┐     /detected_obstacle    ┌─────────────────┐
│    Detection    │ ─────────────────────────▶│     Mapper      │
│    Simulator    │    (camera_link frame)    │     Node        │
└─────────────────┘                           └────────┬────────┘
                                                       │
                                              /obstacle_markers
                                                       │
                                                       ▼
┌─────────────────┐     /patrol_waypoints     ┌─────────────────┐
│   Waypoint      │ ─────────────────────────▶│   Navigator     │
│   Server        │                           │     Node        │
└─────────────────┘                           └─────────────────┘
```

**Requirements:**
- Detected obstacles should appear as red spheres in RViz2
- Navigator should compute safe paths around obstacles
- System should work with TurtleBot3 in Gazebo

**Acceptance Criteria:**
- [ ] All nodes start and communicate
- [ ] Obstacle markers visible in RViz2
- [ ] Robot navigates avoiding marked obstacles
- [ ] Clean shutdown on Ctrl+C

</Exercise>

## Self-Assessment

Rate your confidence (1-5) after completing these exercises:

| Skill | Target | Your Rating |
|-------|--------|-------------|
| Broadcasting static transforms | 4 | ___ |
| Using TF2 buffer and listener | 4 | ___ |
| Writing URDF files | 3 | ___ |
| Using Xacro macros | 3 | ___ |
| Sending Nav2 goals | 4 | ___ |
| Understanding costmaps | 3 | ___ |
| Debugging transform issues | 4 | ___ |

If any rating is below target, review the corresponding lesson material.

## Submission Checklist

Before moving to Week 3, ensure:

- [ ] All exercises compile without warnings
- [ ] URDF passes `check_urdf` validation
- [ ] Transform trees are correct (`view_frames`)
- [ ] Navigation works with TurtleBot3
- [ ] Code follows ROS 2 Python style guidelines

---

**Ready for Week 3?** Continue to Week 3: Perception Pipeline (coming soon) to learn about sensor processing and computer vision.
