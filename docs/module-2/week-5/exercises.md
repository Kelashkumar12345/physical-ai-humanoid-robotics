---
sidebar_position: 4
title: Week 5 Exercises
description: Hands-on exercises for robot models and control
---

# Week 5: Robot Models and Control Exercises

Practice creating robot descriptions and implementing control systems.

## Exercise 1: Build a Custom URDF Robot

**Objective**: Create a URDF description for a simple 3-DOF robot arm.

### Requirements

1. Base link (fixed to world)
2. Three revolute joints
3. Three arm links with proper inertial properties
4. End effector link
5. Gazebo compatibility tags

### Robot Specifications

| Link | Length | Radius | Mass |
|------|--------|--------|------|
| Base | 0.1m height | 0.08m | 2.0kg |
| Link 1 | 0.25m | 0.03m | 0.5kg |
| Link 2 | 0.2m | 0.025m | 0.4kg |
| Link 3 | 0.15m | 0.02m | 0.3kg |

### Starter Template

```xml
<?xml version="1.0"?>
<robot name="simple_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>

  <!-- TODO: Add base_link -->

  <!-- TODO: Add link_1 with revolute joint to base -->

  <!-- TODO: Add link_2 with revolute joint to link_1 -->

  <!-- TODO: Add link_3 with revolute joint to link_2 -->

  <!-- TODO: Add end_effector with fixed joint to link_3 -->

  <!-- TODO: Add Gazebo tags -->

</robot>
```

### Validation

```bash
# Check URDF validity
check_urdf simple_arm.urdf

# View in RViz
ros2 launch urdf_tutorial display.launch.py model:=simple_arm.urdf
```

---

## Exercise 2: Add Joint Controllers

**Objective**: Configure joint position controllers for the robot arm.

### Requirements

1. Add joint state publisher plugin
2. Add position controller for each joint
3. Configure appropriate PID gains
4. Test joint control via command line

### Controller Configuration

```xml
<!-- Add to your model SDF -->
<plugin filename="gz-sim-joint-state-publisher-system"
        name="gz::sim::systems::JointStatePublisher">
  <!-- TODO: Configure joint state publisher -->
</plugin>

<!-- TODO: Add joint position controllers -->
```

### Testing

```bash
# Spawn robot in Gazebo
gz sim -r empty.sdf

# Send joint commands
gz topic -t /joint_1/cmd_pos -m gz.msgs.Double -p 'data: 1.0'
```

<details className="hint-block">
<summary>Hint: PID Tuning</summary>
<div className="hint-content">

Start with these gains and adjust:
- P gain: 50-100 for heavier links
- I gain: 0.1-1.0 (start low)
- D gain: 5-10 (reduces oscillation)

</div>
</details>

---

## Exercise 3: Differential Drive Robot

**Objective**: Create a complete differential drive robot with sensors.

### Requirements

1. Box-shaped chassis (0.3m x 0.2m x 0.1m)
2. Two drive wheels with proper friction
3. One caster wheel for stability
4. Front-facing camera
5. 360-degree LiDAR
6. IMU sensor
7. Differential drive plugin

### Starter Template

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="my_diff_drive_robot">
    <pose>0 0 0.05 0 0 0</pose>

    <!-- Base Link -->
    <link name="base_link">
      <!-- TODO: Add inertial, collision, visual -->
      <!-- TODO: Add IMU sensor -->
    </link>

    <!-- TODO: Add left wheel with joint -->
    <!-- TODO: Add right wheel with joint -->
    <!-- TODO: Add caster wheel -->
    <!-- TODO: Add camera link and sensor -->
    <!-- TODO: Add LiDAR link and sensor -->

    <!-- Plugins -->
    <!-- TODO: Add differential drive plugin -->
    <!-- TODO: Add joint state publisher -->

  </model>
</sdf>
```

### Testing Checklist

- [ ] Robot can drive forward/backward
- [ ] Robot can rotate in place
- [ ] LiDAR scan visible in RViz
- [ ] Camera image visible
- [ ] IMU data publishing
- [ ] Odometry is accurate

---

## Exercise 4: ROS 2 Bridge Setup

**Objective**: Create a complete ROS 2 bridge configuration for the differential drive robot.

### Requirements

1. Bridge all sensor topics to ROS 2
2. Bridge cmd_vel and odometry
3. Create a launch file that starts everything
4. Verify all topics in ROS 2

### Bridge Configuration

```yaml
# my_robot_bridge.yaml
# TODO: Add bridge configurations for:
# - /camera/image -> sensor_msgs/msg/Image
# - /scan -> sensor_msgs/msg/LaserScan
# - /imu -> sensor_msgs/msg/Imu
# - /odom -> nav_msgs/msg/Odometry
# - /cmd_vel <- geometry_msgs/msg/Twist
```

### Launch File

```python
# my_robot_sim_launch.py
from launch import LaunchDescription
# TODO: Complete launch file

def generate_launch_description():
    # TODO: Launch Gazebo with world
    # TODO: Spawn robot
    # TODO: Start bridge
    # TODO: Start RViz

    return LaunchDescription([
        # Add launch actions
    ])
```

---

## Exercise 5: TurtleBot3 Wall Following

**Objective**: Implement a wall-following behavior for TurtleBot3.

### Algorithm

1. Keep a fixed distance from the right wall
2. Use LiDAR to measure wall distance
3. Adjust angular velocity to maintain distance
4. Handle corners and gaps

### Starter Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        # Parameters
        self.declare_parameter('target_distance', 0.5)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 2.0)  # Proportional gain

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def scan_callback(self, msg):
        # TODO: Extract right-side distance from scan
        # TODO: Calculate error from target distance
        # TODO: Apply proportional control
        # TODO: Handle corners and gaps
        # TODO: Publish cmd_vel

        pass

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<details className="solution-block">
<summary>Solution</summary>
<div className="solution-content">

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        self.declare_parameter('target_distance', 0.5)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 2.0)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def get_range_at_angle(self, msg, angle_deg):
        """Get range at specified angle (0=front, 90=left, -90=right)"""
        angle_rad = math.radians(angle_deg)
        index = int((angle_rad - msg.angle_min) / msg.angle_increment)
        index = max(0, min(index, len(msg.ranges) - 1))

        r = msg.ranges[index]
        if math.isinf(r) or math.isnan(r):
            return msg.range_max
        return r

    def scan_callback(self, msg):
        target_dist = self.get_parameter('target_distance').value
        linear_speed = self.get_parameter('linear_speed').value
        kp = self.get_parameter('kp').value

        # Get distances at key angles
        front = self.get_range_at_angle(msg, 0)
        front_right = self.get_range_at_angle(msg, -45)
        right = self.get_range_at_angle(msg, -90)
        back_right = self.get_range_at_angle(msg, -135)

        cmd = Twist()

        # State machine for wall following
        if front < 0.4:
            # Obstacle ahead - turn left
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
            self.get_logger().info('Turning left - obstacle ahead')
        elif right > target_dist * 2:
            # Lost wall - turn right to find it
            cmd.linear.x = linear_speed * 0.5
            cmd.angular.z = -0.3
            self.get_logger().info('Finding wall')
        else:
            # Follow wall with P controller
            error = target_dist - right

            # Use front_right to detect approaching corners
            corner_factor = 0.0
            if front_right < target_dist:
                corner_factor = (target_dist - front_right) * kp

            angular_z = kp * error - corner_factor

            cmd.linear.x = linear_speed
            cmd.angular.z = max(-1.0, min(1.0, angular_z))

            self.get_logger().info(
                f'Following: right={right:.2f}, error={error:.2f}, '
                f'angular={angular_z:.2f}'
            )

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

</div>
</details>

---

## Challenge: Mobile Manipulator

**Objective**: Create a mobile manipulator - differential drive base with 2-DOF arm.

### Requirements

1. Differential drive base (from Exercise 3)
2. 2-DOF arm mounted on top
3. Joint controllers for arm
4. Combined URDF/SDF model
5. ROS 2 interface for all actuators

### Advanced Features

- Arm workspace visualization
- Collision avoidance between arm and base
- Coordinated base-arm motion

### Architecture

```
Mobile Manipulator
├── Base (diff drive)
│   ├── Left wheel
│   ├── Right wheel
│   └── Caster
├── Arm
│   ├── Shoulder joint (revolute)
│   ├── Elbow joint (revolute)
│   └── End effector
└── Sensors
    ├── LiDAR (base)
    ├── Camera (arm)
    └── IMU (base)
```

### Control Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | Twist | Base velocity |
| `/arm/joint1/cmd_pos` | Float64 | Shoulder position |
| `/arm/joint2/cmd_pos` | Float64 | Elbow position |
| `/joint_states` | JointState | All joint states |

---

## Submission Checklist

Before submitting, verify:

- [ ] URDF/SDF files are valid
- [ ] All joints have correct types and limits
- [ ] Inertial properties are physically reasonable
- [ ] Sensors produce correct data
- [ ] Controllers respond to commands
- [ ] ROS 2 bridge is properly configured
- [ ] Launch files work correctly
- [ ] Code is documented

## Resources

- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [SDF Specification](http://sdformat.org/spec)
- [Gazebo Plugins](https://gazebosim.org/docs/harmonic/plugins)
- [TurtleBot3 Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)
