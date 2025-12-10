---
sidebar_position: 2
title: Nodes and Topics
description: Deep dive into ROS 2 nodes, publishers, subscribers, and topic communication
---

# Nodes and Topics

This lesson explores ROS 2's publish-subscribe communication pattern in depth. You'll learn to create publishers and subscribers, understand message types, and build multi-node systems.

<LearningObjectives
  objectives={[
    "Create publisher and subscriber nodes in Python",
    "Understand ROS 2 message types and interfaces",
    "Implement callback-based message handling",
    "Debug communication issues using CLI tools",
    "Design node architectures for robot systems"
  ]}
/>

## Understanding Nodes

A **node** is the fundamental unit of computation in ROS 2. Each node should handle a specific task:

| Node Role | Example | Responsibility |
|-----------|---------|----------------|
| Sensor Driver | camera_driver | Read camera data, publish images |
| Perception | object_detector | Process images, publish detections |
| Planning | path_planner | Calculate routes from A to B |
| Control | motor_controller | Send velocity commands |

### Node Design Principles

1. **Single Responsibility**: Each node does one thing well
2. **Loose Coupling**: Nodes communicate only through topics/services
3. **Reusability**: Generic interfaces enable node reuse
4. **Testability**: Isolated nodes are easier to test

## Publishers and Subscribers

### The Publish-Subscribe Pattern

```
                    Topic: /sensor_data
                    Type: sensor_msgs/LaserScan
                           │
    ┌───────────────┐      │      ┌───────────────┐
    │  LIDAR Driver │──────┼─────▶│   Obstacle    │
    │   (Publisher) │      │      │   Detector    │
    └───────────────┘      │      │  (Subscriber) │
                           │      └───────────────┘
                           │
                           │      ┌───────────────┐
                           └─────▶│   SLAM Node   │
                                  │  (Subscriber) │
                                  └───────────────┘
```

**Key characteristics:**
- Publishers don't know who subscribes
- Subscribers don't know who publishes
- Multiple publishers and subscribers per topic
- Messages are broadcast (one-to-many)

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class TemperatureSensor(Node):
    def __init__(self):
        super().__init__('temperature_sensor')

        # Create publisher
        # Parameters: message_type, topic_name, queue_size
        self.publisher_ = self.create_publisher(
            Float64,
            'temperature',
            10
        )

        # Create timer for periodic publishing
        self.timer = self.create_timer(0.5, self.publish_temperature)
        self.get_logger().info('Temperature sensor started')

    def publish_temperature(self):
        msg = Float64()
        msg.data = 25.5 + (self.get_clock().now().nanoseconds % 100) / 100
        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published: {msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = TemperatureSensor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class TemperatureMonitor(Node):
    def __init__(self):
        super().__init__('temperature_monitor')

        # Create subscription
        self.subscription = self.create_subscription(
            Float64,
            'temperature',
            self.temperature_callback,
            10
        )
        self.get_logger().info('Temperature monitor started')

    def temperature_callback(self, msg: Float64):
        """Called every time a message is received."""
        temp = msg.data

        if temp > 30.0:
            self.get_logger().warn(f'High temperature: {temp:.2f}°C')
        else:
            self.get_logger().info(f'Temperature: {temp:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = TemperatureMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Message Types

### Standard Messages

ROS 2 provides common message types in these packages:

<TopicTable
  topics={[
    { name: "std_msgs", type: "Bool, Int32, Float64, String", description: "Primitive types" },
    { name: "geometry_msgs", type: "Point, Pose, Twist, Transform", description: "Geometric primitives" },
    { name: "sensor_msgs", type: "Image, LaserScan, Imu, PointCloud2", description: "Sensor data" },
    { name: "nav_msgs", type: "Odometry, Path, OccupancyGrid", description: "Navigation data" }
  ]}
/>

### Inspecting Messages

```bash
# List all message types
ros2 interface list | grep msg

# Show message structure
ros2 interface show geometry_msgs/msg/Twist
```

Output:
```
# Linear velocity in m/s
geometry_msgs/Vector3 linear
    float64 x
    float64 y
    float64 z

# Angular velocity in rad/s
geometry_msgs/Vector3 angular
    float64 x
    float64 y
    float64 z
```

### Using Complex Messages

```python
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan

# Create a Twist message (velocity command)
vel_cmd = Twist()
vel_cmd.linear.x = 0.5   # Forward velocity (m/s)
vel_cmd.angular.z = 0.1  # Rotation velocity (rad/s)

# Create a Pose message (position + orientation)
pose = Pose()
pose.position.x = 1.0
pose.position.y = 2.0
pose.position.z = 0.0
pose.orientation.w = 1.0  # Identity quaternion
```

### Creating Custom Messages

For custom data types, create a new package:

```bash
ros2 pkg create --build-type ament_cmake my_interfaces
```

Create `msg/RobotStatus.msg`:
```
# Custom robot status message
string robot_name
float64 battery_level
bool is_moving
float64[3] position
```

Update `CMakeLists.txt`:
```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
)
```

## Multi-Node Communication

### Example: Robot System

Let's build a simple robot system with multiple nodes:

```
┌─────────────┐     /cmd_vel      ┌─────────────┐
│  Teleop     │ ────────────────▶ │   Motion    │
│  Controller │   Twist           │  Controller │
└─────────────┘                   └─────────────┘
                                        │
                                        │ /wheel_speeds
                                        ▼
                                  ┌─────────────┐
                                  │   Motor     │
                                  │   Driver    │
                                  └─────────────┘
```

**Teleop Controller** (`teleop_controller.py`):

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty

class TeleopController(Node):
    def __init__(self):
        super().__init__('teleop_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.read_keyboard)

        self.linear_speed = 0.0
        self.angular_speed = 0.0

        self.get_logger().info('Teleop ready. Use WASD keys.')

    def read_keyboard(self):
        # Simplified - in practice use a proper keyboard library
        msg = Twist()
        msg.linear.x = self.linear_speed
        msg.angular.z = self.angular_speed
        self.publisher_.publish(msg)
```

**Motion Controller** (`motion_controller.py`):

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')

        # Subscribe to velocity commands
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10
        )

        # Publish wheel speeds
        self.wheel_pub = self.create_publisher(
            Float64MultiArray,
            'wheel_speeds',
            10
        )

        # Robot parameters
        self.wheel_base = 0.3  # meters
        self.wheel_radius = 0.05  # meters

    def cmd_callback(self, msg: Twist):
        """Convert Twist to differential drive wheel speeds."""
        linear = msg.linear.x
        angular = msg.angular.z

        # Differential drive kinematics
        left_speed = (linear - angular * self.wheel_base / 2) / self.wheel_radius
        right_speed = (linear + angular * self.wheel_base / 2) / self.wheel_radius

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [left_speed, right_speed]
        self.wheel_pub.publish(wheel_msg)

        self.get_logger().debug(f'Wheels: L={left_speed:.2f}, R={right_speed:.2f}')
```

## Debugging Topics

### Common Issues and Solutions

:::tip No Messages Received
Check if topics match:
```bash
ros2 topic list
ros2 topic info /your_topic
```
Verify QoS compatibility between publisher and subscriber.
:::

:::tip Messages Delayed
Check queue sizes and publishing rates:
```bash
ros2 topic hz /topic_name
ros2 topic bw /topic_name
```
:::

### Visualization Tools

**rqt_graph** - Visualize node connections:
```bash
ros2 run rqt_graph rqt_graph
```

**rqt_topic** - Monitor topic data:
```bash
ros2 run rqt_topic rqt_topic
```

## Exercises

<Exercise title="Temperature Alert System" difficulty="beginner" estimatedTime="20 min">

Create a two-node system:
1. **Publisher**: Simulates temperature readings (random values 20-35°C)
2. **Subscriber**: Logs warnings when temperature exceeds 30°C

Requirements:
- Publishing rate: 2 Hz
- Use `std_msgs/Float64` message type
- Topic name: `/room_temperature`

<Hint>
Use Python's `random.uniform(20, 35)` to generate random temperatures.
</Hint>

<Solution>
Publisher:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import random

class TempPublisher(Node):
    def __init__(self):
        super().__init__('temp_publisher')
        self.pub = self.create_publisher(Float64, 'room_temperature', 10)
        self.timer = self.create_timer(0.5, self.publish)

    def publish(self):
        msg = Float64()
        msg.data = random.uniform(20, 35)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(TempPublisher())
    rclpy.shutdown()
```
</Solution>

</Exercise>

<Exercise title="Velocity Smoother" difficulty="intermediate" estimatedTime="30 min">

Create a node that subscribes to `/cmd_vel_raw` and publishes smoothed velocities to `/cmd_vel`:
- Limit acceleration to 0.5 m/s²
- Limit angular acceleration to 1.0 rad/s²
- Publishing rate: 20 Hz

<Hint>
Store the last published velocity and interpolate toward the target.
</Hint>

</Exercise>

## Summary

Key takeaways from this lesson:

- ✅ Nodes are single-purpose computational units
- ✅ Publishers broadcast messages to topics
- ✅ Subscribers receive messages via callbacks
- ✅ Standard message types cover most use cases
- ✅ Custom messages extend functionality
- ✅ CLI tools help debug communication issues

## Next Steps

Continue to [Services and Actions](/module-1/week-1/services-and-actions) to learn about request-response and long-running task patterns in ROS 2.
