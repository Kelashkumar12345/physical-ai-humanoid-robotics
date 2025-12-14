---
sidebar_position: 3
title: TurtleBot3 Simulation
description: Working with TurtleBot3 in Gazebo simulation
---

# TurtleBot3 Simulation

Learn to simulate the popular TurtleBot3 mobile robot platform in Gazebo.

## Learning Objectives

By the end of this lesson, you will:

- Install and configure TurtleBot3 simulation packages
- Launch TurtleBot3 in various Gazebo worlds
- Control the robot with teleop and programmatic commands
- Integrate with Nav2 for autonomous navigation

## TurtleBot3 Overview

### Robot Models

| Model | Size | Sensors | Use Case |
|-------|------|---------|----------|
| Burger | 138mm x 178mm x 192mm | LiDAR, IMU | Basic navigation |
| Waffle | 281mm x 306mm x 141mm | LiDAR, IMU, Camera | Vision + navigation |
| Waffle Pi | 281mm x 306mm x 141mm | LiDAR, IMU, Camera, RasPi | Full stack |

### Package Structure

```
turtlebot3/
├── turtlebot3_description/     # URDF models
├── turtlebot3_gazebo/          # Gazebo worlds and launch
├── turtlebot3_navigation2/     # Nav2 configuration
├── turtlebot3_cartographer/    # SLAM configuration
├── turtlebot3_teleop/          # Keyboard control
└── turtlebot3_bringup/         # Real robot launch
```

## Installation

### Install TurtleBot3 Packages

```bash
# Install TurtleBot3 packages
sudo apt install ros-humble-turtlebot3* -y

# Install Gazebo packages
sudo apt install ros-humble-turtlebot3-gazebo -y

# Install navigation packages
sudo apt install ros-humble-navigation2 \
                 ros-humble-nav2-bringup -y
```

### Environment Setup

```bash
# Add to ~/.bashrc
export TURTLEBOT3_MODEL=burger  # or waffle, waffle_pi

# Source the setup
source ~/.bashrc
```

## Launching TurtleBot3 in Gazebo

### Empty World

```bash
# Set robot model
export TURTLEBOT3_MODEL=burger

# Launch in empty world
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

### TurtleBot3 World

```bash
# Launch in TurtleBot3 world
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### House World

```bash
# Launch in house environment
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

## Controlling the Robot

### Keyboard Teleop

```bash
# In a new terminal
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

Controls:
```
        w
   a    s    d
        x

w/x : increase/decrease linear velocity
a/d : increase/decrease angular velocity
space key, s : force stop
```

### Programmatic Control

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')

        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('TurtleBot controller started')

    def control_loop(self):
        msg = Twist()

        # Drive forward
        msg.linear.x = 0.2  # m/s
        msg.angular.z = 0.0  # rad/s

        self.publisher.publish(msg)

    def stop(self):
        msg = Twist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Square Pattern

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SquareDriver(Node):
    def __init__(self):
        super().__init__('square_driver')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def drive_square(self, side_length=1.0, speed=0.2):
        for i in range(4):
            # Drive forward
            self.get_logger().info(f'Driving side {i+1}')
            self.drive(speed, 0.0, side_length / speed)

            # Turn 90 degrees
            self.get_logger().info('Turning')
            self.drive(0.0, 0.5, 3.14159 / 2 / 0.5)

        self.stop()

    def drive(self, linear, angular, duration):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        start = time.time()
        while time.time() - start < duration:
            self.publisher.publish(msg)
            time.sleep(0.1)

    def stop(self):
        msg = Twist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SquareDriver()
    node.drive_square()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Topics

### Available Topics

```bash
# List all TurtleBot3 topics
ros2 topic list | grep -E "scan|odom|imu|camera"

# Common topics:
# /scan           - LiDAR data
# /odom           - Odometry
# /imu            - IMU data
# /camera/image   - Camera image (Waffle models)
```

### Reading LiDAR Data

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarReader(Node):
    def __init__(self):
        super().__init__('lidar_reader')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg):
        # Get distance in front
        front_index = len(msg.ranges) // 2
        front_distance = msg.ranges[front_index]

        # Get minimum distance
        valid_ranges = [r for r in msg.ranges if r > msg.range_min]
        min_distance = min(valid_ranges) if valid_ranges else float('inf')

        self.get_logger().info(
            f'Front: {front_distance:.2f}m, Min: {min_distance:.2f}m'
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## SLAM with Cartographer

### Launch SLAM

```bash
# Terminal 1: Launch Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Launch Cartographer
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True

# Terminal 3: Launch RViz
ros2 launch turtlebot3_cartographer cartographer_rviz.launch.py

# Terminal 4: Drive robot with teleop
ros2 run turtlebot3_teleop teleop_keyboard
```

### Save the Map

```bash
# After mapping is complete
ros2 run nav2_map_server map_saver_cli -f ~/map
```

## Navigation with Nav2

### Launch Navigation

```bash
# Terminal 1: Launch Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Launch Navigation
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  use_sim_time:=True \
  map:=$HOME/map.yaml

# Terminal 3: Launch RViz
ros2 launch nav2_bringup rviz_launch.py
```

### Setting Goals in RViz

1. Set initial pose with "2D Pose Estimate" tool
2. Set goal with "Nav2 Goal" tool
3. Robot will autonomously navigate

### Programmatic Navigation

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def go_to_pose(self, x, y, theta):
        self.nav_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        import math
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2)

        self.get_logger().info(f'Navigating to ({x}, {y})')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        return result_future.result().status == 4  # SUCCEEDED

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()

    # Navigate to waypoints
    waypoints = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.57),
        (0.0, 1.0, 3.14),
        (0.0, 0.0, 0.0),
    ]

    for x, y, theta in waypoints:
        navigator.go_to_pose(x, y, theta)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Custom Launch File

```python
# turtlebot3_sim_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='turtlebot3_world')

    # Launch Gazebo with TurtleBot3
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            turtlebot3_gazebo_dir, '/launch/', world, '.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Launch RViz
    rviz_config = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'rviz',
        'tb3_gazebo.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='turtlebot3_world'),
        gazebo_launch,
        rviz_node,
    ])
```

## Obstacle Avoidance

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters
        self.safe_distance = 0.5  # meters
        self.linear_speed = 0.2   # m/s
        self.angular_speed = 0.5  # rad/s

    def scan_callback(self, msg):
        # Divide scan into regions
        ranges = list(msg.ranges)
        n = len(ranges)

        # Front region (center 60 degrees)
        front_start = n // 2 - n // 12
        front_end = n // 2 + n // 12
        front = min(ranges[front_start:front_end])

        # Left region
        left_start = n // 2 + n // 12
        left_end = n // 2 + n // 4
        left = min(ranges[left_start:left_end])

        # Right region
        right_start = n // 2 - n // 4
        right_end = n // 2 - n // 12
        right = min(ranges[right_start:right_end])

        # Decision making
        cmd = Twist()

        if front > self.safe_distance:
            # Path clear, go forward
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif left > right:
            # Turn left
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        else:
            # Turn right
            cmd.linear.x = 0.0
            cmd.angular.z = -self.angular_speed

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this lesson, you learned:

- Installing and configuring TurtleBot3 simulation packages
- Launching TurtleBot3 in different Gazebo worlds
- Controlling the robot with teleop and custom nodes
- Using SLAM with Cartographer
- Autonomous navigation with Nav2
- Implementing obstacle avoidance behaviors

## Next Steps

Continue to [Week 5 Exercises](/module-2/week-5/exercises) to practice robot simulation.
