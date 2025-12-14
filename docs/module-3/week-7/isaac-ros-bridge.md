---
sidebar_position: 3
title: Isaac ROS Bridge
description: Integrating Isaac Sim with ROS 2
---

# Isaac Sim ROS 2 Bridge

Learn to integrate Isaac Sim with ROS 2 for seamless simulation-to-robot development.

## Learning Objectives

By the end of this lesson, you will:

- Enable and configure the ROS 2 bridge in Isaac Sim
- Publish sensor data to ROS 2 topics
- Subscribe to ROS 2 commands for robot control
- Use standard ROS 2 tools with Isaac Sim

## Enabling ROS 2 Bridge

### Extension Setup

```python
from omni.isaac.kit import SimulationApp

# Enable ROS 2 bridge extension
simulation_app = SimulationApp({
    "headless": False,
    "enable_ros2_bridge": True
})

# Import ROS 2 bridge components
import omni.isaac.ros2_bridge as ros2_bridge
```

### Environment Configuration

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Set ROS_DOMAIN_ID if needed
export ROS_DOMAIN_ID=0

# Launch Isaac Sim
./isaac-sim.sh
```

## Publishing Sensor Data

### Camera to ROS 2

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.ros2_bridge import ROSCameraPublisher
import numpy as np

# Create world and camera
world = World()
world.scene.add_default_ground_plane()

camera = Camera(
    prim_path="/World/Camera",
    position=np.array([3.0, 0.0, 1.5]),
    frequency=30,
    resolution=(640, 480)
)
world.scene.add(camera)

# Create ROS 2 publisher
camera_pub = ROSCameraPublisher(
    camera=camera,
    topic_name="/camera/image_raw",
    frame_id="camera_link",
    publish_rgb=True,
    publish_depth=True,
    publish_camera_info=True
)

world.reset()
camera.initialize()

# Run simulation
while simulation_app.is_running():
    world.step(render=True)
    camera_pub.publish()

simulation_app.close()
```

### LiDAR to ROS 2

```python
from omni.isaac.sensor import RotatingLidarPhysX
from omni.isaac.ros2_bridge import ROSLidarPublisher

# Create LiDAR
lidar = RotatingLidarPhysX(
    prim_path="/World/Lidar",
    name="lidar",
    position=np.array([0, 0, 0.5])
)
lidar.set_fov([360.0, 30.0])
lidar.set_resolution([0.4, 1.0])
world.scene.add(lidar)

# Create ROS 2 publisher
lidar_pub = ROSLidarPublisher(
    lidar=lidar,
    topic_name="/scan",
    frame_id="lidar_link"
)

# In simulation loop
lidar_pub.publish()
```

### IMU to ROS 2

```python
from omni.isaac.sensor import IMUSensor
from omni.isaac.ros2_bridge import ROSImuPublisher

# Create IMU
imu = IMUSensor(
    prim_path="/World/Robot/base_link/IMU",
    name="imu",
    frequency=100
)
world.scene.add(imu)

# Create ROS 2 publisher
imu_pub = ROSImuPublisher(
    imu=imu,
    topic_name="/imu/data",
    frame_id="imu_link"
)

# In simulation loop
imu_pub.publish()
```

## Subscribing to ROS 2 Commands

### Twist Commands for Differential Drive

```python
from omni.isaac.wheeled_robots.controllers import DifferentialController
from omni.isaac.ros2_bridge import ROSTwistSubscriber
import rclpy

# Initialize ROS 2
if not rclpy.ok():
    rclpy.init()

# Create differential drive controller
controller = DifferentialController(
    name="diff_controller",
    wheel_radius=0.05,
    wheel_base=0.3
)

# Create twist subscriber
twist_sub = ROSTwistSubscriber(
    topic_name="/cmd_vel",
    callback=lambda msg: controller.forward([msg.linear.x, msg.angular.z])
)

# In simulation loop
world.step(render=True)
twist_sub.spin_once()

# Apply velocities to robot wheels
wheel_velocities = controller.get_wheel_velocities()
# ... apply to robot joints
```

### Joint Commands

```python
from omni.isaac.ros2_bridge import ROSJointStateSubscriber

class JointCommandHandler:
    def __init__(self, robot):
        self.robot = robot
        self.target_positions = None

    def joint_callback(self, msg):
        self.target_positions = dict(zip(msg.name, msg.position))

# Create subscriber
joint_sub = ROSJointStateSubscriber(
    topic_name="/joint_commands",
    callback=handler.joint_callback
)
```

## TF2 Integration

### Publishing TF Transforms

```python
from omni.isaac.ros2_bridge import ROSTF2Publisher
from omni.isaac.core.articulations import ArticulationView

# Create TF publisher
tf_pub = ROSTF2Publisher()

# Get robot articulation
robot_view = ArticulationView(
    prim_paths_expr="/World/Robot",
    name="robot"
)
world.scene.add(robot_view)

# In simulation loop
def publish_tf():
    # Get robot pose
    pose = robot_view.get_world_poses()

    # Publish transforms
    tf_pub.publish_transform(
        parent_frame="world",
        child_frame="base_link",
        position=pose[0][0],
        orientation=pose[1][0]
    )

# Publish link transforms
def publish_robot_tf(robot):
    link_poses = robot.get_link_poses()
    for link_name, pose in link_poses.items():
        tf_pub.publish_transform(
            parent_frame="base_link",
            child_frame=link_name,
            position=pose[:3],
            orientation=pose[3:]
        )
```

## Complete ROS 2 Robot Interface

```python
from omni.isaac.kit import SimulationApp

config = {
    "headless": False,
    "enable_ros2_bridge": True
}
simulation_app = SimulationApp(config)

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera, RotatingLidarPhysX, IMUSensor
from omni.isaac.wheeled_robots.controllers import DifferentialController
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

class IsaacROSBridge(Node):
    def __init__(self, world, robot_prim_path):
        super().__init__('isaac_ros_bridge')

        self.world = world
        self.robot_path = robot_prim_path
        self.bridge = CvBridge()

        # Setup sensors
        self._setup_sensors()

        # Setup publishers
        self._setup_publishers()

        # Setup subscribers
        self._setup_subscribers()

        # Controller
        self.controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.05,
            wheel_base=0.3
        )

        self.cmd_vel = Twist()

    def _setup_sensors(self):
        # Camera
        self.camera = Camera(
            prim_path=f"{self.robot_path}/camera_link/Camera",
            position=np.array([0.1, 0, 0.1]),
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # LiDAR
        self.lidar = RotatingLidarPhysX(
            prim_path=f"{self.robot_path}/lidar_link/Lidar",
            name="lidar",
            position=np.array([0, 0, 0.3])
        )
        self.lidar.set_fov([360.0, 30.0])
        self.lidar.set_resolution([1.0, 1.0])
        self.world.scene.add(self.lidar)

        # IMU
        self.imu = IMUSensor(
            prim_path=f"{self.robot_path}/base_link/IMU",
            name="imu",
            frequency=100
        )
        self.world.scene.add(self.imu)

    def _setup_publishers(self):
        # Camera publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)

        # LiDAR publisher
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # IMU publisher
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Odometry publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

    def _setup_subscribers(self):
        # Velocity command subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg

    def initialize_sensors(self):
        self.camera.initialize()
        self.lidar.initialize()
        self.imu.initialize()

    def publish_sensors(self):
        now = self.get_clock().now().to_msg()

        # Publish camera
        rgb = self.camera.get_rgba()
        if rgb is not None:
            img_msg = self.bridge.cv2_to_imgmsg(rgb[:, :, :3], 'rgb8')
            img_msg.header.stamp = now
            img_msg.header.frame_id = 'camera_link'
            self.image_pub.publish(img_msg)

        # Publish depth
        depth = self.camera.get_depth()
        if depth is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(depth, '32FC1')
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = 'camera_link'
            self.depth_pub.publish(depth_msg)

        # Publish LiDAR
        lidar_data = self.lidar.get_current_frame()
        if lidar_data is not None:
            scan_msg = self.create_laser_scan_msg(lidar_data, now)
            self.scan_pub.publish(scan_msg)

        # Publish IMU
        imu_data = self.imu.get_current_frame()
        if imu_data is not None:
            imu_msg = self.create_imu_msg(imu_data, now)
            self.imu_pub.publish(imu_msg)

    def create_laser_scan_msg(self, lidar_data, stamp):
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = 'lidar_link'
        msg.angle_min = -np.pi
        msg.angle_max = np.pi
        msg.angle_increment = np.radians(1.0)
        msg.range_min = 0.1
        msg.range_max = 100.0

        # Convert point cloud to ranges
        points = lidar_data['points']
        ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        msg.ranges = ranges.tolist()

        return msg

    def create_imu_msg(self, imu_data, stamp):
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'imu_link'

        msg.linear_acceleration.x = float(imu_data['lin_acc'][0])
        msg.linear_acceleration.y = float(imu_data['lin_acc'][1])
        msg.linear_acceleration.z = float(imu_data['lin_acc'][2])

        msg.angular_velocity.x = float(imu_data['ang_vel'][0])
        msg.angular_velocity.y = float(imu_data['ang_vel'][1])
        msg.angular_velocity.z = float(imu_data['ang_vel'][2])

        msg.orientation.x = float(imu_data['orientation'][0])
        msg.orientation.y = float(imu_data['orientation'][1])
        msg.orientation.z = float(imu_data['orientation'][2])
        msg.orientation.w = float(imu_data['orientation'][3])

        return msg

    def get_wheel_commands(self):
        """Convert cmd_vel to wheel velocities."""
        linear = self.cmd_vel.linear.x
        angular = self.cmd_vel.angular.z

        return self.controller.forward([linear, angular])

def main():
    # Initialize ROS 2
    rclpy.init()

    # Create world
    world = World()
    world.scene.add_default_ground_plane()

    # TODO: Add robot from URDF/USD

    # Create bridge
    bridge = IsaacROSBridge(world, "/World/Robot")

    world.reset()
    bridge.initialize_sensors()

    # Main loop
    while simulation_app.is_running():
        world.step(render=True)
        bridge.publish_sensors()
        rclpy.spin_once(bridge, timeout_sec=0)

    bridge.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == '__main__':
    main()
```

## Using ROS 2 Tools with Isaac Sim

### RViz2 Visualization

```bash
# In a separate terminal
ros2 run rviz2 rviz2

# Topics available from Isaac Sim:
# /camera/image_raw
# /camera/depth
# /scan
# /imu/data
# /odom
# /tf
```

### Teleop Control

```bash
# Keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Publishes to /cmd_vel which Isaac Sim subscribes to
```

### Recording Rosbags

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /camera/image_raw /scan /imu/data /odom
```

## Summary

In this lesson, you learned:

- Enabling the ROS 2 bridge extension in Isaac Sim
- Publishing sensor data (camera, LiDAR, IMU) to ROS 2
- Subscribing to velocity commands from ROS 2
- Creating a complete ROS 2 robot interface
- Using standard ROS 2 tools with Isaac Sim

## Next Steps

Continue to [Week 7 Exercises](/module-3/week-7/exercises) to practice Isaac Sim integration.
