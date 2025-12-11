---
sidebar_position: 1
title: Sensor Integration
description: Integrating cameras, LiDAR, and depth sensors with ROS 2
---

# Sensor Integration in ROS 2

This lesson covers integrating various sensors commonly used in humanoid robotics, including cameras, LiDAR, and depth sensors.

## Learning Objectives

By the end of this lesson, you will:

- Understand ROS 2 sensor message types
- Configure and launch camera drivers
- Integrate LiDAR sensors for environment mapping
- Work with depth cameras for 3D perception

## Common Sensor Types in Robotics

### Camera Sensors

Cameras provide visual information essential for:
- Object detection and recognition
- Visual odometry
- Human-robot interaction

### LiDAR Sensors

Light Detection and Ranging (LiDAR) provides:
- Precise distance measurements
- 360-degree environment scanning
- Point cloud data for mapping

### Depth Cameras

Depth cameras (RGB-D) combine:
- Color imagery (RGB)
- Depth information (D)
- Popular models: Intel RealSense, Azure Kinect

## ROS 2 Sensor Message Types

### Image Messages

```python
# sensor_msgs/msg/Image
from sensor_msgs.msg import Image

# Key fields:
# - header: std_msgs/Header (timestamp, frame_id)
# - height: uint32 (image height in pixels)
# - width: uint32 (image width in pixels)
# - encoding: string (pixel encoding, e.g., 'rgb8', 'bgr8', 'mono8')
# - data: uint8[] (actual image data)
```

### Point Cloud Messages

```python
# sensor_msgs/msg/PointCloud2
from sensor_msgs.msg import PointCloud2

# Key fields:
# - header: std_msgs/Header
# - height: uint32
# - width: uint32
# - fields: PointField[] (describes the channels)
# - data: uint8[] (point cloud data)
```

### LaserScan Messages

```python
# sensor_msgs/msg/LaserScan
from sensor_msgs.msg import LaserScan

# Key fields:
# - header: std_msgs/Header
# - angle_min: float32 (start angle of scan)
# - angle_max: float32 (end angle of scan)
# - angle_increment: float32 (angular distance between measurements)
# - ranges: float32[] (range data)
# - intensities: float32[] (intensity data, optional)
```

## Setting Up a USB Camera

### Installing the Camera Driver

```bash
# Install v4l2 camera driver for ROS 2
sudo apt install ros-humble-v4l2-camera

# Install image transport plugins
sudo apt install ros-humble-image-transport-plugins
```

### Launching the Camera

```bash
# Start the v4l2 camera node
ros2 run v4l2_camera v4l2_camera_node

# With parameters
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=/dev/video0 \
  -p image_size:=[640,480] \
  -p camera_frame_id:=camera_link
```

### Camera Launch File

```python
# camera_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera',
            parameters=[{
                'video_device': '/dev/video0',
                'image_size': [640, 480],
                'camera_frame_id': 'camera_link',
                'pixel_format': 'YUYV',
            }],
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ],
        ),
    ])
```

## Intel RealSense Integration

### Installing RealSense ROS 2 Wrapper

```bash
# Install RealSense SDK
sudo apt install ros-humble-realsense2-camera
sudo apt install ros-humble-realsense2-description
```

### Launching RealSense Camera

```bash
# Basic launch
ros2 launch realsense2_camera rs_launch.py

# With depth and color streams
ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  enable_depth:=true \
  enable_color:=true \
  pointcloud.enable:=true
```

### RealSense Launch Configuration

```python
# realsense_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'enable_depth': 'true',
            'enable_color': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
        }.items(),
    )

    return LaunchDescription([
        realsense_launch,
    ])
```

## LiDAR Integration

### Common LiDAR Drivers

| LiDAR Model | ROS 2 Package | Install Command |
|-------------|---------------|-----------------|
| RPLidar A1/A2/A3 | rplidar_ros | `sudo apt install ros-humble-rplidar-ros` |
| Velodyne | velodyne | `sudo apt install ros-humble-velodyne` |
| SICK LMS | sick_scan2 | Build from source |
| Ouster | ros2_ouster | Build from source |

### RPLidar Setup

```bash
# Install RPLidar driver
sudo apt install ros-humble-rplidar-ros

# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER

# Launch RPLidar
ros2 launch rplidar_ros rplidar_a2m8_launch.py
```

### RPLidar Launch File

```python
# rplidar_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'laser_frame',
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
            output='screen',
        ),
    ])
```

## Creating a Sensor Subscriber Node

### Image Subscriber

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Image subscriber initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Process the image
            self.get_logger().info(
                f'Received image: {cv_image.shape[1]}x{cv_image.shape[0]}'
            )

            # Display the image
            cv2.imshow('Camera Feed', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Point Cloud Subscriber

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        self.get_logger().info('Point cloud subscriber initialized')

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = []
        for point in pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True):
            points.append([point[0], point[1], point[2]])

        points_array = np.array(points)

        self.get_logger().info(
            f'Received {len(points_array)} points, '
            f'X range: [{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}]'
        )

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion Basics

### Combining Camera and LiDAR Data

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Create synchronized subscribers
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.scan_sub = Subscriber(self, LaserScan, '/scan')

        # Synchronize messages with 0.1 second tolerance
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        self.get_logger().info('Sensor fusion node initialized')

    def sync_callback(self, image_msg, scan_msg):
        # Process synchronized sensor data
        image_time = image_msg.header.stamp
        scan_time = scan_msg.header.stamp

        self.get_logger().info(
            f'Synchronized data received - '
            f'Image: {image_time.sec}.{image_time.nanosec}, '
            f'Scan: {scan_time.sec}.{scan_time.nanosec}'
        )

        # Perform fusion processing here
        # e.g., project LiDAR points onto image

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Visualizing Sensor Data in RViz2

### RViz2 Configuration for Sensors

```yaml
# sensor_rviz_config.yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Image
      Name: Camera Image
      Topic:
        Value: /camera/image_raw
      Enabled: true

    - Class: rviz_default_plugins/LaserScan
      Name: LiDAR Scan
      Topic:
        Value: /scan
      Size (m): 0.05
      Color Transformer: Intensity
      Enabled: true

    - Class: rviz_default_plugins/PointCloud2
      Name: Depth Point Cloud
      Topic:
        Value: /camera/depth/color/points
      Size (m): 0.01
      Color Transformer: RGB8
      Enabled: true

    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true

  Global Options:
    Fixed Frame: base_link
```

### Launching RViz2 with Configuration

```bash
# Launch RViz2 with sensor configuration
ros2 run rviz2 rviz2 -d sensor_rviz_config.yaml
```

## Sensor Calibration

### Camera Intrinsic Calibration

```bash
# Install camera calibration package
sudo apt install ros-humble-camera-calibration

# Run calibration with checkerboard
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.025 \
  --ros-args -r image:=/camera/image_raw -r camera:=/camera
```

### Camera-LiDAR Extrinsic Calibration

The extrinsic calibration determines the transformation between camera and LiDAR coordinate frames:

1. **Collect calibration data**: Capture synchronized images and point clouds
2. **Identify correspondences**: Match features between image and point cloud
3. **Compute transformation**: Use PnP or other algorithms
4. **Validate**: Project LiDAR points onto image to verify alignment

## Summary

In this lesson, you learned:

- Different sensor types used in robotics and their ROS 2 message types
- How to integrate USB cameras, RealSense, and LiDAR sensors
- Creating subscriber nodes for image and point cloud data
- Basics of sensor fusion and time synchronization
- Visualizing sensor data in RViz2
- Camera calibration procedures

## Next Steps

Continue to [Image Processing](/module-1/week-3/image-processing) to learn about processing camera images with OpenCV.
