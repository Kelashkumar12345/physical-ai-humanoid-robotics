---
sidebar_position: 3
title: Sensor Simulation
description: Simulating cameras, LiDAR, and IMU in Gazebo
---

# Sensor Simulation in Gazebo

Learn to add and configure simulated sensors for realistic robot testing.

## Learning Objectives

By the end of this lesson, you will:

- Add camera sensors to robot models
- Configure LiDAR and depth sensors
- Set up IMU and other proprioceptive sensors
- Bridge sensor data to ROS 2

## Gazebo Sensor System

### Required Plugins

```xml
<!-- Add to world SDF -->
<plugin filename="gz-sim-sensors-system"
        name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>

<plugin filename="gz-sim-imu-system"
        name="gz::sim::systems::Imu">
</plugin>

<plugin filename="gz-sim-contact-system"
        name="gz::sim::systems::Contact">
</plugin>
```

## Camera Sensors

### Basic RGB Camera

```xml
<model name="camera_bot">
  <pose>0 0 0.5 0 0 0</pose>
  <link name="base_link">
    <!-- Base link properties -->
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
      </inertia>
    </inertial>
    <collision name="collision">
      <geometry><box><size>0.2 0.2 0.1</size></box></geometry>
    </collision>
    <visual name="visual">
      <geometry><box><size>0.2 0.2 0.1</size></box></geometry>
    </visual>

    <!-- Camera Sensor -->
    <sensor name="camera" type="camera">
      <pose>0.1 0 0 0 0 0</pose>
      <always_on>true</always_on>
      <update_rate>30</update_rate>

      <camera name="main_camera">
        <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>

      <topic>camera/image</topic>
    </sensor>
  </link>
</model>
```

### Depth Camera (RGB-D)

```xml
<sensor name="rgbd_camera" type="rgbd_camera">
  <pose>0.1 0 0.05 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>30</update_rate>

  <camera name="depth_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <depth_camera>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </depth_camera>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.005</stddev>
    </noise>
  </camera>

  <topic>depth_camera</topic>
</sensor>
```

### Stereo Camera

```xml
<!-- Left Camera -->
<sensor name="left_camera" type="camera">
  <pose>0.1 0.06 0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="left">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <topic>stereo/left/image</topic>
</sensor>

<!-- Right Camera -->
<sensor name="right_camera" type="camera">
  <pose>0.1 -0.06 0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="right">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <topic>stereo/right/image</topic>
</sensor>
```

## LiDAR Sensors

### 2D LiDAR (Laser Scanner)

```xml
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.1 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>

  <lidar>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>0</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </lidar>

  <topic>scan</topic>
</sensor>
```

### 3D LiDAR (Velodyne-style)

```xml
<sensor name="lidar_3d" type="gpu_lidar">
  <pose>0 0 0.3 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>

  <lidar>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- +15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.5</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.02</stddev>
    </noise>
  </lidar>

  <topic>points</topic>
</sensor>
```

## IMU Sensor

### IMU Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <pose>0 0 0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>

  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </z>
    </angular_velocity>

    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>

  <topic>imu</topic>
</sensor>
```

## Other Sensors

### Contact Sensor

```xml
<sensor name="contact_sensor" type="contact">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <contact>
    <collision>base_link_collision</collision>
    <topic>contacts</topic>
  </contact>
</sensor>
```

### GPS Sensor

```xml
<sensor name="gps_sensor" type="navsat">
  <pose>0 0 0.1 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>10</update_rate>

  <navsat>
    <position_sensing>
      <horizontal>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.5</stddev>
        </noise>
      </horizontal>
      <vertical>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.0</stddev>
        </noise>
      </vertical>
    </position_sensing>
    <velocity_sensing>
      <horizontal>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise>
      </horizontal>
      <vertical>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise>
      </vertical>
    </velocity_sensing>
  </navsat>

  <topic>gps</topic>
</sensor>
```

### Altimeter

```xml
<sensor name="altimeter" type="altimeter">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <altimeter>
    <vertical_position>
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.001</stddev>
      </noise>
    </vertical_position>
    <vertical_velocity>
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </vertical_velocity>
  </altimeter>
  <topic>altimeter</topic>
</sensor>
```

### Force/Torque Sensor

```xml
<!-- Add to a joint -->
<joint name="sensor_joint" type="fixed">
  <parent>base_link</parent>
  <child>sensor_link</child>
  <sensor name="force_torque" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>sensor</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <topic>force_torque</topic>
  </sensor>
</joint>
```

## Complete Robot with Sensors

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="sensor_robot">
    <pose>0 0 0.1 0 0 0</pose>

    <!-- Base Link -->
    <link name="base_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.5</ixx><iyy>0.5</iyy><izz>0.5</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.4 0.3 0.1</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.4 0.3 0.1</size></box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.8 1</ambient>
          <diffuse>0.2 0.2 0.8 1</diffuse>
        </material>
      </visual>

      <!-- IMU -->
      <sensor name="imu" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <topic>imu</topic>
      </sensor>
    </link>

    <!-- Camera Mount -->
    <link name="camera_link">
      <pose relative_to="base_link">0.15 0 0.1 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.05 0.05 0.05</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.05 0.05 0.05</size></box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
        </material>
      </visual>

      <!-- RGB Camera -->
      <sensor name="camera" type="camera">
        <pose>0.025 0 0 0 0 0</pose>
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <camera name="main_camera">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <topic>camera/image</topic>
      </sensor>

      <!-- Depth Camera -->
      <sensor name="depth_camera" type="depth_camera">
        <pose>0.025 0 0 0 0 0</pose>
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <camera name="depth">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R_FLOAT32</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10.0</far>
          </clip>
        </camera>
        <topic>camera/depth</topic>
      </sensor>
    </link>

    <joint name="camera_joint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
    </joint>

    <!-- LiDAR Mount -->
    <link name="lidar_link">
      <pose relative_to="base_link">0 0 0.15 0 0 0</pose>
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.04</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.3 0.3 0.3 1</ambient>
          <diffuse>0.3 0.3 0.3 1</diffuse>
        </material>
      </visual>

      <!-- 2D LiDAR -->
      <sensor name="lidar" type="gpu_lidar">
        <pose>0 0 0.025 0 0 0</pose>
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
        </lidar>
        <topic>scan</topic>
      </sensor>
    </link>

    <joint name="lidar_joint" type="fixed">
      <parent>base_link</parent>
      <child>lidar_link</child>
    </joint>

  </model>
</sdf>
```

## Bridging Sensors to ROS 2

### Bridge Configuration

```yaml
# sensor_bridge.yaml
- ros_topic_name: "/camera/image_raw"
  gz_topic_name: "/camera/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/depth"
  gz_topic_name: "/camera/depth"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/scan"
  gz_topic_name: "/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS

- ros_topic_name: "/imu/data"
  gz_topic_name: "/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/camera_info"
  gz_topic_name: "/camera/camera_info"
  ros_type_name: "sensor_msgs/msg/CameraInfo"
  gz_type_name: "gz.msgs.CameraInfo"
  direction: GZ_TO_ROS
```

### Launch File

```python
# sensor_sim_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_robot_sim')

    # Start Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r',
             os.path.join(pkg_dir, 'worlds', 'sensor_world.sdf')],
        output='screen'
    )

    # Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
        ],
        output='screen'
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'sensors.rviz')],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        bridge,
        rviz,
    ])
```

## Sensor Noise Models

### Gaussian Noise

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.01</stddev>
</noise>
```

### Custom Noise Parameters

| Sensor | Typical Noise (stddev) |
|--------|------------------------|
| Camera | 0.005 - 0.01 |
| LiDAR (range) | 0.01 - 0.03 m |
| IMU (angular velocity) | 0.0002 rad/s |
| IMU (linear acceleration) | 0.017 m/s^2 |
| GPS (horizontal) | 0.5 - 2.0 m |
| Depth camera | 0.001 - 0.01 m |

## Summary

In this lesson, you learned:

- How to add various sensors to robot models
- Configuring camera, LiDAR, IMU, and other sensors
- Setting up realistic sensor noise
- Bridging sensor data to ROS 2

## Next Steps

Continue to [Week 4 Exercises](/module-2/week-4/exercises) to practice creating simulated robots with sensors.
