---
sidebar_position: 1
title: Introduction to Gazebo
description: Getting started with Gazebo simulation for robotics
---

# Introduction to Gazebo Simulation

Gazebo is a powerful 3D robotics simulator that provides physics simulation, sensor modeling, and visualization for testing robot algorithms before deployment.

## Learning Objectives

By the end of this lesson, you will:

- Understand Gazebo architecture and components
- Install and configure Gazebo Harmonic
- Create and navigate simulation worlds
- Understand the relationship between Gazebo and ROS 2

## What is Gazebo?

Gazebo is an open-source 3D robotics simulator that offers:

- **Physics Simulation**: Realistic dynamics using DART, ODE, Bullet, or Simbody
- **Sensor Simulation**: Cameras, LiDAR, IMU, depth sensors, and more
- **Environment Modeling**: Complex indoor and outdoor environments
- **ROS 2 Integration**: Seamless communication with ROS 2 nodes
- **Plugin System**: Extensible architecture for custom functionality

### Gazebo vs. Gazebo Classic

| Feature | Gazebo (Harmonic) | Gazebo Classic |
|---------|-------------------|----------------|
| Physics Engine | DART (default), Bullet | ODE (default), Bullet, DART |
| Transport | Ignition Transport | Gazebo Transport |
| GUI Framework | Qt5 with Ogre2 | Qt5 with Ogre 1.x |
| ROS 2 Bridge | ros_gz | gazebo_ros_pkgs |
| Status | Active development | Maintenance mode |

## Installing Gazebo Harmonic

### Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required tools
sudo apt install -y wget lsb-release gnupg
```

### Installation

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update and install
sudo apt update
sudo apt install gz-harmonic -y
```

### Verify Installation

```bash
# Check Gazebo version
gz sim --version

# Launch empty world
gz sim
```

### ROS 2 Integration Packages

```bash
# Install ROS-Gazebo bridge
sudo apt install ros-humble-ros-gz -y

# Install additional tools
sudo apt install ros-humble-ros-gz-bridge \
                 ros-humble-ros-gz-image \
                 ros-humble-ros-gz-sim -y
```

## Gazebo Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Gazebo Simulator                      │
├──────────────┬──────────────┬──────────────┬────────────┤
│   Physics    │   Sensors    │   Rendering  │   GUI      │
│   Engine     │   System     │   Engine     │   Client   │
├──────────────┴──────────────┴──────────────┴────────────┤
│                   Transport Layer                        │
│              (Ignition Transport / Topics)               │
├─────────────────────────────────────────────────────────┤
│                   Plugin System                          │
│        (System, Visual, Sensor, Model Plugins)          │
└─────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **World**: The complete simulation environment
2. **Model**: A robot or object in the simulation
3. **Link**: A rigid body within a model
4. **Joint**: Connection between links
5. **Sensor**: Device that generates data (camera, LiDAR, IMU)
6. **Plugin**: Code that extends Gazebo functionality

## Launching Gazebo

### Basic Launch

```bash
# Launch with empty world
gz sim empty.sdf

# Launch with specific world
gz sim shapes.sdf

# Launch in verbose mode
gz sim -v 4 empty.sdf
```

### Common World Files

```bash
# List available worlds
ls /usr/share/gz/gz-sim*/worlds/

# Common worlds:
# - empty.sdf: Empty environment
# - shapes.sdf: Basic shapes for testing
# - tunnel.sdf: Underground tunnel
# - warehouse.sdf: Indoor warehouse
```

## Creating Your First World

### SDF Format Basics

Simulation Description Format (SDF) is the XML format used to describe Gazebo worlds:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="my_first_world">
    <!-- Physics settings -->
    <physics name="default_physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- A simple box -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.167</ixx>
            <iyy>0.167</iyy>
            <izz>0.167</izz>
          </inertia>
        </inertial>
      </link>
    </model>

  </world>
</sdf>
```

### Launching Custom World

```bash
# Save the above as my_world.sdf
gz sim my_world.sdf
```

## Gazebo GUI Overview

### Main Interface

```
┌─────────────────────────────────────────────────────────┐
│  Menu Bar: File | Edit | View | Window | Help           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    3D Viewport                          │
│                                                         │
│   [Navigate with mouse:                                 │
│    - Left click + drag: Rotate                          │
│    - Right click + drag: Zoom                           │
│    - Middle click + drag: Pan                           │
│    - Scroll wheel: Zoom]                                │
│                                                         │
├──────────────────────┬──────────────────────────────────┤
│   Entity Tree        │   Component Inspector            │
│   - World            │   - Name                         │
│     - Models         │   - Pose                         │
│     - Lights         │   - Physics                      │
│     - Sensors        │   - Properties                   │
└──────────────────────┴──────────────────────────────────┘
```

### Useful Plugins

Access via Window menu:

| Plugin | Description |
|--------|-------------|
| Entity Tree | View world hierarchy |
| Component Inspector | Edit entity properties |
| Transform Control | Move/rotate entities |
| Grid Config | Configure ground grid |
| World Control | Play/pause/step simulation |
| World Stats | View simulation statistics |
| View Angle | Quick camera positioning |

## Gazebo Topics and Services

### Viewing Topics

```bash
# List all Gazebo topics
gz topic -l

# Echo a topic
gz topic -e -t /world/default/stats

# Get topic info
gz topic -i -t /world/default/stats
```

### Common Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/world/*/stats` | gz.msgs.WorldStatistics | Simulation statistics |
| `/world/*/clock` | gz.msgs.Clock | Simulation time |
| `/world/*/pose/info` | gz.msgs.Pose_V | Entity poses |
| `/model/*/joint_state` | gz.msgs.Model | Joint states |

### Services

```bash
# List services
gz service -l

# Call a service
gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf: "..."'
```

## ROS 2 - Gazebo Bridge

### Bridge Configuration

```yaml
# bridge_config.yaml
- ros_topic_name: "/scan"
  gz_topic_name: "/world/default/model/robot/link/base_link/sensor/lidar/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS

- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/robot/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ
```

### Launching the Bridge

```bash
# Run bridge with config file
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge_config.yaml

# Or specify topics directly
ros2 run ros_gz_bridge parameter_bridge /scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan
```

### Common Bridge Mappings

| ROS 2 Type | Gazebo Type |
|------------|-------------|
| `std_msgs/msg/String` | `gz.msgs.StringMsg` |
| `geometry_msgs/msg/Twist` | `gz.msgs.Twist` |
| `sensor_msgs/msg/Image` | `gz.msgs.Image` |
| `sensor_msgs/msg/LaserScan` | `gz.msgs.LaserScan` |
| `sensor_msgs/msg/PointCloud2` | `gz.msgs.PointCloudPacked` |
| `nav_msgs/msg/Odometry` | `gz.msgs.Odometry` |

## Controlling the Simulation

### Play/Pause

```bash
# Pause simulation
gz service -s /world/default/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 1000 --req 'pause: true'

# Resume simulation
gz service -s /world/default/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 1000 --req 'pause: false'
```

### Spawning Models

```bash
# Spawn a model from SDF file
gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf_filename: "model.sdf", pose: {position: {x: 0, y: 0, z: 1}}'
```

### ROS 2 Spawn Service

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity

class SpawnService(Node):
    def __init__(self):
        super().__init__('spawn_service')
        self.cli = self.create_client(SpawnEntity, '/world/default/create')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')

    def spawn_model(self, name, sdf_string, pose):
        req = SpawnEntity.Request()
        req.name = name
        req.string = sdf_string
        req.initial_pose = pose

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

## Summary

In this lesson, you learned:

- Gazebo's architecture and key components
- How to install Gazebo Harmonic with ROS 2 integration
- Creating basic simulation worlds using SDF
- Navigating the Gazebo GUI
- Understanding Gazebo topics, services, and the ROS 2 bridge

## Next Steps

Continue to [World Building](/module-2/week-4/world-building) to learn how to create complex simulation environments.
