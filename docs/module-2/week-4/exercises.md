---
sidebar_position: 4
title: Week 4 Exercises
description: Hands-on exercises for Gazebo simulation basics
---

# Week 4: Gazebo Simulation Exercises

Practice creating simulation environments and configuring sensors.

## Exercise 1: Custom World Creation

**Objective**: Create a simulation world for testing mobile robot navigation.

### Requirements

1. Create a 15m x 15m indoor environment
2. Add at least 4 walls with one doorway
3. Include 5-10 static obstacles (boxes, cylinders)
4. Add appropriate lighting
5. Configure physics for real-time simulation

### Starter Template

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="navigation_test">

    <!-- TODO: Add physics configuration -->

    <!-- TODO: Add required plugins -->

    <!-- TODO: Add lighting -->

    <!-- TODO: Add ground plane -->

    <!-- TODO: Add walls -->

    <!-- TODO: Add obstacles -->

  </world>
</sdf>
```

### Verification

```bash
# Launch your world
gz sim my_navigation_world.sdf

# Check it runs in real-time
# Observe physics stats in GUI
```

<details className="solution-block">
<summary>Solution</summary>
<div className="solution-content">

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="navigation_test">

    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>

    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <light name="ceiling_1" type="point">
      <pose>-3 -3 2.8 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <attenuation>
        <range>20</range>
        <constant>0.3</constant>
        <linear>0.01</linear>
      </attenuation>
    </light>

    <light name="ceiling_2" type="point">
      <pose>3 3 2.8 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <attenuation>
        <range>20</range>
        <constant>0.3</constant>
        <linear>0.01</linear>
      </attenuation>
    </light>

    <!-- Floor -->
    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>15 15 0.1</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>15 15 0.1</size></box></geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- North Wall -->
    <model name="north_wall">
      <static>true</static>
      <pose>0 7.5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>15 0.2 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>15 0.2 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- South Wall -->
    <model name="south_wall">
      <static>true</static>
      <pose>0 -7.5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>15 0.2 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>15 0.2 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- East Wall (with doorway) -->
    <model name="east_wall_1">
      <static>true</static>
      <pose>7.5 4 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.2 7 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 7 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <model name="east_wall_2">
      <static>true</static>
      <pose>7.5 -4 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.2 7 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 7 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- West Wall -->
    <model name="west_wall">
      <static>true</static>
      <pose>-7.5 0 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.2 15 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 15 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="box_1">
      <static>true</static>
      <pose>-3 2 0.5 0 0 0.3</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>1 1 1</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>1 1 1</size></box></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="box_2">
      <static>true</static>
      <pose>2 -3 0.75 0 0 -0.2</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>1.5 0.8 1.5</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>1.5 0.8 1.5</size></box></geometry>
          <material><ambient>0.2 0.8 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="cylinder_1">
      <static>true</static>
      <pose>-2 -4 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.4</radius><length>1</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.4</radius><length>1</length></cylinder>
          </geometry>
          <material><ambient>0.2 0.2 0.8 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="cylinder_2">
      <static>true</static>
      <pose>4 4 0.6 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.5</radius><length>1.2</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.5</radius><length>1.2</length></cylinder>
          </geometry>
          <material><ambient>0.8 0.8 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="box_3">
      <static>true</static>
      <pose>5 -2 0.4 0 0 0.5</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.8 0.8 0.8</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.8 0.8 0.8</size></box></geometry>
          <material><ambient>0.6 0.3 0.8 1</ambient></material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

</div>
</details>

---

## Exercise 2: Robot Model with Sensors

**Objective**: Create a simple robot model with camera and LiDAR sensors.

### Requirements

1. Create a box-shaped robot base (0.4m x 0.3m x 0.1m)
2. Add a camera sensor facing forward
3. Add a 2D LiDAR sensor on top
4. Include an IMU sensor
5. Configure appropriate sensor parameters

### Starter Template

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="sensor_robot">
    <pose>0 0 0.1 0 0 0</pose>

    <!-- Base Link -->
    <link name="base_link">
      <!-- TODO: Add inertial -->
      <!-- TODO: Add collision -->
      <!-- TODO: Add visual -->
      <!-- TODO: Add IMU sensor -->
    </link>

    <!-- Camera Link -->
    <link name="camera_link">
      <!-- TODO: Define camera mount -->
      <!-- TODO: Add camera sensor -->
    </link>

    <!-- TODO: Add camera joint -->

    <!-- LiDAR Link -->
    <link name="lidar_link">
      <!-- TODO: Define LiDAR mount -->
      <!-- TODO: Add LiDAR sensor -->
    </link>

    <!-- TODO: Add LiDAR joint -->

  </model>
</sdf>
```

### Verification

```bash
# Spawn robot in Gazebo
gz sim -r empty.sdf

# In another terminal, spawn the model
gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf_filename: "sensor_robot.sdf"'

# Check sensor topics
gz topic -l | grep -E "camera|scan|imu"
```

---

## Exercise 3: ROS 2 Bridge Configuration

**Objective**: Set up a complete ROS 2 bridge for sensor data.

### Requirements

1. Bridge camera image to `/camera/image_raw`
2. Bridge depth image to `/camera/depth`
3. Bridge LiDAR scan to `/scan`
4. Bridge IMU data to `/imu/data`
5. Create a launch file that starts Gazebo and the bridge

### Starter Code

```python
# sensor_bridge_launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # TODO: Start Gazebo with your world

    # TODO: Configure and start the bridge

    # TODO: Start RViz for visualization

    return LaunchDescription([
        # Add your launch actions here
    ])
```

### Verification

```bash
# Run the launch file
ros2 launch my_robot_sim sensor_bridge_launch.py

# Check ROS 2 topics
ros2 topic list

# Echo sensor data
ros2 topic echo /camera/image_raw --no-arr
ros2 topic echo /scan
ros2 topic echo /imu/data
```

<details className="hint-block">
<summary>Hint: Bridge Arguments</summary>
<div className="hint-content">

```python
bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/camera/depth@sensor_msgs/msg/Image@gz.msgs.Image',
        '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
    ],
    output='screen'
)
```

</div>
</details>

---

## Exercise 4: Dynamic Object Interaction

**Objective**: Create a world with dynamic objects that the robot can interact with.

### Requirements

1. Create a flat platform
2. Add 5 dynamic cubes that can be pushed
3. Add proper inertial properties so physics works correctly
4. Include friction settings for realistic interaction
5. Test that objects respond to collisions

### Physics Considerations

- Mass should be realistic (1-5 kg for small objects)
- Inertia must be calculated correctly for stable simulation
- Friction coefficients affect how objects slide

### Inertia Calculation Reference

For a box with dimensions (w, h, d) and mass m:
- Ixx = (1/12) * m * (h^2 + d^2)
- Iyy = (1/12) * m * (w^2 + d^2)
- Izz = (1/12) * m * (w^2 + h^2)

For a cylinder with radius r, length l, and mass m:
- Ixx = Iyy = (1/12) * m * (3*r^2 + l^2)
- Izz = (1/2) * m * r^2

---

## Exercise 5: Multi-Robot Environment

**Objective**: Create a world with multiple robot instances.

### Requirements

1. Create a simple differential drive robot model
2. Spawn 3 instances in different locations
3. Ensure each robot has unique topic namespaces
4. Set up bridges for all robots

### Namespacing

```xml
<!-- In your model SDF -->
<model name="robot_1">
  <!-- Use topic remapping in the bridge -->
</model>
```

```python
# In your bridge configuration
bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    namespace='robot_1',
    arguments=[
        '/robot_1/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        # ... more topics
    ],
)
```

---

## Challenge: Complete Simulation Setup

**Objective**: Create a complete simulation environment for robot navigation testing.

### Requirements

1. Warehouse-style world (minimum 20m x 20m)
2. Multiple rooms connected by doorways
3. Various obstacles (static and dynamic)
4. A robot with:
   - RGB camera
   - Depth camera
   - 2D LiDAR
   - IMU
5. ROS 2 bridge for all sensors
6. RViz configuration file for visualization

### Deliverables

1. `warehouse_world.sdf` - World file
2. `sensor_robot.sdf` - Robot model
3. `simulation_launch.py` - Launch file
4. `sensors.rviz` - RViz configuration
5. `bridge_config.yaml` - Bridge configuration

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| World runs at real-time factor | 20 |
| All sensors produce data | 25 |
| Bridge correctly configured | 20 |
| RViz shows all sensor data | 15 |
| Code organization and documentation | 10 |
| Creative world design | 10 |

---

## Submission Checklist

Before submitting, verify:

- [ ] All SDF files are valid (use `gz sdf -k <file>`)
- [ ] Simulation runs without errors
- [ ] Physics operates at real-time or faster
- [ ] All sensors produce data on correct topics
- [ ] ROS 2 bridge correctly forwards all data
- [ ] Launch files work on clean ROS 2 environment
- [ ] Code is commented and organized

## Resources

- [Gazebo SDF Specification](http://sdformat.org/spec)
- [ROS-Gazebo Bridge Documentation](https://gazebosim.org/docs/harmonic/ros2_integration)
- [Gazebo Fuel Models](https://app.gazebosim.org/fuel)
