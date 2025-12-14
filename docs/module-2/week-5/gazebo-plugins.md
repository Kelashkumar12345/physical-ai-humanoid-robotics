---
sidebar_position: 2
title: Gazebo Plugins
description: Adding functionality with Gazebo system plugins
---

# Gazebo Plugins

Learn to use and configure Gazebo plugins to add realistic robot behaviors.

## Learning Objectives

By the end of this lesson, you will:

- Understand the Gazebo plugin architecture
- Configure common robot plugins (diff drive, joint control)
- Add sensor plugins for data generation
- Create custom plugin configurations

## Plugin Architecture

### Plugin Types

| Type | Level | Purpose |
|------|-------|---------|
| System | World | Physics, sensors, scene broadcasting |
| Model | Model | Robot controllers, actuators |
| Sensor | Sensor | Data processing, noise |
| Visual | Visual | Rendering effects |

### Plugin Loading

```xml
<!-- World-level plugin -->
<world name="my_world">
  <plugin filename="gz-sim-physics-system"
          name="gz::sim::systems::Physics">
  </plugin>
</world>

<!-- Model-level plugin -->
<model name="my_robot">
  <plugin filename="gz-sim-diff-drive-system"
          name="gz::sim::systems::DiffDrive">
    <!-- Configuration -->
  </plugin>
</model>

<!-- Sensor-level plugin -->
<sensor name="camera" type="camera">
  <plugin filename="gz-sim-camera-system"
          name="gz::sim::systems::Camera">
  </plugin>
</sensor>
```

## Essential World Plugins

### Physics System

```xml
<plugin filename="gz-sim-physics-system"
        name="gz::sim::systems::Physics">
</plugin>
```

### User Commands (Spawn/Delete)

```xml
<plugin filename="gz-sim-user-commands-system"
        name="gz::sim::systems::UserCommands">
</plugin>
```

### Scene Broadcaster (Visualization)

```xml
<plugin filename="gz-sim-scene-broadcaster-system"
        name="gz::sim::systems::SceneBroadcaster">
</plugin>
```

### Contact Detection

```xml
<plugin filename="gz-sim-contact-system"
        name="gz::sim::systems::Contact">
</plugin>
```

### Sensor System

```xml
<plugin filename="gz-sim-sensors-system"
        name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>
```

## Differential Drive Plugin

### Configuration

```xml
<model name="diff_drive_robot">
  <!-- Robot links and joints -->

  <plugin filename="gz-sim-diff-drive-system"
          name="gz::sim::systems::DiffDrive">
    <!-- Joint names -->
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>

    <!-- Wheel geometry -->
    <wheel_separation>0.32</wheel_separation>
    <wheel_radius>0.05</wheel_radius>

    <!-- Control limits -->
    <max_linear_acceleration>1.0</max_linear_acceleration>
    <max_angular_acceleration>2.0</max_angular_acceleration>

    <!-- Topics -->
    <topic>cmd_vel</topic>
    <odom_topic>odom</odom_topic>

    <!-- TF frames -->
    <frame_id>odom</frame_id>
    <child_frame_id>base_link</child_frame_id>

    <!-- Publishing rate -->
    <odom_publish_frequency>50</odom_publish_frequency>
  </plugin>
</model>
```

### ROS 2 Bridge

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry
```

## Ackermann Steering Plugin

### Configuration

```xml
<plugin filename="gz-sim-ackermann-steering-system"
        name="gz::sim::systems::AckermannSteering">
  <!-- Steering joints -->
  <left_steering_joint>left_steering_joint</left_steering_joint>
  <right_steering_joint>right_steering_joint</right_steering_joint>

  <!-- Drive joints -->
  <left_joint>rear_left_wheel_joint</left_joint>
  <right_joint>rear_right_wheel_joint</right_joint>

  <!-- Geometry -->
  <wheel_separation>0.5</wheel_separation>
  <wheel_base>0.8</wheel_base>
  <wheel_radius>0.1</wheel_radius>

  <!-- Steering limits -->
  <steering_limit>0.5</steering_limit>

  <!-- Topics -->
  <topic>cmd_vel</topic>
  <odom_topic>odom</odom_topic>
</plugin>
```

## Joint State Publisher

### Configuration

```xml
<plugin filename="gz-sim-joint-state-publisher-system"
        name="gz::sim::systems::JointStatePublisher">
  <topic>joint_states</topic>
  <joint_name>joint_1</joint_name>
  <joint_name>joint_2</joint_name>
  <joint_name>joint_3</joint_name>
</plugin>
```

### ROS 2 Bridge

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /joint_states@sensor_msgs/msg/JointState@gz.msgs.Model
```

## Joint Position Controller

### Configuration

```xml
<plugin filename="gz-sim-joint-position-controller-system"
        name="gz::sim::systems::JointPositionController">
  <joint_name>arm_joint_1</joint_name>
  <topic>arm_joint_1/cmd_pos</topic>
  <p_gain>10.0</p_gain>
  <i_gain>0.1</i_gain>
  <d_gain>1.0</d_gain>
  <i_max>1.0</i_max>
  <i_min>-1.0</i_min>
  <cmd_max>100.0</cmd_max>
  <cmd_min>-100.0</cmd_min>
</plugin>
```

### Multiple Joints

```xml
<!-- One plugin per joint -->
<plugin filename="gz-sim-joint-position-controller-system"
        name="gz::sim::systems::JointPositionController">
  <joint_name>joint_1</joint_name>
  <topic>joint_1/cmd_pos</topic>
  <p_gain>50.0</p_gain>
  <i_gain>0.5</i_gain>
  <d_gain>5.0</d_gain>
</plugin>

<plugin filename="gz-sim-joint-position-controller-system"
        name="gz::sim::systems::JointPositionController">
  <joint_name>joint_2</joint_name>
  <topic>joint_2/cmd_pos</topic>
  <p_gain>30.0</p_gain>
  <i_gain>0.3</i_gain>
  <d_gain>3.0</d_gain>
</plugin>
```

## Joint Trajectory Controller

### Configuration

```xml
<plugin filename="gz-sim-joint-trajectory-controller-system"
        name="gz::sim::systems::JointTrajectoryController">
  <topic>arm_controller/joint_trajectory</topic>

  <joint_name>joint_1</joint_name>
  <initial_position>0.0</initial_position>
  <position_p_gain>100.0</position_p_gain>
  <position_i_gain>0.1</position_i_gain>
  <position_d_gain>10.0</position_d_gain>

  <joint_name>joint_2</joint_name>
  <initial_position>0.0</initial_position>
  <position_p_gain>100.0</position_p_gain>
  <position_i_gain>0.1</position_i_gain>
  <position_d_gain>10.0</position_d_gain>
</plugin>
```

## IMU Plugin

### Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>imu</topic>

  <imu>
    <angular_velocity>
      <x><noise type="gaussian"><mean>0</mean><stddev>0.0002</stddev></noise></x>
      <y><noise type="gaussian"><mean>0</mean><stddev>0.0002</stddev></noise></y>
      <z><noise type="gaussian"><mean>0</mean><stddev>0.0002</stddev></noise></z>
    </angular_velocity>
    <linear_acceleration>
      <x><noise type="gaussian"><mean>0</mean><stddev>0.017</stddev></noise></x>
      <y><noise type="gaussian"><mean>0</mean><stddev>0.017</stddev></noise></y>
      <z><noise type="gaussian"><mean>0</mean><stddev>0.017</stddev></noise></z>
    </linear_acceleration>
  </imu>
</sensor>

<!-- World plugin required -->
<plugin filename="gz-sim-imu-system"
        name="gz::sim::systems::Imu">
</plugin>
```

## Gripper Plugin

### Detach Plugin for Pick and Place

```xml
<plugin filename="gz-sim-detachable-joint-system"
        name="gz::sim::systems::DetachableJoint">
  <parent_link>gripper_link</parent_link>
  <child_model>target_object</child_model>
  <child_link>object_link</child_link>
  <topic>gripper/attach</topic>
  <suppress_child_weight>true</suppress_child_weight>
</plugin>
```

### Touch Plugin

```xml
<plugin filename="gz-sim-touchplugin-system"
        name="gz::sim::systems::TouchPlugin">
  <target>target_object</target>
  <namespace>gripper</namespace>
  <time>0.5</time>
  <enabled>true</enabled>
</plugin>
```

## Apply Force/Torque

```xml
<plugin filename="gz-sim-apply-joint-force-system"
        name="gz::sim::systems::ApplyJointForce">
  <joint_name>propeller_joint</joint_name>
</plugin>
```

## Lift Drag Plugin (Aerial Vehicles)

```xml
<plugin filename="gz-sim-lift-drag-system"
        name="gz::sim::systems::LiftDrag">
  <link_name>wing_link</link_name>
  <air_density>1.2041</air_density>
  <area>0.5</area>
  <a0>0.05984</a0>
  <cla>4.752798</cla>
  <cda>0.6417112</cda>
  <cp>0.0 0 0</cp>
  <forward>1 0 0</forward>
  <upward>0 0 1</upward>
</plugin>
```

## Complete Robot Example with Plugins

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="mobile_manipulator">
    <pose>0 0 0.1 0 0 0</pose>

    <!-- Base Link -->
    <link name="base_link">
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.5</ixx><iyy>0.5</iyy><izz>0.5</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>0.5 0.4 0.15</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.5 0.4 0.15</size></box></geometry>
        <material><ambient>0.2 0.2 0.8 1</ambient></material>
      </visual>

      <!-- IMU Sensor -->
      <sensor name="imu" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <topic>imu</topic>
      </sensor>
    </link>

    <!-- Left Wheel -->
    <link name="left_wheel">
      <pose relative_to="base_link">0 0.22 -0.05 -1.5708 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.05</radius><length>0.04</length></cylinder></geometry>
        <surface>
          <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.05</radius><length>0.04</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient></material>
      </visual>
    </link>

    <joint name="left_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>

    <!-- Right Wheel -->
    <link name="right_wheel">
      <pose relative_to="base_link">0 -0.22 -0.05 -1.5708 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.05</radius><length>0.04</length></cylinder></geometry>
        <surface>
          <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.05</radius><length>0.04</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient></material>
      </visual>
    </link>

    <joint name="right_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>right_wheel</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>

    <!-- Caster -->
    <link name="caster">
      <pose relative_to="base_link">-0.2 0 -0.075 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><sphere><radius>0.025</radius></sphere></geometry>
        <surface>
          <friction><ode><mu>0.01</mu><mu2>0.01</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><sphere><radius>0.025</radius></sphere></geometry>
      </visual>
    </link>

    <joint name="caster_joint" type="fixed">
      <parent>base_link</parent>
      <child>caster</child>
    </joint>

    <!-- Arm Base -->
    <link name="arm_base">
      <pose relative_to="base_link">0.15 0 0.1 0 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.04</radius><length>0.05</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.04</radius><length>0.05</length></cylinder></geometry>
        <material><ambient>0.3 0.3 0.3 1</ambient></material>
      </visual>
    </link>

    <joint name="arm_base_joint" type="fixed">
      <parent>base_link</parent>
      <child>arm_base</child>
    </joint>

    <!-- Arm Link 1 -->
    <link name="arm_link_1">
      <pose relative_to="arm_base">0 0 0.075 0 0 0</pose>
      <inertial>
        <origin xyz="0 0 0.1"/>
        <mass>0.3</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <origin xyz="0 0 0.1"/>
        <geometry><box><size>0.04 0.04 0.2</size></box></geometry>
      </collision>
      <visual name="visual">
        <origin xyz="0 0 0.1"/>
        <geometry><box><size>0.04 0.04 0.2</size></box></geometry>
        <material><ambient>0.8 0.2 0.2 1</ambient></material>
      </visual>
    </link>

    <joint name="arm_joint_1" type="revolute">
      <parent>arm_base</parent>
      <child>arm_link_1</child>
      <axis><xyz>0 0 1</xyz></axis>
      <limit><lower>-3.14</lower><upper>3.14</upper><effort>50</effort></limit>
    </joint>

    <!-- Arm Link 2 -->
    <link name="arm_link_2">
      <pose relative_to="arm_link_1">0 0 0.2 0 0 0</pose>
      <inertial>
        <origin xyz="0 0 0.075"/>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <origin xyz="0 0 0.075"/>
        <geometry><box><size>0.03 0.03 0.15</size></box></geometry>
      </collision>
      <visual name="visual">
        <origin xyz="0 0 0.075"/>
        <geometry><box><size>0.03 0.03 0.15</size></box></geometry>
        <material><ambient>0.2 0.8 0.2 1</ambient></material>
      </visual>
    </link>

    <joint name="arm_joint_2" type="revolute">
      <parent>arm_link_1</parent>
      <child>arm_link_2</child>
      <axis><xyz>0 1 0</xyz></axis>
      <limit><lower>-1.57</lower><upper>1.57</upper><effort>30</effort></limit>
    </joint>

    <!-- LiDAR -->
    <link name="lidar_link">
      <pose relative_to="base_link">0.2 0 0.1 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.03</radius><length>0.04</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.03</radius><length>0.04</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient></material>
      </visual>

      <sensor name="lidar" type="gpu_lidar">
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
          </range>
        </lidar>
        <topic>scan</topic>
      </sensor>
    </link>

    <joint name="lidar_joint" type="fixed">
      <parent>base_link</parent>
      <child>lidar_link</child>
    </joint>

    <!-- PLUGINS -->

    <!-- Differential Drive -->
    <plugin filename="gz-sim-diff-drive-system"
            name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.44</wheel_separation>
      <wheel_radius>0.05</wheel_radius>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <frame_id>odom</frame_id>
      <child_frame_id>base_link</child_frame_id>
      <odom_publish_frequency>50</odom_publish_frequency>
    </plugin>

    <!-- Joint State Publisher -->
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher">
      <topic>joint_states</topic>
      <joint_name>arm_joint_1</joint_name>
      <joint_name>arm_joint_2</joint_name>
    </plugin>

    <!-- Arm Joint Controllers -->
    <plugin filename="gz-sim-joint-position-controller-system"
            name="gz::sim::systems::JointPositionController">
      <joint_name>arm_joint_1</joint_name>
      <topic>arm_joint_1/cmd_pos</topic>
      <p_gain>50.0</p_gain>
      <i_gain>0.5</i_gain>
      <d_gain>5.0</d_gain>
    </plugin>

    <plugin filename="gz-sim-joint-position-controller-system"
            name="gz::sim::systems::JointPositionController">
      <joint_name>arm_joint_2</joint_name>
      <topic>arm_joint_2/cmd_pos</topic>
      <p_gain>30.0</p_gain>
      <i_gain>0.3</i_gain>
      <d_gain>3.0</d_gain>
    </plugin>

  </model>
</sdf>
```

## Summary

In this lesson, you learned:

- Gazebo plugin architecture and types
- Configuring differential drive for mobile robots
- Setting up joint controllers for manipulators
- Adding sensor plugins with appropriate noise models
- Combining multiple plugins for complex robots

## Next Steps

Continue to [TurtleBot3 Simulation](/module-2/week-5/turtlebot3-simulation) to work with a complete robot platform.
