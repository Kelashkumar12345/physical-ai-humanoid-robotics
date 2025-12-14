---
sidebar_position: 1
title: Robot Description
description: Creating robot models with URDF and SDF
---

# Robot Description in Gazebo

Learn to create detailed robot models for simulation using URDF and SDF formats.

## Learning Objectives

By the end of this lesson, you will:

- Understand URDF and SDF robot description formats
- Create articulated robot models with joints
- Define visual, collision, and inertial properties
- Convert between URDF and SDF formats

## URDF vs SDF

### Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| Origin | ROS | Gazebo |
| Joint types | revolute, continuous, prismatic, fixed, floating, planar | All URDF types + ball, screw, universal, gearbox |
| Sensors | Limited (via plugins) | Native support |
| Closed loops | Not supported | Supported |
| Multiple robots | Separate files | Single world file |
| ROS integration | Native | Via ros_gz |

### When to Use Each

- **URDF**: ROS 2 robot_state_publisher, MoveIt, Nav2
- **SDF**: Gazebo simulation, complex mechanisms, sensor definition

## URDF Basics

### Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">

  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>...</visual>
    <collision>...</collision>
    <inertial>...</inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

</robot>
```

### Link Properties

```xml
<link name="arm_link">
  <!-- Visual: What you see in visualization -->
  <visual>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.5"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>

  <!-- Collision: Used for physics simulation -->
  <collision>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.5"/>
    </geometry>
  </collision>

  <!-- Inertial: Mass and inertia for dynamics -->
  <inertial>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.02" ixy="0" ixz="0"
             iyy="0.02" iyz="0"
             izz="0.001"/>
  </inertial>
</link>
```

### Joint Types

```xml
<!-- Revolute: Rotation with limits -->
<joint name="shoulder_joint" type="revolute">
  <parent link="base_link"/>
  <child link="upper_arm"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
</joint>

<!-- Continuous: Unlimited rotation (wheels) -->
<joint name="wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="wheel"/>
  <origin xyz="0.1 0 0" rpy="-1.5708 0 0"/>
  <axis xyz="0 0 1"/>
</joint>

<!-- Prismatic: Linear motion -->
<joint name="linear_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="slider"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="100" velocity="0.5"/>
</joint>

<!-- Fixed: No motion (rigid connection) -->
<joint name="camera_mount" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0.05" rpy="0 0 0"/>
</joint>
```

## Complete URDF Example: Simple Robot Arm

```xml
<?xml version="1.0"?>
<robot name="simple_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="red">
    <color rgba="0.8 0 0 1"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.005" ixy="0" ixz="0"
               iyy="0.005" iyz="0"
               izz="0.01"/>
    </inertial>
  </link>

  <!-- Shoulder Link -->
  <link name="shoulder_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0"
               izz="0.0005"/>
    </inertial>
  </link>

  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.025" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="1.0"/>
  </joint>

  <!-- Upper Arm -->
  <link name="upper_arm_link">
    <visual>
      <origin xyz="0.15 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0.15 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.15 0 0"/>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.003" iyz="0"
               izz="0.003"/>
    </inertial>
  </link>

  <joint name="shoulder_lift_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <!-- Forearm -->
  <link name="forearm_link">
    <visual>
      <origin xyz="0.125 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.25"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0.125 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.125 0 0"/>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.002" iyz="0"
               izz="0.002"/>
    </inertial>
  </link>

  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="30" velocity="1.0"/>
  </joint>

  <!-- End Effector -->
  <link name="end_effector_link">
    <visual>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.00004" ixy="0" ixz="0"
               iyy="0.00004" iyz="0"
               izz="0.00004"/>
    </inertial>
  </link>

  <joint name="wrist_joint" type="revolute">
    <parent link="forearm_link"/>
    <child link="end_effector_link"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2.0"/>
  </joint>

</robot>
```

## Xacro for Reusable Models

### Using Xacro Macros

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot">

  <!-- Properties -->
  <xacro:property name="link_radius" value="0.03"/>
  <xacro:property name="link_length" value="0.3"/>
  <xacro:property name="link_mass" value="0.5"/>

  <!-- Macro for cylindrical link -->
  <xacro:macro name="cylinder_link" params="name length radius mass color">
    <link name="${name}">
      <visual>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
        <material name="${color}"/>
      </visual>
      <collision>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
      </collision>
      <inertial>
        <origin xyz="0 0 ${length/2}"/>
        <mass value="${mass}"/>
        <xacro:property name="ixx" value="${(1/12)*mass*(3*radius*radius + length*length)}"/>
        <xacro:property name="izz" value="${(1/2)*mass*radius*radius}"/>
        <inertia ixx="${ixx}" ixy="0" ixz="0"
                 iyy="${ixx}" iyz="0"
                 izz="${izz}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Macro for revolute joint -->
  <xacro:macro name="revolute_joint" params="name parent child xyz rpy axis lower upper">
    <joint name="${name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${child}"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="${axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="50" velocity="1.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macros -->
  <xacro:cylinder_link name="link_1" length="0.3" radius="0.03" mass="0.5" color="blue"/>
  <xacro:cylinder_link name="link_2" length="0.25" radius="0.025" mass="0.4" color="white"/>

  <xacro:revolute_joint name="joint_1"
                        parent="link_1" child="link_2"
                        xyz="0 0 0.3" rpy="0 0 0"
                        axis="0 1 0"
                        lower="-1.57" upper="1.57"/>

</robot>
```

### Processing Xacro

```bash
# Convert xacro to URDF
xacro robot.urdf.xacro > robot.urdf

# With arguments
xacro robot.urdf.xacro arm_length:=0.5 > robot.urdf
```

## Converting URDF to SDF

### Using gz sdf

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf
```

### Adding Gazebo-Specific Elements

```xml
<!-- In URDF, add gazebo tags -->
<robot name="my_robot">
  <!-- ... links and joints ... -->

  <!-- Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.5</mu1>
    <mu2>0.5</mu2>
  </gazebo>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher">
      <topic>joint_states</topic>
    </plugin>
  </gazebo>

</robot>
```

## Differential Drive Robot Example

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Properties -->
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_length" value="0.4"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.05"/>
  <xacro:property name="wheel_width" value="0.025"/>
  <xacro:property name="wheel_separation" value="0.32"/>

  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="black">
    <color rgba="0.1 0.1 0.1 1"/>
  </material>

  <!-- Wheel Macro -->
  <xacro:macro name="wheel" params="name x_offset y_offset">
    <link name="${name}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.0002" ixy="0" ixz="0"
                 iyy="0.0002" iyz="0"
                 izz="0.00025"/>
      </inertial>
    </link>

    <joint name="${name}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${name}_wheel"/>
      <origin xyz="${x_offset} ${y_offset} 0" rpy="-1.5708 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <gazebo reference="${name}_wheel">
      <mu1>1.0</mu1>
      <mu2>1.0</mu2>
    </gazebo>
  </xacro:macro>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0"
               iyy="0.08" iyz="0"
               izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <xacro:wheel name="left" x_offset="0" y_offset="${wheel_separation/2}"/>
  <xacro:wheel name="right" x_offset="0" y_offset="${-wheel_separation/2}"/>

  <!-- Caster Wheel -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.025"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.00001" ixy="0" ixz="0"
               iyy="0.00001" iyz="0"
               izz="0.00001"/>
    </inertial>
  </link>

  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="-0.15 0 -0.025" rpy="0 0 0"/>
  </joint>

  <gazebo reference="caster_wheel">
    <mu1>0.01</mu1>
    <mu2>0.01</mu2>
  </gazebo>

  <!-- Gazebo Differential Drive Plugin -->
  <gazebo>
    <plugin filename="gz-sim-diff-drive-system"
            name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>${wheel_separation}</wheel_separation>
      <wheel_radius>${wheel_radius}</wheel_radius>
      <odom_publish_frequency>50</odom_publish_frequency>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <frame_id>odom</frame_id>
      <child_frame_id>base_link</child_frame_id>
    </plugin>
  </gazebo>

</robot>
```

## Validating Robot Models

### URDF Validation

```bash
# Check URDF syntax
check_urdf robot.urdf

# Visualize in RViz
ros2 launch urdf_tutorial display.launch.py model:=robot.urdf
```

### SDF Validation

```bash
# Validate SDF
gz sdf -k robot.sdf

# Print parsed SDF
gz sdf -p robot.sdf
```

## Summary

In this lesson, you learned:

- URDF and SDF format differences and use cases
- Creating links with visual, collision, and inertial properties
- Defining different joint types
- Using Xacro for modular robot descriptions
- Converting between URDF and SDF formats

## Next Steps

Continue to [Gazebo Plugins](/module-2/week-5/gazebo-plugins) to learn how to add functionality to your robots.
