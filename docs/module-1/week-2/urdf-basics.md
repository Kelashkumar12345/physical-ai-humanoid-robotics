---
sidebar_position: 2
title: URDF Basics
description: Creating robot descriptions with URDF and Xacro
---

# URDF Basics

The **Unified Robot Description Format (URDF)** is an XML format for describing robot structure. This lesson covers creating URDF files for visualization and simulation.

<LearningObjectives
  objectives={[
    "Write URDF files describing robot links and joints",
    "Use Xacro for modular, reusable robot descriptions",
    "Visualize robots in RViz2 with joint state publisher",
    "Understand visual, collision, and inertial properties",
    "Create launch files for robot visualization"
  ]}
/>

## URDF Structure

A URDF file defines a robot as a tree of **links** connected by **joints**.

### Basic Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">

  <!-- Links define physical components -->
  <link name="base_link">
    <visual>
      <!-- What it looks like -->
    </visual>
    <collision>
      <!-- Collision geometry for physics -->
    </collision>
    <inertial>
      <!-- Mass and inertia for dynamics -->
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <!-- wheel definition -->
  </link>

</robot>
```

## Link Properties

### Visual Geometry

Defines how the link appears in visualization:

```xml
<link name="chassis">
  <visual>
    <!-- Position offset from link origin -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Geometry: box, cylinder, sphere, or mesh -->
    <geometry>
      <box size="0.4 0.3 0.1"/>
    </geometry>

    <!-- Material (color) -->
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>
</link>
```

### Geometry Types

| Type | Parameters | Example |
|------|------------|---------|
| `box` | size (x y z) | `<box size="0.5 0.3 0.1"/>` |
| `cylinder` | length, radius | `<cylinder length="0.1" radius="0.05"/>` |
| `sphere` | radius | `<sphere radius="0.1"/>` |
| `mesh` | filename, scale | `<mesh filename="package://pkg/mesh.dae"/>` |

### Collision Geometry

For physics simulation (often simplified from visual):

```xml
<link name="arm_link">
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <!-- Simplified collision shape -->
      <cylinder length="0.3" radius="0.03"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

Required for dynamics simulation:

```xml
<link name="base_link">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="5.0"/>
    <inertia
      ixx="0.1" ixy="0" ixz="0"
      iyy="0.1" iyz="0"
      izz="0.1"/>
  </inertial>
</link>
```

:::tip Calculating Inertia
For common shapes:
- **Box**: `ixx = m/12 * (y² + z²)`
- **Cylinder (Z-axis)**: `ixx = m/12 * (3r² + h²)`, `izz = m/2 * r²`
- **Sphere**: `ixx = 2/5 * m * r²`
:::

## Joint Types

| Type | Description | Limits |
|------|-------------|--------|
| `fixed` | No movement | None |
| `continuous` | Unlimited rotation | None |
| `revolute` | Limited rotation | upper, lower |
| `prismatic` | Linear sliding | upper, lower |
| `floating` | 6-DOF (uncommon) | None |
| `planar` | 2D motion (uncommon) | None |

### Joint Definition

```xml
<joint name="shoulder_joint" type="revolute">
  <!-- Parent and child links -->
  <parent link="base_link"/>
  <child link="upper_arm"/>

  <!-- Position and orientation of joint -->
  <origin xyz="0 0 0.5" rpy="0 0 0"/>

  <!-- Axis of rotation/translation -->
  <axis xyz="0 1 0"/>

  <!-- Limits (for revolute/prismatic) -->
  <limit lower="-1.57" upper="1.57"
         effort="100" velocity="1.0"/>

  <!-- Optional dynamics -->
  <dynamics damping="0.1" friction="0.05"/>
</joint>
```

## Complete Example: Simple Robot Arm

```xml
<?xml version="1.0"?>
<robot name="simple_arm">

  <!-- Base (fixed to world) -->
  <link name="world"/>

  <joint name="fixed_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.15"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0"
               iyy="0.01" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Shoulder Joint -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="upper_arm">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.008" ixy="0" ixz="0"
               iyy="0.008" iyz="0" izz="0.0004"/>
    </inertial>
  </link>

  <!-- Elbow Joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="30" velocity="1.5"/>
  </joint>

  <link name="lower_arm">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.04 0.04 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

</robot>
```

## Xacro: Modular Robot Descriptions

**Xacro** (XML Macro) adds variables, math, and includes to URDF.

### Basic Xacro Features

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Properties (variables) -->
  <xacro:property name="wheel_radius" value="0.05"/>
  <xacro:property name="wheel_width" value="0.02"/>
  <xacro:property name="base_length" value="0.3"/>

  <!-- Math expressions -->
  <xacro:property name="wheel_y_offset" value="${base_length/2 + wheel_width/2}"/>

  <!-- Macros (reusable blocks) -->
  <xacro:macro name="wheel" params="prefix y_offset">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
        </geometry>
      </visual>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="0 ${y_offset} ${wheel_radius}" rpy="${-pi/2} 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="left" y_offset="${wheel_y_offset}"/>
  <xacro:wheel prefix="right" y_offset="${-wheel_y_offset}"/>

</robot>
```

### Including Files

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="full_robot">

  <!-- Include common definitions -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/sensors.xacro"/>

  <!-- Include sub-assemblies -->
  <xacro:include filename="base.urdf.xacro"/>
  <xacro:include filename="arm.urdf.xacro"/>

</robot>
```

### Processing Xacro

```bash
# Convert xacro to URDF
xacro robot.urdf.xacro > robot.urdf

# Or use in launch file (automatic processing)
```

## Launch File for Visualization

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_path = get_package_share_directory('my_robot_description')
    urdf_file = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')

    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Joint State Publisher GUI (for testing)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_path, 'rviz', 'robot.rviz')]
        ),
    ])
```

## Validating URDF

```bash
# Check URDF for errors
check_urdf robot.urdf

# Visualize link/joint tree
urdf_to_graphviz robot.urdf
```

## Exercises

<Exercise title="Simple Mobile Robot" difficulty="beginner" estimatedTime="25 min">

Create a URDF for a differential drive robot with:
- Rectangular chassis (0.3 x 0.2 x 0.1 m)
- Two drive wheels (radius 0.05m, width 0.02m)
- One caster wheel (sphere, radius 0.02m)

Visualize in RViz2 with joint_state_publisher_gui.

<Hint>
Start with the chassis, then add wheels using fixed joints for the caster and continuous joints for drive wheels.
</Hint>

</Exercise>

<Exercise title="Xacro Refactor" difficulty="intermediate" estimatedTime="30 min">

Convert the simple arm URDF to Xacro:
1. Create properties for dimensions
2. Create a macro for arm segments
3. Use math expressions for inertia calculations

<Hint>
Inertia for a box: `ixx = m/12 * (y² + z²)`
Use `${m/12 * (y*y + z*z)}` in Xacro.
</Hint>

</Exercise>

## Summary

Key concepts covered:

- ✅ URDF describes robots as links connected by joints
- ✅ Links have visual, collision, and inertial properties
- ✅ Joint types control allowed motion
- ✅ Xacro adds variables, macros, and includes
- ✅ robot_state_publisher broadcasts TF from URDF

## Next Steps

Continue to [Nav2 Introduction](/module-1/week-2/nav2-introduction) to learn about autonomous navigation with the Navigation2 stack.
