---
sidebar_position: 2
title: World Building
description: Creating simulation environments with SDF
---

# World Building in Gazebo

Learn to create rich, interactive simulation environments for testing robot behaviors.

## Learning Objectives

By the end of this lesson, you will:

- Create complex simulation worlds from scratch
- Add static and dynamic objects
- Configure lighting and environmental effects
- Use include files and model references

## SDF World Structure

### Complete World Template

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="robot_world">

    <!-- Physics Configuration -->
    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Required Plugins -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics">
    </plugin>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact">
    </plugin>

    <!-- Ambient Lighting -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Sun Light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
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
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
            </friction>
          </surface>
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
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Adding Objects to the World

### Basic Shapes

```xml
<!-- Box -->
<model name="red_box">
  <pose>2 0 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box><size>1 1 1</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>1 1 1</size></box>
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

<!-- Sphere -->
<model name="blue_sphere">
  <pose>0 2 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <sphere><radius>0.5</radius></sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere><radius>0.5</radius></sphere>
      </geometry>
      <material>
        <ambient>0 0 1 1</ambient>
        <diffuse>0 0 1 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.1</ixx>
        <iyy>0.1</iyy>
        <izz>0.1</izz>
      </inertia>
    </inertial>
  </link>
</model>

<!-- Cylinder -->
<model name="green_cylinder">
  <pose>-2 0 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.3</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.3</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>0 1 0 1</ambient>
        <diffuse>0 1 0 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.112</ixx>
        <iyy>0.112</iyy>
        <izz>0.045</izz>
      </inertia>
    </inertial>
  </link>
</model>
```

### Static vs Dynamic Objects

```xml
<!-- Static object (won't move) -->
<model name="wall">
  <static>true</static>  <!-- Key difference -->
  <pose>5 0 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box><size>0.1 10 2</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>0.1 10 2</size></box>
      </geometry>
    </visual>
  </link>
</model>

<!-- Dynamic object (affected by physics) -->
<model name="falling_cube">
  <static>false</static>  <!-- Default, can be omitted -->
  <pose>0 0 3 0 0 0</pose>
  <link name="link">
    <!-- Must have inertial for physics -->
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.167</ixx>
        <iyy>0.167</iyy>
        <izz>0.167</izz>
      </inertia>
    </inertial>
    <collision name="collision">
      <geometry>
        <box><size>1 1 1</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>1 1 1</size></box>
      </geometry>
    </visual>
  </link>
</model>
```

## Building an Indoor Environment

### Room with Walls

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="indoor_world">

    <!-- Include physics and plugins from previous example -->

    <!-- Floor -->
    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 10 0.1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 10 0.1</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- North Wall -->
    <model name="north_wall">
      <static>true</static>
      <pose>0 5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- South Wall -->
    <model name="south_wall">
      <static>true</static>
      <pose>0 -5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- East Wall -->
    <model name="east_wall">
      <static>true</static>
      <pose>5 0 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 3</size></box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- West Wall (with doorway) -->
    <model name="west_wall_left">
      <static>true</static>
      <pose>-5 -3 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 4 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 4 3</size></box>
          </geometry>
        </visual>
      </link>
    </model>

    <model name="west_wall_right">
      <static>true</static>
      <pose>-5 3 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 4 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 4 3</size></box>
          </geometry>
        </visual>
      </link>
    </model>

    <model name="west_wall_top">
      <static>true</static>
      <pose>-5 0 2.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 2 1</size></box>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Indoor Lighting -->
    <light name="ceiling_light" type="point">
      <pose>0 0 2.8 0 0 0</pose>
      <diffuse>1 1 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>

  </world>
</sdf>
```

## Using Mesh Files

### Including 3D Meshes

```xml
<model name="custom_object">
  <static>true</static>
  <pose>0 0 0 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <mesh>
          <uri>model://my_models/meshes/object_collision.stl</uri>
          <scale>0.001 0.001 0.001</scale>
        </mesh>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <mesh>
          <uri>model://my_models/meshes/object_visual.dae</uri>
          <scale>0.001 0.001 0.001</scale>
        </mesh>
      </geometry>
    </visual>
  </link>
</model>
```

### Model Database Structure

```
my_models/
├── model.config
├── model.sdf
└── meshes/
    ├── visual.dae
    └── collision.stl
```

**model.config:**
```xml
<?xml version="1.0"?>
<model>
  <name>My Custom Model</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>email@example.com</email>
  </author>
  <description>A custom model for simulation</description>
</model>
```

## Including Models from Fuel

### Accessing Fuel Models

```xml
<!-- Include a model from Gazebo Fuel -->
<include>
  <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Table</uri>
  <name>dining_table</name>
  <pose>2 3 0 0 0 0</pose>
</include>

<!-- Include another instance with different name -->
<include>
  <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Chair</uri>
  <name>chair_1</name>
  <pose>2 2 0 0 0 0</pose>
</include>
```

### Environment Variables

```bash
# Set model path
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/my_models

# Download from Fuel
gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/Table
```

## Lighting Techniques

### Light Types

```xml
<!-- Directional Light (Sun) -->
<light name="sun" type="directional">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>1 1 1 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <direction>-0.5 0.1 -0.9</direction>
</light>

<!-- Point Light (Bulb) -->
<light name="bulb" type="point">
  <pose>0 0 3 0 0 0</pose>
  <diffuse>1 1 0.9 1</diffuse>
  <specular>0.1 0.1 0.1 1</specular>
  <attenuation>
    <range>20</range>
    <constant>0.5</constant>
    <linear>0.05</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>

<!-- Spot Light (Lamp) -->
<light name="spotlight" type="spot">
  <pose>0 0 5 0 0.785 0</pose>  <!-- Angled down -->
  <diffuse>1 1 1 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <direction>0 0 -1</direction>
  <attenuation>
    <range>30</range>
    <constant>0.1</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <spot>
    <inner_angle>0.3</inner_angle>
    <outer_angle>0.5</outer_angle>
    <falloff>1</falloff>
  </spot>
</light>
```

## Surface Properties

### Friction Configuration

```xml
<collision name="collision">
  <geometry>
    <box><size>1 1 1</size></box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>      <!-- Friction coefficient 1 -->
        <mu2>0.5</mu2>    <!-- Friction coefficient 2 -->
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.0</restitution_coefficient>
      <threshold>1e5</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1e12</kp>
        <kd>1.0</kd>
        <max_vel>0.01</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Material Properties

```xml
<visual name="visual">
  <geometry>
    <box><size>1 1 1</size></box>
  </geometry>
  <material>
    <ambient>0.3 0.3 0.3 1</ambient>
    <diffuse>0.7 0.7 0.7 1</diffuse>
    <specular>0.01 0.01 0.01 1</specular>
    <emissive>0 0 0 1</emissive>

    <!-- PBR Material (optional) -->
    <pbr>
      <metal>
        <albedo_map>materials/textures/metal_albedo.png</albedo_map>
        <normal_map>materials/textures/metal_normal.png</normal_map>
        <metalness_map>materials/textures/metal_metalness.png</metalness_map>
        <roughness_map>materials/textures/metal_roughness.png</roughness_map>
      </metal>
    </pbr>
  </material>
</visual>
```

## Creating a Complete Test Environment

### Warehouse World Example

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="warehouse">

    <physics name="physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Essential plugins -->
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- Environment -->
    <scene>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.3 0.3 0.3 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Warehouse Lighting -->
    <light name="light_1" type="point">
      <pose>-5 -5 4 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>30</range>
        <constant>0.3</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>

    <light name="light_2" type="point">
      <pose>5 5 4 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>30</range>
        <constant>0.3</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>

    <!-- Floor -->
    <model name="warehouse_floor">
      <static>true</static>
      <pose>0 0 -0.05 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>20 20 0.1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>20 20 0.1</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Shelving Units -->
    <model name="shelf_1">
      <static>true</static>
      <pose>-6 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.3 0.2 1</ambient>
            <diffuse>0.4 0.3 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="shelf_2">
      <static>true</static>
      <pose>6 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.3 0.2 1</ambient>
            <diffuse>0.4 0.3 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Boxes on floor -->
    <model name="box_1">
      <pose>-2 -2 0.25 0 0 0.3</pose>
      <link name="link">
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.2</ixx><iyy>0.2</iyy><izz>0.2</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="box_2">
      <pose>2 3 0.3 0 0 -0.2</pose>
      <link name="link">
        <inertial>
          <mass>3.0</mass>
          <inertia>
            <ixx>0.15</ixx><iyy>0.2</iyy><izz>0.15</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>0.4 0.6 0.6</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.4 0.6 0.6</size></box></geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Pallet -->
    <model name="pallet">
      <static>true</static>
      <pose>0 -5 0.075 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>1.2 1 0.15</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>1.2 1 0.15</size></box></geometry>
          <material>
            <ambient>0.5 0.4 0.3 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Summary

In this lesson, you learned:

- SDF world file structure and syntax
- Creating static and dynamic objects
- Building indoor environments with walls and floors
- Using mesh files and Fuel models
- Configuring lighting for different scenarios
- Setting up surface properties for realistic physics

## Next Steps

Continue to [Sensor Simulation](/module-2/week-4/sensor-simulation) to learn how to add sensors to your simulation.
