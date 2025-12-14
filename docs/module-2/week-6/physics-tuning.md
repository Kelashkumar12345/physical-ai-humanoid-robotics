---
sidebar_position: 1
title: Physics Tuning
description: Optimizing Gazebo physics for realistic simulation
---

# Physics Tuning in Gazebo

Learn to configure and optimize physics parameters for realistic and stable simulations.

## Learning Objectives

By the end of this lesson, you will:

- Understand Gazebo physics engines and their parameters
- Configure physics for real-time performance
- Tune contact and friction parameters
- Debug and fix common physics issues

## Physics Engines

### Available Engines

| Engine | Strengths | Use Cases |
|--------|-----------|-----------|
| **DART** | Accuracy, joint constraints | Humanoids, manipulators |
| **ODE** | Speed, compatibility | General purpose |
| **Bullet** | Game physics, soft bodies | Interactive demos |
| **Simbody** | Biomechanics | Human modeling |

### Selecting Physics Engine

```xml
<physics name="my_physics" type="dart">
  <!-- DART is default in Gazebo Harmonic -->
</physics>

<physics name="ode_physics" type="ode">
  <!-- ODE for compatibility -->
</physics>
```

## Time Step Configuration

### Understanding Step Size

```xml
<physics name="1ms" type="dart">
  <!-- Time step for each physics update -->
  <max_step_size>0.001</max_step_size>

  <!-- Target ratio of sim time to real time -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Maximum updates per second (for GUI responsiveness) -->
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

### Trade-offs

| Step Size | Accuracy | Performance | Use Case |
|-----------|----------|-------------|----------|
| 0.0001s | Very High | Slow | High-precision |
| 0.001s | High | Good | Default |
| 0.005s | Medium | Fast | Simple robots |
| 0.01s | Low | Very Fast | Large scenes |

### Performance Tuning

```xml
<!-- High performance (for complex scenes) -->
<physics name="fast" type="dart">
  <max_step_size>0.004</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>

<!-- High accuracy (for precise manipulation) -->
<physics name="accurate" type="dart">
  <max_step_size>0.0005</max_step_size>
  <real_time_factor>0.5</real_time_factor>
</physics>
```

## Solver Configuration

### DART Solver Options

```xml
<physics name="dart_physics" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>

  <dart>
    <!-- Constraint solver iterations -->
    <solver>
      <solver_type>dantzig</solver_type>
    </solver>

    <!-- Collision detection -->
    <collision_detector>fcl</collision_detector>
  </dart>
</physics>
```

### ODE Solver Options

```xml
<physics name="ode_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>

  <ode>
    <solver>
      <!-- quickstep for speed, world for accuracy -->
      <type>quick</type>
      <iters>50</iters>
      <sor>1.3</sor>
    </solver>

    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
    </constraints>
  </ode>
</physics>
```

## Contact Parameters

### Surface Properties

```xml
<collision name="collision">
  <geometry>
    <box><size>1 1 1</size></box>
  </geometry>

  <surface>
    <!-- Friction -->
    <friction>
      <ode>
        <mu>1.0</mu>       <!-- Primary friction coefficient -->
        <mu2>1.0</mu2>     <!-- Secondary friction coefficient -->
        <fdir1>0 0 1</fdir1>  <!-- Primary friction direction -->
        <slip1>0.0</slip1>    <!-- First slip coefficient -->
        <slip2>0.0</slip2>    <!-- Second slip coefficient -->
      </ode>
    </friction>

    <!-- Bounce/restitution -->
    <bounce>
      <restitution_coefficient>0.0</restitution_coefficient>
      <threshold>1e5</threshold>
    </bounce>

    <!-- Contact parameters -->
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>  <!-- Constraint force mixing -->
        <soft_erp>0.2</soft_erp>  <!-- Error reduction parameter -->
        <kp>1e12</kp>              <!-- Contact stiffness -->
        <kd>1.0</kd>               <!-- Contact damping -->
        <max_vel>0.01</max_vel>    <!-- Max contact correction velocity -->
        <min_depth>0.001</min_depth>  <!-- Minimum penetration depth -->
      </ode>
    </contact>
  </surface>
</collision>
```

### Friction Guidelines

| Material Pair | mu Value |
|--------------|----------|
| Rubber on concrete | 0.8 - 1.2 |
| Metal on metal (dry) | 0.4 - 0.7 |
| Metal on metal (lubricated) | 0.1 - 0.2 |
| Wood on wood | 0.3 - 0.5 |
| Ice | 0.01 - 0.05 |

### Wheel Friction Example

```xml
<!-- High friction for drive wheels -->
<collision name="wheel_collision">
  <geometry>
    <cylinder>
      <radius>0.05</radius>
      <length>0.03</length>
    </cylinder>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.5</mu>
        <mu2>1.5</mu2>
      </ode>
    </friction>
  </surface>
</collision>

<!-- Low friction for caster wheel -->
<collision name="caster_collision">
  <geometry>
    <sphere><radius>0.02</radius></sphere>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>0.01</mu>
        <mu2>0.01</mu2>
      </ode>
    </friction>
  </surface>
</collision>
```

## Inertia Configuration

### Calculating Inertia

For common shapes:

**Box (w × h × d, mass m):**
```
Ixx = (1/12) × m × (h² + d²)
Iyy = (1/12) × m × (w² + d²)
Izz = (1/12) × m × (w² + h²)
```

**Cylinder (radius r, length l, mass m):**
```
Ixx = Iyy = (1/12) × m × (3r² + l²)
Izz = (1/2) × m × r²
```

**Sphere (radius r, mass m):**
```
Ixx = Iyy = Izz = (2/5) × m × r²
```

### Inertia Helper Script

```python
#!/usr/bin/env python3

def box_inertia(mass, w, h, d):
    """Calculate inertia for a box."""
    ixx = (1/12) * mass * (h**2 + d**2)
    iyy = (1/12) * mass * (w**2 + d**2)
    izz = (1/12) * mass * (w**2 + h**2)
    return ixx, iyy, izz

def cylinder_inertia(mass, radius, length):
    """Calculate inertia for a cylinder (along z-axis)."""
    ixx = (1/12) * mass * (3 * radius**2 + length**2)
    iyy = ixx
    izz = (1/2) * mass * radius**2
    return ixx, iyy, izz

def sphere_inertia(mass, radius):
    """Calculate inertia for a solid sphere."""
    i = (2/5) * mass * radius**2
    return i, i, i

# Example usage
mass = 1.0
w, h, d = 0.1, 0.1, 0.2

ixx, iyy, izz = box_inertia(mass, w, h, d)
print(f"Box ({w}x{h}x{d}, {mass}kg):")
print(f"  Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}")
```

### Common Mistakes

```xml
<!-- BAD: Zero or near-zero inertia -->
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0</ixx><iyy>0</iyy><izz>0</izz>
  </inertia>
</inertial>

<!-- BAD: Mass at wrong origin -->
<inertial>
  <mass>1.0</mass>
  <!-- Origin should be at center of mass -->
  <origin xyz="0 0 0"/>
</inertial>

<!-- GOOD: Proper inertia and origin -->
<inertial>
  <origin xyz="0 0 0.1"/>  <!-- CoM location -->
  <mass>1.0</mass>
  <inertia>
    <ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz>
    <iyy>0.01</iyy><iyz>0</iyz>
    <izz>0.005</izz>
  </inertia>
</inertial>
```

## Debugging Physics Issues

### Common Problems

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Robot flies away | Missing/bad inertia | Check inertial values |
| Objects fall through floor | Step size too large | Decrease max_step_size |
| Jittery motion | Solver not converging | Increase iterations |
| Wheels slip | Low friction | Increase mu values |
| Robot tilts | CoM offset | Adjust inertial origin |

### Visualization Tools

```bash
# Enable physics debugging in Gazebo
gz sim -v 4 world.sdf  # Verbose output

# View collision shapes
# In Gazebo GUI: View -> Collisions

# View inertia visualization
# In Gazebo GUI: View -> Inertias
```

### Physics Validation

```python
#!/usr/bin/env python3
"""Validate physics properties of a robot model."""

import xml.etree.ElementTree as ET

def validate_urdf(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    issues = []

    for link in root.findall('.//link'):
        name = link.get('name')

        # Check inertial
        inertial = link.find('inertial')
        if inertial is None:
            issues.append(f"Link '{name}' missing inertial")
            continue

        # Check mass
        mass = inertial.find('mass')
        if mass is not None:
            m = float(mass.get('value', 0))
            if m <= 0:
                issues.append(f"Link '{name}' has invalid mass: {m}")

        # Check inertia
        inertia = inertial.find('inertia')
        if inertia is not None:
            ixx = float(inertia.get('ixx', 0))
            iyy = float(inertia.get('iyy', 0))
            izz = float(inertia.get('izz', 0))

            if ixx <= 0 or iyy <= 0 or izz <= 0:
                issues.append(f"Link '{name}' has invalid inertia")

    return issues

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        issues = validate_urdf(sys.argv[1])
        if issues:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No issues found!")
```

## Performance Optimization

### Simplify Collision Geometry

```xml
<!-- Instead of complex mesh -->
<collision name="collision">
  <geometry>
    <mesh><uri>model://robot/meshes/arm.stl</uri></mesh>
  </geometry>
</collision>

<!-- Use simple primitives -->
<collision name="collision_simplified">
  <geometry>
    <cylinder>
      <radius>0.05</radius>
      <length>0.3</length>
    </cylinder>
  </geometry>
</collision>
```

### Reduce Collision Pairs

```xml
<!-- Disable self-collision for adjacent links -->
<joint name="joint_1" type="revolute">
  <parent>link_0</parent>
  <child>link_1</child>
  <!-- ... -->
</joint>

<!-- In Gazebo, use disable_collision tags -->
<gazebo>
  <disable_collision>
    <link1>link_0</link1>
    <link2>link_1</link2>
  </disable_collision>
</gazebo>
```

### Multi-threaded Physics

```xml
<!-- Enable threading for large simulations -->
<physics name="threaded" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>
```

## Summary

In this lesson, you learned:

- Different physics engines and their characteristics
- Configuring time step and solver parameters
- Setting up friction and contact properties
- Calculating and validating inertial properties
- Debugging common physics issues
- Optimizing physics performance

## Next Steps

Continue to [Domain Randomization](/module-2/week-6/domain-randomization) to learn simulation-to-real transfer techniques.
