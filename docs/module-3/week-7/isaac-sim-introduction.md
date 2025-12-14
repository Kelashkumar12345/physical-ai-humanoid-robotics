---
sidebar_position: 1
title: Introduction to Isaac Sim
description: Getting started with NVIDIA Isaac Sim for robotics
---

# Introduction to NVIDIA Isaac Sim

Isaac Sim is NVIDIA's advanced robotics simulation platform built on Omniverse, offering photorealistic rendering, GPU-accelerated physics, and deep learning integration.

## Learning Objectives

By the end of this lesson, you will:

- Understand Isaac Sim's architecture and capabilities
- Install and configure Isaac Sim
- Navigate the Isaac Sim interface
- Create your first Isaac Sim scene

## What is Isaac Sim?

Isaac Sim is part of NVIDIA's Isaac robotics platform:

| Component | Purpose |
|-----------|---------|
| **Isaac Sim** | High-fidelity simulation |
| **Isaac ROS** | ROS 2 hardware acceleration |
| **Isaac SDK** | Robot development framework |
| **Isaac Gym** | RL training environments |

### Key Features

- **Photorealistic Rendering**: RTX ray tracing for realistic visuals
- **PhysX 5**: GPU-accelerated physics simulation
- **Synthetic Data Generation**: Automatic labeling for ML
- **ROS 2 Integration**: Native bridge to ROS ecosystem
- **Multi-Robot Simulation**: Thousands of robots in parallel
- **Digital Twin**: Connect to real robots and environments

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 2070 or higher |
| VRAM | 8 GB |
| CPU | Intel i7 / AMD Ryzen 7 |
| RAM | 32 GB |
| Storage | 50 GB SSD |
| OS | Ubuntu 22.04 / Windows 10/11 |

### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA RTX 3080/4080 or higher |
| VRAM | 12+ GB |
| CPU | Intel i9 / AMD Ryzen 9 |
| RAM | 64 GB |
| Storage | 100 GB NVMe SSD |

## Installation

### Option 1: Omniverse Launcher (Recommended)

1. Download NVIDIA Omniverse Launcher from [nvidia.com/omniverse](https://www.nvidia.com/omniverse)

2. Install the launcher:
```bash
# Linux
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```

3. Sign in with NVIDIA account

4. Navigate to Exchange → Isaac Sim → Install

5. Launch Isaac Sim from Library

### Option 2: Docker Container

```bash
# Login to NVIDIA NGC
docker login nvcr.io

# Pull Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# Run Isaac Sim
docker run --name isaac-sim --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  nvcr.io/nvidia/isaac-sim:2023.1.1
```

### Option 3: Workstation Install

```bash
# Download Isaac Sim package
wget https://install.launcher.omniverse.nvidia.com/isaac-sim/isaac_sim-2023.1.1-linux-x86_64.tar.gz

# Extract
tar -xzf isaac_sim-2023.1.1-linux-x86_64.tar.gz

# Run
cd isaac_sim-2023.1.1
./isaac-sim.sh
```

## Isaac Sim Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA Omniverse                          │
├─────────────────────────────────────────────────────────────┤
│  Isaac Sim                                                   │
│  ├── Kit (Application Framework)                             │
│  ├── USD (Universal Scene Description)                       │
│  ├── PhysX 5 (Physics Engine)                               │
│  ├── RTX Renderer (Ray Tracing)                              │
│  └── Extensions                                              │
│      ├── omni.isaac.core                                     │
│      ├── omni.isaac.robot                                    │
│      ├── omni.isaac.sensor                                   │
│      ├── omni.isaac.ros2_bridge                              │
│      └── omni.replicator (Synthetic Data)                   │
├─────────────────────────────────────────────────────────────┤
│  Nucleus (Collaboration Server)                              │
└─────────────────────────────────────────────────────────────┘
```

### Universal Scene Description (USD)

USD is the scene format used by Isaac Sim:

```python
# Example USD structure
/World
├── /World/GroundPlane
├── /World/Robot
│   ├── /World/Robot/base_link
│   ├── /World/Robot/wheel_left
│   └── /World/Robot/wheel_right
├── /World/Camera
└── /World/Lights
```

## Isaac Sim Interface

### Main Components

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar: File | Edit | Window | Isaac | Help              │
├────────────────────────────────────┬────────────────────────┤
│                                    │                        │
│         Viewport                   │    Stage               │
│         (3D Scene)                 │    (Scene Tree)        │
│                                    │                        │
│                                    ├────────────────────────┤
│                                    │    Property            │
│                                    │    (Selected Object)   │
├────────────────────────────────────┴────────────────────────┤
│  Console | Script Editor | Timeline | Content Browser       │
└─────────────────────────────────────────────────────────────┘
```

### Key Windows

| Window | Purpose |
|--------|---------|
| **Viewport** | 3D visualization and interaction |
| **Stage** | Scene hierarchy (USD tree) |
| **Property** | Edit selected object attributes |
| **Content Browser** | Asset library and files |
| **Script Editor** | Python scripting |
| **Console** | Command output and logs |

### Navigation Controls

| Action | Mouse/Keyboard |
|--------|---------------|
| Orbit | Alt + Left Mouse |
| Pan | Alt + Middle Mouse |
| Zoom | Alt + Right Mouse / Scroll |
| Focus on object | F |
| Reset view | Home |

## Creating Your First Scene

### Using the GUI

1. **Create Ground Plane**
   - Menu: Create → Physics → Ground Plane

2. **Add a Cube**
   - Menu: Create → Shapes → Cube
   - Position above ground (Y = 2.0)

3. **Add Physics**
   - Select cube
   - Menu: Add → Physics → Rigid Body
   - Menu: Add → Physics → Collider

4. **Play Simulation**
   - Press Play button or Space

### Using Python

```python
from omni.isaac.kit import SimulationApp

# Create simulation app
simulation_app = SimulationApp({"headless": False})

# Import after SimulationApp is created
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# Create world
world = World()

# Add ground plane
world.scene.add_default_ground_plane()

# Add a cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=[0, 0, 1.0],
        size=0.5,
        color=[1.0, 0, 0]  # Red
    )
)

# Reset world
world.reset()

# Simulation loop
while simulation_app.is_running():
    world.step(render=True)

# Cleanup
simulation_app.close()
```

## Physics Configuration

### World Physics Settings

```python
from omni.isaac.core import World
from omni.isaac.core.utils.physics import set_physics_properties

# Create world with custom physics
world = World(
    physics_dt=1.0/120.0,  # 120 Hz physics
    rendering_dt=1.0/60.0,  # 60 Hz rendering
    stage_units_in_meters=1.0
)

# Or set physics properties directly
set_physics_properties(
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
    enable_gpu_dynamics=True,
    enable_scene_query_support=True
)
```

### Rigid Body Properties

```python
from omni.isaac.core.utils.prims import create_prim
from pxr import UsdPhysics, PhysxSchema

# Create a prim
prim = create_prim("/World/Box", "Cube")

# Add rigid body
rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
rigid_body_api.CreateVelocityAttr().Set((0, 0, 0))
rigid_body_api.CreateAngularVelocityAttr().Set((0, 0, 0))

# Add mass
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr().Set(1.0)

# Add collision
collision_api = UsdPhysics.CollisionAPI.Apply(prim)

# PhysX-specific properties
physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
physx_rigid_body_api.CreateLinearDampingAttr().Set(0.1)
physx_rigid_body_api.CreateAngularDampingAttr().Set(0.1)
```

## Loading Robot Models

### From URDF

```python
from omni.isaac.urdf import _urdf

# URDF import configuration
urdf_config = _urdf.ImportConfig()
urdf_config.merge_fixed_joints = True
urdf_config.fix_base = False
urdf_config.make_default_prim = True
urdf_config.create_physics_scene = True

# Import URDF
result, prim_path = _urdf.import_robot(
    asset_root="/path/to/robot",
    asset_file="robot.urdf",
    import_config=urdf_config
)

print(f"Robot imported at: {prim_path}")
```

### From USD

```python
from omni.isaac.core.utils.stage import add_reference_to_stage

# Add robot from USD
robot_prim_path = "/World/Robot"
robot_usd_path = "omniverse://localhost/Library/Robots/turtlebot3.usd"

add_reference_to_stage(
    usd_path=robot_usd_path,
    prim_path=robot_prim_path
)
```

### Using Isaac Assets

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path

# Get Isaac Sim assets root
assets_root = get_assets_root_path()

# Available robot assets
robots = {
    "carter": f"{assets_root}/Isaac/Robots/Carter/carter_v1.usd",
    "franka": f"{assets_root}/Isaac/Robots/Franka/franka.usd",
    "jetbot": f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd",
    "ur10": f"{assets_root}/Isaac/Robots/UR10/ur10.usd",
}
```

## Running Headless Simulation

```python
from omni.isaac.kit import SimulationApp

# Headless mode (no GUI)
config = {
    "headless": True,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(config)

from omni.isaac.core import World

world = World()
world.scene.add_default_ground_plane()
world.reset()

# Run simulation steps
for i in range(1000):
    world.step(render=False)

    if i % 100 == 0:
        print(f"Step {i}")

simulation_app.close()
```

## Summary

In this lesson, you learned:

- Isaac Sim's architecture and key features
- System requirements and installation options
- The Isaac Sim interface and navigation
- Creating basic scenes with physics
- Loading robot models from URDF and USD

## Next Steps

Continue to [Isaac Sim Sensors](/module-3/week-7/isaac-sim-sensors) to learn about sensor simulation.
