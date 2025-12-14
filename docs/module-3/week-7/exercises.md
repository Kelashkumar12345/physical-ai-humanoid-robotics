---
sidebar_position: 4
title: Week 7 Exercises
description: Hands-on exercises for Isaac Sim basics
---

# Week 7: Isaac Sim Exercises

Practice creating simulations and integrating with ROS 2 in NVIDIA Isaac Sim.

## Exercise 1: First Isaac Sim Scene

**Objective**: Create a basic Isaac Sim scene with physics objects.

### Requirements

1. Create a ground plane
2. Add 5 dynamic cubes at different positions
3. Add 2 static obstacles
4. Run physics simulation for 10 seconds
5. Print object positions after simulation

### Starter Code

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
import numpy as np

# TODO: Create world

# TODO: Add ground plane

# TODO: Add 5 dynamic cubes

# TODO: Add 2 static obstacles

# TODO: Reset and run simulation

# TODO: Print final positions

simulation_app.close()
```

### Verification

- All cubes should fall and settle on ground
- Static obstacles should not move
- No objects should clip through each other

<details className="solution-block">
<summary>Solution</summary>
<div className="solution-content">

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
import numpy as np

# Create world
world = World(physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
world.scene.add_default_ground_plane()

# Add dynamic cubes
cubes = []
colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
]

for i in range(5):
    cube = world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/Cube_{i}",
            name=f"cube_{i}",
            position=np.array([i - 2, 0, 1.0 + i * 0.5]),
            size=0.3,
            color=np.array(colors[i])
        )
    )
    cubes.append(cube)

# Add static obstacles
obstacle1 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Obstacle_1",
        name="obstacle_1",
        position=np.array([0, 2, 0.5]),
        size=1.0,
        color=np.array([0.5, 0.5, 0.5])
    )
)

obstacle2 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Obstacle_2",
        name="obstacle_2",
        position=np.array([0, -2, 0.25]),
        scale=np.array([2, 0.5, 0.5]),
        color=np.array([0.3, 0.3, 0.3])
    )
)

# Reset world
world.reset()

# Run simulation for 10 seconds (10 * 60 = 600 frames at 60 Hz)
for i in range(600):
    world.step(render=True)

    if i % 60 == 0:
        print(f"Time: {i/60:.1f}s")

# Print final positions
print("\nFinal positions:")
for cube in cubes:
    pos, _ = cube.get_world_pose()
    print(f"  {cube.name}: {pos}")

simulation_app.close()
```

</div>
</details>

---

## Exercise 2: Camera Sensor Setup

**Objective**: Configure a camera and capture images with different render products.

### Requirements

1. Create a camera at position (3, 0, 2)
2. Point camera at origin
3. Capture RGB, depth, and semantic images
4. Save images to files
5. Display camera intrinsics

### Starter Code

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import cv2

# Create world with objects

# TODO: Create camera

# TODO: Initialize camera

# TODO: Run simulation and capture images

# TODO: Save images

# TODO: Print camera parameters

simulation_app.close()
```

<details className="hint-block">
<summary>Hint: Camera Orientation</summary>
<div className="hint-content">

To point the camera at the origin from position (3, 0, 2):

```python
from scipy.spatial.transform import Rotation

# Calculate direction to origin
pos = np.array([3, 0, 2])
target = np.array([0, 0, 0])
direction = target - pos
direction = direction / np.linalg.norm(direction)

# Create rotation to align camera forward with direction
# This depends on Isaac Sim's camera convention
```

Or use `look_at` method if available.

</div>
</details>

---

## Exercise 3: LiDAR Integration

**Objective**: Set up a rotating LiDAR and visualize point cloud data.

### Requirements

1. Create rotating LiDAR with 360° FOV
2. Configure angular resolution of 0.5°
3. Set range from 0.1m to 50m
4. Visualize point cloud in Isaac Sim
5. Calculate point cloud statistics

### Starter Code

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import RotatingLidarPhysX
from omni.isaac.core.objects import FixedCuboid
import numpy as np

# Create world with obstacles

# TODO: Create LiDAR

# TODO: Configure LiDAR parameters

# TODO: Initialize and run

# TODO: Process point cloud data

# TODO: Calculate statistics (point count, range distribution)

simulation_app.close()
```

---

## Exercise 4: ROS 2 Bridge Setup

**Objective**: Create a complete ROS 2 interface for Isaac Sim.

### Requirements

1. Enable ROS 2 bridge
2. Publish camera images to `/camera/image_raw`
3. Publish LiDAR scans to `/scan`
4. Subscribe to `/cmd_vel` for robot control
5. Verify with ROS 2 CLI tools

### Test Commands

```bash
# In separate terminal
ros2 topic list

# View camera
ros2 run image_view image_view --ros-args -r image:=/camera/image_raw

# Echo scan
ros2 topic echo /scan --once

# Send velocity command
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "linear: {x: 0.5}"
```

### Starter Code

```python
from omni.isaac.kit import SimulationApp

config = {
    "headless": False,
    "enable_ros2_bridge": True
}
simulation_app = SimulationApp(config)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

# TODO: Create world and sensors

# TODO: Create ROS node with publishers/subscribers

# TODO: Main loop publishing sensor data

simulation_app.close()
```

---

## Exercise 5: Multi-Robot Simulation

**Objective**: Spawn multiple robots with independent sensors.

### Requirements

1. Create 3 robot instances
2. Each robot has camera and LiDAR
3. Use namespaced ROS 2 topics
4. Control robots independently

### Architecture

```
/robot_1/
├── /camera/image_raw
├── /scan
└── /cmd_vel

/robot_2/
├── /camera/image_raw
├── /scan
└── /cmd_vel

/robot_3/
├── /camera/image_raw
├── /scan
└── /cmd_vel
```

### Starter Code

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import numpy as np

class RobotInstance:
    def __init__(self, world, name, position):
        self.name = name
        self.position = position
        # TODO: Create robot, sensors, ROS interface

    def update(self):
        # TODO: Publish sensor data, process commands
        pass

# Create world
world = World()

# TODO: Create 3 robot instances at different positions

# TODO: Main loop

simulation_app.close()
```

---

## Challenge: Warehouse Navigation

**Objective**: Create a warehouse environment with autonomous navigation.

### Requirements

1. Warehouse environment (20m x 15m)
2. Shelving units and obstacles
3. Robot with full sensor suite
4. ROS 2 navigation integration
5. Goal-based navigation demo

### Environment Design

```
+--------------------+
|  [S]    [S]    [S] |
|                    |
|  [S]    [S]    [S] |
|        R          |
|  [S]    [S]    [S] |
|                    |
|  [S]    [S]  G [S] |
+--------------------+

[S] = Shelf unit
R = Robot start
G = Goal position
```

### Deliverables

1. `warehouse_env.py` - Environment creation
2. `robot_interface.py` - ROS 2 bridge
3. `navigation_demo.py` - Navigation script
4. Documentation of approach

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Environment realism | 20 |
| Sensor functionality | 25 |
| ROS 2 integration | 25 |
| Navigation success | 20 |
| Code quality | 10 |

---

## Submission Checklist

- [ ] All scripts run without errors
- [ ] Sensors produce correct data
- [ ] ROS 2 topics publish correctly
- [ ] Code is documented
- [ ] README with setup instructions

## Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [USD Specification](https://graphics.pixar.com/usd/docs/index.html)
