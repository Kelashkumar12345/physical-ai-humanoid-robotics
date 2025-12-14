---
sidebar_position: 3
title: Week 8 Exercises
description: Hands-on exercises for humanoid robots in Isaac Sim
---

# Week 8: Humanoid Robot Exercises

Practice working with humanoid robot models and locomotion control.

## Exercise 1: Import and Configure Humanoid

**Objective**: Import a humanoid model and configure joint controllers.

### Requirements

1. Import a humanoid URDF model
2. Print all joint names and limits
3. Configure PD gains for each joint group
4. Test joint position control

### Starter Code

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.urdf import _urdf
from omni.isaac.core.articulations import ArticulationView
import numpy as np

# Create world
world = World()
world.scene.add_default_ground_plane()

# TODO: Import humanoid URDF

# TODO: Create ArticulationView

# TODO: Print joint information

# TODO: Configure gains per joint group

# TODO: Test joint control

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

### Expected Output

```
Humanoid joints:
  - left_hip_yaw: limit [-1.57, 1.57]
  - left_hip_roll: limit [-0.5, 0.5]
  ...

Joint groups configured:
  - left_leg: kp=500, kd=50
  - right_leg: kp=500, kd=50
  - left_arm: kp=200, kd=20
  ...
```

---

## Exercise 2: Standing Balance

**Objective**: Implement a basic balance controller for standing.

### Requirements

1. Detect when the robot is tilting
2. Apply ankle torques to maintain balance
3. Keep COM within support polygon
4. Recover from small pushes

### Starter Code

```python
class BalanceController:
    def __init__(self, humanoid):
        self.humanoid = humanoid

        # Balance parameters
        self.kp_ankle = 500.0
        self.kd_ankle = 50.0

    def update(self, dt):
        """Compute balance torques."""
        # TODO: Get current body orientation

        # TODO: Compute tilt error

        # TODO: Calculate corrective ankle torques

        # TODO: Apply torques

        pass

# TODO: Test balance with small perturbations
```

### Testing

1. Robot should stand stably
2. Apply small forces to torso
3. Robot should recover balance
4. Print COM position and ZMP

---

## Exercise 3: Joint Group Control

**Objective**: Create organized joint control for different body parts.

### Requirements

1. Define joint groups (arms, legs, torso, head)
2. Create functions to control each group independently
3. Implement basic poses (stand, raise arms, crouch)
4. Smooth transitions between poses

### Starter Code

```python
class HumanoidPoseController:
    def __init__(self, humanoid):
        self.humanoid = humanoid

        # TODO: Define joint groups

        # TODO: Define named poses

    def move_to_pose(self, pose_name, duration=1.0):
        """Smoothly move to named pose."""
        # TODO: Get target positions for pose

        # TODO: Interpolate from current to target

        pass

    def move_group(self, group_name, positions, duration=1.0):
        """Move specific joint group."""
        pass

# Poses to implement:
# - "stand": neutral standing
# - "arms_up": raise both arms
# - "arms_out": arms out to sides
# - "crouch": bend knees
# - "wave": one arm waving position
```

---

## Exercise 4: Foot Trajectory Generation

**Objective**: Generate smooth foot trajectories for walking.

### Requirements

1. Implement swing foot trajectory (lift and forward)
2. Implement stance foot trajectory (support phase)
3. Ensure smooth transitions
4. Visualize trajectories

### Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt

class FootTrajectory:
    def __init__(self, step_length, step_height, step_time):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time

    def swing_foot(self, t, start_pos, end_pos):
        """
        Compute swing foot position at time t.

        Args:
            t: time in [0, step_time]
            start_pos: [x, y, z] start position
            end_pos: [x, y, z] end position

        Returns:
            pos: [x, y, z] foot position
        """
        # TODO: Implement smooth swing trajectory

        pass

    def plot_trajectory(self):
        """Visualize foot trajectory."""
        times = np.linspace(0, self.step_time, 100)
        positions = []

        start = np.array([0, 0, 0])
        end = np.array([self.step_length, 0, 0])

        for t in times:
            pos = self.swing_foot(t, start, end)
            positions.append(pos)

        positions = np.array(positions)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Side view (x-z)
        axes[0].plot(positions[:, 0], positions[:, 2])
        axes[0].set_xlabel('X (forward)')
        axes[0].set_ylabel('Z (height)')
        axes[0].set_title('Foot Trajectory (Side View)')

        # Height over time
        axes[1].plot(times, positions[:, 2])
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Height')
        axes[1].set_title('Foot Height')

        plt.tight_layout()
        plt.show()
```

<details className="hint-block">
<summary>Hint: Smooth Trajectory</summary>
<div className="hint-content">

Use a polynomial or sinusoidal trajectory for smooth motion:

```python
# Normalized phase
phase = t / self.step_time

# Smooth position interpolation (cubic)
s = 3 * phase**2 - 2 * phase**3
x = start[0] + s * (end[0] - start[0])

# Height trajectory (parabolic)
z = 4 * self.step_height * phase * (1 - phase)
```

</div>
</details>

---

## Exercise 5: Simple Walking Gait

**Objective**: Implement basic walking gait generation.

### Requirements

1. Implement gait state machine
2. Generate alternating foot trajectories
3. Maintain balance during walking
4. Control walking speed

### Starter Code

```python
class SimpleWalkingController:
    def __init__(self, humanoid):
        self.humanoid = humanoid

        # Gait parameters
        self.step_length = 0.15  # meters
        self.step_height = 0.05  # meters
        self.step_time = 0.5     # seconds
        self.double_support_time = 0.1  # seconds

        # State
        self.time = 0
        self.phase = "double_support"
        self.swing_foot = "right"

        # Foot positions
        self.left_foot = np.array([0, 0.1, 0])
        self.right_foot = np.array([0, -0.1, 0])

    def update(self, dt, velocity_cmd):
        """
        Update walking controller.

        Args:
            dt: time step
            velocity_cmd: [vx, vy, omega] velocity command

        Returns:
            joint_targets: target joint positions
        """
        self.time += dt

        # TODO: Update gait phase

        # TODO: Generate foot trajectories

        # TODO: Compute IK for leg joints

        # TODO: Return joint targets

        pass

    def get_phase(self):
        """Determine current gait phase."""
        cycle_time = 2 * self.step_time + 2 * self.double_support_time
        phase_time = self.time % cycle_time

        # TODO: Determine phase based on time

        pass
```

---

## Challenge: Full Locomotion System

**Objective**: Create a complete humanoid locomotion system with variable speed and turning.

### Requirements

1. Walking at variable speeds (0 to 0.5 m/s)
2. Turning in place
3. Curved walking paths
4. Start/stop transitions
5. Balance recovery from disturbances

### System Architecture

```
┌─────────────────────────────────────────────────┐
│           High-Level Locomotion Controller       │
├─────────────────────────────────────────────────┤
│  Velocity Command → Footstep Planner            │
│       ↓                                          │
│  Footsteps → Trajectory Generator               │
│       ↓                                          │
│  Trajectories → Inverse Kinematics              │
│       ↓                                          │
│  Joint Targets → Low-Level Controllers          │
└─────────────────────────────────────────────────┘
```

### Deliverables

1. `locomotion_controller.py` - Main controller
2. `gait_planner.py` - Gait state machine
3. `trajectory_generator.py` - Foot trajectories
4. `balance_controller.py` - Balance maintenance
5. Demo video of walking robot

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Stable walking | 25 |
| Variable speed | 20 |
| Turning capability | 20 |
| Start/stop transitions | 15 |
| Balance recovery | 10 |
| Code quality | 10 |

---

## Submission Checklist

- [ ] All controllers implemented
- [ ] Robot walks stably in simulation
- [ ] Code handles edge cases
- [ ] Documentation complete
- [ ] Demo video recorded

## Resources

- [Humanoid Robotics Book](https://link.springer.com/book/10.1007/978-3-642-54536-8)
- [ZMP Walking Tutorial](https://ieeexplore.ieee.org/document/1241826)
- [Isaac Sim Humanoid Examples](https://docs.omniverse.nvidia.com/isaacsim/)
