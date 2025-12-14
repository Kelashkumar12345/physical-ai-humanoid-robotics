---
sidebar_position: 2
title: Humanoid Locomotion
description: Walking and movement control for humanoid robots
---

# Humanoid Locomotion

Learn the fundamentals of humanoid walking, balance control, and gait generation.

## Learning Objectives

By the end of this lesson, you will:

- Understand bipedal walking dynamics
- Implement basic walking gait generation
- Use Zero Moment Point (ZMP) for stability
- Create simple locomotion controllers

## Bipedal Walking Fundamentals

### Walking Phases

```
┌─────────────────────────────────────────────────────┐
│                Walking Gait Cycle                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Double Support → Single Support → Double Support   │
│       (DS)           (SS)              (DS)         │
│                                                      │
│  ┌───┐      ┌───┐                ┌───┐      ┌───┐  │
│  │   │      │   │                │   │      │   │  │
│  │   │      │   │ ──→            │   │      │   │  │
│  │   │      │   │                │   │      │   │  │
│  └─┬─┘      └─┬─┘                └─┬─┘      └─┬─┘  │
│    │          │                    │          │    │
│  ──┴──────────┴──            ──────┴──────────┴──  │
│                                                      │
│   Both feet      One foot        Both feet          │
│   on ground      on ground       on ground          │
└─────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **COM** | Center of Mass - average position of all mass |
| **COP** | Center of Pressure - point where GRF acts |
| **ZMP** | Zero Moment Point - where horizontal moment is zero |
| **Support Polygon** | Area enclosed by contact points |

## Zero Moment Point (ZMP)

### ZMP Calculation

```python
import numpy as np

def compute_zmp(com_pos, com_acc, foot_positions, gravity=9.81):
    """
    Compute Zero Moment Point from COM dynamics.

    Args:
        com_pos: [x, y, z] center of mass position
        com_acc: [ax, ay, az] center of mass acceleration
        foot_positions: list of foot contact positions
        gravity: gravitational acceleration

    Returns:
        zmp: [x, y] zero moment point
    """
    # ZMP formula (simplified, flat ground)
    zmp_x = com_pos[0] - (com_pos[2] / gravity) * com_acc[0]
    zmp_y = com_pos[1] - (com_pos[2] / gravity) * com_acc[1]

    return np.array([zmp_x, zmp_y])

def is_stable(zmp, support_polygon):
    """
    Check if ZMP is inside support polygon.

    Args:
        zmp: [x, y] zero moment point
        support_polygon: list of [x, y] points defining polygon

    Returns:
        bool: True if stable
    """
    from shapely.geometry import Point, Polygon

    point = Point(zmp)
    polygon = Polygon(support_polygon)

    return polygon.contains(point)
```

### Support Polygon

```python
def get_support_polygon(left_foot_pos, right_foot_pos, foot_size):
    """
    Get support polygon from foot positions.

    Args:
        left_foot_pos: [x, y, theta] left foot pose
        right_foot_pos: [x, y, theta] right foot pose
        foot_size: [length, width] foot dimensions

    Returns:
        vertices: list of polygon vertices
    """
    vertices = []

    for foot_pos in [left_foot_pos, right_foot_pos]:
        x, y, theta = foot_pos
        l, w = foot_size[0] / 2, foot_size[1] / 2

        # Foot corners in foot frame
        corners = [
            [l, w], [l, -w], [-l, -w], [-l, w]
        ]

        # Rotate and translate to world frame
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for cx, cy in corners:
            wx = x + cx * cos_t - cy * sin_t
            wy = y + cx * sin_t + cy * cos_t
            vertices.append([wx, wy])

    # Compute convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    return [vertices[i] for i in hull.vertices]
```

## Basic Walking Controller

### Foot Trajectory Generation

```python
class FootTrajectoryGenerator:
    """Generate foot trajectories for walking."""

    def __init__(self, step_length=0.2, step_height=0.05, step_time=0.5):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time

    def generate_swing_trajectory(self, start_pos, end_pos, phase):
        """
        Generate swing foot trajectory.

        Args:
            start_pos: [x, y, z] start position
            end_pos: [x, y, z] end position
            phase: 0 to 1 phase through swing

        Returns:
            pos: [x, y, z] current foot position
        """
        # Linear interpolation for x, y
        x = start_pos[0] + phase * (end_pos[0] - start_pos[0])
        y = start_pos[1] + phase * (end_pos[1] - start_pos[1])

        # Parabolic trajectory for z (foot lift)
        z = start_pos[2] + 4 * self.step_height * phase * (1 - phase)

        return np.array([x, y, z])

    def generate_stance_trajectory(self, start_pos, velocity, phase):
        """
        Generate stance foot trajectory (moves backward relative to body).

        Args:
            start_pos: [x, y, z] start position
            velocity: walking velocity
            phase: 0 to 1 phase through stance

        Returns:
            pos: [x, y, z] current foot position (relative to pelvis)
        """
        # Foot moves backward as body moves forward
        x = start_pos[0] - velocity * phase * self.step_time
        y = start_pos[1]
        z = start_pos[2]

        return np.array([x, y, z])
```

### Gait State Machine

```python
from enum import Enum

class GaitPhase(Enum):
    LEFT_STANCE = 0
    RIGHT_STANCE = 1
    DOUBLE_SUPPORT_LR = 2  # Left to Right
    DOUBLE_SUPPORT_RL = 3  # Right to Left

class GaitStateMachine:
    """State machine for walking gait."""

    def __init__(self, cycle_time=1.0, double_support_ratio=0.2):
        self.cycle_time = cycle_time
        self.double_support_ratio = double_support_ratio

        self.phase = GaitPhase.DOUBLE_SUPPORT_LR
        self.phase_time = 0.0

    def update(self, dt):
        """Update gait state."""
        self.phase_time += dt

        # Calculate phase durations
        ds_time = self.cycle_time * self.double_support_ratio / 2
        ss_time = self.cycle_time * (1 - self.double_support_ratio) / 2

        # State transitions
        if self.phase == GaitPhase.DOUBLE_SUPPORT_LR:
            if self.phase_time >= ds_time:
                self.phase = GaitPhase.RIGHT_STANCE
                self.phase_time = 0.0

        elif self.phase == GaitPhase.RIGHT_STANCE:
            if self.phase_time >= ss_time:
                self.phase = GaitPhase.DOUBLE_SUPPORT_RL
                self.phase_time = 0.0

        elif self.phase == GaitPhase.DOUBLE_SUPPORT_RL:
            if self.phase_time >= ds_time:
                self.phase = GaitPhase.LEFT_STANCE
                self.phase_time = 0.0

        elif self.phase == GaitPhase.LEFT_STANCE:
            if self.phase_time >= ss_time:
                self.phase = GaitPhase.DOUBLE_SUPPORT_LR
                self.phase_time = 0.0

    def get_phase_ratio(self):
        """Get progress through current phase (0 to 1)."""
        ds_time = self.cycle_time * self.double_support_ratio / 2
        ss_time = self.cycle_time * (1 - self.double_support_ratio) / 2

        if self.phase in [GaitPhase.DOUBLE_SUPPORT_LR, GaitPhase.DOUBLE_SUPPORT_RL]:
            return self.phase_time / ds_time
        else:
            return self.phase_time / ss_time

    def is_left_stance(self):
        return self.phase in [GaitPhase.LEFT_STANCE, GaitPhase.DOUBLE_SUPPORT_LR]

    def is_right_stance(self):
        return self.phase in [GaitPhase.RIGHT_STANCE, GaitPhase.DOUBLE_SUPPORT_RL]
```

### Walking Controller

```python
class WalkingController:
    """Basic walking controller for humanoid."""

    def __init__(self, humanoid, step_length=0.2, step_height=0.05):
        self.humanoid = humanoid
        self.gait = GaitStateMachine()
        self.trajectory_gen = FootTrajectoryGenerator(
            step_length=step_length,
            step_height=step_height
        )

        # Foot positions (relative to pelvis)
        self.hip_width = 0.15
        self.left_foot = np.array([0, self.hip_width, 0])
        self.right_foot = np.array([0, -self.hip_width, 0])

        # Target foot positions
        self.left_target = self.left_foot.copy()
        self.right_target = self.right_foot.copy()

    def update(self, dt, velocity_cmd):
        """
        Update walking controller.

        Args:
            dt: time step
            velocity_cmd: [vx, vy, omega] velocity command

        Returns:
            joint_targets: target joint positions
        """
        # Update gait state machine
        self.gait.update(dt)

        # Calculate step targets based on velocity
        step_length = velocity_cmd[0] * self.gait.cycle_time

        # Get foot trajectories based on current phase
        phase_ratio = self.gait.get_phase_ratio()

        if self.gait.phase == GaitPhase.LEFT_STANCE:
            # Right foot swinging
            self.right_foot = self.trajectory_gen.generate_swing_trajectory(
                self.right_foot, self.right_target, phase_ratio
            )
        elif self.gait.phase == GaitPhase.RIGHT_STANCE:
            # Left foot swinging
            self.left_foot = self.trajectory_gen.generate_swing_trajectory(
                self.left_foot, self.left_target, phase_ratio
            )

        # Use IK to compute joint angles
        left_joint_angles = self.compute_leg_ik(self.left_foot, 'left')
        right_joint_angles = self.compute_leg_ik(self.right_foot, 'right')

        # Combine into full joint target vector
        joint_targets = self.humanoid.get_joint_positions()
        joint_targets = self.set_leg_joints(joint_targets, left_joint_angles, 'left')
        joint_targets = self.set_leg_joints(joint_targets, right_joint_angles, 'right')

        return joint_targets

    def compute_leg_ik(self, foot_pos, side):
        """Compute leg IK (simplified analytical solution)."""
        # This is a placeholder - actual IK depends on robot kinematics
        hip_to_foot = foot_pos.copy()

        # Simple geometric IK for 6-DOF leg
        # In practice, use proper IK solver

        return np.zeros(6)  # Placeholder

    def set_leg_joints(self, all_joints, leg_joints, side):
        """Set leg joint values in full joint vector."""
        # Map leg joints to indices based on robot model
        return all_joints
```

## Linear Inverted Pendulum Model (LIPM)

### LIPM Dynamics

```python
class LIPMController:
    """Linear Inverted Pendulum Model for walking."""

    def __init__(self, com_height=0.9, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity

        # Natural frequency
        self.omega = np.sqrt(gravity / com_height)

    def compute_capture_point(self, com_pos, com_vel):
        """
        Compute capture point (Divergent Component of Motion).

        The capture point is where the robot needs to step to stop.

        Args:
            com_pos: [x, y] COM position
            com_vel: [vx, vy] COM velocity

        Returns:
            capture_point: [x, y] capture point
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega

        return np.array([cp_x, cp_y])

    def compute_desired_cop(self, com_pos, com_vel, target_vel):
        """
        Compute desired COP to achieve target velocity.

        Args:
            com_pos: current COM position
            com_vel: current COM velocity
            target_vel: desired COM velocity

        Returns:
            cop: desired center of pressure
        """
        # P control on capture point
        kp = 2.0

        cp = self.compute_capture_point(com_pos, com_vel)
        target_cp = com_pos + target_vel / self.omega

        cop = cp - kp * (cp - target_cp)

        return cop

    def predict_trajectory(self, com_pos, com_vel, cop, duration, dt=0.01):
        """
        Predict COM trajectory under constant COP.

        Args:
            com_pos: initial COM position
            com_vel: initial COM velocity
            cop: center of pressure
            duration: prediction horizon
            dt: time step

        Returns:
            trajectory: list of (pos, vel) tuples
        """
        trajectory = []
        pos = com_pos.copy()
        vel = com_vel.copy()

        for t in np.arange(0, duration, dt):
            # LIPM dynamics: x'' = omega^2 * (x - cop)
            acc = self.omega**2 * (pos - cop)
            vel = vel + acc * dt
            pos = pos + vel * dt

            trajectory.append((pos.copy(), vel.copy()))

        return trajectory
```

## Complete Walking Demo

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
import numpy as np

class HumanoidWalkingDemo:
    def __init__(self, world, humanoid_path):
        self.world = world

        # Create articulation
        self.humanoid = ArticulationView(
            prim_paths_expr=humanoid_path,
            name="humanoid"
        )
        world.scene.add(self.humanoid)

        # Controllers
        self.gait = GaitStateMachine(cycle_time=1.0)
        self.trajectory_gen = FootTrajectoryGenerator()
        self.lipm = LIPMController(com_height=0.9)

        # State
        self.left_foot_pos = np.array([0, 0.15, 0])
        self.right_foot_pos = np.array([0, -0.15, 0])

    def initialize(self):
        self.humanoid.initialize()

        # Set initial standing pose
        standing_pose = self.get_standing_pose()
        self.humanoid.set_joint_positions(standing_pose)

        # Set gains
        kps = np.full(self.humanoid.num_dof, 500.0)
        kds = np.full(self.humanoid.num_dof, 50.0)
        self.humanoid.set_gains(kps=kps, kds=kds)

    def get_standing_pose(self):
        """Get joint positions for stable standing."""
        return np.zeros(self.humanoid.num_dof)

    def update(self, dt, velocity_cmd):
        """Update walking controller."""
        # Get current state
        com_pos = self.estimate_com()
        com_vel = self.estimate_com_velocity()

        # Update gait timing
        self.gait.update(dt)

        # Compute foot targets
        phase = self.gait.get_phase_ratio()

        if self.gait.is_left_stance():
            # Swing right foot
            target = self.compute_step_target('right', velocity_cmd)
            self.right_foot_pos = self.trajectory_gen.generate_swing_trajectory(
                self.right_foot_pos, target, phase
            )
        else:
            # Swing left foot
            target = self.compute_step_target('left', velocity_cmd)
            self.left_foot_pos = self.trajectory_gen.generate_swing_trajectory(
                self.left_foot_pos, target, phase
            )

        # Compute joint targets via IK
        joint_targets = self.compute_joint_targets()

        # Apply to robot
        self.humanoid.set_joint_position_targets(joint_targets)

    def estimate_com(self):
        """Estimate center of mass position."""
        pose, _ = self.humanoid.get_world_poses()
        return pose[0][:2]  # Just x, y

    def estimate_com_velocity(self):
        """Estimate center of mass velocity."""
        vel = self.humanoid.get_velocities()
        return vel[0][:2]  # Just vx, vy

    def compute_step_target(self, foot, velocity_cmd):
        """Compute target step location."""
        step_length = velocity_cmd[0] * self.gait.cycle_time

        if foot == 'left':
            target = self.left_foot_pos + np.array([step_length, 0, 0])
        else:
            target = self.right_foot_pos + np.array([step_length, 0, 0])

        return target

    def compute_joint_targets(self):
        """Compute joint targets from foot positions via IK."""
        # Placeholder - implement actual IK
        return np.zeros(self.humanoid.num_dof)

# Main
world = World(physics_dt=1/240.0, rendering_dt=1/60.0)
world.scene.add_default_ground_plane()

# Import humanoid (adjust path)
# demo = HumanoidWalkingDemo(world, "/World/Humanoid")

world.reset()
# demo.initialize()

velocity_cmd = np.array([0.3, 0, 0])  # 0.3 m/s forward

while simulation_app.is_running():
    world.step(render=True)
    # demo.update(1/240.0, velocity_cmd)

simulation_app.close()
```

## Summary

In this lesson, you learned:

- Bipedal walking phases and dynamics
- Zero Moment Point for stability analysis
- Foot trajectory generation for walking
- Gait state machine implementation
- Linear Inverted Pendulum Model basics

## Next Steps

Continue to [Week 8 Exercises](/module-3/week-8/exercises) to practice humanoid control.
