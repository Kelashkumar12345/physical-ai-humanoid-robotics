---
sidebar_position: 1
title: Humanoid Robot Models
description: Working with humanoid robots in Isaac Sim
---

# Humanoid Robot Models in Isaac Sim

Learn to import, configure, and simulate humanoid robots in NVIDIA Isaac Sim.

## Learning Objectives

By the end of this lesson, you will:

- Understand humanoid robot kinematics
- Import humanoid models from URDF/MJCF
- Configure joint controllers for humanoids
- Set up inverse kinematics for manipulation

## Humanoid Robot Architecture

### Typical Humanoid Structure

```
Humanoid Robot
├── Torso (base)
│   ├── Head
│   │   └── Neck joints (2 DOF)
│   ├── Left Arm (7 DOF)
│   │   ├── Shoulder (3 DOF)
│   │   ├── Elbow (1 DOF)
│   │   ├── Wrist (3 DOF)
│   │   └── Hand/Gripper
│   ├── Right Arm (7 DOF)
│   └── Waist joints (2-3 DOF)
├── Left Leg (6 DOF)
│   ├── Hip (3 DOF)
│   ├── Knee (1 DOF)
│   └── Ankle (2 DOF)
└── Right Leg (6 DOF)
```

### Degrees of Freedom

| Robot Type | Total DOF | Locomotion | Manipulation |
|------------|-----------|------------|--------------|
| Simple biped | 12 | 12 | 0 |
| Basic humanoid | 25 | 12 | 13 |
| Advanced humanoid | 40+ | 12 | 28+ |
| Full dexterous | 50+ | 12 | 38+ |

## Available Humanoid Models

### NVIDIA Isaac Gym Assets

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path

assets_root = get_assets_root_path()

humanoid_assets = {
    "humanoid": f"{assets_root}/Isaac/Robots/Humanoid/humanoid.usd",
    "allegro_hand": f"{assets_root}/Isaac/Robots/AllegroHand/allegro_hand.usd",
}
```

### Popular Open-Source Humanoids

| Robot | DOF | Source | Format |
|-------|-----|--------|--------|
| Atlas | 30 | Boston Dynamics | URDF |
| Digit | 30 | Agility Robotics | URDF/MJCF |
| Cassie | 20 | Agility Robotics | URDF/MJCF |
| Talos | 32 | PAL Robotics | URDF |
| iCub | 53 | IIT | URDF |
| Unitree H1 | 19 | Unitree | URDF |

## Importing Humanoid Models

### From URDF

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.urdf import _urdf

# Create world
world = World()
world.scene.add_default_ground_plane()

# URDF import configuration for humanoid
urdf_config = _urdf.ImportConfig()
urdf_config.merge_fixed_joints = False  # Keep all joints
urdf_config.fix_base = False  # Allow floating base
urdf_config.make_default_prim = True
urdf_config.create_physics_scene = True
urdf_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

# Import humanoid URDF
result, prim_path = _urdf.import_robot(
    asset_root="/path/to/humanoid",
    asset_file="humanoid.urdf",
    import_config=urdf_config,
    position=[0, 0, 1.0]  # Start above ground
)

print(f"Humanoid imported at: {prim_path}")

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

### From MuJoCo (MJCF)

```python
from omni.isaac.mjcf import _mjcf

# MJCF import configuration
mjcf_config = _mjcf.ImportConfig()
mjcf_config.fix_base = False
mjcf_config.make_default_prim = True

# Import from MJCF
result, prim_path = _mjcf.import_robot(
    asset_root="/path/to/humanoid",
    asset_file="humanoid.xml",
    import_config=mjcf_config
)
```

## Humanoid Articulation

### Creating Articulation View

```python
from omni.isaac.core.articulations import ArticulationView
import numpy as np

class HumanoidRobot:
    def __init__(self, world, prim_path):
        self.world = world
        self.prim_path = prim_path

        # Create articulation view
        self.articulation = ArticulationView(
            prim_paths_expr=prim_path,
            name="humanoid"
        )
        world.scene.add(self.articulation)

        # Store joint information
        self.num_dof = None
        self.joint_names = None

    def initialize(self):
        self.articulation.initialize()
        self.num_dof = self.articulation.num_dof
        self.joint_names = self.articulation.dof_names

        print(f"Humanoid initialized with {self.num_dof} DOF")
        print(f"Joints: {self.joint_names}")

    def get_joint_positions(self):
        return self.articulation.get_joint_positions()

    def get_joint_velocities(self):
        return self.articulation.get_joint_velocities()

    def set_joint_positions(self, positions):
        self.articulation.set_joint_positions(positions)

    def set_joint_velocities(self, velocities):
        self.articulation.set_joint_velocities(velocities)

    def apply_joint_efforts(self, efforts):
        self.articulation.apply_joint_efforts(efforts)
```

### Joint Grouping

```python
class HumanoidJointGroups:
    """Organize humanoid joints into functional groups."""

    def __init__(self, joint_names):
        self.all_joints = joint_names

        # Define joint groups (adjust based on your robot)
        self.groups = {
            'left_arm': [
                'left_shoulder_pitch',
                'left_shoulder_roll',
                'left_shoulder_yaw',
                'left_elbow',
                'left_wrist_roll',
                'left_wrist_pitch',
                'left_wrist_yaw'
            ],
            'right_arm': [
                'right_shoulder_pitch',
                'right_shoulder_roll',
                'right_shoulder_yaw',
                'right_elbow',
                'right_wrist_roll',
                'right_wrist_pitch',
                'right_wrist_yaw'
            ],
            'left_leg': [
                'left_hip_yaw',
                'left_hip_roll',
                'left_hip_pitch',
                'left_knee',
                'left_ankle_pitch',
                'left_ankle_roll'
            ],
            'right_leg': [
                'right_hip_yaw',
                'right_hip_roll',
                'right_hip_pitch',
                'right_knee',
                'right_ankle_pitch',
                'right_ankle_roll'
            ],
            'torso': [
                'waist_yaw',
                'waist_pitch'
            ],
            'head': [
                'neck_yaw',
                'neck_pitch'
            ]
        }

        # Create index mappings
        self.group_indices = {}
        for group_name, group_joints in self.groups.items():
            indices = []
            for joint in group_joints:
                if joint in self.all_joints:
                    indices.append(self.all_joints.index(joint))
            self.group_indices[group_name] = indices

    def get_group_positions(self, all_positions, group_name):
        """Get positions for a specific joint group."""
        indices = self.group_indices[group_name]
        return all_positions[indices]

    def set_group_positions(self, all_positions, group_name, group_positions):
        """Set positions for a specific joint group."""
        indices = self.group_indices[group_name]
        all_positions[indices] = group_positions
        return all_positions
```

## Joint Control

### Position Control

```python
from omni.isaac.core.controllers import ArticulationController

class HumanoidController:
    def __init__(self, articulation):
        self.articulation = articulation

        # Create controller
        self.controller = ArticulationController(
            name="humanoid_controller",
            articulation=articulation
        )

        # Default gains
        self.kp = 1000.0  # Position gain
        self.kd = 100.0   # Damping gain

    def set_gains(self, kp, kd):
        """Set PD gains for all joints."""
        num_dof = self.articulation.num_dof
        self.articulation.set_gains(
            kps=np.full(num_dof, kp),
            kds=np.full(num_dof, kd)
        )

    def move_to_position(self, target_positions):
        """Move all joints to target positions."""
        self.articulation.set_joint_position_targets(target_positions)

    def move_group(self, group_name, target_positions):
        """Move specific joint group."""
        current = self.articulation.get_joint_positions()
        indices = self.joint_groups.group_indices[group_name]
        current[indices] = target_positions
        self.move_to_position(current)
```

### Effort Control

```python
class HumanoidEffortController:
    def __init__(self, articulation):
        self.articulation = articulation

    def apply_efforts(self, efforts):
        """Apply torques to all joints."""
        self.articulation.set_joint_efforts(efforts)

    def gravity_compensation(self):
        """Compute gravity compensation torques."""
        # Get current positions
        positions = self.articulation.get_joint_positions()

        # Compute gravity torques (simplified)
        # In practice, use inverse dynamics
        gravity_torques = self.compute_gravity_torques(positions)

        return gravity_torques

    def compute_gravity_torques(self, positions):
        """Compute gravity compensation (placeholder)."""
        # This would use the robot's mass matrix and gravity vector
        # For now, return zeros
        return np.zeros(self.articulation.num_dof)
```

## Standing Balance

### Basic Balance Controller

```python
class BalanceController:
    """Simple balance controller for humanoid standing."""

    def __init__(self, humanoid):
        self.humanoid = humanoid

        # Balance parameters
        self.com_height = 0.9  # Target COM height
        self.ankle_kp = 500.0
        self.ankle_kd = 50.0

    def compute_balance_torques(self):
        """Compute torques to maintain balance."""
        # Get current state
        base_pose = self.humanoid.get_base_pose()
        base_vel = self.humanoid.get_base_velocity()

        # Compute COM position error
        com_pos = self.estimate_com_position()
        com_error = self.com_height - com_pos[2]

        # Simple ankle strategy
        # In practice, use full-body inverse dynamics
        ankle_torque = self.ankle_kp * com_error

        # Distribute to ankle joints
        torques = np.zeros(self.humanoid.num_dof)
        left_ankle_idx = self.humanoid.get_joint_index('left_ankle_pitch')
        right_ankle_idx = self.humanoid.get_joint_index('right_ankle_pitch')

        torques[left_ankle_idx] = ankle_torque / 2
        torques[right_ankle_idx] = ankle_torque / 2

        return torques

    def estimate_com_position(self):
        """Estimate center of mass position."""
        # Simplified: use base position
        # In practice, compute from all link masses
        base_pos, _ = self.humanoid.articulation.get_world_poses()
        return base_pos[0]
```

## Inverse Kinematics

### IK for Arm Manipulation

```python
from omni.isaac.motion_generation import LulaKinematicsSolver

class HumanoidIK:
    def __init__(self, robot_description_path, urdf_path):
        # Create IK solver
        self.ik_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        # Arm end effector frames
        self.left_hand_frame = "left_hand"
        self.right_hand_frame = "right_hand"

    def solve_left_arm(self, target_position, target_orientation=None):
        """Solve IK for left arm."""
        return self.ik_solver.compute_inverse_kinematics(
            frame_name=self.left_hand_frame,
            target_position=target_position,
            target_orientation=target_orientation
        )

    def solve_right_arm(self, target_position, target_orientation=None):
        """Solve IK for right arm."""
        return self.ik_solver.compute_inverse_kinematics(
            frame_name=self.right_hand_frame,
            target_position=target_position,
            target_orientation=target_orientation
        )

    def reach_target(self, target_pos, arm='right'):
        """Move arm end effector to target position."""
        if arm == 'right':
            joint_positions = self.solve_right_arm(target_pos)
        else:
            joint_positions = self.solve_left_arm(target_pos)

        return joint_positions
```

## Complete Humanoid Example

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.urdf import _urdf
import numpy as np

class SimpleHumanoid:
    def __init__(self, world, urdf_path, position=[0, 0, 1.0]):
        self.world = world

        # Import URDF
        urdf_config = _urdf.ImportConfig()
        urdf_config.fix_base = False
        _, self.prim_path = _urdf.import_robot(
            asset_root=urdf_path.rsplit('/', 1)[0],
            asset_file=urdf_path.rsplit('/', 1)[1],
            import_config=urdf_config,
            position=position
        )

        # Create articulation
        self.articulation = ArticulationView(
            prim_paths_expr=self.prim_path,
            name="humanoid"
        )
        world.scene.add(self.articulation)

    def initialize(self):
        self.articulation.initialize()

        # Set default pose
        default_positions = np.zeros(self.articulation.num_dof)
        self.articulation.set_joint_positions(default_positions)

        # Set PD gains
        kps = np.full(self.articulation.num_dof, 1000.0)
        kds = np.full(self.articulation.num_dof, 100.0)
        self.articulation.set_gains(kps=kps, kds=kds)

    def stand(self):
        """Move to standing pose."""
        standing_pose = self.get_standing_pose()
        self.articulation.set_joint_position_targets(standing_pose)

    def get_standing_pose(self):
        """Get joint positions for standing."""
        positions = np.zeros(self.articulation.num_dof)
        # Adjust specific joints for standing
        # This depends on the robot model
        return positions

    def wave(self, arm='right'):
        """Simple wave motion."""
        shoulder_idx = self.get_joint_index(f'{arm}_shoulder_pitch')
        elbow_idx = self.get_joint_index(f'{arm}_elbow')

        targets = self.articulation.get_joint_positions()
        targets[shoulder_idx] = -1.5  # Raise arm
        targets[elbow_idx] = 1.0      # Bend elbow

        self.articulation.set_joint_position_targets(targets)

    def get_joint_index(self, joint_name):
        """Get index of joint by name."""
        names = self.articulation.dof_names
        if joint_name in names:
            return names.index(joint_name)
        return None

# Main
world = World()
world.scene.add_default_ground_plane()

# Create humanoid (adjust path to your model)
humanoid = SimpleHumanoid(
    world,
    urdf_path="/path/to/humanoid.urdf",
    position=[0, 0, 1.0]
)

world.reset()
humanoid.initialize()

# Simulation loop
step = 0
while simulation_app.is_running():
    world.step(render=True)

    if step == 100:
        humanoid.stand()
    elif step == 300:
        humanoid.wave('right')

    step += 1

simulation_app.close()
```

## Summary

In this lesson, you learned:

- Humanoid robot architecture and joint organization
- Importing humanoid models from URDF and MJCF
- Creating articulation views for control
- Joint grouping for organized control
- Basic balance and inverse kinematics concepts

## Next Steps

Continue to [Humanoid Locomotion](/module-3/week-8/humanoid-locomotion) to learn walking and movement control.
