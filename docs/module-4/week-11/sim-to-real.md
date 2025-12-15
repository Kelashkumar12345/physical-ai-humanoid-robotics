---
sidebar_position: 2
title: Sim-to-Real Transfer
description: Deploying simulation-trained policies on real robots
---

# Sim-to-Real Transfer

Learn techniques for successfully deploying policies trained in simulation to real robot hardware.

## Learning Objectives

By the end of this lesson, you will:

- Understand the sim-to-real gap
- Implement domain randomization strategies
- Apply system identification techniques
- Deploy and debug policies on real hardware

## The Sim-to-Real Gap

### Sources of Reality Gap

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sim-to-Real Gap Sources                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Perception    │    │    Dynamics     │    │   Control   │ │
│  │      Gap        │    │      Gap        │    │     Gap     │ │
│  └────────┬────────┘    └────────┬────────┘    └──────┬──────┘ │
│           │                      │                    │         │
│  • Lighting         • Friction           • Latency              │
│  • Camera noise     • Mass/inertia       • Discretization       │
│  • Occlusion        • Joint dynamics     • Actuator limits      │
│  • Textures         • Contact forces     • Communication        │
│  • Lens distortion  • Damping            • PD gains             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Mitigation Strategies                      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  1. Domain Randomization (DR)                               ││
│  │  2. System Identification (SysID)                           ││
│  │  3. Domain Adaptation                                       ││
│  │  4. Real-world fine-tuning                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Domain Randomization

### Visual Randomization

```python
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class VisualRandomizationConfig:
    """Configuration for visual domain randomization."""

    # Lighting
    brightness_range: tuple = (0.7, 1.3)
    contrast_range: tuple = (0.8, 1.2)

    # Colors
    hue_shift_range: tuple = (-20, 20)
    saturation_range: tuple = (0.8, 1.2)

    # Noise
    gaussian_noise_std: tuple = (0, 0.05)
    salt_pepper_prob: float = 0.01

    # Geometric
    random_crop_scale: tuple = (0.9, 1.0)
    rotation_range: tuple = (-5, 5)

    # Occlusion
    random_patch_prob: float = 0.3
    num_patches: int = 3


class VisualRandomizer:
    """Apply visual domain randomization to images."""

    def __init__(self, config: VisualRandomizationConfig):
        self.config = config

    def randomize(self, image):
        """Apply random visual transformations."""
        image = image.astype(np.float32) / 255.0

        # Brightness and contrast
        brightness = np.random.uniform(*self.config.brightness_range)
        contrast = np.random.uniform(*self.config.contrast_range)
        image = contrast * (image - 0.5) + 0.5 + (brightness - 1.0)

        # Color jitter
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_shift = np.random.uniform(*self.config.hue_shift_range)
        sat_scale = np.random.uniform(*self.config.saturation_range)
        image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_shift) % 180
        image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * sat_scale, 0, 1)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Gaussian noise
        noise_std = np.random.uniform(*self.config.gaussian_noise_std)
        noise = np.random.randn(*image.shape) * noise_std
        image = np.clip(image + noise, 0, 1)

        # Random patches (occlusion)
        if np.random.random() < self.config.random_patch_prob:
            image = self._add_random_patches(image)

        return (image * 255).astype(np.uint8)

    def _add_random_patches(self, image):
        """Add random rectangular patches as occlusion."""
        h, w = image.shape[:2]

        for _ in range(self.config.num_patches):
            # Random patch size and position
            patch_h = np.random.randint(h // 20, h // 5)
            patch_w = np.random.randint(w // 20, w // 5)
            y = np.random.randint(0, h - patch_h)
            x = np.random.randint(0, w - patch_w)

            # Random color
            color = np.random.rand(3)
            image[y:y+patch_h, x:x+patch_w] = color

        return image
```

### Physics Randomization

```python
@dataclass
class PhysicsRandomizationConfig:
    """Configuration for physics domain randomization."""

    # Mass
    mass_scale_range: tuple = (0.8, 1.2)

    # Friction
    friction_range: tuple = (0.5, 1.5)
    rolling_friction_range: tuple = (0.001, 0.01)

    # Joint dynamics
    damping_scale_range: tuple = (0.8, 1.2)
    stiffness_scale_range: tuple = (0.9, 1.1)

    # External forces
    external_force_magnitude: tuple = (0, 50)  # Newtons
    force_application_prob: float = 0.1

    # Motor parameters
    motor_strength_range: tuple = (0.9, 1.1)
    action_delay_range: tuple = (0, 3)  # timesteps


class PhysicsRandomizer:
    """Apply physics domain randomization to simulation."""

    def __init__(self, config: PhysicsRandomizationConfig, sim):
        self.config = config
        self.sim = sim

    def randomize_episode(self):
        """Randomize physics parameters at episode start."""
        self._randomize_masses()
        self._randomize_friction()
        self._randomize_joint_dynamics()
        self._randomize_motor()

    def _randomize_masses(self):
        """Randomize body masses."""
        for body_id in range(self.sim.num_bodies):
            original_mass = self.sim.get_body_mass(body_id)
            scale = np.random.uniform(*self.config.mass_scale_range)
            self.sim.set_body_mass(body_id, original_mass * scale)

    def _randomize_friction(self):
        """Randomize surface friction."""
        for geom_id in range(self.sim.num_geoms):
            friction = np.random.uniform(*self.config.friction_range)
            rolling = np.random.uniform(*self.config.rolling_friction_range)
            self.sim.set_geom_friction(geom_id, friction, rolling)

    def _randomize_joint_dynamics(self):
        """Randomize joint damping and stiffness."""
        for joint_id in range(self.sim.num_joints):
            damping = self.sim.get_joint_damping(joint_id)
            stiffness = self.sim.get_joint_stiffness(joint_id)

            damping_scale = np.random.uniform(*self.config.damping_scale_range)
            stiffness_scale = np.random.uniform(*self.config.stiffness_scale_range)

            self.sim.set_joint_damping(joint_id, damping * damping_scale)
            self.sim.set_joint_stiffness(joint_id, stiffness * stiffness_scale)

    def _randomize_motor(self):
        """Randomize motor strength."""
        scale = np.random.uniform(*self.config.motor_strength_range)
        self.motor_strength = scale

        delay = np.random.randint(*self.config.action_delay_range)
        self.action_delay = delay

    def step(self, action):
        """Apply action with randomized dynamics."""
        # Apply motor strength scaling
        scaled_action = action * self.motor_strength

        # Apply action delay
        if self.action_delay > 0:
            if not hasattr(self, 'action_buffer'):
                self.action_buffer = [np.zeros_like(action)] * self.action_delay

            self.action_buffer.append(scaled_action)
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = scaled_action

        # Random external force
        if np.random.random() < self.config.force_application_prob:
            magnitude = np.random.uniform(*self.config.external_force_magnitude)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            self.sim.apply_external_force(magnitude * direction)

        return self.sim.step(delayed_action)
```

### Automatic Domain Randomization (ADR)

```python
class AutomaticDomainRandomization:
    """Automatically adjust randomization ranges based on performance."""

    def __init__(self, initial_ranges, adjustment_rate=0.1):
        self.ranges = initial_ranges.copy()
        self.adjustment_rate = adjustment_rate

        # Performance tracking
        self.success_history = []
        self.window_size = 100

        # Thresholds
        self.lower_threshold = 0.3  # Narrow ranges if success < 30%
        self.upper_threshold = 0.8  # Expand ranges if success > 80%

    def update(self, success):
        """Update ranges based on episode success."""
        self.success_history.append(float(success))

        if len(self.success_history) < self.window_size:
            return

        # Compute recent success rate
        recent_success = np.mean(self.success_history[-self.window_size:])

        if recent_success < self.lower_threshold:
            self._narrow_ranges()
        elif recent_success > self.upper_threshold:
            self._expand_ranges()

    def _narrow_ranges(self):
        """Narrow randomization ranges (easier)."""
        for key, (low, high) in self.ranges.items():
            center = (low + high) / 2
            width = (high - low) / 2

            # Reduce width
            new_width = width * (1 - self.adjustment_rate)

            self.ranges[key] = (center - new_width, center + new_width)

        print(f"ADR: Narrowed ranges")

    def _expand_ranges(self):
        """Expand randomization ranges (harder)."""
        for key, (low, high) in self.ranges.items():
            center = (low + high) / 2
            width = (high - low) / 2

            # Increase width
            new_width = width * (1 + self.adjustment_rate)

            self.ranges[key] = (center - new_width, center + new_width)

        print(f"ADR: Expanded ranges")

    def get_ranges(self):
        """Get current randomization ranges."""
        return self.ranges.copy()
```

## System Identification

### Parameter Estimation

```python
import scipy.optimize as opt

class SystemIdentification:
    """Identify physical parameters from real robot data."""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Parameters to identify
        self.param_names = [
            'mass_scale',
            'friction',
            'damping',
            'motor_strength'
        ]

        # Initial guesses
        self.initial_params = [1.0, 0.8, 0.5, 1.0]

        # Bounds
        self.bounds = [
            (0.5, 2.0),   # mass_scale
            (0.1, 2.0),   # friction
            (0.1, 2.0),   # damping
            (0.5, 1.5),   # motor_strength
        ]

    def collect_calibration_data(self, robot, trajectories):
        """Collect real robot data for system identification."""
        calibration_data = []

        for trajectory in trajectories:
            # Execute trajectory on real robot
            real_data = robot.execute_trajectory(trajectory)

            calibration_data.append({
                'commands': trajectory,
                'joint_positions': real_data['positions'],
                'joint_velocities': real_data['velocities'],
                'timestamps': real_data['timestamps']
            })

        return calibration_data

    def simulate_with_params(self, params, trajectory):
        """Simulate trajectory with given parameters."""
        # Apply parameters to simulation
        self.robot_model.set_mass_scale(params[0])
        self.robot_model.set_friction(params[1])
        self.robot_model.set_damping(params[2])
        self.robot_model.set_motor_strength(params[3])

        # Simulate
        sim_positions = []
        sim_velocities = []

        self.robot_model.reset()

        for command in trajectory:
            self.robot_model.step(command)
            sim_positions.append(self.robot_model.get_joint_positions())
            sim_velocities.append(self.robot_model.get_joint_velocities())

        return np.array(sim_positions), np.array(sim_velocities)

    def objective(self, params, calibration_data):
        """Compute simulation-to-real error."""
        total_error = 0

        for data in calibration_data:
            sim_pos, sim_vel = self.simulate_with_params(
                params, data['commands']
            )

            # Position error
            pos_error = np.mean((sim_pos - data['joint_positions'])**2)

            # Velocity error
            vel_error = np.mean((sim_vel - data['joint_velocities'])**2)

            total_error += pos_error + 0.1 * vel_error

        return total_error

    def identify(self, calibration_data):
        """Run system identification optimization."""
        result = opt.minimize(
            self.objective,
            self.initial_params,
            args=(calibration_data,),
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100}
        )

        identified_params = dict(zip(self.param_names, result.x))

        print("Identified parameters:")
        for name, value in identified_params.items():
            print(f"  {name}: {value:.4f}")

        return identified_params
```

### Online Adaptation

```python
class OnlineSystemAdaptation:
    """Adapt simulation parameters online during deployment."""

    def __init__(self, initial_params, learning_rate=0.01):
        self.params = initial_params.copy()
        self.learning_rate = learning_rate

        # Prediction history
        self.prediction_buffer = []
        self.reality_buffer = []
        self.buffer_size = 100

    def predict(self, observation, action):
        """Predict next state using current parameters."""
        # Use current parameters for prediction
        predicted_state = self.model.predict(observation, action, self.params)

        self.prediction_buffer.append({
            'observation': observation,
            'action': action,
            'prediction': predicted_state
        })

        return predicted_state

    def observe(self, actual_state):
        """Observe actual state and update parameters."""
        self.reality_buffer.append(actual_state)

        # Update when buffer is full
        if len(self.reality_buffer) >= self.buffer_size:
            self._update_parameters()

    def _update_parameters(self):
        """Update parameters using prediction errors."""
        # Compute prediction errors
        errors = []
        for pred, actual in zip(self.prediction_buffer, self.reality_buffer):
            error = actual - pred['prediction']
            errors.append(error)

        errors = np.array(errors)

        # Compute parameter gradients (numerical approximation)
        gradients = self._compute_gradients(errors)

        # Update parameters
        for key in self.params:
            self.params[key] -= self.learning_rate * gradients[key]

        # Clear buffers
        self.prediction_buffer = []
        self.reality_buffer = []

    def _compute_gradients(self, errors):
        """Compute gradients via finite differences."""
        gradients = {}
        epsilon = 0.01

        for key, value in self.params.items():
            # Perturb parameter
            params_plus = self.params.copy()
            params_plus[key] = value + epsilon

            params_minus = self.params.copy()
            params_minus[key] = value - epsilon

            # Compute perturbed predictions
            error_plus = self._compute_error_with_params(params_plus)
            error_minus = self._compute_error_with_params(params_minus)

            # Finite difference gradient
            gradients[key] = (error_plus - error_minus) / (2 * epsilon)

        return gradients
```

## Real Robot Deployment

### Safety Wrapper

```python
class SafetyWrapper:
    """Safety wrapper for real robot deployment."""

    def __init__(self, robot, config):
        self.robot = robot
        self.config = config

        # Limits
        self.position_limits = config.position_limits
        self.velocity_limits = config.velocity_limits
        self.torque_limits = config.torque_limits
        self.workspace_bounds = config.workspace_bounds

        # Emergency stop flag
        self.e_stop = False

    def check_action(self, action):
        """Check if action is safe."""
        violations = []

        # Check joint limits
        current_pos = self.robot.get_joint_positions()
        next_pos = current_pos + action[:len(current_pos)]  # Assuming delta

        for i, (pos, (low, high)) in enumerate(zip(next_pos, self.position_limits)):
            if pos < low or pos > high:
                violations.append(f"Joint {i} position out of bounds")

        # Check velocity limits
        velocities = action / self.config.dt
        for i, (vel, limit) in enumerate(zip(velocities, self.velocity_limits)):
            if abs(vel) > limit:
                violations.append(f"Joint {i} velocity exceeds limit")

        # Check workspace
        ee_pos = self.robot.forward_kinematics(next_pos)
        if not self._in_workspace(ee_pos):
            violations.append("End-effector outside workspace")

        return len(violations) == 0, violations

    def _in_workspace(self, position):
        """Check if position is within workspace bounds."""
        for i, (pos, (low, high)) in enumerate(
            zip(position, self.workspace_bounds)
        ):
            if pos < low or pos > high:
                return False
        return True

    def safe_step(self, action):
        """Execute action with safety checks."""
        if self.e_stop:
            return None, "E-STOP active"

        safe, violations = self.check_action(action)

        if not safe:
            # Scale down action or stop
            scaled_action = self._scale_to_safe(action)
            print(f"Safety: Scaled action due to {violations}")
            action = scaled_action

        return self.robot.step(action)

    def _scale_to_safe(self, action):
        """Scale action to be within safety limits."""
        scale = 1.0

        current_pos = self.robot.get_joint_positions()
        next_pos = current_pos + action[:len(current_pos)]

        for i, (pos, (low, high)) in enumerate(zip(next_pos, self.position_limits)):
            if pos < low:
                required_scale = (low - current_pos[i]) / action[i]
                scale = min(scale, required_scale * 0.9)
            elif pos > high:
                required_scale = (high - current_pos[i]) / action[i]
                scale = min(scale, required_scale * 0.9)

        return action * scale

    def emergency_stop(self):
        """Trigger emergency stop."""
        self.e_stop = True
        self.robot.stop()
        print("EMERGENCY STOP ACTIVATED")

    def reset_e_stop(self):
        """Reset emergency stop."""
        self.e_stop = False
        print("E-stop reset")
```

### Deployment Pipeline

```python
class Sim2RealDeployment:
    """Complete sim-to-real deployment pipeline."""

    def __init__(self, policy, robot, config):
        self.policy = policy
        self.robot = robot
        self.config = config

        # Safety wrapper
        self.safety = SafetyWrapper(robot, config.safety)

        # Observation processing
        self.obs_normalizer = self._load_normalizer(config.normalizer_path)

        # Action processing
        self.action_smoothing = ExponentialSmoothing(config.smoothing_alpha)

        # Monitoring
        self.monitor = DeploymentMonitor()

    def _load_normalizer(self, path):
        """Load observation normalizer from training."""
        return torch.load(path)

    def preprocess_observation(self, raw_obs):
        """Preprocess real robot observation."""
        # Convert to tensor
        obs_tensor = torch.from_numpy(raw_obs).float()

        # Normalize using training statistics
        normalized = self.obs_normalizer.normalize(obs_tensor)

        return normalized

    def postprocess_action(self, raw_action):
        """Postprocess policy action for robot."""
        action = raw_action.cpu().numpy()

        # Smooth action
        smoothed = self.action_smoothing(action)

        # Clip to limits
        clipped = np.clip(
            smoothed,
            -self.config.max_action,
            self.config.max_action
        )

        return clipped

    def run_episode(self, task_instruction, max_steps=500):
        """Run deployment episode."""
        obs = self.robot.reset()
        total_reward = 0
        trajectory = []

        for step in range(max_steps):
            # Preprocess observation
            processed_obs = self.preprocess_observation(obs)

            # Get action from policy
            with torch.no_grad():
                raw_action = self.policy(processed_obs, task_instruction)

            # Postprocess action
            action = self.postprocess_action(raw_action)

            # Safety check and execute
            result = self.safety.safe_step(action)

            if result is None:
                print("Episode terminated due to safety violation")
                break

            obs, reward, done, info = result

            # Log
            trajectory.append({
                'observation': obs,
                'action': action,
                'reward': reward
            })
            total_reward += reward

            # Monitor
            self.monitor.update(obs, action, reward)

            if done:
                break

        return {
            'success': info.get('success', False),
            'total_reward': total_reward,
            'steps': step + 1,
            'trajectory': trajectory
        }

    def evaluate(self, tasks, episodes_per_task=10):
        """Evaluate policy on multiple tasks."""
        results = {}

        for task in tasks:
            task_results = []

            for ep in range(episodes_per_task):
                print(f"Task: {task}, Episode: {ep}")
                result = self.run_episode(task)
                task_results.append(result)

            results[task] = {
                'success_rate': np.mean([r['success'] for r in task_results]),
                'avg_reward': np.mean([r['total_reward'] for r in task_results]),
                'avg_steps': np.mean([r['steps'] for r in task_results])
            }

            print(f"  Success rate: {results[task]['success_rate']:.2%}")

        return results


class ExponentialSmoothing:
    """Exponential smoothing for action sequences."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_action = None

    def __call__(self, action):
        if self.prev_action is None:
            self.prev_action = action
            return action

        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed

        return smoothed
```

## Debugging and Monitoring

### Deployment Monitor

```python
import matplotlib.pyplot as plt
from collections import deque

class DeploymentMonitor:
    """Monitor and visualize deployment performance."""

    def __init__(self, window_size=100):
        self.window_size = window_size

        # Metrics buffers
        self.action_history = deque(maxlen=window_size)
        self.observation_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)

        # Performance tracking
        self.step_count = 0
        self.episode_count = 0

    def update(self, observation, action, reward):
        """Update monitoring buffers."""
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.step_count += 1

    def plot_diagnostics(self):
        """Plot diagnostic information."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Action distribution
        actions = np.array(list(self.action_history))
        axes[0, 0].boxplot(actions)
        axes[0, 0].set_title('Action Distribution')
        axes[0, 0].set_xlabel('Dimension')

        # Action smoothness (rate of change)
        if len(actions) > 1:
            action_diff = np.diff(actions, axis=0)
            axes[0, 1].plot(np.linalg.norm(action_diff, axis=1))
            axes[0, 1].set_title('Action Rate of Change')
            axes[0, 1].set_xlabel('Step')

        # Reward curve
        rewards = list(self.reward_history)
        axes[1, 0].plot(rewards)
        axes[1, 0].set_title('Reward History')
        axes[1, 0].set_xlabel('Step')

        # Cumulative reward
        cumulative = np.cumsum(rewards)
        axes[1, 1].plot(cumulative)
        axes[1, 1].set_title('Cumulative Reward')
        axes[1, 1].set_xlabel('Step')

        plt.tight_layout()
        plt.savefig('deployment_diagnostics.png')
        plt.show()

    def check_anomalies(self):
        """Check for deployment anomalies."""
        anomalies = []

        # Check action magnitude
        actions = np.array(list(self.action_history))
        if len(actions) > 10:
            action_magnitude = np.linalg.norm(actions, axis=1)
            if action_magnitude[-1] > 2 * np.mean(action_magnitude[:-10]):
                anomalies.append("Large action spike detected")

        # Check reward
        rewards = list(self.reward_history)
        if len(rewards) > 10:
            if np.mean(rewards[-10:]) < np.mean(rewards) * 0.5:
                anomalies.append("Reward degradation detected")

        return anomalies
```

## Summary

In this lesson, you learned:

- Understanding the sim-to-real gap sources
- Visual and physics domain randomization
- Automatic domain randomization (ADR)
- System identification techniques
- Safe deployment practices
- Monitoring and debugging tools

## Next Steps

Continue to [Week 11 Exercises](/module-4/week-11/exercises) to practice sim-to-real transfer techniques.
