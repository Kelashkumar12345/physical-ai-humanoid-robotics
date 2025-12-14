---
sidebar_position: 1
title: Isaac Gym Basics
description: GPU-accelerated reinforcement learning for robotics
---

# Isaac Gym for Reinforcement Learning

Learn to use NVIDIA Isaac Gym for massively parallel robot training with GPU acceleration.

## Learning Objectives

By the end of this lesson, you will:

- Understand Isaac Gym architecture
- Set up parallel simulation environments
- Define observation and action spaces
- Create basic RL training loops

## What is Isaac Gym?

Isaac Gym enables training thousands of robot instances in parallel on a single GPU:

| Feature | Benefit |
|---------|---------|
| **GPU Physics** | PhysX on GPU for massive parallelism |
| **Vectorized Environments** | Thousands of envs simultaneously |
| **Direct Tensor Access** | No CPU-GPU transfer overhead |
| **RL Integration** | Works with PyTorch/TensorFlow |

### Performance Comparison

| Framework | Environments | FPS |
|-----------|--------------|-----|
| MuJoCo (CPU) | 1 | ~1000 |
| PyBullet (CPU) | 8 | ~500 |
| Isaac Gym (GPU) | 4096 | ~50000+ |

## Installation

```bash
# Download Isaac Gym from NVIDIA
# https://developer.nvidia.com/isaac-gym

# Extract and install
cd isaacgym/python
pip install -e .

# Verify installation
python examples/joint_monkey.py
```

## Basic Structure

### Environment Template

```python
import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

class RobotEnv:
    def __init__(self, num_envs=4096, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Simulation parameters
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1/60
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.use_gpu = True

        # Create sim
        self.sim = self.gym.create_sim(
            0, 0, gymapi.SIM_PHYSX, self.sim_params
        )

        # Create environments
        self._create_envs()

        # Prepare tensors
        self.gym.prepare_sim(self.sim)
        self._init_tensors()

    def _create_envs(self):
        """Create parallel environments."""
        # Load robot asset
        asset_root = "assets"
        asset_file = "robot.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # Environment spacing
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Create robot actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)
            actor = self.gym.create_actor(env, robot_asset, pose, f"robot_{i}", i, 1)

            self.envs.append(env)
            self.actors.append(actor)

    def _init_tensors(self):
        """Initialize GPU tensors for state/action access."""
        # Get root states tensor
        self.root_states = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )

        # Get DOF states tensor
        self.dof_states = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )

        # Split into position and velocity
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)

    def step(self, actions):
        """Step simulation with actions."""
        # Apply actions (torques)
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(actions)
        )

        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Compute observations and rewards
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()

        return obs, rewards, dones, {}

    def reset(self, env_ids=None):
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # Reset DOF states
        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0

        # Apply resets
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        return self._compute_observations()

    def _compute_observations(self):
        """Compute observations from state."""
        # Example: joint positions and velocities
        return torch.cat([self.dof_pos, self.dof_vel], dim=-1)

    def _compute_rewards(self):
        """Compute reward for each environment."""
        # Placeholder - implement task-specific reward
        return torch.zeros(self.num_envs, device=self.device)

    def _compute_dones(self):
        """Compute termination conditions."""
        # Placeholder - implement termination logic
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def render(self):
        """Render visualization."""
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
```

## Observation Space

### Common Observations for Robots

```python
def _compute_observations(self):
    """Compute comprehensive robot observations."""
    obs_components = []

    # 1. Base state (position, orientation, velocities)
    base_pos = self.root_states[:, :3]
    base_quat = self.root_states[:, 3:7]
    base_lin_vel = self.root_states[:, 7:10]
    base_ang_vel = self.root_states[:, 10:13]

    obs_components.append(base_pos)
    obs_components.append(base_quat)
    obs_components.append(base_lin_vel)
    obs_components.append(base_ang_vel)

    # 2. Joint states
    obs_components.append(self.dof_pos)
    obs_components.append(self.dof_vel)

    # 3. Previous action
    obs_components.append(self.last_actions)

    # 4. Command (for locomotion)
    obs_components.append(self.commands)

    # 5. Body heights (for terrain awareness)
    if hasattr(self, 'measured_heights'):
        obs_components.append(self.measured_heights)

    # Concatenate all observations
    obs = torch.cat(obs_components, dim=-1)

    return obs
```

### Observation Normalization

```python
class ObservationNormalizer:
    def __init__(self, obs_dim, device="cuda:0"):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 1e-4

    def update(self, obs):
        """Update running statistics."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        """Normalize observations."""
        return (obs - self.mean) / (torch.sqrt(self.var) + 1e-8)
```

## Action Space

### Position vs Torque Control

```python
class ActionHandler:
    def __init__(self, env, control_type="torque"):
        self.env = env
        self.control_type = control_type

        # Get joint limits
        self.dof_limits = env.gym.get_actor_dof_properties(
            env.envs[0], env.actors[0]
        )

    def process_actions(self, raw_actions):
        """Convert network outputs to control signals."""
        if self.control_type == "torque":
            # Scale to torque limits
            torque_limits = torch.tensor(
                [d.effort for d in self.dof_limits],
                device=raw_actions.device
            )
            actions = raw_actions * torque_limits

        elif self.control_type == "position":
            # Scale to position limits
            lower = torch.tensor(
                [d.lower for d in self.dof_limits],
                device=raw_actions.device
            )
            upper = torch.tensor(
                [d.upper for d in self.dof_limits],
                device=raw_actions.device
            )
            # Tanh output in [-1, 1] -> position in [lower, upper]
            actions = lower + (raw_actions + 1) * 0.5 * (upper - lower)

        elif self.control_type == "position_delta":
            # Apply delta to current position
            delta_scale = 0.1  # Max change per step
            current_pos = self.env.dof_pos
            actions = current_pos + raw_actions * delta_scale

        return actions
```

## Reward Design

### Locomotion Reward Example

```python
def compute_locomotion_reward(self):
    """Compute reward for locomotion task."""
    rewards = {}

    # Target velocity tracking
    lin_vel_error = torch.sum(
        torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
        dim=1
    )
    rewards['tracking_lin_vel'] = torch.exp(-lin_vel_error / 0.25)

    # Angular velocity tracking
    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    rewards['tracking_ang_vel'] = torch.exp(-ang_vel_error / 0.25)

    # Base height
    height_error = torch.square(self.base_pos[:, 2] - self.target_height)
    rewards['base_height'] = torch.exp(-height_error / 0.1)

    # Orientation (stay upright)
    # Penalize roll and pitch
    roll, pitch, _ = self.euler_from_quat(self.base_quat)
    rewards['orientation'] = torch.exp(-torch.abs(roll) - torch.abs(pitch))

    # Energy efficiency
    rewards['energy'] = -torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    # Smooth actions
    action_diff = self.actions - self.last_actions
    rewards['action_smoothness'] = -torch.sum(torch.square(action_diff), dim=1)

    # Total reward (weighted sum)
    total_reward = (
        2.0 * rewards['tracking_lin_vel'] +
        1.0 * rewards['tracking_ang_vel'] +
        0.5 * rewards['base_height'] +
        0.5 * rewards['orientation'] +
        0.001 * rewards['energy'] +
        0.01 * rewards['action_smoothness']
    )

    return total_reward
```

## Training Loop

### Basic PPO Training

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Actor (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(self, obs, deterministic=False):
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_log_std)

        if deterministic:
            return action_mean, value

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


class PPOTrainer:
    def __init__(self, env, policy, device="cuda:0"):
        self.env = env
        self.policy = policy.to(device)
        self.device = device

        # PPO hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.learning_rate = 3e-4
        self.num_epochs = 5
        self.batch_size = 4096

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.learning_rate
        )

    def collect_rollout(self, num_steps):
        """Collect experience from parallel environments."""
        obs_buf = []
        action_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        log_prob_buf = []

        obs = self.env.reset()

        for step in range(num_steps):
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs)

            next_obs, reward, done, info = self.env.step(action)

            obs_buf.append(obs)
            action_buf.append(action)
            reward_buf.append(reward)
            done_buf.append(done)
            value_buf.append(value.squeeze())
            log_prob_buf.append(log_prob)

            obs = next_obs

            # Reset done environments
            if done.any():
                obs = self.env.reset(done.nonzero().squeeze())

        return {
            'obs': torch.stack(obs_buf),
            'actions': torch.stack(action_buf),
            'rewards': torch.stack(reward_buf),
            'dones': torch.stack(done_buf),
            'values': torch.stack(value_buf),
            'log_probs': torch.stack(log_prob_buf),
        }

    def compute_gae(self, rollout):
        """Compute Generalized Advantage Estimation."""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (~dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (~dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """Update policy with PPO."""
        advantages, returns = self.compute_gae(rollout)

        # Flatten batch
        obs = rollout['obs'].view(-1, rollout['obs'].shape[-1])
        actions = rollout['actions'].view(-1, rollout['actions'].shape[-1])
        old_log_probs = rollout['log_probs'].view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(self.num_epochs):
            # Random permutation
            perm = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = perm[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Forward pass
                action_mean, values = self.policy(batch_obs)
                action_std = torch.exp(self.policy.actor_log_std)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
```

## Summary

In this lesson, you learned:

- Isaac Gym's GPU-accelerated simulation architecture
- Creating parallel robot environments
- Defining observation and action spaces
- Designing reward functions for locomotion
- Basic PPO training loop implementation

## Next Steps

Continue to [RL Policy Training](/module-3/week-9/rl-policy-training) to train a walking policy.
