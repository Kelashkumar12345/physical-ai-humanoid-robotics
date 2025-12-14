---
sidebar_position: 3
title: Week 9 Exercises
description: Hands-on exercises for reinforcement learning in Isaac Gym
---

# Week 9: Reinforcement Learning Exercises

Practice training robot policies with GPU-accelerated reinforcement learning.

## Exercise 1: Simple Pendulum Environment

**Objective**: Create and train a simple inverted pendulum environment.

### Requirements

1. Create pendulum environment in Isaac Gym
2. Define observation space (angle, angular velocity)
3. Define action space (torque)
4. Implement reward function (keep upright)
5. Train policy to balance pendulum

### Starter Code

```python
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

class PendulumEnv:
    def __init__(self, num_envs=1024, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # TODO: Set up simulation parameters

        # TODO: Create simulation

        # TODO: Create pendulum asset and environments

        # TODO: Initialize tensors

    def step(self, actions):
        # TODO: Apply torques

        # TODO: Step simulation

        # TODO: Compute observations

        # TODO: Compute rewards

        # TODO: Check termination

        pass

    def reset(self, env_ids=None):
        # TODO: Reset pendulum to random position

        pass

    def _compute_observations(self):
        """Return [angle, angular_velocity]."""
        # TODO: Get joint state

        pass

    def _compute_reward(self):
        """Reward for staying upright."""
        # TODO: Implement reward

        pass

# TODO: Train policy using PPO
```

<details className="solution-block">
<summary>Solution Hints</summary>
<div className="solution-content">

```python
def _compute_reward(self):
    # Get angle from horizontal (upright = 0)
    angle = self.dof_pos[:, 0]
    angular_vel = self.dof_vel[:, 0]

    # Reward for staying upright
    upright_reward = torch.cos(angle)

    # Penalty for high angular velocity
    vel_penalty = 0.1 * torch.square(angular_vel)

    # Penalty for control effort
    effort_penalty = 0.001 * torch.square(self.actions[:, 0])

    return upright_reward - vel_penalty - effort_penalty
```

</div>
</details>

---

## Exercise 2: Reward Function Design

**Objective**: Design and compare different reward functions for locomotion.

### Requirements

1. Implement 3 different reward formulations
2. Train policies with each reward
3. Compare learning curves
4. Analyze resulting behaviors

### Reward Variants

```python
class RewardVariants:
    @staticmethod
    def sparse_reward(env):
        """Sparse: reward only for reaching goal."""
        goal_reached = env.base_pos[:, 0] > env.goal_distance
        return goal_reached.float() * 10.0

    @staticmethod
    def dense_reward(env):
        """Dense: continuous feedback on progress."""
        # Forward velocity reward
        vel_reward = env.base_lin_vel[:, 0]

        # Survival reward
        alive_reward = 1.0

        return vel_reward + alive_reward

    @staticmethod
    def shaped_reward(env):
        """Shaped: comprehensive guidance."""
        # TODO: Implement multi-component reward
        # - Velocity tracking
        # - Orientation penalty
        # - Energy penalty
        # - Smoothness bonus

        pass
```

### Analysis Questions

1. Which reward learns fastest?
2. Which produces most natural motion?
3. How does reward shaping affect final performance?

---

## Exercise 3: Domain Randomization

**Objective**: Implement domain randomization and measure robustness.

### Requirements

1. Implement mass randomization
2. Implement friction randomization
3. Implement external force perturbations
4. Compare policy robustness with/without randomization

### Starter Code

```python
class RobustnessTester:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def test_mass_variation(self, mass_scales):
        """Test policy under different mass values."""
        results = {}

        for scale in mass_scales:
            self.env.set_mass_scale(scale)
            reward = self.evaluate()
            results[scale] = reward

        return results

    def test_friction_variation(self, friction_values):
        """Test policy under different friction."""
        # TODO: Implement

        pass

    def test_perturbation_resistance(self, force_magnitudes):
        """Test recovery from pushes."""
        # TODO: Implement

        pass

    def evaluate(self, num_episodes=100):
        """Evaluate policy performance."""
        total_reward = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.policy.get_action(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes

# TODO: Compare randomized vs non-randomized training
```

---

## Exercise 4: Curriculum Learning

**Objective**: Implement progressive difficulty curriculum.

### Requirements

1. Define difficulty levels for locomotion
2. Implement automatic level progression
3. Track learning progress per level
4. Compare with non-curriculum training

### Curriculum Definition

```python
class LocomotionCurriculum:
    def __init__(self):
        self.levels = [
            # Level 0: Standing
            {
                'command_range': [0, 0],
                'terrain': 'flat',
                'perturbation': 0,
            },
            # Level 1: Slow walking
            {
                'command_range': [0, 0.5],
                'terrain': 'flat',
                'perturbation': 0,
            },
            # Level 2: Fast walking
            {
                'command_range': [0, 1.0],
                'terrain': 'flat',
                'perturbation': 50,
            },
            # Level 3: Rough terrain
            {
                'command_range': [0, 1.0],
                'terrain': 'rough',
                'perturbation': 100,
            },
        ]

        self.current_level = 0
        self.level_success_history = []

    def update(self, success_rate):
        """Update level based on performance."""
        # TODO: Implement level progression logic

        pass

    def get_config(self):
        """Get current level configuration."""
        return self.levels[self.current_level]
```

---

## Exercise 5: Policy Visualization

**Objective**: Visualize trained policy behavior and learned features.

### Requirements

1. Record policy rollouts as video
2. Visualize action distributions
3. Plot reward components over time
4. Analyze failure cases

### Starter Code

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PolicyVisualizer:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.history = []

    def record_episode(self):
        """Record one episode of policy execution."""
        obs = self.env.reset()
        done = False

        frames = []
        actions = []
        rewards = []

        while not done:
            action = self.policy.get_action(obs)
            obs, reward, done, info = self.env.step(action)

            frames.append(self.env.render())
            actions.append(action.cpu().numpy())
            rewards.append(reward.cpu().numpy())

        return {
            'frames': frames,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
        }

    def plot_action_distribution(self, episode_data):
        """Plot action statistics over episode."""
        # TODO: Implement

        pass

    def plot_reward_breakdown(self, episode_data):
        """Plot individual reward components."""
        # TODO: Implement

        pass

    def create_video(self, episode_data, filename):
        """Save episode as video."""
        # TODO: Implement using matplotlib animation

        pass
```

---

## Challenge: Humanoid Walking from Scratch

**Objective**: Train a humanoid robot to walk using RL from random initialization.

### Requirements

1. Import humanoid model
2. Design comprehensive reward function
3. Implement curriculum learning
4. Train stable walking policy
5. Achieve forward velocity > 0.5 m/s

### Evaluation Metrics

| Metric | Target |
|--------|--------|
| Forward velocity | > 0.5 m/s |
| Survival time | > 1000 steps |
| Velocity tracking error | < 0.2 m/s |
| Energy efficiency | < 100 W |
| Fall recovery | > 50% success |

### Starter Structure

```python
# humanoid_walking.py

from isaacgym import gymapi, gymtorch
import torch

class HumanoidWalkingEnv:
    """Environment for humanoid walking."""

    def __init__(self, cfg):
        # TODO: Initialize environment

        pass

    def _compute_observations(self):
        # TODO: Include:
        # - Base state (pos, ori, vel)
        # - Joint states
        # - Foot contacts
        # - Commands

        pass

    def _compute_reward(self):
        # TODO: Implement reward with:
        # - Velocity tracking
        # - Balance (orientation)
        # - Energy efficiency
        # - Smoothness

        pass

def train_humanoid():
    """Train humanoid walking policy."""
    cfg = HumanoidConfig()
    env = HumanoidWalkingEnv(cfg)

    policy = ActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )

    trainer = PPOTrainer(env, policy, cfg)

    for iteration in range(cfg.max_iterations):
        rollout = trainer.collect_rollout()
        stats = trainer.update(rollout)

        if iteration % 100 == 0:
            print(f"Iter {iteration}: reward={rollout['rewards'].mean():.3f}")

    return policy

if __name__ == '__main__':
    policy = train_humanoid()
    torch.save(policy.state_dict(), 'humanoid_policy.pt')
```

### Tips

1. Start with a simple standing reward before locomotion
2. Use curriculum to gradually increase velocity commands
3. Add significant orientation penalty to prevent falling
4. Tune action rate penalty for smooth motion
5. Use domain randomization from the start

---

## Submission Checklist

- [ ] Pendulum environment works correctly
- [ ] Reward variants implemented and compared
- [ ] Domain randomization improves robustness
- [ ] Curriculum learning accelerates training
- [ ] Visualization tools functional
- [ ] Challenge: humanoid achieves target metrics

## Resources

- [Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- [Domain Randomization](https://arxiv.org/abs/1703.06907)
