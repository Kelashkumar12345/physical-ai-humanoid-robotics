---
sidebar_position: 2
title: RL Policy Training
description: Training locomotion policies with reinforcement learning
---

# Training Locomotion Policies

Learn to train robust locomotion policies for humanoid robots using reinforcement learning.

## Learning Objectives

By the end of this lesson, you will:

- Design reward functions for locomotion
- Implement curriculum learning
- Use domain randomization for robustness
- Deploy trained policies

## Locomotion Task Setup

### Environment Configuration

```python
class HumanoidLocomotionEnv:
    def __init__(self, cfg):
        self.cfg = cfg

        # Task parameters
        self.max_episode_length = cfg.episode_length
        self.command_ranges = {
            'lin_vel_x': [-1.0, 1.0],   # m/s
            'lin_vel_y': [-0.5, 0.5],   # m/s
            'ang_vel': [-1.0, 1.0],     # rad/s
        }

        # Reward scales
        self.reward_scales = {
            'tracking_lin_vel': 1.0,
            'tracking_ang_vel': 0.5,
            'lin_vel_z': -2.0,
            'ang_vel_xy': -0.05,
            'orientation': -1.0,
            'base_height': -30.0,
            'torques': -0.0001,
            'dof_acc': -2.5e-7,
            'action_rate': -0.01,
            'collision': -1.0,
            'feet_air_time': 1.0,
            'stumble': -0.5,
        }

        # Termination conditions
        self.termination_height = 0.3
        self.termination_contact_forces = 100.0

    def _compute_observations(self):
        """Compute privileged observations."""
        obs = torch.cat([
            # Base state
            self.base_lin_vel * self.obs_scales['lin_vel'],
            self.base_ang_vel * self.obs_scales['ang_vel'],
            self.projected_gravity,

            # Commands
            self.commands[:, :3] * self.obs_scales['commands'],

            # Joint state
            (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
            self.dof_vel * self.obs_scales['dof_vel'],

            # Previous action
            self.actions,

            # Foot contacts
            self.feet_contact_filt.float(),
        ], dim=-1)

        return obs

    def _resample_commands(self, env_ids):
        """Resample velocity commands."""
        self.commands[env_ids, 0] = torch.uniform(
            self.command_ranges['lin_vel_x'][0],
            self.command_ranges['lin_vel_x'][1],
            (len(env_ids),), device=self.device
        )
        self.commands[env_ids, 1] = torch.uniform(
            self.command_ranges['lin_vel_y'][0],
            self.command_ranges['lin_vel_y'][1],
            (len(env_ids),), device=self.device
        )
        self.commands[env_ids, 2] = torch.uniform(
            self.command_ranges['ang_vel'][0],
            self.command_ranges['ang_vel'][1],
            (len(env_ids),), device=self.device
        )

        # Small commands set to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1, keepdim=True) > 0.2
        )
```

## Reward Function Design

### Comprehensive Locomotion Reward

```python
def compute_reward(self):
    """Compute comprehensive locomotion reward."""

    # Velocity tracking reward
    lin_vel_error = torch.sum(
        torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
    )
    lin_vel_reward = torch.exp(-lin_vel_error / 0.25)

    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    ang_vel_reward = torch.exp(-ang_vel_error / 0.25)

    # Penalize vertical velocity
    lin_vel_z_penalty = torch.square(self.base_lin_vel[:, 2])

    # Penalize xy angular velocity
    ang_vel_xy_penalty = torch.sum(
        torch.square(self.base_ang_vel[:, :2]), dim=1
    )

    # Orientation reward (stay upright)
    orientation_penalty = torch.sum(
        torch.square(self.projected_gravity[:, :2]), dim=1
    )

    # Base height reward
    base_height_error = torch.square(
        self.base_pos[:, 2] - self.cfg.base_height_target
    )

    # Energy penalty (torque * velocity)
    torque_penalty = torch.sum(torch.square(self.torques), dim=1)

    # Joint acceleration penalty (smooth motion)
    dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
    dof_acc_penalty = torch.sum(torch.square(dof_acc), dim=1)

    # Action rate penalty (smooth actions)
    action_rate_penalty = torch.sum(
        torch.square(self.actions - self.last_actions), dim=1
    )

    # Collision penalty
    collision_penalty = torch.sum(
        1.0 * (self.contact_forces[:, self.penalize_contacts] > 1.0), dim=1
    )

    # Feet air time reward (encourage stepping)
    first_contact = (self.feet_air_time > 0) * self.feet_contact_filt
    feet_air_time_reward = torch.sum(
        (self.feet_air_time - 0.5) * first_contact, dim=1
    )

    # Stumble penalty (lateral foot movement)
    stumble_penalty = torch.any(
        torch.norm(self.foot_velocities[:, :, :2], dim=2) > 2.0,
        dim=1
    ).float()

    # Combine rewards
    reward = (
        self.reward_scales['tracking_lin_vel'] * lin_vel_reward +
        self.reward_scales['tracking_ang_vel'] * ang_vel_reward +
        self.reward_scales['lin_vel_z'] * lin_vel_z_penalty +
        self.reward_scales['ang_vel_xy'] * ang_vel_xy_penalty +
        self.reward_scales['orientation'] * orientation_penalty +
        self.reward_scales['base_height'] * base_height_error +
        self.reward_scales['torques'] * torque_penalty +
        self.reward_scales['dof_acc'] * dof_acc_penalty +
        self.reward_scales['action_rate'] * action_rate_penalty +
        self.reward_scales['collision'] * collision_penalty +
        self.reward_scales['feet_air_time'] * feet_air_time_reward +
        self.reward_scales['stumble'] * stumble_penalty
    )

    return reward
```

### Reward Curriculum

```python
class RewardCurriculum:
    """Gradually increase reward complexity."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.current_level = 0
        self.levels = [
            # Level 0: Basic standing
            {
                'tracking_lin_vel': 0.0,
                'tracking_ang_vel': 0.0,
                'orientation': -5.0,
                'base_height': -50.0,
            },
            # Level 1: Slow walking
            {
                'tracking_lin_vel': 1.0,
                'tracking_ang_vel': 0.5,
                'orientation': -2.0,
                'base_height': -30.0,
            },
            # Level 2: Full locomotion
            {
                'tracking_lin_vel': 2.0,
                'tracking_ang_vel': 1.0,
                'orientation': -1.0,
                'base_height': -20.0,
            },
        ]

    def get_reward_scales(self):
        """Get current reward scales."""
        return self.levels[self.current_level]

    def update(self, mean_reward, success_rate):
        """Update curriculum level based on performance."""
        if success_rate > 0.8 and self.current_level < len(self.levels) - 1:
            self.current_level += 1
            print(f"Curriculum: Advanced to level {self.current_level}")
```

## Domain Randomization

### Physics Randomization

```python
class DomainRandomization:
    """Randomize simulation parameters for robust training."""

    def __init__(self, cfg):
        self.cfg = cfg

    def randomize_physics(self, env):
        """Randomize physics parameters."""
        for env_id in range(env.num_envs):
            # Mass randomization
            if self.cfg.randomize_mass:
                mass_scale = torch.uniform(0.8, 1.2)
                # Apply to all links
                for link_id in range(env.num_bodies):
                    props = env.gym.get_actor_rigid_body_properties(
                        env.envs[env_id], env.actors[env_id]
                    )
                    props[link_id].mass *= mass_scale
                    env.gym.set_actor_rigid_body_properties(
                        env.envs[env_id], env.actors[env_id], props
                    )

            # Friction randomization
            if self.cfg.randomize_friction:
                friction = torch.uniform(0.5, 1.5)
                for shape_id in range(env.num_shapes):
                    props = env.gym.get_actor_rigid_shape_properties(
                        env.envs[env_id], env.actors[env_id]
                    )
                    props[shape_id].friction = friction
                    env.gym.set_actor_rigid_shape_properties(
                        env.envs[env_id], env.actors[env_id], props
                    )

    def randomize_external_forces(self, env):
        """Apply random external disturbances."""
        if not self.cfg.apply_external_forces:
            return

        # Random push interval
        if env.episode_length % 50 == 0:
            force_mag = torch.uniform(50, 200, (env.num_envs,))
            force_dir = torch.randn(env.num_envs, 3)
            force_dir = force_dir / torch.norm(force_dir, dim=1, keepdim=True)

            forces = force_dir * force_mag.unsqueeze(1)
            forces[:, 2] = 0  # No vertical forces

            env.gym.apply_rigid_body_force_tensors(
                env.sim, gymtorch.unwrap_tensor(forces),
                None, gymapi.ENV_SPACE
            )

    def add_observation_noise(self, obs):
        """Add noise to observations."""
        if not self.cfg.add_noise:
            return obs

        noise = torch.randn_like(obs) * self.cfg.noise_scales
        return obs + noise
```

### Terrain Randomization

```python
class TerrainGenerator:
    """Generate varied terrain for training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.terrain_types = ['flat', 'rough', 'stairs', 'slope']

    def generate_terrain(self, env):
        """Generate procedural terrain."""
        terrain_width = self.cfg.terrain_width
        terrain_length = self.cfg.terrain_length
        resolution = self.cfg.terrain_resolution

        heightfield = np.zeros((terrain_width, terrain_length))

        for terrain_type in self.terrain_types:
            if terrain_type == 'flat':
                pass  # Already zeros

            elif terrain_type == 'rough':
                # Random bumps
                heightfield += np.random.uniform(
                    -0.05, 0.05, heightfield.shape
                )

            elif terrain_type == 'stairs':
                # Staircase
                step_height = 0.1
                step_width = 0.3
                for i in range(terrain_width):
                    stair_num = int(i / (step_width / resolution))
                    heightfield[i, :] = stair_num * step_height

            elif terrain_type == 'slope':
                # Inclined plane
                slope_angle = np.radians(10)
                for i in range(terrain_width):
                    heightfield[i, :] = i * resolution * np.tan(slope_angle)

        return heightfield
```

## Training Configuration

### Hyperparameters

```python
class TrainingConfig:
    # Environment
    num_envs = 4096
    episode_length = 1000

    # PPO
    learning_rate = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    num_epochs = 5
    batch_size = 4096
    entropy_coef = 0.01

    # Network
    hidden_dims = [512, 256, 128]
    activation = 'elu'

    # Training
    max_iterations = 10000
    save_interval = 500
    eval_interval = 100

    # Domain randomization
    randomize_mass = True
    mass_range = [0.8, 1.2]
    randomize_friction = True
    friction_range = [0.5, 1.5]
    apply_external_forces = True
    force_magnitude = [50, 200]
    add_noise = True
    noise_scale = 0.01

    # Curriculum
    use_curriculum = True
    curriculum_levels = 3
```

### Training Loop

```python
def train(cfg):
    """Main training loop."""
    # Create environment
    env = HumanoidLocomotionEnv(cfg)

    # Create policy network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = ActorCritic(obs_dim, action_dim, cfg.hidden_dims)

    # Create trainer
    trainer = PPOTrainer(env, policy, cfg)

    # Domain randomization
    domain_rand = DomainRandomization(cfg)

    # Curriculum
    curriculum = RewardCurriculum(cfg) if cfg.use_curriculum else None

    # Training loop
    for iteration in range(cfg.max_iterations):
        # Randomize physics at start of iteration
        if cfg.randomize_mass or cfg.randomize_friction:
            domain_rand.randomize_physics(env)

        # Collect rollout
        rollout = trainer.collect_rollout(cfg.episode_length)

        # Update policy
        stats = trainer.update(rollout)

        # Update curriculum
        if curriculum:
            mean_reward = rollout['rewards'].mean()
            success_rate = compute_success_rate(env)
            curriculum.update(mean_reward, success_rate)
            env.reward_scales = curriculum.get_reward_scales()

        # Logging
        if iteration % 10 == 0:
            print(f"Iteration {iteration}")
            print(f"  Mean reward: {rollout['rewards'].mean():.3f}")
            print(f"  Policy loss: {stats['policy_loss']:.4f}")
            print(f"  Value loss: {stats['value_loss']:.4f}")

        # Save checkpoint
        if iteration % cfg.save_interval == 0:
            save_checkpoint(policy, iteration)

        # Evaluation
        if iteration % cfg.eval_interval == 0:
            eval_reward = evaluate(env, policy)
            print(f"  Eval reward: {eval_reward:.3f}")

    return policy
```

## Policy Deployment

### Exporting Trained Policy

```python
def export_policy(policy, path):
    """Export policy for deployment."""
    # Save PyTorch model
    torch.save(policy.state_dict(), f"{path}/policy.pt")

    # Export to ONNX for embedded deployment
    dummy_input = torch.randn(1, policy.obs_dim)
    torch.onnx.export(
        policy,
        dummy_input,
        f"{path}/policy.onnx",
        input_names=['observation'],
        output_names=['action_mean', 'value'],
        dynamic_axes={
            'observation': {0: 'batch'},
            'action_mean': {0: 'batch'},
            'value': {0: 'batch'},
        }
    )

    # Save normalization parameters
    torch.save({
        'obs_mean': policy.obs_normalizer.mean,
        'obs_var': policy.obs_normalizer.var,
    }, f"{path}/normalization.pt")
```

### Real-Time Inference

```python
class PolicyRunner:
    """Run trained policy in real-time."""

    def __init__(self, policy_path, device="cuda:0"):
        self.device = device

        # Load policy
        self.policy = ActorCritic(obs_dim, action_dim)
        self.policy.load_state_dict(torch.load(f"{policy_path}/policy.pt"))
        self.policy.to(device)
        self.policy.eval()

        # Load normalization
        norm_params = torch.load(f"{policy_path}/normalization.pt")
        self.obs_mean = norm_params['obs_mean'].to(device)
        self.obs_var = norm_params['obs_var'].to(device)

    def get_action(self, obs):
        """Get action from observation."""
        with torch.no_grad():
            # Normalize observation
            obs_normalized = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

            # Get action
            action_mean, _ = self.policy(obs_normalized.unsqueeze(0))

            return action_mean.squeeze(0)
```

## Summary

In this lesson, you learned:

- Designing comprehensive locomotion reward functions
- Implementing curriculum learning for progressive difficulty
- Using domain randomization for robust policies
- Training configurations and hyperparameters
- Deploying trained policies for real-time control

## Next Steps

Continue to [Week 9 Exercises](/module-3/week-9/exercises) to train your own locomotion policy.
