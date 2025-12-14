---
sidebar_position: 3
title: Week 6 Exercises
description: Hands-on exercises for advanced Gazebo simulation
---

# Week 6: Advanced Simulation Exercises

Practice physics tuning, domain randomization, and sim-to-real techniques.

## Exercise 1: Physics Parameter Optimization

**Objective**: Tune physics parameters for stable robot simulation.

### Requirements

1. Load the provided unstable robot model
2. Identify and fix physics issues
3. Achieve stable simulation at real-time factor
4. Document your parameter changes

### Problem Symptoms

- Robot jitters when stationary
- Wheels slip excessively
- Robot tips over unexpectedly
- Simulation runs slower than real-time

### Starter Configuration

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="physics_test">
    <physics name="broken_physics" type="dart">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- TODO: Fix physics configuration -->

    <model name="unstable_robot">
      <pose>0 0 0.1 0 0 0</pose>

      <link name="base_link">
        <inertial>
          <!-- TODO: Fix inertial properties -->
          <mass>0.001</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <iyy>0.0001</iyy>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>0.4 0.3 0.1</size></box></geometry>
          <surface>
            <friction>
              <ode><mu>0.01</mu><mu2>0.01</mu2></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.4 0.3 0.1</size></box></geometry>
        </visual>
      </link>

      <!-- Wheels with bad parameters -->
      <!-- TODO: Fix wheel configuration -->

    </model>
  </world>
</sdf>
```

### Validation Checklist

- [ ] Robot remains stable when stationary
- [ ] Wheels grip properly during acceleration
- [ ] Simulation maintains real-time factor > 0.9
- [ ] No physics warnings in console

<details className="solution-block">
<summary>Solution Guidelines</summary>
<div className="solution-content">

Key fixes needed:
1. **Step size**: Reduce to 0.001s for stability
2. **Mass**: Increase to realistic value (5-10 kg)
3. **Inertia**: Calculate properly based on geometry
4. **Friction**: Increase wheel friction to 1.0+
5. **Solver iterations**: Increase if needed

```xml
<physics name="fixed_physics" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>

<inertial>
  <mass>5.0</mass>
  <inertia>
    <ixx>0.05</ixx>
    <iyy>0.08</iyy>
    <izz>0.1</izz>
  </inertia>
</inertial>
```

</div>
</details>

---

## Exercise 2: Inertia Calculator

**Objective**: Create a tool to calculate correct inertial properties.

### Requirements

1. Support box, cylinder, and sphere geometries
2. Calculate inertia tensor from mass and dimensions
3. Output valid SDF/URDF format
4. Validate against known values

### Starter Code

```python
#!/usr/bin/env python3
"""Inertia calculator for robot links."""
import math

class InertiaCalculator:
    @staticmethod
    def box(mass, width, height, depth):
        """Calculate inertia for a box.

        Args:
            mass: Mass in kg
            width: X dimension in m
            height: Y dimension in m
            depth: Z dimension in m

        Returns:
            Tuple of (ixx, iyy, izz)
        """
        # TODO: Implement
        pass

    @staticmethod
    def cylinder(mass, radius, length, axis='z'):
        """Calculate inertia for a cylinder.

        Args:
            mass: Mass in kg
            radius: Radius in m
            length: Length in m
            axis: Axis of symmetry ('x', 'y', or 'z')

        Returns:
            Tuple of (ixx, iyy, izz)
        """
        # TODO: Implement
        pass

    @staticmethod
    def sphere(mass, radius):
        """Calculate inertia for a solid sphere.

        Args:
            mass: Mass in kg
            radius: Radius in m

        Returns:
            Tuple of (ixx, iyy, izz)
        """
        # TODO: Implement
        pass

    @staticmethod
    def to_sdf(ixx, iyy, izz, ixy=0, ixz=0, iyz=0):
        """Format inertia as SDF XML."""
        return f"""<inertia>
  <ixx>{ixx:.6f}</ixx>
  <ixy>{ixy:.6f}</ixy>
  <ixz>{ixz:.6f}</ixz>
  <iyy>{iyy:.6f}</iyy>
  <iyz>{iyz:.6f}</iyz>
  <izz>{izz:.6f}</izz>
</inertia>"""

# Test cases
if __name__ == '__main__':
    calc = InertiaCalculator()

    # Test box: 1kg, 0.1m cube
    ixx, iyy, izz = calc.box(1.0, 0.1, 0.1, 0.1)
    print(f"Box: Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}")
    # Expected: all ≈ 0.001667

    # Test cylinder: 1kg, r=0.05m, l=0.2m
    ixx, iyy, izz = calc.cylinder(1.0, 0.05, 0.2)
    print(f"Cylinder: Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}")
    # Expected: Ixx=Iyy≈0.003958, Izz≈0.00125

    # Test sphere: 1kg, r=0.1m
    ixx, iyy, izz = calc.sphere(1.0, 0.1)
    print(f"Sphere: Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}")
    # Expected: all ≈ 0.004
```

---

## Exercise 3: Visual Domain Randomization

**Objective**: Implement texture and lighting randomization.

### Requirements

1. Generate random textures programmatically
2. Apply textures to simulation objects
3. Randomize lighting conditions
4. Create a training data collector

### Starter Code

```python
#!/usr/bin/env python3
"""Visual domain randomization for training data collection."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import random
import os

class VisualRandomizer(Node):
    def __init__(self):
        super().__init__('visual_randomizer')

        self.bridge = CvBridge()

        # Image subscriber
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Output directory
        self.output_dir = 'training_data'
        os.makedirs(self.output_dir, exist_ok=True)

        self.frame_count = 0
        self.randomization_interval = 50  # Frames between randomizations

    def image_callback(self, msg):
        # TODO: Convert to OpenCV
        # TODO: Apply random augmentations
        # TODO: Save with metadata

        self.frame_count += 1

        if self.frame_count % self.randomization_interval == 0:
            self.randomize_environment()

    def randomize_environment(self):
        """Trigger environment randomization."""
        # TODO: Call Gazebo services to:
        # - Change lighting
        # - Move objects
        # - Change textures

        self.get_logger().info('Environment randomized')

    def apply_augmentation(self, image):
        """Apply random image augmentations."""
        augmented = image.copy()

        # TODO: Implement augmentations:
        # - Brightness adjustment
        # - Contrast adjustment
        # - Gaussian noise
        # - Color jitter

        return augmented

def main(args=None):
    rclpy.init(args=args)
    node = VisualRandomizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Exercise 4: Dynamics Randomization

**Objective**: Implement physics parameter randomization for robust policy training.

### Requirements

1. Randomize mass and inertia within bounds
2. Randomize friction coefficients
3. Randomize actuator properties
4. Log all randomized parameters

### Configuration Format

```yaml
# dynamics_randomization.yaml
randomization:
  mass:
    base_link:
      nominal: 5.0
      range: [0.8, 1.2]  # Multiplier range

  friction:
    wheels:
      nominal: 1.0
      range: [0.5, 1.5]
    floor:
      nominal: 0.8
      range: [0.3, 1.2]

  actuators:
    torque_scale:
      nominal: 1.0
      range: [0.9, 1.1]
    delay:
      nominal: 0.0
      range: [0.0, 0.02]  # Seconds
```

### Starter Code

```python
#!/usr/bin/env python3
"""Dynamics randomization system."""
import yaml
import random
import json
from datetime import datetime

class DynamicsRandomizer:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        self.history = []

    def randomize(self):
        """Generate randomized parameters."""
        params = {}

        # TODO: Implement randomization for each category
        # - Mass
        # - Friction
        # - Actuators

        # Log parameters
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'params': params
        })

        return params

    def apply_to_simulation(self, params):
        """Apply parameters to Gazebo simulation."""
        # TODO: Implement Gazebo service calls
        pass

    def save_history(self, filename):
        """Save randomization history for analysis."""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)

if __name__ == '__main__':
    randomizer = DynamicsRandomizer('dynamics_randomization.yaml')

    # Generate 10 random configurations
    for i in range(10):
        params = randomizer.randomize()
        print(f"Config {i+1}: {params}")

    randomizer.save_history('randomization_history.json')
```

---

## Exercise 5: Sim-to-Real Transfer Test

**Objective**: Evaluate policy robustness across domain variations.

### Requirements

1. Train a simple policy in simulation
2. Test across multiple randomized environments
3. Measure performance degradation
4. Identify failure modes

### Test Protocol

```python
#!/usr/bin/env python3
"""Sim-to-real transfer evaluation."""
import numpy as np

class TransferEvaluator:
    def __init__(self, policy, env_factory, randomizer):
        self.policy = policy
        self.env_factory = env_factory
        self.randomizer = randomizer

    def evaluate(self, n_envs=100, n_episodes_per_env=10):
        """Evaluate policy across randomized environments."""
        results = []

        for env_idx in range(n_envs):
            # Randomize environment
            params = self.randomizer.randomize()
            env = self.env_factory(params)

            env_results = []
            for ep in range(n_episodes_per_env):
                reward = self.run_episode(env)
                env_results.append(reward)

            results.append({
                'params': params,
                'rewards': env_results,
                'mean_reward': np.mean(env_results),
                'std_reward': np.std(env_results)
            })

        return self.analyze_results(results)

    def run_episode(self, env, max_steps=1000):
        """Run single episode and return total reward."""
        obs = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = self.policy.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        return total_reward

    def analyze_results(self, results):
        """Analyze transfer performance."""
        all_rewards = [r['mean_reward'] for r in results]

        analysis = {
            'overall_mean': np.mean(all_rewards),
            'overall_std': np.std(all_rewards),
            'worst_case': np.min(all_rewards),
            'best_case': np.max(all_rewards),
            'failure_rate': sum(1 for r in all_rewards if r < 0) / len(all_rewards)
        }

        # TODO: Identify which parameters cause failures
        # TODO: Generate recommendations for improving robustness

        return analysis
```

---

## Challenge: Complete Simulation Pipeline

**Objective**: Build a complete training pipeline with domain randomization.

### Requirements

1. Configurable randomization system
2. Parallel environment execution
3. Automatic data collection
4. Performance monitoring
5. Checkpoint saving

### Architecture

```
Training Pipeline
├── Environment Manager
│   ├── Gazebo instances (parallel)
│   ├── ROS 2 bridges
│   └── Domain randomizer
├── Data Collector
│   ├── Observations
│   ├── Actions
│   └── Rewards
├── Policy Trainer
│   ├── Neural network
│   ├── Optimizer
│   └── Checkpoints
└── Evaluator
    ├── Test environments
    ├── Metrics
    └── Reports
```

### Deliverables

1. `randomization_config.yaml` - Randomization parameters
2. `environment_manager.py` - Environment orchestration
3. `data_collector.py` - Training data collection
4. `evaluator.py` - Transfer evaluation
5. Documentation of results

---

## Submission Checklist

- [ ] Physics parameters are correctly tuned
- [ ] Inertia calculator passes all test cases
- [ ] Visual randomization generates diverse images
- [ ] Dynamics randomization covers parameter ranges
- [ ] Transfer evaluation produces meaningful metrics
- [ ] Code is documented and tested

## Resources

- [Domain Randomization Paper](https://arxiv.org/abs/1703.06907)
- [Sim-to-Real Transfer Survey](https://arxiv.org/abs/2009.13303)
- [Gazebo Physics Documentation](https://gazebosim.org/docs)
