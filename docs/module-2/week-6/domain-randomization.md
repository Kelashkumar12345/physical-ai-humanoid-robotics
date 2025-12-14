---
sidebar_position: 2
title: Domain Randomization
description: Sim-to-real transfer techniques for robust robot learning
---

# Domain Randomization

Learn to use domain randomization for training robust robot policies that transfer from simulation to the real world.

## Learning Objectives

By the end of this lesson, you will:

- Understand the sim-to-real gap problem
- Implement visual domain randomization
- Apply physics and dynamics randomization
- Configure randomization in Gazebo

## The Sim-to-Real Gap

### Why Simulation Differs from Reality

| Factor | Simulation | Reality |
|--------|------------|---------|
| **Physics** | Simplified models | Complex, noisy |
| **Visuals** | Clean renders | Variable lighting, occlusions |
| **Sensors** | Idealized noise | Calibration errors, drift |
| **Actuators** | Perfect response | Delays, backlash, wear |

### Domain Randomization Strategy

Train policies that are robust to variations by exposing them to many different simulated environments.

```
┌─────────────────────────────────────────────────────┐
│                  Training Loop                       │
├─────────────────────────────────────────────────────┤
│  For each episode:                                  │
│    1. Randomize environment parameters              │
│    2. Run robot policy                              │
│    3. Collect experience                            │
│    4. Update policy                                 │
│                                                     │
│  Result: Policy robust to parameter variations      │
└─────────────────────────────────────────────────────┘
```

## Visual Domain Randomization

### Lighting Variations

```xml
<!-- Base light configuration -->
<light name="sun" type="directional">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <direction>-0.5 0.1 -0.9</direction>
</light>
```

```python
#!/usr/bin/env python3
"""Randomize lighting in Gazebo."""
import random
import subprocess

def randomize_light():
    # Random intensity
    intensity = random.uniform(0.5, 1.2)

    # Random direction
    dir_x = random.uniform(-1, 1)
    dir_y = random.uniform(-1, 1)
    dir_z = random.uniform(-0.5, -1)

    # Random color temperature (warm to cool)
    temp = random.uniform(0, 1)
    r = 1.0 - 0.2 * temp
    g = 1.0 - 0.1 * temp
    b = 1.0 + 0.2 * temp

    # Apply via Gazebo service
    # (Implementation depends on Gazebo version)
    print(f"Light: intensity={intensity:.2f}, "
          f"color=({r:.2f},{g:.2f},{b:.2f})")

    return intensity, (r, g, b), (dir_x, dir_y, dir_z)
```

### Texture Randomization

```python
#!/usr/bin/env python3
"""Generate random textures for domain randomization."""
import numpy as np
from PIL import Image
import os

def generate_random_texture(size=(256, 256)):
    """Generate a random texture."""
    # Random base color
    base_color = np.random.randint(0, 255, 3)

    # Create noise pattern
    noise = np.random.randint(-50, 50, (*size, 3))

    # Combine
    texture = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(texture)

def generate_checkerboard(size=(256, 256), squares=8):
    """Generate checkerboard pattern."""
    texture = np.zeros((*size, 3), dtype=np.uint8)
    sq_size = size[0] // squares

    color1 = np.random.randint(100, 255, 3)
    color2 = np.random.randint(0, 100, 3)

    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                texture[i*sq_size:(i+1)*sq_size,
                       j*sq_size:(j+1)*sq_size] = color1
            else:
                texture[i*sq_size:(i+1)*sq_size,
                       j*sq_size:(j+1)*sq_size] = color2

    return Image.fromarray(texture)

def save_texture(texture, path):
    texture.save(path)

# Generate random textures
for i in range(10):
    tex = generate_random_texture()
    save_texture(tex, f'textures/random_{i}.png')

    checker = generate_checkerboard()
    save_texture(checker, f'textures/checker_{i}.png')
```

### Camera Parameter Randomization

```python
#!/usr/bin/env python3
"""Randomize camera intrinsics and extrinsics."""
import random
import numpy as np

class CameraRandomizer:
    def __init__(self, base_params):
        self.base_params = base_params

    def randomize(self):
        params = self.base_params.copy()

        # Focal length variation (±5%)
        params['fx'] *= random.uniform(0.95, 1.05)
        params['fy'] *= random.uniform(0.95, 1.05)

        # Principal point variation (±2 pixels)
        params['cx'] += random.uniform(-2, 2)
        params['cy'] += random.uniform(-2, 2)

        # Slight pose variation
        params['position'] = [
            p + random.gauss(0, 0.005)
            for p in self.base_params['position']
        ]

        params['rotation'] = [
            r + random.gauss(0, 0.01)
            for r in self.base_params['rotation']
        ]

        return params

# Example usage
base = {
    'fx': 525.0, 'fy': 525.0,
    'cx': 319.5, 'cy': 239.5,
    'position': [0.1, 0, 0.15],
    'rotation': [0, 0, 0]
}

randomizer = CameraRandomizer(base)
random_params = randomizer.randomize()
```

## Physics Domain Randomization

### Mass and Inertia Randomization

```python
#!/usr/bin/env python3
"""Randomize robot dynamics parameters."""
import random

class DynamicsRandomizer:
    def __init__(self, nominal_params, ranges):
        """
        Args:
            nominal_params: Dict of nominal values
            ranges: Dict of (min_mult, max_mult) tuples
        """
        self.nominal = nominal_params
        self.ranges = ranges

    def randomize(self):
        params = {}

        for key, value in self.nominal.items():
            if key in self.ranges:
                min_mult, max_mult = self.ranges[key]
                multiplier = random.uniform(min_mult, max_mult)
                params[key] = value * multiplier
            else:
                params[key] = value

        return params

# Example: Robot arm dynamics
nominal = {
    'link1_mass': 1.0,
    'link2_mass': 0.8,
    'link3_mass': 0.5,
    'joint1_friction': 0.1,
    'joint2_friction': 0.1,
    'joint3_friction': 0.1,
    'joint1_damping': 0.5,
    'joint2_damping': 0.4,
    'joint3_damping': 0.3,
}

ranges = {
    'link1_mass': (0.8, 1.2),      # ±20%
    'link2_mass': (0.8, 1.2),
    'link3_mass': (0.8, 1.2),
    'joint1_friction': (0.5, 2.0), # 50-200%
    'joint2_friction': (0.5, 2.0),
    'joint3_friction': (0.5, 2.0),
    'joint1_damping': (0.7, 1.3),
    'joint2_damping': (0.7, 1.3),
    'joint3_damping': (0.7, 1.3),
}

randomizer = DynamicsRandomizer(nominal, ranges)
```

### Friction Randomization

```python
def randomize_friction(base_mu=1.0, variation=0.3):
    """Randomize friction coefficient."""
    return base_mu * random.uniform(1 - variation, 1 + variation)

def generate_friction_sdf(mu, mu2=None):
    """Generate SDF friction element."""
    if mu2 is None:
        mu2 = mu

    return f"""
    <surface>
      <friction>
        <ode>
          <mu>{mu:.3f}</mu>
          <mu2>{mu2:.3f}</mu2>
        </ode>
      </friction>
    </surface>
    """
```

### Actuator Randomization

```python
class ActuatorRandomizer:
    """Randomize actuator characteristics."""

    def __init__(self, nominal_torque, nominal_delay):
        self.nominal_torque = nominal_torque
        self.nominal_delay = nominal_delay

    def get_params(self):
        return {
            # Torque variation (±10%)
            'max_torque': self.nominal_torque * random.uniform(0.9, 1.1),

            # Response delay (0-20ms additional)
            'delay': self.nominal_delay + random.uniform(0, 0.02),

            # Noise in torque output
            'torque_noise_std': random.uniform(0.01, 0.05),

            # Backlash (0-2 degrees)
            'backlash': random.uniform(0, 0.035),
        }
```

## Sensor Noise Randomization

### LiDAR Noise

```python
class LidarRandomizer:
    """Add realistic noise to LiDAR readings."""

    def __init__(self, base_noise_std=0.02):
        self.base_noise_std = base_noise_std

    def add_noise(self, ranges):
        """Add Gaussian noise to range readings."""
        noise_std = self.base_noise_std * random.uniform(0.5, 2.0)
        noise = np.random.normal(0, noise_std, len(ranges))
        return ranges + noise

    def add_dropouts(self, ranges, dropout_rate=0.01):
        """Randomly drop some readings."""
        mask = np.random.random(len(ranges)) > dropout_rate
        result = ranges.copy()
        result[~mask] = float('inf')
        return result

    def add_multipath(self, ranges, multipath_rate=0.005):
        """Add occasional multipath errors."""
        mask = np.random.random(len(ranges)) < multipath_rate
        result = ranges.copy()
        result[mask] *= random.uniform(0.5, 0.9)  # Shorter than actual
        return result
```

### Camera Noise

```python
class CameraNoiseRandomizer:
    """Add realistic noise to camera images."""

    def add_gaussian_noise(self, image, std_range=(5, 25)):
        """Add Gaussian noise."""
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def add_motion_blur(self, image, kernel_size_range=(3, 7)):
        """Add motion blur."""
        import cv2
        size = random.choice(range(*kernel_size_range, 2))
        kernel = np.zeros((size, size))

        # Random direction
        if random.random() > 0.5:
            kernel[size//2, :] = 1.0 / size  # Horizontal
        else:
            kernel[:, size//2] = 1.0 / size  # Vertical

        return cv2.filter2D(image, -1, kernel)

    def adjust_brightness(self, image, factor_range=(0.7, 1.3)):
        """Randomly adjust brightness."""
        factor = random.uniform(*factor_range)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """Randomly adjust contrast."""
        factor = random.uniform(*factor_range)
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
```

## Complete Randomization System

```python
#!/usr/bin/env python3
"""Complete domain randomization system for Gazebo."""
import rclpy
from rclpy.node import Node
import random
import json

class DomainRandomizer(Node):
    def __init__(self):
        super().__init__('domain_randomizer')

        # Load configuration
        self.declare_parameter('config_file', 'randomization_config.json')
        config_file = self.get_parameter('config_file').value

        with open(config_file) as f:
            self.config = json.load(f)

        # Timer for periodic randomization
        self.declare_parameter('randomize_interval', 10.0)
        interval = self.get_parameter('randomize_interval').value

        self.timer = self.create_timer(interval, self.randomize_all)

        self.get_logger().info('Domain randomizer initialized')

    def randomize_all(self):
        """Randomize all configured parameters."""
        self.randomize_lighting()
        self.randomize_physics()
        self.randomize_objects()

        self.get_logger().info('Environment randomized')

    def randomize_lighting(self):
        """Randomize lighting conditions."""
        if 'lighting' not in self.config:
            return

        cfg = self.config['lighting']

        # Intensity
        intensity = random.uniform(
            cfg.get('intensity_min', 0.5),
            cfg.get('intensity_max', 1.5)
        )

        # Color temperature
        temp = random.uniform(0, 1)
        # TODO: Apply via Gazebo service

    def randomize_physics(self):
        """Randomize physics parameters."""
        if 'physics' not in self.config:
            return

        cfg = self.config['physics']

        # Friction
        if 'friction' in cfg:
            mu = random.uniform(
                cfg['friction']['min'],
                cfg['friction']['max']
            )
            # TODO: Apply via Gazebo service

    def randomize_objects(self):
        """Randomize object positions and properties."""
        if 'objects' not in self.config:
            return

        for obj_cfg in self.config['objects']:
            # Random position within bounds
            x = random.uniform(obj_cfg['x_min'], obj_cfg['x_max'])
            y = random.uniform(obj_cfg['y_min'], obj_cfg['y_max'])
            z = obj_cfg.get('z', 0)

            # Random rotation
            yaw = random.uniform(0, 6.28)

            # TODO: Apply via Gazebo service

def main(args=None):
    rclpy.init(args=args)
    node = DomainRandomizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Configuration File

```json
{
  "lighting": {
    "intensity_min": 0.5,
    "intensity_max": 1.5,
    "color_temp_min": 3000,
    "color_temp_max": 7000
  },
  "physics": {
    "friction": {
      "min": 0.5,
      "max": 1.5
    },
    "mass_variation": 0.2
  },
  "objects": [
    {
      "name": "target_object",
      "x_min": -1.0,
      "x_max": 1.0,
      "y_min": -1.0,
      "y_max": 1.0,
      "z": 0.5
    }
  ],
  "camera": {
    "noise_std": [5, 25],
    "brightness_range": [0.7, 1.3]
  }
}
```

## Best Practices

### Randomization Schedule

```python
class ProgressiveRandomizer:
    """Gradually increase randomization during training."""

    def __init__(self, max_steps, warmup_steps=1000):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_randomization_scale(self):
        """Get current randomization scale (0 to 1)."""
        if self.current_step < self.warmup_steps:
            return 0.0  # No randomization during warmup

        progress = (self.current_step - self.warmup_steps) / \
                   (self.max_steps - self.warmup_steps)

        return min(1.0, progress)

    def step(self):
        self.current_step += 1
```

### Parameter Ranges

| Parameter | Conservative | Aggressive |
|-----------|--------------|------------|
| Mass | ±10% | ±30% |
| Friction | ±20% | ±50% |
| Damping | ±15% | ±40% |
| Sensor noise | ×1-2 | ×1-5 |
| Lighting | ±30% | ±60% |

## Summary

In this lesson, you learned:

- The sim-to-real gap and why domain randomization helps
- Visual randomization techniques (lighting, textures, camera)
- Physics randomization (mass, friction, actuators)
- Sensor noise randomization
- Building a complete randomization system

## Next Steps

Continue to [Week 6 Exercises](/module-2/week-6/exercises) to practice simulation techniques.
