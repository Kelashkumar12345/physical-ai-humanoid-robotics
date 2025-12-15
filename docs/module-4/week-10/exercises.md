---
sidebar_position: 3
title: Week 10 Exercises
description: Hands-on exercises for VLA foundations and fine-tuning
---

# Week 10: VLA Foundations Exercises

Practice working with Vision-Language-Action models for robot control.

## Exercise 1: VLA Model Exploration

**Objective**: Explore pretrained VLA model capabilities.

### Requirements

1. Load a pretrained VLA model (OpenVLA or Octo)
2. Test inference with different images and instructions
3. Analyze action outputs for various commands
4. Measure inference latency

### Starter Code

```python
import torch
import numpy as np
from PIL import Image
import time

class VLAExplorer:
    def __init__(self, model_name="openvla/openvla-7b"):
        # TODO: Load model and processor

        pass

    def test_instruction(self, image_path, instruction):
        """Test model with image and instruction."""
        # TODO: Load image

        # TODO: Run inference

        # TODO: Return action and timing

        pass

    def compare_instructions(self, image_path, instructions):
        """Compare actions for different instructions on same image."""
        results = {}

        for instruction in instructions:
            action, latency = self.test_instruction(image_path, instruction)
            results[instruction] = {
                'action': action,
                'latency': latency
            }

        return results

    def analyze_action_space(self, image_path, base_instruction):
        """Analyze action output distribution."""
        # TODO: Run multiple inferences
        # TODO: Analyze variance
        # TODO: Visualize action distribution

        pass

# Test cases
explorer = VLAExplorer()

# Test different manipulation instructions
instructions = [
    "pick up the red cup",
    "move the cup to the left",
    "push the cup forward",
    "grasp the handle",
    "open the drawer"
]

results = explorer.compare_instructions("test_scene.jpg", instructions)

for instruction, result in results.items():
    print(f"\n{instruction}:")
    print(f"  Action: {result['action']}")
    print(f"  Latency: {result['latency']:.3f}s")
```

### Analysis Questions

1. How do actions differ for similar instructions?
2. What is the inference latency? Is it suitable for real-time control?
3. How does the model handle ambiguous instructions?

---

## Exercise 2: Demonstration Collection

**Objective**: Build a demonstration collection pipeline.

### Requirements

1. Create a simulated environment for data collection
2. Implement keyboard/mouse teleoperation
3. Record image-action pairs with language labels
4. Save demonstrations in HDF5 format

### Starter Code

```python
import gymnasium as gym
import h5py
import numpy as np
import cv2
from pynput import keyboard

class DemoCollector:
    def __init__(self, env_name="FetchPickAndPlace-v2"):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.demonstrations = []
        self.current_demo = None

        # Keyboard state
        self.keys_pressed = set()

        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            self.keys_pressed.add(key)

    def _on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            self.keys_pressed.discard(key)

    def get_teleop_action(self):
        """Get action from keyboard input."""
        action = np.zeros(4)  # [dx, dy, dz, gripper]

        # TODO: Map keys to actions
        # w/s: forward/backward
        # a/d: left/right
        # q/e: up/down
        # space: toggle gripper

        return action

    def collect_demo(self, task_instruction):
        """Collect a single demonstration."""
        obs, _ = self.env.reset()
        demo = {
            'instruction': task_instruction,
            'observations': [],
            'actions': [],
            'success': False
        }

        print(f"Collecting: {task_instruction}")
        print("Controls: WASD + QE for movement, SPACE for gripper")
        print("Press 'r' to reset, 's' for success, 'f' for failure")

        done = False
        while not done:
            # Get image
            image = self.env.render()

            # Get teleop action
            action = self.get_teleop_action()

            # Store observation-action pair
            demo['observations'].append({
                'image': image.copy(),
                'proprio': obs['observation'].copy()
            })
            demo['actions'].append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Check for end conditions
            if 's' in self.keys_pressed:
                demo['success'] = True
                done = True
            elif 'f' in self.keys_pressed:
                demo['success'] = False
                done = True
            elif 'r' in self.keys_pressed:
                obs, _ = self.env.reset()
                demo = {
                    'instruction': task_instruction,
                    'observations': [],
                    'actions': [],
                    'success': False
                }

            # Display
            cv2.imshow('Collection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        self.demonstrations.append(demo)
        print(f"Demo recorded: {'Success' if demo['success'] else 'Failure'}")

        return demo

    def save_dataset(self, path):
        """Save all demonstrations to HDF5."""
        # TODO: Implement HDF5 saving

        pass

# Collect demonstrations
collector = DemoCollector()

tasks = [
    "pick up the cube",
    "place the cube on the target",
    "push the cube to the right"
]

for task in tasks:
    for i in range(5):  # 5 demos per task
        collector.collect_demo(task)

collector.save_dataset("demos.hdf5")
```

---

## Exercise 3: Custom VLA Fine-tuning

**Objective**: Fine-tune a VLA model on collected demonstrations.

### Requirements

1. Load demonstration dataset
2. Implement LoRA fine-tuning
3. Train for specified epochs
4. Evaluate before/after performance

### Starter Code

```python
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

class DemoDataset(Dataset):
    """Dataset from collected demonstrations."""

    def __init__(self, hdf5_path, processor):
        self.processor = processor
        self.samples = self._load_data(hdf5_path)

    def _load_data(self, path):
        # TODO: Load from HDF5
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: Return processed sample
        pass


class VLAFineTuner:
    def __init__(self, model_name, lora_rank=8):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def train(self, train_dataset, val_dataset, config):
        """Fine-tune the model."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size']
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(config['epochs']):
            # Training
            self.model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                # TODO: Forward pass
                # TODO: Compute loss
                # TODO: Backward pass
                # TODO: Update weights

                pass

            # Validation
            val_loss = self.evaluate(val_loader, device)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        return self.model

    def evaluate(self, dataloader, device):
        """Evaluate model on dataset."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                # TODO: Compute validation loss
                pass

        return total_loss / len(dataloader)

    def save(self, path):
        """Save fine-tuned model."""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)


# Fine-tune
config = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'epochs': 10
}

finetuner = VLAFineTuner("openvla/openvla-7b")

train_dataset = DemoDataset("demos_train.hdf5", finetuner.processor)
val_dataset = DemoDataset("demos_val.hdf5", finetuner.processor)

finetuner.train(train_dataset, val_dataset, config)
finetuner.save("finetuned_vla")
```

---

## Exercise 4: Language Grounding Analysis

**Objective**: Analyze how VLA models ground language to actions.

### Requirements

1. Create test scenes with multiple objects
2. Test object reference understanding
3. Test spatial reasoning (left, right, above)
4. Visualize attention or action patterns

### Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt

class LanguageGroundingAnalyzer:
    def __init__(self, model):
        self.model = model

    def test_object_grounding(self, scene_image, objects):
        """Test if model can distinguish between objects."""
        results = {}

        for obj in objects:
            instruction = f"pick up the {obj}"
            action = self.model.predict_action(scene_image, instruction)
            results[obj] = action

        return results

    def test_spatial_reasoning(self, scene_image):
        """Test spatial instruction understanding."""
        spatial_instructions = [
            "move to the left",
            "move to the right",
            "move forward",
            "move backward",
            "move up",
            "move down"
        ]

        results = {}
        for instruction in spatial_instructions:
            action = self.model.predict_action(scene_image, instruction)
            results[instruction] = action

        return results

    def visualize_action_patterns(self, results):
        """Visualize action patterns for different instructions."""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for ax, (instruction, action) in zip(axes.flat, results.items()):
            # Plot action vector
            ax.bar(range(len(action)), action)
            ax.set_title(instruction)
            ax.set_xlabel("Action Dimension")
            ax.set_ylabel("Value")

        plt.tight_layout()
        plt.savefig("action_patterns.png")
        plt.show()

    def consistency_test(self, scene_image, instruction, num_samples=10):
        """Test action consistency for same input."""
        actions = []

        for _ in range(num_samples):
            action = self.model.predict_action(scene_image, instruction)
            actions.append(action)

        actions = np.array(actions)

        return {
            'mean': actions.mean(axis=0),
            'std': actions.std(axis=0),
            'max_variance': actions.std(axis=0).max()
        }

# Run analysis
analyzer = LanguageGroundingAnalyzer(model)

# Test with scene containing red cup, blue cube, green ball
objects = ["red cup", "blue cube", "green ball"]
grounding_results = analyzer.test_object_grounding("multi_object_scene.jpg", objects)

# Analyze
for obj, action in grounding_results.items():
    print(f"{obj}: action = {action[:3]}")  # First 3 dimensions

# Spatial reasoning
spatial_results = analyzer.test_spatial_reasoning("scene.jpg")
analyzer.visualize_action_patterns(spatial_results)
```

---

## Exercise 5: Real-Time VLA Pipeline

**Objective**: Build a real-time VLA control pipeline with ROS 2.

### Requirements

1. Create ROS 2 node for VLA inference
2. Subscribe to camera images
3. Accept language commands via topic/service
4. Publish control commands at consistent rate

### Starter Code

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import threading
import time

class VLAControlNode(Node):
    def __init__(self):
        super().__init__('vla_controller')

        # TODO: Load VLA model

        # CV Bridge
        self.bridge = CvBridge()

        # State
        self.latest_image = None
        self.current_instruction = "stay still"
        self.image_lock = threading.Lock()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.instruction_sub = self.create_subscription(
            String,
            '/vla/instruction',
            self.instruction_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/robot/cmd_vel',
            10
        )

        # Control loop timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Metrics
        self.inference_times = []

        self.get_logger().info("VLA Controller initialized")

    def image_callback(self, msg):
        """Store latest camera image."""
        with self.image_lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def instruction_callback(self, msg):
        """Update current instruction."""
        self.current_instruction = msg.data
        self.get_logger().info(f"New instruction: {self.current_instruction}")

    def control_loop(self):
        """Main control loop - runs at fixed rate."""
        with self.image_lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()

        # TODO: Run VLA inference
        start_time = time.time()

        # action = self.model.predict_action(image, self.current_instruction)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # TODO: Convert action to TwistStamped
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        # cmd.twist.linear.x = action[0]
        # cmd.twist.linear.y = action[1]
        # cmd.twist.angular.z = action[5]

        self.cmd_pub.publish(cmd)

        # Log metrics periodically
        if len(self.inference_times) % 100 == 0:
            avg_time = sum(self.inference_times[-100:]) / 100
            self.get_logger().info(f"Avg inference: {avg_time*1000:.1f}ms")

def main():
    rclpy.init()
    node = VLAControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Challenge: Multi-Task VLA System

**Objective**: Build a complete VLA system that handles multiple manipulation tasks.

### Requirements

1. Support at least 5 different tasks
2. Implement task selection via voice or text
3. Real-time visual feedback of model predictions
4. Performance logging and analysis

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Task VLA System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │    Speech    │     │    Vision    │     │    Robot     │   │
│   │  Recognition │     │    Input     │     │   Sensors    │   │
│   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘   │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                    VLA Controller                       │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│   │  │    Task     │  │   Action    │  │   Safety    │    │   │
│   │  │  Selector   │  │  Generator  │  │   Monitor   │    │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│   └────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                   Robot Execution                       │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

| Metric | Target |
|--------|--------|
| Inference latency | < 100ms |
| Task success rate | > 80% |
| Language understanding | > 90% accuracy |
| Control frequency | 10 Hz |
| Multi-task generalization | 5+ tasks |

---

## Submission Checklist

- [ ] VLA model exploration completed
- [ ] Demonstration collection pipeline working
- [ ] Fine-tuning implemented with LoRA
- [ ] Language grounding analysis done
- [ ] Real-time ROS 2 pipeline functional
- [ ] Challenge system operational

## Resources

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [Octo Paper](https://arxiv.org/abs/2405.12213)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
