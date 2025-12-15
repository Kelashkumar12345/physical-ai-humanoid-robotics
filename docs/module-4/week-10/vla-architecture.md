---
sidebar_position: 1
title: VLA Architecture
description: Understanding Vision-Language-Action models for robotics
---

# Week 10: Vision-Language-Action (VLA) Architecture

<WeekHeader
  week={10}
  title="VLA Foundations"
  module={4}
  estimatedHours={8}
  skills={["VLA Models", "Ollama", "Prompt Engineering", "Robot Control"]}
/>

<LearningObjectives
  week={10}
  objectives={[
    "Understand VLA model architectures and capabilities",
    "Set up local LLM inference with Ollama",
    "Design vision-language pipelines for robot control",
    "Implement action generation from language commands",
    "Integrate VLA models with ROS 2 systems"
  ]}
/>

## What are Vision-Language-Action (VLA) Models?

**Vision-Language-Action (VLA)** models represent a breakthrough in robotics AI, combining computer vision, natural language understanding, and action generation in a single neural network.

### Traditional vs. VLA Approaches

<TerminalOutput>
Traditional Approach:
  Perception → Planning → Control (separate modules)

VLA Approach:
  [Image + Text] → [Action] (end-to-end)
</TerminalOutput>

### Key VLA Architectures

| Model | Architecture | Strengths | Limitations |
|-------|-------------|-----------|-------------|
| **RT-2** | Vision + Language → Actions | Generalization, few-shot learning | Requires large datasets |
| **PaLM-E** | Embodied language model | Multimodal integration | Computational requirements |
| **VoxPoser** | Visual reasoning + manipulation | Precise manipulation | Limited to grasping tasks |
| **LM-Nav** | Language-guided navigation | Natural language navigation | Indoor environments only |

## Ollama for Local VLA Inference

**Ollama** provides local LLM inference, making VLA models accessible without cloud dependencies.

### Installation

```bash
# Download from https://ollama.ai or use package manager
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl start ollama  # Linux
# Or on Windows: Start the Ollama service from the installer
```

### Model Selection for Robotics

```bash
# Pull vision-language models
ollama pull llava:13b    # Vision + Language Understanding
ollama pull llava:34b    # More capable vision model
ollama pull llama3:8b    # Language-only (for action planning)

# For robotics applications, consider:
ollama pull phi3:mini     # Lightweight, fast inference
ollama pull mistral:7b    # Good balance of capability/speed
```

### Basic Ollama Usage

```python
import ollama

def describe_scene(image_path):
    """Describe what the robot sees."""
    response = ollama.chat(
        model='llava:13b',
        messages=[
            {
                'role': 'user',
                'content': 'Describe this robot scene in detail. Focus on objects, their positions, and potential interactions.',
                'images': [image_path]
            }
        ]
    )
    return response['message']['content']

def generate_action(command, scene_description):
    """Generate robot actions from language command."""
    prompt = f"""
    Robot scene: {scene_description}
    Command: {command}

    Generate a sequence of robot actions to accomplish the task.
    Format: [action1, action2, action3, ...]
    Actions: ['move_forward', 'turn_left', 'grasp_object', 'release_object', 'rotate_gripper', 'stop']
    """

    response = ollama.chat(
        model='llama3:8b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']
```

## Vision-Language Pipeline

### Camera Integration

```python
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VLAPipeline:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.vla_publisher = self.create_publisher(String, '/vla/commands', 10)

    def image_callback(self, msg):
        """Process incoming camera image."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Add robot state context
        robot_state = self.get_robot_state()

        # Generate VLA command
        command = self.process_with_vla(cv_image, robot_state)

        # Publish for execution
        self.vla_publisher.publish(String(data=command))
```

### Prompt Engineering for Robotics

Effective prompts guide the LLM to generate appropriate robot actions:

```python
def create_robotic_prompt(task, observation, robot_state):
    """Create structured prompt for robotic action generation."""

    prompt = f"""
    # ROBOTIC TASK INSTRUCTION
    Task: {task}

    # OBSERVATION
    {observation}

    # ROBOT STATE
    Position: {robot_state['position']}
    Orientation: {robot_state['orientation']}
    Battery: {robot_state['battery']}%
    Gripper: {robot_state['gripper_state']}

    # ACTION CONSTRAINTS
    - Use only available actions: move_forward, turn_left, turn_right, grasp, release
    - Consider safety: avoid obstacles, don't exceed joint limits
    - Be precise: specify distances/angles when needed

    # OUTPUT FORMAT
    [action1, action2, action3, ...]

    # RESPONSE:
    """

    return prompt
```

## Action Generation and Execution

### Converting Language to Actions

```python
import json
import re

class ActionGenerator:
    def __init__(self):
        self.action_mapping = {
            'move forward': 'move_forward',
            'go forward': 'move_forward',
            'move backward': 'move_backward',
            'turn left': 'turn_left',
            'turn right': 'turn_right',
            'grasp': 'grasp_object',
            'pick up': 'grasp_object',
            'release': 'release_object',
            'drop': 'release_object',
            'stop': 'stop_robot'
        }

    def parse_actions(self, vla_output):
        """Parse VLA output into executable robot actions."""
        try:
            # Try to parse as JSON first
            actions = json.loads(vla_output)
            return actions
        except json.JSONDecodeError:
            # Parse as text list
            text = vla_output.lower()
            actions = []

            for cmd, action in self.action_mapping.items():
                if cmd in text:
                    actions.append(action)

            return actions
```

### Safety Constraints

```python
class SafeActionExecutor:
    def __init__(self):
        self.safety_checker = SafetyConstraints()

    def execute_with_safety(self, actions):
        """Execute actions with safety validation."""
        for action in actions:
            if self.safety_checker.is_safe(action):
                self.execute_action(action)
            else:
                self.get_logger().warn(f'Safety violation for action: {action}')
                continue

class SafetyConstraints:
    def is_safe(self, action):
        """Validate action against safety constraints."""
        # Check for obstacles
        if action in ['move_forward', 'move_backward']:
            if self.detect_obstacle_in_path():
                return False

        # Check battery level
        if self.get_battery_level() < 15:  # 15% threshold
            if action not in ['return_home', 'stop_robot']:
                return False

        # Check joint limits
        if action == 'grasp_object':
            if self.would_exceed_joint_limits():
                return False

        return True
```

## Integration with ROS 2

### VLA Node Architecture

```
┌─────────────────┐    /camera/image    ┌─────────────────┐
│   Camera Node   │ ──────────────────▶ │   VLA Node      │
│                 │                     │                 │
└─────────────────┘                     │ 1. Process img  │
                                        │ 2. Query LLM    │
┌─────────────────┐    /robot/state     │ 3. Generate cmd │
│ State Publisher │ ──────────────────▶ │ 4. Validate     │
│                 │                     │ 5. Publish      │
└─────────────────┘                     └─────────────────┘
                                               │
                                               ▼
┌─────────────────┐                    ┌─────────────────┐
│ Action Executor │ ◀────────────────── │   Command       │
│                 │    /vla/command    │   Publisher     │
└─────────────────┘                    └─────────────────┘
```

### ROS 2 VLA Node Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import ollama

class VLARobotNode(Node):
    def __init__(self):
        super().__init__('vla_robot_node')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vla_cmd_pub = self.create_publisher(String, '/vla/commands', 10)

        # Components
        self.bridge = CvBridge()
        self.vla_pipeline = VLAPipeline()

        self.get_logger().info('VLA Robot Node initialized')

    def image_callback(self, msg):
        """Process image and generate VLA commands."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Get robot state
            robot_state = self.get_robot_state()

            # Generate action with VLA model
            command = self.vla_pipeline.generate_action(
                cv_image, robot_state
            )

            # Publish for execution
            self.vla_cmd_pub.publish(String(data=command))

        except Exception as e:
            self.get_logger().error(f'Error in VLA pipeline: {e}')
```

## Exercises

<Exercise title="Simple VLA Command Generator" difficulty="intermediate" estimatedTime="30 min">

Create a VLA node that:
1. Takes an image and text command as input
2. Uses Ollama to generate a sequence of robot actions
3. Publishes actions to a ROS 2 topic
4. Includes basic safety validation

**Requirements:**
- Use `llava:13b` for vision processing
- Use `llama3:8b` for action planning
- Implement safety checks for obstacles
- Test with sample images and commands

<Hint>
Structure your prompt to include context about the robot's capabilities and environment.
</Hint>

</Exercise>

<Exercise title="VLA Navigation System" difficulty="advanced" estimatedTime="60 min">

Create a complete VLA navigation system that:
1. Processes camera input continuously
2. Accepts natural language navigation commands
3. Generates and executes navigation actions
4. Handles dynamic obstacles and replanning

**Requirements:**
- Integrate with Nav2 for path planning
- Use costmap for obstacle avoidance
- Implement feedback loop for course correction
- Handle command interruption and cancellation

</Exercise>

## Summary

Key concepts covered:

- ✅ VLA model architectures and capabilities
- ✅ Local inference with Ollama
- ✅ Vision-language pipeline design
- ✅ Action generation from language
- ✅ Safety constraints and validation
- ✅ ROS 2 integration patterns

## Next Steps

Continue to [Ollama Setup](/module-4/week-10/ollama-setup) to learn how to configure and optimize your local LLM inference system.