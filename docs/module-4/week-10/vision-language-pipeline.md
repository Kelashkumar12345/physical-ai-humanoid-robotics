---
sidebar_position: 3
title: Vision-Language Pipeline
description: Building complete vision-language systems for robot control
---

# Vision-Language Pipeline

This lesson covers building complete vision-language pipelines that process camera input and generate robot commands from natural language instructions.

<LearningObjectives
  objectives={[
    "Design vision-language pipeline architectures",
    "Implement real-time image processing for robotics",
    "Create multimodal fusion for vision and language",
    "Build end-to-end VLA systems with feedback loops",
    "Optimize pipelines for real-time performance"
  ]}
/>

## Vision-Language Pipeline Architecture

### End-to-End Pipeline Design

<ArchitectureDiagram title="VLA Pipeline Architecture">
{`
┌─────────────────┐    Camera    ┌─────────────────┐
│   Robot Camera  │ ──────────▶ │  Preprocessing  │
│                 │             │                 │
└─────────────────┘             │ 1. Resize       │
                                │ 2. Normalize    │
                                │ 3. Format       │
┌─────────────────┐             └─────────────────┘
│ Natural Language│ ──────────▶ │                 │
│   Command       │   Text      │  Vision-Language│
│                 │             │     Fusion      │
└─────────────────┘             │                 │
                                │ 1. Embed images │
                                │ 2. Embed text   │
                                │ 3. Combine      │
                                │ 4. Generate     │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │   Action        │
                                │   Generation    │
                                │                 │
                                │ 1. Parse output │
                                │ 2. Validate     │
                                │ 3. Execute      │
                                └─────────────────┘
`}
</ArchitectureDiagram>

### Pipeline Components

| Component | Function | Implementation |
|-----------|----------|----------------|
| **Image Preprocessing** | Resize, normalize, format | OpenCV, PIL |
| **Text Preprocessing** | Tokenization, formatting | Transformers |
| **Multimodal Fusion** | Combine vision and language | VLA models |
| **Action Parsing** | Convert output to robot commands | Regex, JSON |

## Real-Time Image Processing

### Camera Data Pipeline

```python
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import time

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        # Processing parameters
        self.target_fps = 5  # Process every 200ms
        self.last_process_time = 0
        self.image_queue = []

    def image_callback(self, msg):
        """Handle incoming camera images."""
        current_time = time.time()

        # Throttle processing rate
        if current_time - self.last_process_time < 1.0 / self.target_fps:
            return

        self.last_process_time = current_time

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Preprocess image for VLA model
            processed_image = self.preprocess_for_vla(cv_image)

            # Publish for VLA processing
            self.process_for_vla(processed_image)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def preprocess_for_vla(self, image):
        """Preprocess image for VLA model input."""
        # Resize to model input size (typically 224x224 or 448x448)
        resized = cv2.resize(image, (224, 224))

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        # Convert BGR to RGB if needed
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)

        return rgb_image
```

### Object Detection Integration

```python
import torch
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForObjectDetection

class ObjectDetector:
    def __init__(self, model_name="microsoft/conditional-detr-resnet-50"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.eval()

    def detect_objects(self, image):
        """Detect objects in image and return structured output."""
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs to get bounding boxes and labels
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        objects = []
        for box, label, score in zip(
            results["boxes"], results["labels"], results["scores"]
        ):
            objects.append({
                'label': self.model.config.id2label[label.item()],
                'confidence': score.item(),
                'bbox': box.tolist(),
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            })

        return objects
```

## Text Processing and Understanding

### Natural Language Command Parser

```python
import re
from typing import Dict, List, Tuple

class NLCommandParser:
    def __init__(self):
        self.action_patterns = {
            'move_forward': [
                r'go forward',
                r'move forward',
                r'go straight',
                r'go ahead'
            ],
            'turn_left': [
                r'turn left',
                r'go left',
                r'rotate left'
            ],
            'turn_right': [
                r'turn right',
                r'go right',
                r'rotate right'
            ],
            'grasp_object': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'get (.+)',
                r'grab (.+)'
            ],
            'navigate_to': [
                r'go to (.+)',
                r'move to (.+)',
                r'navigate to (.+)'
            ]
        }

    def parse_command(self, command: str) -> Dict:
        """Parse natural language command into structured action."""
        command = command.lower().strip()

        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command)
                if match:
                    return {
                        'action': action,
                        'parameters': match.groups() if match.groups() else [],
                        'confidence': 0.9
                    }

        # If no pattern matches, return as general command
        return {
            'action': 'general_command',
            'parameters': [command],
            'confidence': 0.5
        }
```

### Context-Aware Command Understanding

```python
class ContextualCommandProcessor:
    def __init__(self):
        self.robot_state = {}
        self.environment_context = {}
        self.command_history = []

    def process_command_with_context(self, command: str, image_objects: List[Dict]):
        """Process command considering robot state and environment."""
        # Add context to command
        context_prompt = f"""
        Robot State: {self.robot_state}
        Environment Objects: {image_objects}
        Command: {command}

        Generate a specific action considering the current context.
        """

        # This would be processed by VLA model
        structured_command = self.query_vla_model(context_prompt)
        return structured_command

    def query_vla_model(self, prompt: str):
        """Query VLA model with context."""
        # Implementation would call Ollama or other VLA model
        pass
```

## Multimodal Fusion

### Vision-Language Embedding

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MultimodalFusion:
    def __init__(self):
        self.text_embeddings = {}
        self.image_embeddings = {}

    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text command."""
        # This would use a text encoder like SentenceTransformers
        # or call the text encoder portion of VLA model
        pass

    def embed_image(self, image) -> np.ndarray:
        """Create embedding for image."""
        # This would use a vision encoder like CLIP
        # or call the vision encoder portion of VLA model
        pass

    def fuse_embeddings(self, text_emb: np.ndarray, image_emb: np.ndarray) -> np.ndarray:
        """Combine text and image embeddings."""
        # Simple concatenation (more sophisticated methods exist)
        combined = np.concatenate([text_emb, image_emb])
        return combined / np.linalg.norm(combined)  # Normalize
```

### Attention Mechanisms

```python
import torch
import torch.nn as nn

class VisionLanguageAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear projections for attention
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, text_features, image_features):
        """Compute attention between text and image features."""
        # Project features
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)

        # Compute attention weights
        attention_weights = torch.softmax(
            torch.matmul(text_proj, image_proj.transpose(-2, -1)) /
            torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32)),
            dim=-1
        )

        # Apply attention
        attended_image = torch.matmul(attention_weights, image_features)

        # Fuse features
        fused = torch.cat([text_features, attended_image], dim=-1)
        output = self.fusion_proj(fused)

        return output
```

## Action Generation and Execution

### Command-to-Action Mapping

```python
class ActionGenerator:
    def __init__(self):
        self.action_space = {
            'navigation': ['move_forward', 'turn_left', 'turn_right', 'stop'],
            'manipulation': ['grasp', 'release', 'rotate_gripper'],
            'inspection': ['look_at', 'approach', 'scan'],
            'communication': ['speak', 'signal', 'report']
        }

    def generate_robot_actions(self, vla_output: str, robot_state: Dict):
        """Convert VLA output to executable robot actions."""
        try:
            # Parse JSON output if available
            actions = self.parse_json_output(vla_output)
        except:
            # Fallback to text parsing
            actions = self.parse_text_output(vla_output)

        # Validate actions against robot capabilities
        valid_actions = self.validate_actions(actions, robot_state)

        return valid_actions

    def parse_json_output(self, output: str):
        """Parse JSON-formatted action sequence."""
        import json
        data = json.loads(output)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'actions' in data:
            return data['actions']
        else:
            raise ValueError("Invalid JSON format")

    def validate_actions(self, actions: List[str], robot_state: Dict) -> List[str]:
        """Validate actions against robot capabilities and safety constraints."""
        valid_actions = []

        for action in actions:
            if self.is_action_valid(action, robot_state):
                valid_actions.append(action)
            else:
                self.get_logger().warn(f"Invalid action skipped: {action}")

        return valid_actions
```

### Safety-Constrained Execution

```python
class SafeActionExecutor:
    def __init__(self):
        self.safety_constraints = SafetyConstraints()
        self.robot_interface = RobotInterface()

    def execute_with_safety(self, actions: List[str], robot_state: Dict):
        """Execute actions with safety validation."""
        for action in actions:
            # Check safety constraints
            if not self.safety_constraints.is_safe(action, robot_state):
                self.get_logger().warn(f"Action blocked by safety: {action}")
                continue

            # Execute action
            try:
                self.robot_interface.execute_action(action)

                # Wait for completion or timeout
                if not self.wait_for_action_completion(action, timeout=10.0):
                    self.get_logger().error(f"Action timeout: {action}")
                    break

            except Exception as e:
                self.get_logger().error(f"Action execution failed: {e}")
                break

class SafetyConstraints:
    def is_safe(self, action: str, robot_state: Dict) -> bool:
        """Check if action is safe to execute."""
        # Check for obstacles in path
        if action in ['move_forward', 'move_backward']:
            if self.would_hit_obstacle(action, robot_state):
                return False

        # Check battery level
        if robot_state.get('battery', 100) < 10:  # 10% threshold
            if action not in ['return_home', 'dock']:
                return False

        # Check joint limits
        if action in ['grasp', 'rotate_gripper']:
            if self.would_exceed_limits(action, robot_state):
                return False

        return True
```

## Pipeline Optimization

### Real-Time Performance

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class OptimizedVLAPipeline:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.input_queue = queue.Queue(maxsize=10)  # Bounded queue
        self.output_queue = queue.Queue(maxsize=10)

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_pipeline, daemon=True
        )
        self.processing_thread.start()

    def process_pipeline(self):
        """Process pipeline in separate thread."""
        while True:
            try:
                # Get input from queue
                data = self.input_queue.get(timeout=1.0)

                # Process with VLA model
                result = self.vla_process(data)

                # Put result in output queue
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    # Drop old result if queue is full
                    pass

                self.input_queue.task_done()

            except queue.Empty:
                continue  # Check again

    def vla_process(self, data):
        """Process data through VLA pipeline."""
        # Implementation of VLA processing
        pass
```

### Memory Management

```python
import gc
import torch

class MemoryEfficientVLA:
    def __init__(self):
        self.model_cache = {}
        self.max_cache_size = 2  # Maximum models to keep in memory

    def process_with_model(self, model_name: str, inputs):
        """Process inputs with specified model, managing memory efficiently."""
        # Check if model is already loaded
        if model_name not in self.model_cache:
            # Load model if not in cache
            self.load_model(model_name)

        # Process inputs
        result = self.model_cache[model_name](inputs)

        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear Python garbage collector
        gc.collect()

        return result

    def load_model(self, model_name: str):
        """Load model with memory management."""
        if len(self.model_cache) >= self.max_cache_size:
            # Remove oldest model
            oldest_model = next(iter(self.model_cache))
            del self.model_cache[oldest_model]

        # Load new model
        # Implementation depends on your model loading method
        pass
```

## Exercises

<Exercise title="Real-Time VLA Pipeline" difficulty="advanced" estimatedTime="90 min">

Create a complete real-time VLA pipeline that:
1. Processes camera images at 5 FPS
2. Accepts natural language commands
3. Generates and executes robot actions
4. Includes safety constraints and validation
5. Handles pipeline failures gracefully

**Requirements:**
- Use threading for real-time processing
- Implement action queuing and validation
- Include performance monitoring
- Add error recovery mechanisms

<Hint>
Structure your pipeline with separate threads for:
- Image acquisition
- VLA processing
- Action execution
- Safety monitoring
</Hint>

</Exercise>

<Exercise title="Multimodal Attention System" difficulty="advanced" estimatedTime="120 min">

Build a sophisticated multimodal fusion system that:
1. Uses attention mechanisms to combine vision and language
2. Implements cross-attention between modalities
3. Provides interpretability for decision-making
4. Optimizes for real-time performance

**Requirements:**
- PyTorch implementation of attention mechanisms
- Visualization of attention weights
- Performance optimization
- Integration with ROS 2

</Exercise>

## Summary

Key concepts covered:

- ✅ Vision-language pipeline architecture
- ✅ Real-time image processing
- ✅ Text command understanding
- ✅ Multimodal fusion techniques
- ✅ Action generation and safety
- ✅ Performance optimization

## Next Steps

Continue to [Week 10 Exercises](/module-4/week-10/exercises) to practice building complete vision-language systems for robot control.