---
sidebar_position: 1
title: VLA Foundations
description: Introduction to Vision-Language-Action models for robotics
---

# Vision-Language-Action Foundations

Understand the fundamentals of VLA models that enable robots to understand natural language instructions and visual scenes to generate appropriate actions.

## Learning Objectives

By the end of this lesson, you will:

- Understand VLA model architecture and components
- Learn about multimodal representation learning
- Explore language-conditioned robot control
- Implement basic VLA inference pipelines

## What are VLA Models?

Vision-Language-Action (VLA) models are multimodal AI systems that:

1. **See**: Process visual input (images, video, depth)
2. **Understand**: Interpret natural language instructions
3. **Act**: Generate robot actions or control commands

```
┌─────────────────────────────────────────────────────────────────┐
│                    VLA Model Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐ │
│   │  Camera  │    │ Language │    │                          │ │
│   │  Images  │    │Instruction│   │    Multimodal Encoder    │ │
│   └────┬─────┘    └────┬─────┘    │                          │ │
│        │               │          │  ┌────────┐ ┌─────────┐  │ │
│        ▼               ▼          │  │ Vision │ │Language │  │ │
│   ┌─────────┐    ┌─────────┐     │  │Encoder │ │ Encoder │  │ │
│   │  Vision │    │  Text   │     │  └───┬────┘ └────┬────┘  │ │
│   │ Encoder │    │ Encoder │     │      │           │       │ │
│   │  (ViT)  │    │  (BERT) │     │      └─────┬─────┘       │ │
│   └────┬────┘    └────┬────┘     │            │             │ │
│        │              │          │     ┌──────▼──────┐      │ │
│        └──────┬───────┘          │     │   Fusion    │      │ │
│               │                  │     │   Module    │      │ │
│        ┌──────▼──────┐           │     └──────┬──────┘      │ │
│        │   Fusion    │           └────────────┼─────────────┘ │
│        │   Layer     │                        │               │
│        └──────┬──────┘                        ▼               │
│               │                        ┌─────────────┐        │
│        ┌──────▼──────┐                 │   Action    │        │
│        │   Policy    │                 │   Decoder   │        │
│        │   Head      │                 └──────┬──────┘        │
│        └──────┬──────┘                        │               │
│               │                               ▼               │
│               ▼                        ┌─────────────┐        │
│        ┌─────────────┐                 │   Robot     │        │
│        │  Actions    │                 │   Actions   │        │
│        │ [x,y,z,θ]   │                 └─────────────┘        │
│        └─────────────┘                                        │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Key VLA Model Families

### 1. RT-2 (Robotics Transformer 2)

Google's VLA model based on PaLM-E:

```python
# Conceptual RT-2 architecture
class RT2Model:
    """RT-2: Vision-Language-Action model."""

    def __init__(self):
        # Vision encoder (ViT)
        self.vision_encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            dim=1024,
            depth=24,
            heads=16
        )

        # Language model (PaLM)
        self.language_model = PaLMDecoder(
            dim=1024,
            depth=32,
            heads=16,
            vocab_size=32000
        )

        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(
            num_bins=256,  # Discretize continuous actions
            action_dim=7   # 6-DOF + gripper
        )

    def forward(self, image, instruction):
        """Generate action from image and instruction."""
        # Encode image
        vision_tokens = self.vision_encoder(image)

        # Tokenize instruction
        text_tokens = self.tokenizer(instruction)

        # Concatenate and process
        combined = torch.cat([vision_tokens, text_tokens], dim=1)

        # Generate action tokens
        action_tokens = self.language_model.generate(
            combined,
            max_new_tokens=7  # 7-DOF action
        )

        # Decode to continuous actions
        actions = self.action_tokenizer.decode(action_tokens)

        return actions
```

### 2. OpenVLA

Open-source VLA based on LLaMA:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

class OpenVLAInference:
    """OpenVLA model for robot control."""

    def __init__(self, model_path="openvla/openvla-7b"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def predict_action(self, image, instruction):
        """Predict robot action from observation and instruction."""
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate action tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Decode action
        action = self.processor.decode_action(outputs)

        return action
```

### 3. Octo

Generalist robot policy from UC Berkeley:

```python
from octo.model.octo_model import OctoModel

class OctoController:
    """Octo generalist robot controller."""

    def __init__(self, model_path="hf://rail-berkeley/octo-base"):
        self.model = OctoModel.load_pretrained(model_path)

    def get_action(self, observation, task_description):
        """Get action for given observation and task."""
        # Format observation
        obs = {
            "image_primary": observation["rgb"],
            "image_wrist": observation.get("wrist_rgb"),
            "proprio": observation["joint_positions"]
        }

        # Create task
        task = self.model.create_tasks(texts=[task_description])

        # Predict action
        action = self.model.sample_actions(
            obs,
            task,
            rng=jax.random.PRNGKey(0)
        )

        return action
```

## Multimodal Representation Learning

### Vision-Language Alignment

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class VisionLanguageEncoder:
    """Encode images and text into aligned embedding space."""

    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image):
        """Encode image to embedding."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, text):
        """Encode text to embedding."""
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    def similarity(self, image, text):
        """Compute image-text similarity."""
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        return (img_emb @ txt_emb.T).item()
```

### Contrastive Learning for Robotics

```python
class RoboticsContrastiveLoss(nn.Module):
    """Contrastive loss for robot observation-action pairs."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, obs_embeddings, action_embeddings, labels):
        """
        Compute contrastive loss.

        Args:
            obs_embeddings: [B, D] observation embeddings
            action_embeddings: [B, D] action embeddings
            labels: [B] matching indices
        """
        # Normalize embeddings
        obs_norm = obs_embeddings / obs_embeddings.norm(dim=-1, keepdim=True)
        act_norm = action_embeddings / action_embeddings.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        similarity = obs_norm @ act_norm.T / self.temperature

        # Contrastive loss
        loss = nn.functional.cross_entropy(similarity, labels)

        return loss
```

## Language-Conditioned Control

### Task Embedding

```python
class TaskEmbedding(nn.Module):
    """Embed language instructions for robot control."""

    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()

        # Use pretrained language model
        self.language_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Project to control embedding
        self.projection = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, instructions):
        """
        Embed instruction text.

        Args:
            instructions: List of instruction strings

        Returns:
            task_embeddings: [B, embed_dim] task embeddings
        """
        # Get sentence embeddings
        with torch.no_grad():
            sentence_emb = self.language_encoder.encode(
                instructions,
                convert_to_tensor=True
            )

        # Project to control space
        task_emb = self.projection(sentence_emb)

        return task_emb
```

### Conditioned Policy Network

```python
class LanguageConditionedPolicy(nn.Module):
    """Policy network conditioned on language instructions."""

    def __init__(self, obs_dim, action_dim, task_dim=512, hidden_dim=256):
        super().__init__()

        # Task embedding
        self.task_encoder = TaskEmbedding(embed_dim=task_dim)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Film conditioning layers
        self.film_gamma = nn.Linear(task_dim, hidden_dim)
        self.film_beta = nn.Linear(task_dim, hidden_dim)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observation, instruction):
        """
        Generate action conditioned on instruction.

        Args:
            observation: [B, obs_dim] robot observation
            instruction: List of instruction strings

        Returns:
            action: [B, action_dim] robot action
        """
        # Encode task
        task_emb = self.task_encoder(instruction)

        # Encode observation
        obs_features = self.obs_encoder(observation)

        # FiLM conditioning
        gamma = self.film_gamma(task_emb)
        beta = self.film_beta(task_emb)
        conditioned = gamma * obs_features + beta

        # Generate action
        action = self.policy_head(conditioned)

        return action
```

## VLA Inference Pipeline

### Complete Pipeline

```python
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class VLAConfig:
    """VLA inference configuration."""
    model_name: str = "openvla-7b"
    image_size: tuple = (224, 224)
    action_dim: int = 7
    max_action: float = 1.0
    control_frequency: float = 10.0  # Hz

class VLAController:
    """Complete VLA controller for robot manipulation."""

    def __init__(self, config: VLAConfig):
        self.config = config

        # Load model
        self.model = self._load_model(config.model_name)

        # Action history for smoothing
        self.action_history = []
        self.max_history = 5

    def _load_model(self, model_name):
        """Load VLA model."""
        # Model-specific loading logic
        if "openvla" in model_name:
            return OpenVLAInference(model_name)
        elif "octo" in model_name:
            return OctoController(model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Resize
        image = cv2.resize(image, self.config.image_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def get_action(self, image, instruction):
        """
        Get robot action from image and instruction.

        Args:
            image: RGB image from camera
            instruction: Natural language task description

        Returns:
            action: 7-DOF action [x, y, z, roll, pitch, yaw, gripper]
        """
        # Preprocess
        processed_image = self.preprocess_image(image)

        # Get model prediction
        raw_action = self.model.predict_action(processed_image, instruction)

        # Post-process action
        action = self._postprocess_action(raw_action)

        # Smooth action
        smoothed_action = self._smooth_action(action)

        return smoothed_action

    def _postprocess_action(self, action):
        """Post-process model output to valid action."""
        # Clip to valid range
        action = np.clip(action, -self.config.max_action, self.config.max_action)

        # Ensure correct dimension
        if len(action) != self.config.action_dim:
            action = np.zeros(self.config.action_dim)
            action[:len(action)] = action

        return action

    def _smooth_action(self, action):
        """Apply exponential smoothing to actions."""
        self.action_history.append(action)

        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        # Weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.action_history)))
        weights /= weights.sum()

        smoothed = np.zeros_like(action)
        for w, a in zip(weights, self.action_history):
            smoothed += w * a

        return smoothed
```

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge

class VLANode(Node):
    """ROS 2 node for VLA-based robot control."""

    def __init__(self):
        super().__init__('vla_controller')

        # Initialize VLA
        config = VLAConfig()
        self.vla = VLAController(config)

        # CV bridge
        self.bridge = CvBridge()

        # Current instruction
        self.current_instruction = "stay still"

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
        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.latest_image = None

    def image_callback(self, msg):
        """Store latest image."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def instruction_callback(self, msg):
        """Update current instruction."""
        self.current_instruction = msg.data
        self.get_logger().info(f"New instruction: {self.current_instruction}")

    def control_loop(self):
        """Main control loop."""
        if self.latest_image is None:
            return

        # Get action from VLA
        action = self.vla.get_action(
            self.latest_image,
            self.current_instruction
        )

        # Convert to Twist message
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.y = float(action[1])
        cmd.angular.z = float(action[5])

        # Publish
        self.action_pub.publish(cmd)

def main():
    rclpy.init()
    node = VLANode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Evaluation Metrics

### Action Prediction Accuracy

```python
class VLAMetrics:
    """Metrics for VLA model evaluation."""

    @staticmethod
    def action_mse(predicted, ground_truth):
        """Mean squared error between actions."""
        return np.mean((predicted - ground_truth) ** 2)

    @staticmethod
    def success_rate(episodes):
        """
        Compute task success rate.

        Args:
            episodes: List of episode results with 'success' key
        """
        successes = sum(1 for ep in episodes if ep['success'])
        return successes / len(episodes)

    @staticmethod
    def trajectory_similarity(predicted_traj, ground_truth_traj):
        """
        Dynamic Time Warping distance between trajectories.
        """
        from dtw import dtw

        distance, _, _, _ = dtw(
            predicted_traj,
            ground_truth_traj,
            dist=lambda x, y: np.linalg.norm(x - y)
        )

        return distance

    @staticmethod
    def language_grounding_score(model, test_pairs):
        """
        Evaluate language understanding.

        Args:
            test_pairs: List of (image, instruction, expected_action) tuples
        """
        scores = []

        for image, instruction, expected in test_pairs:
            predicted = model.predict_action(image, instruction)
            score = 1.0 - VLAMetrics.action_mse(predicted, expected)
            scores.append(max(0, score))

        return np.mean(scores)
```

## Summary

In this lesson, you learned:

- VLA model architectures (RT-2, OpenVLA, Octo)
- Multimodal representation learning techniques
- Language-conditioned robot control
- Building VLA inference pipelines
- ROS 2 integration for VLA controllers

## Next Steps

Continue to [VLA Fine-tuning](/module-4/week-10/vla-finetuning) to learn how to adapt VLA models for specific robots and tasks.
