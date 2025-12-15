---
sidebar_position: 1
title: Multimodal Integration
description: Integrating vision, language, and proprioception for robot control
---

# Multimodal Integration for Robotics

Learn to integrate multiple sensing modalities with language understanding for robust robot control.

## Learning Objectives

By the end of this lesson, you will:

- Understand multimodal fusion architectures
- Implement sensor fusion for robot control
- Combine proprioceptive and visual feedback
- Build end-to-end multimodal pipelines

## Multimodal Sensor Fusion

### Fusion Architectures

```
┌─────────────────────────────────────────────────────────────────┐
│                   Multimodal Fusion Strategies                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Early Fusion              Mid-Level Fusion       Late Fusion   │
│  ────────────              ────────────────       ───────────   │
│                                                                  │
│  ┌─────┐                   ┌─────┐               ┌─────┐        │
│  │ RGB │──┐                │ RGB │──►Encoder     │ RGB │──►Dec  │
│  └─────┘  │                └─────┘      │        └─────┘   │    │
│           │  ┌───────┐          ┌──────▼──────┐       ┌───▼───┐ │
│  ┌─────┐  ├──►Concat │──►Model  │   Fusion    │──►Dec │Combine│ │
│  │Depth│  │  └───────┘          │   Module    │       └───▲───┘ │
│  └─────┘  │                     └──────▲──────┘           │     │
│           │                │           │        ┌─────┐   │     │
│  ┌─────┐  │                │ Depth │──►Encoder │Depth│──►Dec    │
│  │Prop │──┘                └─────┘                └─────┘       │
│  └─────┘                                                         │
│                                                                  │
│  Pros: Simple             Pros: Flexible        Pros: Modular   │
│  Cons: High dim           Cons: Complex         Cons: No early  │
│                                                    interaction   │
└─────────────────────────────────────────────────────────────────┘
```

### Early Fusion Implementation

```python
import torch
import torch.nn as nn

class EarlyFusionEncoder(nn.Module):
    """Early fusion of multiple modalities."""

    def __init__(self, config):
        super().__init__()

        # Individual preprocessing
        self.rgb_preprocess = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.depth_preprocess = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.proprio_preprocess = nn.Sequential(
            nn.Linear(config.proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Fused encoder
        self.fused_encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # 32 + 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Final projection
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7 + 32, config.embed_dim),
            nn.ReLU()
        )

    def forward(self, rgb, depth, proprio):
        """
        Forward pass with early fusion.

        Args:
            rgb: [B, 3, H, W] RGB images
            depth: [B, 1, H, W] depth images
            proprio: [B, proprio_dim] proprioceptive state
        """
        # Preprocess each modality
        rgb_feat = self.rgb_preprocess(rgb)
        depth_feat = self.depth_preprocess(depth)
        proprio_feat = self.proprio_preprocess(proprio)

        # Concatenate visual features (channel-wise)
        visual_concat = torch.cat([rgb_feat, depth_feat], dim=1)

        # Encode fused visual features
        visual_encoded = self.fused_encoder(visual_concat)

        # Flatten and combine with proprio
        visual_flat = visual_encoded.flatten(start_dim=1)
        combined = torch.cat([visual_flat, proprio_feat], dim=1)

        # Project to embedding
        embedding = self.projector(combined)

        return embedding
```

### Cross-Modal Attention

```python
class CrossModalAttention(nn.Module):
    """Attention-based fusion between modalities."""

    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for query, key, value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query_tokens, context_tokens):
        """
        Cross-attention from query to context modality.

        Args:
            query_tokens: [B, N_q, D] query modality tokens
            context_tokens: [B, N_c, D] context modality tokens

        Returns:
            attended: [B, N_q, D] query tokens with context information
        """
        B, N_q, D = query_tokens.shape
        N_c = context_tokens.shape[1]

        # Normalize
        query_tokens = self.norm1(query_tokens)
        context_tokens = self.norm2(context_tokens)

        # Project
        q = self.q_proj(query_tokens)  # [B, N_q, D]
        k = self.k_proj(context_tokens)  # [B, N_c, D]
        v = self.v_proj(context_tokens)  # [B, N_c, D]

        # Reshape for multi-head attention
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)

        return out


class MultimodalTransformer(nn.Module):
    """Transformer with cross-modal attention layers."""

    def __init__(self, dim, num_layers=6, num_heads=8):
        super().__init__()

        # Vision encoder
        self.vision_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads, dim * 4)
            for _ in range(num_layers // 2)
        ])

        # Language encoder
        self.language_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads, dim * 4)
            for _ in range(num_layers // 2)
        ])

        # Cross-modal layers
        self.vision_to_language = nn.ModuleList([
            CrossModalAttention(dim, num_heads)
            for _ in range(num_layers // 2)
        ])

        self.language_to_vision = nn.ModuleList([
            CrossModalAttention(dim, num_heads)
            for _ in range(num_layers // 2)
        ])

        # Final fusion
        self.fusion_layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4)

    def forward(self, vision_tokens, language_tokens):
        """
        Process vision and language with cross-modal attention.

        Args:
            vision_tokens: [B, N_v, D] vision tokens
            language_tokens: [B, N_l, D] language tokens

        Returns:
            fused: [B, N_v + N_l, D] fused representation
        """
        # Self-attention in each modality
        for v_layer, l_layer in zip(self.vision_encoder, self.language_encoder):
            vision_tokens = v_layer(vision_tokens)
            language_tokens = l_layer(language_tokens)

        # Cross-modal attention
        for v2l, l2v in zip(self.vision_to_language, self.language_to_vision):
            # Vision attends to language
            vision_update = l2v(vision_tokens, language_tokens)
            vision_tokens = vision_tokens + vision_update

            # Language attends to vision
            language_update = v2l(language_tokens, vision_tokens)
            language_tokens = language_tokens + language_update

        # Concatenate and fuse
        combined = torch.cat([vision_tokens, language_tokens], dim=1)
        fused = self.fusion_layer(combined)

        return fused
```

## Proprioceptive Integration

### Proprioceptive State Encoding

```python
class ProprioceptionEncoder(nn.Module):
    """Encode robot proprioceptive state."""

    def __init__(self, config):
        super().__init__()

        self.joint_encoder = nn.Sequential(
            nn.Linear(config.num_joints * 2, 128),  # pos + vel
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.ee_encoder = nn.Sequential(
            nn.Linear(7, 64),  # position (3) + quaternion (4)
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.force_encoder = nn.Sequential(
            nn.Linear(6, 32),  # force/torque (6)
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, config.proprio_embed_dim),
            nn.ReLU()
        )

    def forward(self, joint_state, ee_pose, force_torque):
        """
        Encode proprioceptive information.

        Args:
            joint_state: [B, num_joints * 2] joint positions and velocities
            ee_pose: [B, 7] end-effector pose
            force_torque: [B, 6] force/torque sensor readings
        """
        joint_feat = self.joint_encoder(joint_state)
        ee_feat = self.ee_encoder(ee_pose)
        force_feat = self.force_encoder(force_torque)

        combined = torch.cat([joint_feat, ee_feat, force_feat], dim=-1)
        proprio_embed = self.fusion(combined)

        return proprio_embed


class ProprioVisualFusion(nn.Module):
    """Fuse proprioceptive and visual information."""

    def __init__(self, vision_dim, proprio_dim, output_dim):
        super().__init__()

        # Film conditioning
        self.gamma = nn.Linear(proprio_dim, vision_dim)
        self.beta = nn.Linear(proprio_dim, vision_dim)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, vision_features, proprio_features):
        """
        Condition vision on proprioception using FiLM.

        Args:
            vision_features: [B, vision_dim] visual features
            proprio_features: [B, proprio_dim] proprioceptive features

        Returns:
            fused: [B, output_dim] fused features
        """
        gamma = self.gamma(proprio_features)
        beta = self.beta(proprio_features)

        # FiLM modulation
        modulated = gamma * vision_features + beta

        return self.output(modulated)
```

## Language Conditioning

### Instruction-Conditioned Policy

```python
from transformers import AutoModel, AutoTokenizer

class LanguageConditionedPolicy(nn.Module):
    """Policy conditioned on language instructions."""

    def __init__(self, config):
        super().__init__()

        # Language encoder
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model)
        self.language_encoder = AutoModel.from_pretrained(config.language_model)

        # Freeze language encoder
        for param in self.language_encoder.parameters():
            param.requires_grad = False

        # Vision encoder
        self.vision_encoder = self._build_vision_encoder(config)

        # Proprio encoder
        self.proprio_encoder = ProprioceptionEncoder(config)

        # Multimodal fusion
        self.cross_attention = CrossModalAttention(
            config.embed_dim,
            num_heads=config.num_heads
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.action_dim)
        )

    def _build_vision_encoder(self, config):
        """Build vision encoder."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, config.embed_dim)
        )

    def encode_language(self, instructions):
        """Encode language instructions."""
        # Tokenize
        tokens = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Encode
        with torch.no_grad():
            outputs = self.language_encoder(**tokens)

        # Use CLS token or mean pooling
        language_embed = outputs.last_hidden_state[:, 0, :]

        return language_embed

    def forward(self, image, proprio, instruction):
        """
        Generate action from multimodal input.

        Args:
            image: [B, 3, H, W] RGB image
            proprio: dict with joint_state, ee_pose, force_torque
            instruction: list of instruction strings

        Returns:
            action: [B, action_dim] predicted action
        """
        # Encode each modality
        vision_feat = self.vision_encoder(image)
        proprio_feat = self.proprio_encoder(
            proprio['joint_state'],
            proprio['ee_pose'],
            proprio['force_torque']
        )
        language_feat = self.encode_language(instruction)

        # Combine vision and proprio
        vision_proprio = vision_feat + proprio_feat

        # Cross-attention with language
        # Reshape for attention: [B, 1, D]
        vision_proprio = vision_proprio.unsqueeze(1)
        language_feat = language_feat.unsqueeze(1)

        attended = self.cross_attention(vision_proprio, language_feat)

        # Generate action
        action = self.policy_head(attended.squeeze(1))

        return action
```

## Multi-Camera Fusion

### Spatial View Aggregation

```python
class MultiCameraEncoder(nn.Module):
    """Encode and fuse multiple camera views."""

    def __init__(self, config):
        super().__init__()

        # Shared vision backbone
        self.backbone = self._build_backbone(config)

        # Camera-specific projections
        self.camera_projections = nn.ModuleDict({
            'front': nn.Linear(config.backbone_dim, config.embed_dim),
            'wrist': nn.Linear(config.backbone_dim, config.embed_dim),
            'overhead': nn.Linear(config.backbone_dim, config.embed_dim)
        })

        # Spatial attention for view fusion
        self.view_attention = nn.MultiheadAttention(
            config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        # View position embeddings
        self.view_embeddings = nn.Parameter(
            torch.randn(3, config.embed_dim)
        )

    def _build_backbone(self, config):
        """Build shared vision backbone."""
        from torchvision.models import resnet18

        backbone = resnet18(pretrained=True)
        # Remove final classification layer
        return nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, camera_images):
        """
        Encode and fuse multiple camera views.

        Args:
            camera_images: dict with 'front', 'wrist', 'overhead' images

        Returns:
            fused: [B, embed_dim] fused multi-view features
        """
        view_features = []

        for i, (camera_name, image) in enumerate(camera_images.items()):
            # Encode with shared backbone
            feat = self.backbone(image).squeeze(-1).squeeze(-1)

            # Camera-specific projection
            proj_feat = self.camera_projections[camera_name](feat)

            # Add view position embedding
            proj_feat = proj_feat + self.view_embeddings[i]

            view_features.append(proj_feat)

        # Stack views: [B, num_views, embed_dim]
        view_stack = torch.stack(view_features, dim=1)

        # Self-attention across views
        attended, _ = self.view_attention(
            view_stack, view_stack, view_stack
        )

        # Mean pool across views
        fused = attended.mean(dim=1)

        return fused
```

## End-to-End Pipeline

### Complete Multimodal Robot Controller

```python
class MultimodalRobotController(nn.Module):
    """Complete multimodal robot controller."""

    def __init__(self, config):
        super().__init__()

        # Multi-camera encoder
        self.camera_encoder = MultiCameraEncoder(config)

        # Proprio encoder
        self.proprio_encoder = ProprioceptionEncoder(config)

        # Language encoder
        self.language_encoder = SentenceTransformer(config.language_model)

        # Multimodal transformer
        self.multimodal_transformer = MultimodalTransformer(
            config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads
        )

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(config.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.action_dim)
        )

        # Action chunking
        self.chunk_size = config.action_chunk_size
        self.chunk_decoder = nn.Linear(128, config.action_dim * self.chunk_size)

    def forward(self, observation, instruction):
        """
        Generate action from observation and instruction.

        Args:
            observation: dict with cameras, proprio
            instruction: language instruction string

        Returns:
            action: [B, action_dim] or [B, chunk_size, action_dim]
        """
        # Encode cameras
        camera_feat = self.camera_encoder(observation['cameras'])

        # Encode proprio
        proprio_feat = self.proprio_encoder(
            observation['proprio']['joint_state'],
            observation['proprio']['ee_pose'],
            observation['proprio']['force_torque']
        )

        # Encode language
        with torch.no_grad():
            language_feat = self.language_encoder.encode(
                instruction,
                convert_to_tensor=True
            )

        # Create tokens
        vision_tokens = torch.cat([
            camera_feat.unsqueeze(1),
            proprio_feat.unsqueeze(1)
        ], dim=1)

        language_tokens = language_feat.unsqueeze(1)

        # Multimodal fusion
        fused = self.multimodal_transformer(vision_tokens, language_tokens)

        # Use first token for action
        action_features = fused[:, 0]

        # Decode action
        action = self.action_decoder(action_features)

        return action

    def predict_action_chunk(self, observation, instruction):
        """Predict chunk of future actions."""
        # Similar forward pass
        action_features = self._get_action_features(observation, instruction)

        # Decode chunk
        chunk = self.chunk_decoder(action_features)
        chunk = chunk.view(-1, self.chunk_size, self.action_dim)

        return chunk
```

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

class MultimodalControllerNode(Node):
    """ROS 2 node for multimodal robot control."""

    def __init__(self):
        super().__init__('multimodal_controller')

        # Load model
        self.model = MultimodalRobotController(config)
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()

        # CV bridge
        self.bridge = CvBridge()

        # State
        self.cameras = {}
        self.proprio = {}
        self.instruction = "stay still"

        # Subscribers - cameras
        self.front_sub = self.create_subscription(
            Image, '/camera/front/image_raw',
            lambda msg: self.camera_callback(msg, 'front'), 10
        )
        self.wrist_sub = self.create_subscription(
            Image, '/camera/wrist/image_raw',
            lambda msg: self.camera_callback(msg, 'wrist'), 10
        )

        # Subscribers - proprio
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )
        self.ee_sub = self.create_subscription(
            PoseStamped, '/end_effector/pose',
            self.ee_callback, 10
        )
        self.ft_sub = self.create_subscription(
            WrenchStamped, '/force_torque_sensor',
            self.ft_callback, 10
        )

        # Subscriber - instruction
        self.instruction_sub = self.create_subscription(
            String, '/instruction',
            self.instruction_callback, 10
        )

        # Publisher
        self.action_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def camera_callback(self, msg, camera_name):
        """Store camera image."""
        self.cameras[camera_name] = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def joint_callback(self, msg):
        """Store joint state."""
        self.proprio['joint_positions'] = np.array(msg.position)
        self.proprio['joint_velocities'] = np.array(msg.velocity)

    def ee_callback(self, msg):
        """Store end-effector pose."""
        self.proprio['ee_pose'] = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

    def ft_callback(self, msg):
        """Store force-torque readings."""
        self.proprio['force_torque'] = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

    def instruction_callback(self, msg):
        """Update instruction."""
        self.instruction = msg.data
        self.get_logger().info(f"Instruction: {self.instruction}")

    def control_loop(self):
        """Main control loop."""
        if not self._has_all_data():
            return

        # Prepare observation
        observation = {
            'cameras': self.cameras,
            'proprio': self.proprio
        }

        # Get action
        with torch.no_grad():
            action = self.model(observation, self.instruction)

        # Publish command
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.position = action.cpu().numpy().tolist()

        self.action_pub.publish(cmd)

    def _has_all_data(self):
        """Check if all required data is available."""
        required_cameras = ['front', 'wrist']
        required_proprio = ['joint_positions', 'ee_pose', 'force_torque']

        for cam in required_cameras:
            if cam not in self.cameras:
                return False

        for prop in required_proprio:
            if prop not in self.proprio:
                return False

        return True


def main():
    rclpy.init()
    node = MultimodalControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

## Summary

In this lesson, you learned:

- Multimodal fusion architectures (early, mid, late)
- Cross-modal attention mechanisms
- Proprioceptive state encoding
- Language-conditioned policies
- Multi-camera view fusion
- End-to-end multimodal control pipelines

## Next Steps

Continue to [Sim-to-Real Transfer](/module-4/week-11/sim-to-real) to learn how to deploy VLA models on real robots.
