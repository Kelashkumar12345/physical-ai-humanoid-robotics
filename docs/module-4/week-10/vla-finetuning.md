---
sidebar_position: 2
title: VLA Fine-tuning
description: Adapting VLA models for specific robots and tasks
---

# Fine-tuning VLA Models

Learn to adapt pretrained VLA models to your specific robot hardware and manipulation tasks.

## Learning Objectives

By the end of this lesson, you will:

- Understand fine-tuning strategies for VLA models
- Prepare robot demonstration datasets
- Implement efficient fine-tuning techniques
- Evaluate fine-tuned model performance

## Fine-tuning Strategies

### Full Fine-tuning vs. Parameter-Efficient Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                Fine-tuning Strategy Comparison                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Full Fine-tuning          LoRA                   Adapters      │
│  ─────────────────        ─────                   ────────      │
│  ┌───────────────┐        ┌─────────────┐        ┌──────────┐  │
│  │ All weights   │        │   Original  │        │ Original │  │
│  │ updated       │        │   weights   │        │ weights  │  │
│  │               │        │  (frozen)   │        │ (frozen) │  │
│  │  ●●●●●●●●●●  │        │   ○○○○○○○   │        │  ○○○○○○  │  │
│  │  ●●●●●●●●●●  │        │      ↓      │        │     ↓    │  │
│  │  ●●●●●●●●●●  │        │  ┌──────┐   │        │ ┌──────┐ │  │
│  └───────────────┘        │  │LoRA │   │        │ │Adapt │ │  │
│                           │  │ A×B │   │        │ │      │ │  │
│  Parameters: ~7B          │  └──────┘   │        │ └──────┘ │  │
│  Memory: ~60GB            └─────────────┘        └──────────┘  │
│  Training time: Days      Parameters: ~1%        Parameters:~2%│
│                           Memory: ~16GB          Memory: ~20GB │
│                           Time: Hours            Time: Hours   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA (Low-Rank Adaptation)

```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()

        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Get dimensions
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # Original forward
        original_output = self.original(x)

        # LoRA adaptation
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return original_output + lora_output


def apply_lora_to_vla(model, rank=8, target_modules=None):
    """Apply LoRA to VLA model."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return peft_model
```

## Dataset Preparation

### Demonstration Collection

```python
import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Demonstration:
    """Single robot demonstration."""
    task_instruction: str
    observations: List[Dict[str, np.ndarray]]
    actions: List[np.ndarray]
    rewards: List[float]
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class DemonstrationCollector:
    """Collect robot demonstrations for fine-tuning."""

    def __init__(self, robot_interface, save_path):
        self.robot = robot_interface
        self.save_path = save_path
        self.demonstrations = []

    def collect_demonstration(self, task_instruction):
        """Collect a single demonstration."""
        observations = []
        actions = []
        rewards = []

        print(f"Collecting demonstration for: {task_instruction}")
        print("Press 'q' to finish, 's' for success, 'f' for failure")

        # Reset robot
        self.robot.reset()

        collecting = True
        while collecting:
            # Get current observation
            obs = {
                'rgb': self.robot.get_camera_image(),
                'depth': self.robot.get_depth_image(),
                'proprioception': self.robot.get_joint_states()
            }
            observations.append(obs)

            # Get action from teleop or demonstration device
            action = self.robot.get_teleop_action()
            actions.append(action)

            # Execute action
            reward = self.robot.step(action)
            rewards.append(reward)

            # Check for end condition
            key = self.robot.check_keyboard()
            if key == 'q':
                collecting = False
            elif key == 's':
                collecting = False
                success = True
            elif key == 'f':
                collecting = False
                success = False

        # Create demonstration
        demo = Demonstration(
            task_instruction=task_instruction,
            observations=observations,
            actions=actions,
            rewards=rewards,
            success=success,
            metadata={
                'robot': self.robot.name,
                'timestamp': time.time()
            }
        )

        self.demonstrations.append(demo)

        return demo

    def save_dataset(self, filename):
        """Save demonstrations to HDF5."""
        filepath = f"{self.save_path}/{filename}.hdf5"

        with h5py.File(filepath, 'w') as f:
            for i, demo in enumerate(self.demonstrations):
                grp = f.create_group(f'demo_{i}')

                # Task instruction
                grp.attrs['instruction'] = demo.task_instruction
                grp.attrs['success'] = demo.success

                # Observations
                obs_grp = grp.create_group('observations')
                for j, obs in enumerate(demo.observations):
                    step_grp = obs_grp.create_group(f'step_{j}')
                    for key, value in obs.items():
                        step_grp.create_dataset(key, data=value)

                # Actions
                grp.create_dataset('actions', data=np.array(demo.actions))

                # Rewards
                grp.create_dataset('rewards', data=np.array(demo.rewards))

        print(f"Saved {len(self.demonstrations)} demonstrations to {filepath}")
```

### Dataset Formatting

```python
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

class VLAFineTuningDataset(Dataset):
    """Dataset for VLA fine-tuning."""

    def __init__(self, hdf5_path, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length

        # Load demonstrations
        self.samples = self._load_demonstrations(hdf5_path)

    def _load_demonstrations(self, path):
        """Load demonstrations from HDF5."""
        samples = []

        with h5py.File(path, 'r') as f:
            for demo_key in f.keys():
                demo = f[demo_key]
                instruction = demo.attrs['instruction']

                obs_grp = demo['observations']
                actions = demo['actions'][:]

                # Create samples for each timestep
                for i, step_key in enumerate(sorted(obs_grp.keys())):
                    step = obs_grp[step_key]

                    samples.append({
                        'instruction': instruction,
                        'image': step['rgb'][:],
                        'action': actions[i],
                        'timestep': i,
                        'total_steps': len(actions)
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Process image and instruction
        inputs = self.processor(
            images=sample['image'],
            text=sample['instruction'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add action labels
        inputs['action_labels'] = torch.tensor(
            sample['action'],
            dtype=torch.float32
        )

        return inputs


def create_dataloader(dataset, batch_size=8, shuffle=True):
    """Create DataLoader with proper collation."""

    def collate_fn(batch):
        """Custom collation for VLA data."""
        collated = {}

        for key in batch[0].keys():
            if key == 'action_labels':
                collated[key] = torch.stack([b[key] for b in batch])
            else:
                collated[key] = torch.stack([b[key] for b in batch])

        return collated

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
```

## Training Pipeline

### Training Configuration

```python
from dataclasses import dataclass

@dataclass
class VLATrainingConfig:
    """Configuration for VLA fine-tuning."""

    # Model
    model_name: str = "openvla/openvla-7b"
    lora_rank: int = 8
    lora_alpha: int = 16

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Data
    train_data_path: str = "data/train.hdf5"
    val_data_path: str = "data/val.hdf5"
    max_length: int = 2048

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    # Output
    output_dir: str = "checkpoints/"
```

### Training Loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

class VLATrainer:
    """Trainer for VLA fine-tuning."""

    def __init__(self, config: VLATrainingConfig):
        self.config = config

        # Load model
        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(config.model_name)

        # Apply LoRA
        self.model = apply_lora_to_vla(
            self.model,
            rank=config.lora_rank
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Load datasets
        self.train_dataset = VLAFineTuningDataset(
            config.train_data_path,
            self.processor
        )
        self.val_dataset = VLAFineTuningDataset(
            config.val_data_path,
            self.processor
        )

        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_model(self):
        """Load pretrained VLA model."""
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    def train(self):
        """Run training loop."""
        wandb.init(project="vla-finetuning", config=self.config.__dict__)

        global_step = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )

                # Compute action loss
                action_loss = self._compute_action_loss(
                    outputs,
                    batch['action_labels']
                )

                # Total loss
                loss = outputs.loss + action_loss
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

                # Logging
                if global_step % self.config.log_interval == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    }, step=global_step)

                # Evaluation
                if global_step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    wandb.log(val_metrics, step=global_step)

                # Save checkpoint
                if global_step % self.config.save_interval == 0:
                    self.save_checkpoint(global_step)

            print(f"Epoch {epoch} - Loss: {epoch_loss / len(self.train_loader):.4f}")

        # Save final model
        self.save_checkpoint("final")
        wandb.finish()

    def _compute_action_loss(self, outputs, action_labels):
        """Compute action prediction loss."""
        # Extract action predictions from model outputs
        # This depends on model architecture
        action_preds = self.model.decode_actions(outputs.logits)

        # MSE loss for continuous actions
        loss = nn.functional.mse_loss(action_preds, action_labels)

        return loss

    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        action_errors = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )

                total_loss += outputs.loss.item()

                # Action error
                action_preds = self.model.decode_actions(outputs.logits)
                error = torch.abs(action_preds - batch['action_labels']).mean()
                action_errors.append(error.item())

        self.model.train()

        return {
            'val/loss': total_loss / len(self.val_loader),
            'val/action_mae': np.mean(action_errors)
        }

    def save_checkpoint(self, step):
        """Save model checkpoint."""
        path = f"{self.config.output_dir}/checkpoint_{step}"

        # Save LoRA weights only
        self.model.save_pretrained(path)

        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step
        }, f"{path}/training_state.pt")

        print(f"Saved checkpoint to {path}")
```

## Data Augmentation

### Image Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VLAAugmentation:
    """Data augmentation for VLA training."""

    def __init__(self, image_size=224):
        self.transform = A.Compose([
            # Geometric
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),

            # Color
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

        # Corresponding action transforms
        self.action_transforms = {
            'horizontal_flip': self._flip_action,
            'rotate': self._rotate_action
        }

    def __call__(self, image, action):
        """Apply augmentation to image and action."""
        # Get augmentation parameters
        transformed = self.transform(image=image)

        # Transform action accordingly
        if transformed.get('horizontal_flip'):
            action = self._flip_action(action)

        if transformed.get('rotate'):
            action = self._rotate_action(action, transformed['rotate'])

        return transformed['image'], action

    def _flip_action(self, action):
        """Flip action for horizontal flip."""
        action = action.copy()
        action[1] = -action[1]  # Flip y
        action[4] = -action[4]  # Flip roll
        return action

    def _rotate_action(self, action, angle):
        """Rotate action for image rotation."""
        # Rotate xy components
        rad = np.radians(angle)
        cos, sin = np.cos(rad), np.sin(rad)

        x, y = action[0], action[1]
        action[0] = cos * x - sin * y
        action[1] = sin * x + cos * y

        # Adjust yaw
        action[5] -= rad

        return action
```

### Action Chunking

```python
class ActionChunking:
    """Predict sequences of actions for temporal consistency."""

    def __init__(self, chunk_size=4, overlap=1):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, actions):
        """Create overlapping action chunks."""
        chunks = []

        for i in range(0, len(actions) - self.chunk_size + 1,
                       self.chunk_size - self.overlap):
            chunk = actions[i:i + self.chunk_size]
            chunks.append(chunk)

        return np.array(chunks)

    def execute_chunk(self, chunk, robot):
        """Execute action chunk with temporal ensemble."""
        executed = []

        for i, action in enumerate(chunk):
            # Weight decreases for later actions
            weight = 1.0 - (i / self.chunk_size) * 0.5

            # Blend with previous action
            if executed:
                action = weight * action + (1 - weight) * executed[-1]

            robot.step(action)
            executed.append(action)

        return executed
```

## Evaluation

### In-Simulation Evaluation

```python
class VLAEvaluator:
    """Evaluate fine-tuned VLA model."""

    def __init__(self, model, env, tasks):
        self.model = model
        self.env = env
        self.tasks = tasks

    def evaluate(self, num_episodes=50):
        """Run evaluation episodes."""
        results = {task: [] for task in self.tasks}

        for task in self.tasks:
            for _ in range(num_episodes):
                success, metrics = self.run_episode(task)
                results[task].append({
                    'success': success,
                    **metrics
                })

        return self.aggregate_results(results)

    def run_episode(self, task, max_steps=200):
        """Run single evaluation episode."""
        obs = self.env.reset()
        total_reward = 0
        trajectory = []

        for step in range(max_steps):
            # Get action from model
            action = self.model.predict_action(
                obs['rgb'],
                task
            )

            trajectory.append({
                'observation': obs,
                'action': action
            })

            # Execute action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break

        return info.get('success', False), {
            'reward': total_reward,
            'steps': step + 1,
            'trajectory': trajectory
        }

    def aggregate_results(self, results):
        """Aggregate evaluation results."""
        summary = {}

        for task, episodes in results.items():
            successes = [ep['success'] for ep in episodes]
            rewards = [ep['reward'] for ep in episodes]
            steps = [ep['steps'] for ep in episodes]

            summary[task] = {
                'success_rate': np.mean(successes),
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_steps': np.mean(steps)
            }

        # Overall metrics
        all_successes = [
            ep['success']
            for episodes in results.values()
            for ep in episodes
        ]
        summary['overall'] = {
            'success_rate': np.mean(all_successes),
            'num_tasks': len(self.tasks)
        }

        return summary
```

## Summary

In this lesson, you learned:

- Fine-tuning strategies (full, LoRA, adapters)
- Collecting and formatting demonstration datasets
- Implementing VLA training pipelines
- Data augmentation for robotics
- Evaluating fine-tuned models

## Next Steps

Continue to [Week 10 Exercises](/module-4/week-10/exercises) to practice fine-tuning VLA models.
