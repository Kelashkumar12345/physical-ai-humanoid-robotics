---
sidebar_position: 2
title: Ollama Setup
description: Installing and configuring Ollama for local LLM inference
---

# Ollama Setup

This lesson covers installing, configuring, and optimizing Ollama for robotics applications with Vision-Language-Action models.

<LearningObjectives
  objectives={[
    "Install Ollama on Ubuntu 22.04 with GPU acceleration",
    "Configure models for robotics vision-language tasks",
    "Optimize performance for real-time inference",
    "Implement model management and switching",
    "Create robust error handling for LLM services"
  ]}
/>

## Installing Ollama

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | NVIDIA GTX 1660 (6GB) | RTX 3080 (10GB+) |
| **VRAM** | 4 GB | 8+ GB for vision models |
| **Storage** | 20 GB | 50+ GB for multiple models |

### Installation Methods

**Method 1: Official Installation Script**

```bash
# Download and run the official installer
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

**Method 2: Manual Installation (Ubuntu)**

```bash
# Download the latest release
wget https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz
sudo mv ollama /usr/local/bin/

# Install systemd service
sudo systemctl enable ollama
sudo systemctl start ollama
```

### GPU Acceleration Setup

**For NVIDIA GPUs:**

```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Ollama automatically detects CUDA-capable GPUs
# Verify GPU detection:
ollama run llama3:8b
# Look for "using GPU" in the output
```

**CUDA Version Compatibility:**

| CUDA Version | Ollama Support | Notes |
|--------------|----------------|-------|
| 11.8 | ✅ Full support | Recommended |
| 12.0 | ✅ Full support | Default for Ubuntu 22.04 |
| 12.1+ | ✅ Full support | Latest tested |

## Model Selection for Robotics

### Recommended Models

| Model | Size | Use Case | VRAM | Performance |
|-------|------|----------|------|-------------|
| `phi3:mini` | 3.8B | Fast responses, simple tasks | 2GB | Very fast |
| `mistral:7b` | 7B | General robotics tasks | 4GB | Fast |
| `llama3:8b` | 8B | Complex reasoning | 5GB | Good balance |
| `llava:13b` | 13B | Vision-language tasks | 8GB | Vision tasks |
| `llava:34b` | 34B | Advanced vision understanding | 18GB | High capability |

### Downloading Models

```bash
# Pull models in the background
ollama pull phi3:mini &    # Small, fast
ollama pull mistral:7b &   # Balanced
ollama pull llava:13b &    # Vision tasks
ollama pull llama3:8b &    # General tasks

# Check available models
ollama list
```

### Custom Model Configuration

Create a `Modelfile` for robotics-specific tuning:

```
# Modelfile for Robotics VLA
FROM llama3:8b

# System prompt for robotics
SYSTEM """
You are a helpful robotics assistant. Generate clear, executable commands for robots.
- Be concise but specific
- Consider safety in all responses
- Use structured output when possible
- Acknowledge limitations of the robot
"""

# Parameters for consistent output
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER seed 42
```

Build the custom model:

```bash
# Create and build custom model
ollama create robotics-llama3 -f Modelfile
```

## Performance Optimization

### GPU Memory Management

```bash
# Check GPU memory usage
nvidia-smi

# Run with specific GPU (if multiple GPUs)
CUDA_VISIBLE_DEVICES=0 ollama run llava:13b

# Limit context size to reduce memory usage
ollama run --options='{"num_ctx": 1024}' llama3:8b
```

### Concurrency and Batch Processing

```python
import asyncio
import aiohttp
from typing import List, Dict

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def generate_batch(self, prompts: List[str], model: str = "llama3:8b"):
        """Generate responses for multiple prompts concurrently."""
        tasks = [
            self.generate_single(prompt, model)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def generate_single(self, prompt: str, model: str):
        """Generate a single response."""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=data
        ) as response:
            result = await response.json()
            return result["response"]
```

### Model Quantization

```bash
# Pull quantized models for faster inference
ollama pull llama3:8b-text-q4_0  # Smaller, slightly less accurate
ollama pull mistral:7b-instruct-q4_K_M  # Quantized for speed
```

## ROS 2 Integration

### Ollama Service Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ollama
import json

class OllamaServiceNode(Node):
    def __init__(self):
        super().__init__('ollama_service')

        # Subscribers
        self.text_sub = self.create_subscription(
            String, 'llm_input', self.text_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image', self.image_callback, 10
        )

        # Publishers
        self.response_pub = self.create_publisher(String, 'llm_response', 10)

        # Components
        self.bridge = CvBridge()
        self.active_model = "llama3:8b"

        self.get_logger().info('Ollama Service Node started')

    def text_callback(self, msg):
        """Handle text-based LLM requests."""
        try:
            response = ollama.chat(
                model=self.active_model,
                messages=[{'role': 'user', 'content': msg.data}]
            )

            response_msg = String()
            response_msg.data = response['message']['content']
            self.response_pub.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')

    def image_callback(self, msg):
        """Handle image-based VLA requests."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Convert to format Ollama expects
            temp_path = '/tmp/current_image.jpg'
            cv2.imwrite(temp_path, cv_image)

            response = ollama.chat(
                model='llava:13b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this scene for a robot. What objects are visible and where are they located?',
                        'images': [temp_path]
                    }
                ]
            )

            response_msg = String()
            response_msg.data = response['message']['content']
            self.response_pub.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'VLA error: {e}')
```

### Model Management

```python
class ModelManager:
    def __init__(self):
        self.available_models = self.get_available_models()
        self.current_model = "llama3:8b"

    def get_available_models(self):
        """Get list of locally available models."""
        result = ollama.list()
        return [model['name'] for model in result['models']]

    def switch_model(self, model_name):
        """Switch to a different model."""
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        else:
            raise ValueError(f"Model {model_name} not available")

    def get_model_info(self, model_name):
        """Get information about a specific model."""
        info = ollama.show(model_name)
        return info
```

## Troubleshooting

### Common Issues

:::tip GPU Not Detected
If Ollama doesn't detect your GPU:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Restart Ollama service
sudo systemctl restart ollama
```
:::

:::tip Out of Memory
Reduce context size or use smaller models:
```bash
# Run with smaller context
ollama run --options='{"num_ctx": 512}' llama3:8b

# Use quantized models
ollama run llama3:8b-text-q4_0
```
:::

:::tip Slow Inference
Optimize for speed:
```bash
# Use smaller models for faster responses
ollama run phi3:mini

# Reduce temperature for faster generation
ollama run --options='{"temperature": 0.1}' llama3:8b
```
:::

### Health Checks

```bash
# Check if Ollama service is running
systemctl status ollama

# Test basic functionality
ollama run llama3:8b
# Type: "Hello, are you working?"
# Should respond appropriately

# Check available memory
free -h
nvidia-smi
```

## Exercises

<Exercise title="Ollama Model Benchmark" difficulty="intermediate" estimatedTime="45 min">

Create a benchmark script that:
1. Tests different models with the same prompt
2. Measures response time and quality
3. Monitors GPU/CPU usage during inference
4. Recommends optimal model for robotics tasks

**Requirements:**
- Test at least 3 different models
- Measure time and memory usage
- Compare output quality for robotics commands
- Create a recommendation system

<Hint>
Use Python's `time` module and `psutil` for monitoring:
```python
import time
import psutil

start_time = time.time()
response = ollama.generate(model="...", prompt=...)
end_time = time.time()
```
</Hint>

</Exercise>

<Exercise title="Robust LLM Service" difficulty="advanced" estimatedTime="60 min">

Create a fault-tolerant Ollama service that:
1. Handles model failures gracefully
2. Falls back to different models when needed
3. Monitors service health continuously
4. Implements request queuing for high load

**Requirements:**
- Automatic model switching on failure
- Health monitoring with alerts
- Request queue with timeout handling
- Performance metrics logging

</Exercise>

## Summary

Key concepts covered:

- ✅ Ollama installation with GPU acceleration
- ✅ Model selection for robotics applications
- ✅ Performance optimization techniques
- ✅ ROS 2 integration patterns
- ✅ Error handling and troubleshooting

## Next Steps

Continue to [Vision-Language Pipeline](/module-4/week-10/vision-language-pipeline) to learn how to build complete vision-language systems for robot control.