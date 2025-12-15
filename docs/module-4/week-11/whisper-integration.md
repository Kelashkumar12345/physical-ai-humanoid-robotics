---
sidebar_position: 1
title: Whisper Integration
description: Adding speech recognition to VLA systems with Whisper
---

# Week 11: Whisper Integration

<WeekHeader
  week={11}
  title="Multimodal Integration"
  module={4}
  estimatedHours={8}
  skills={["Whisper", "Speech Recognition", "Multimodal Systems", "Audio Processing"]}
/>

<LearningObjectives
  week={11}
  objectives={[
    "Integrate Whisper for speech-to-text in robotics",
    "Process audio streams in real-time for robot commands",
    "Combine speech recognition with vision-language models",
    "Implement voice-based robot interaction systems",
    "Optimize audio processing for embedded robotics"
  ]}
/>

## Introduction to Whisper for Robotics

**Whisper** is OpenAI's automatic speech recognition (ASR) system that can transcribe speech to text with high accuracy. For robotics applications, it enables voice-based interaction and command input.

### Whisper Model Variants

| Model | Size | Speed | Accuracy | VRAM | Use Case |
|-------|------|-------|----------|------|----------|
| `tiny` | 39M | Fast | Good | 1GB | Embedded robotics |
| `base` | 74M | Fast | Better | 1GB | Mobile robots |
| `small` | 244M | Medium | High | 2GB | Stationary robots |
| `medium` | 769M | Slow | Very high | 5GB | High-accuracy tasks |
| `large` | 1550M | Slow | Highest | 10GB | Professional applications |

### Why Whisper for Robotics?

- **Robustness**: Handles various accents and background noise
- **Multi-language**: Supports 99+ languages
- **Offline**: Can run without internet connection
- **Customizable**: Fine-tuning for domain-specific vocabularies

## Installing and Setting Up Whisper

### Installation Options

**Option 1: OpenAI Whisper (Python)**

```bash
pip install openai-whisper
# Note: Requires ffmpeg for audio processing
```

**Option 2: Faster Whisper (Recommended for robotics)**

```bash
pip install faster-whisper
# More efficient, GPU acceleration support
```

### Basic Whisper Usage

```python
from faster_whisper import WhisperModel

# Load model (first run downloads the model)
model = WhisperModel("base", device="cuda", compute_type="float16")

# Transcribe audio file
segments, info = model.transcribe("robot_command.wav", beam_size=5)
transcription = " ".join([segment.text for segment in segments])

print(f"Detected language: {info.language}")
print(f"Transcription: {transcription}")
```

## Real-Time Audio Processing

### Audio Stream Processing

```python
import pyaudio
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel

class AudioProcessor:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # Audio parameters
        self.rate = 16000  # Hz
        self.chunk = 1024  # samples
        self.format = pyaudio.paInt16
        self.channels = 1

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Processing queues
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def start_stream(self):
        """Start audio stream."""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

    def process_audio(self):
        """Process audio chunks in background."""
        while True:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe
                segments, _ = self.model.transcribe(
                    audio_array,
                    beam_size=5,
                    language="en"
                )

                transcription = " ".join([seg.text for seg in segments])

                # Put result in queue
                self.result_queue.put(transcription)

            except queue.Empty:
                continue

    def get_transcription(self, timeout=5.0):
        """Get latest transcription with timeout."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

### Voice Activity Detection (VAD)

```python
import webrtcvad
from collections import deque

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.frame_size = int(sample_rate * frame_duration / 1000)

        # Speech detection history
        self.speech_history = deque(maxlen=10)
        self.is_speaking = False

    def is_voice_active(self, audio_chunk):
        """Check if voice is active in audio chunk."""
        # Convert to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()

        # Check for voice activity
        try:
            is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
            self.speech_history.append(is_speech)

            # Update speaking state based on majority of recent frames
            recent_speech = sum(self.speech_history) / len(self.speech_history)
            self.is_speaking = recent_speech > 0.3  # 30% threshold

            return self.is_speaking
        except:
            return False
```

## ROS 2 Integration

### Audio Input Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')

        # Publishers
        self.transcription_pub = self.create_publisher(String, 'transcription', 10)
        self.command_pub = self.create_publisher(String, 'robot_command', 10)

        # Whisper model
        self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

        # Audio parameters
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # Process audio every 2 seconds
        self.timer = self.create_timer(2.0, self.process_audio_chunk)

        # Audio buffer
        self.audio_buffer = []

        self.get_logger().info('Whisper Node initialized')

    def process_audio_chunk(self):
        """Process accumulated audio and transcribe."""
        # Read audio data
        data = self.stream.read(self.chunk * 8)  # 0.5 seconds of audio
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to buffer
        self.audio_buffer.extend(audio_array)

        # If buffer is large enough, transcribe
        if len(self.audio_buffer) > self.rate * 2:  # 2 seconds of audio
            audio_segment = np.array(self.audio_buffer[-self.rate*2:])  # Last 2 seconds

            try:
                segments, _ = self.whisper_model.transcribe(
                    audio_segment,
                    beam_size=5,
                    language="en"
                )

                transcription = " ".join([seg.text for seg in segments]).strip()

                if transcription:  # Only publish if there's content
                    # Publish transcription
                    trans_msg = String()
                    trans_msg.data = transcription
                    self.transcription_pub.publish(trans_msg)

                    # Process as robot command
                    self.process_robot_command(transcription)

            except Exception as e:
                self.get_logger().error(f'Transcription error: {e}')

            # Clear buffer to avoid accumulating too much
            self.audio_buffer = self.audio_buffer[-self.rate:]  # Keep last 1 second

    def process_robot_command(self, transcription):
        """Process transcription as robot command."""
        # Simple command extraction
        if any(word in transcription.lower() for word in ['move', 'go', 'forward', 'backward', 'left', 'right']):
            cmd_msg = String()
            cmd_msg.data = transcription
            self.command_pub.publish(cmd_msg)
            self.get_logger().info(f'Robot command: {transcription}')
```

### Voice Command Parser

```python
import re
from typing import Dict, List

class VoiceCommandParser:
    def __init__(self):
        self.command_patterns = {
            # Navigation commands
            'move_forward': [
                r'go forward',
                r'move forward',
                r'go straight',
                r'go ahead',
                r'move straight'
            ],
            'turn_left': [
                r'turn left',
                r'go left',
                r'turn to the left'
            ],
            'turn_right': [
                r'turn right',
                r'go right',
                r'turn to the right'
            ],
            'move_backward': [
                r'go backward',
                r'move backward',
                r'go back',
                r'go backwards'
            ],
            'stop': [
                r'stop',
                r'hold',
                r'wait',
                r'pause'
            ],
            # Object interaction
            'grasp_object': [
                r'pick up (.+)',
                r'grab (.+)',
                r'get (.+)',
                r'pick (.+)'
            ],
            'describe_scene': [
                r'what do you see',
                r'describe the scene',
                r'what is there',
                r'tell me what you see'
            ],
            # Navigation to objects
            'navigate_to_object': [
                r'go to the (.+)',
                r'go to (.+)',
                r'move to the (.+)'
            ]
        }

    def parse_voice_command(self, transcription: str) -> Dict:
        """Parse voice transcription into structured command."""
        transcription = transcription.strip()
        if not transcription:
            return {'action': 'none', 'parameters': [], 'confidence': 0.0}

        # Clean transcription
        transcription = re.sub(r'[^\w\s]', ' ', transcription.lower())

        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, transcription)
                if match:
                    return {
                        'action': action,
                        'parameters': list(match.groups()) if match.groups() else [],
                        'confidence': 0.9,
                        'original': transcription
                    }

        # If no specific pattern matches, return as general command
        return {
            'action': 'general_command',
            'parameters': [transcription],
            'confidence': 0.6,
            'original': transcription
        }
```

## Multimodal Integration

### Combining Voice, Vision, and Language

```python
class MultimodalRobotController:
    def __init__(self):
        self.whisper_node = WhisperNode()
        self.vision_processor = VisionLanguageProcessor()
        self.command_parser = VoiceCommandParser()
        self.action_executor = ActionExecutor()

    def process_multimodal_command(self, voice_input, image_input):
        """Process combined voice and vision input."""
        # Process voice command
        voice_result = self.whisper_node.process_audio(voice_input)
        parsed_command = self.command_parser.parse_voice_command(voice_result)

        # Process visual scene
        scene_description = self.vision_processor.describe_scene(image_input)

        # Combine information for action generation
        action_context = {
            'voice_command': parsed_command,
            'scene_description': scene_description,
            'robot_state': self.get_robot_state()
        }

        # Generate and execute action
        action = self.generate_action(action_context)
        self.action_executor.execute(action)

    def generate_action(self, context):
        """Generate action based on multimodal context."""
        voice_cmd = context['voice_command']
        scene_desc = context['scene_description']

        # If voice command is high confidence, use it directly
        if voice_cmd['confidence'] > 0.8:
            return voice_cmd

        # Otherwise, combine with visual context
        prompt = f"""
        Voice command: {voice_cmd['original']}
        Scene description: {scene_desc}
        Robot state: {context['robot_state']}

        Generate appropriate robot action considering both the voice command and visual scene.
        """

        # This would call VLA model to generate action
        return self.query_vla_model(prompt)
```

## Performance Optimization

### Efficient Audio Processing

```python
import threading
import time
from collections import deque

class OptimizedAudioProcessor:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # Audio parameters
        self.rate = 16000
        self.chunk_size = 1024
        self.buffer_duration = 2.0  # seconds

        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.rate * self.buffer_duration))

        # Processing state
        self.is_processing = False
        self.last_transcription_time = 0
        self.min_transcription_interval = 1.0  # seconds

        # Threading
        self.processing_lock = threading.Lock()
        self.transcription_callback = None

    def add_audio_chunk(self, chunk):
        """Add audio chunk to buffer."""
        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        for sample in audio_data:
            self.audio_buffer.append(sample)

    def process_if_ready(self):
        """Process audio if conditions are met."""
        current_time = time.time()

        # Check if enough time has passed since last transcription
        if current_time - self.last_transcription_time < self.min_transcription_interval:
            return False

        # Check if buffer has enough data
        if len(self.audio_buffer) < self.rate * 1.0:  # At least 1 second
            return False

        # Check if not already processing
        if self.is_processing:
            return False

        # Process in background
        with self.processing_lock:
            if not self.is_processing:
                self.is_processing = True
                threading.Thread(target=self._transcribe_async, daemon=True).start()
                return True

        return False

    def _transcribe_async(self):
        """Transcribe audio in background thread."""
        try:
            # Convert buffer to numpy array
            audio_array = np.array(list(self.audio_buffer))

            # Transcribe
            segments, _ = self.model.transcribe(audio_array, beam_size=5)
            transcription = " ".join([seg.text for seg in segments]).strip()

            # Update last transcription time
            self.last_transcription_time = time.time()

            # Call callback if available
            if self.transcription_callback:
                self.transcription_callback(transcription)

        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            self.is_processing = False
```

### Memory Management

```python
import gc
import torch

class MemoryEfficientWhisper:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load Whisper model with memory optimization."""
        if torch.cuda.is_available():
            # Clear GPU cache before loading
            torch.cuda.empty_cache()

        self.model = WhisperModel(
            self.model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )

    def transcribe_with_memory_management(self, audio_data):
        """Transcribe with memory management."""
        try:
            segments, _ = self.model.transcribe(audio_data, beam_size=5)
            result = " ".join([seg.text for seg in segments])

            # Clear memory periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Clear cache and try again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Retry with smaller beam size
                segments, _ = self.model.transcribe(audio_data, beam_size=3)
                return " ".join([seg.text for seg in segments])
            else:
                raise e
```

## Exercises

<Exercise title="Voice-Controlled Robot" difficulty="intermediate" estimatedTime="60 min">

Create a system that:
1. Captures audio from microphone continuously
2. Uses Whisper to transcribe voice commands
3. Converts voice commands to robot actions
4. Includes voice activity detection to avoid processing silence

**Requirements:**
- Real-time audio capture
- Whisper transcription
- Command parsing
- Voice activity detection
- Action execution on robot

<Hint>
Use a state machine approach:
```python
class VoiceControlState(Enum):
    LISTENING = 1
    PROCESSING = 2
    EXECUTING = 3
```
</Hint>

</Exercise>

<Exercise title="Multimodal Command System" difficulty="advanced" estimatedTime="90 min">

Build a complete system that:
1. Processes voice commands and camera images simultaneously
2. Uses both modalities to generate robot actions
3. Implements attention mechanisms to weigh modalities
4. Provides feedback to user about command understanding

**Requirements:**
- Synchronized audio and video processing
- Multimodal fusion
- Attention-based weighting
- User feedback system

</Exercise>

## Troubleshooting

### Common Issues

:::tip Audio Quality Issues
For better transcription accuracy:
- Use directional microphone
- Reduce background noise
- Ensure 16kHz sampling rate
- Normalize audio levels
:::

:::tip Performance Issues
To improve transcription speed:
- Use smaller model (tiny/base)
- Reduce beam_size parameter
- Process shorter audio segments
- Use GPU acceleration
:::

:::tip Memory Issues
For memory-constrained systems:
- Use tiny or base models
- Clear GPU cache regularly
- Process audio in smaller chunks
- Consider offline processing
:::

## Summary

Key concepts covered:

- ✅ Whisper installation and setup for robotics
- ✅ Real-time audio processing
- ✅ Voice activity detection
- ✅ ROS 2 integration patterns
- ✅ Multimodal fusion techniques
- ✅ Performance optimization

## Next Steps

Continue to [Action Generation](/module-4/week-11/action-generation) to learn how to convert multimodal inputs into robot actions.