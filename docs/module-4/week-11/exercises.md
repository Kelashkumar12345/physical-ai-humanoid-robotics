---
sidebar_position: 4
title: Week 11 Exercises
description: Hands-on exercises for multimodal integration and safety systems
---

# Week 11 Exercises

Complete these exercises to build practical experience with multimodal integration and safety systems for VLA-powered robots.

<Prerequisites
  items={[
    "Completed all Week 11 lessons",
    "Whisper models installed and tested",
    "Ollama with VLA models running",
    "ROS 2 Humble with audio/sensor simulation"
  ]}
/>

## Exercise 1: Voice-Controlled Navigation

<Exercise title="Voice Command Navigation System" difficulty="intermediate" estimatedTime="60 min">

Create a system that:
1. Captures voice commands and transcribes them with Whisper
2. Uses VLA models to generate navigation actions
3. Executes navigation with safety validation
4. Provides voice feedback to user

**Requirements:**
- Real-time speech-to-text processing
- Natural language command interpretation
- Navigation action generation
- Safety constraint validation
- Voice feedback system

**Acceptance Criteria:**
- [ ] Voice commands correctly transcribed
- [ ] Navigation commands executed safely
- [ ] Safety constraints enforced
- [ ] Voice feedback provided to user

<Hint>
Structure your system with these components:
- Audio input node
- Whisper transcription node
- Command interpretation node
- Navigation safety validator
- Feedback generation node
</Hint>

<Solution>
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import threading
import queue

class VoiceNavigationNode(Node):
    def __init__(self):
        super().__init__('voice_navigation')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, 'speak_text', 10)
        self.status_pub = self.create_publisher(String, 'voice_status', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )

        # Whisper model
        self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

        # Audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.obstacle_distance = float('inf')

        # Start audio processing
        self.start_audio_capture()

        self.get_logger().info('Voice Navigation Node initialized')

    def laser_callback(self, msg):
        """Update obstacle distance from laser scan."""
        if len(msg.ranges) > 0:
            front_ranges = msg.ranges[:30] + msg.ranges[-30:]  # Front 60 degrees
            valid_ranges = [r for r in front_ranges if 0 < r < float('inf')]
            if valid_ranges:
                self.obstacle_distance = min(valid_ranges)

    def start_audio_capture(self):
        """Start audio capture in background thread."""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self.process_transcriptions, daemon=True)
        self.transcription_thread.start()

    def process_audio(self):
        """Process audio chunks."""
        while True:
            # Read audio chunk
            data = self.stream.read(1024 * 8)  # 0.5 seconds
            self.audio_queue.put(data)

    def process_transcriptions(self):
        """Process transcriptions and execute commands."""
        while True:
            try:
                # Get audio data
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Convert to numpy
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe
                segments, _ = self.whisper_model.transcribe(audio_array, beam_size=5)
                transcription = " ".join([seg.text for seg in segments]).strip()

                if transcription:
                    self.result_queue.put(transcription)
                    self.process_voice_command(transcription)

            except queue.Empty:
                continue

    def process_voice_command(self, command):
        """Process voice command and execute navigation."""
        command_lower = command.lower()

        # Publish status
        status_msg = String()
        status_msg.data = f"Heard: {command}"
        self.status_pub.publish(status_msg)

        # Parse command and execute
        cmd = Twist()
        success = False

        if "go forward" in command_lower or "move forward" in command_lower:
            if self.obstacle_distance > 0.8:  # 80cm safety
                cmd.linear.x = 0.3
                success = True
            else:
                self.speak(f"Cannot go forward, obstacle at {self.obstacle_distance:.2f} meters")
        elif "turn left" in command_lower:
            cmd.angular.z = 0.5
            success = True
        elif "turn right" in command_lower:
            cmd.angular.z = -0.5
            success = True
        elif "stop" in command_lower or "halt" in command_lower:
            # cmd is already zero
            success = True

        if success:
            self.cmd_vel_pub.publish(cmd)
            self.speak(f"Executing: {command}")
        else:
            self.speak(f"Command not executed: {command}")

    def speak(self, text):
        """Publish text for speech synthesis."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

def main():
    rclpy.init()
    node = VoiceNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```
</Solution>

</Exercise>

## Exercise 2: Multimodal Safety System

<Exercise title="Integrated Safety Validation" difficulty="advanced" estimatedTime="90 min">

Create a comprehensive safety system that:
1. Validates VLA outputs against multiple safety constraints
2. Monitors robot state in real-time
3. Integrates vision, audio, and sensor data for safety
4. Implements emergency response procedures

**Requirements:**
- Multi-modal input processing
- Real-time constraint validation
- Emergency response system
- Safety logging and monitoring

**Acceptance Criteria:**
- [ ] VLA output validation working
- [ ] Real-time monitoring active
- [ ] Emergency responses triggered
- [ ] Safety logs maintained

</Exercise>

## Exercise 3: Action Generation Pipeline

<Exercise title="End-to-End Action Pipeline" difficulty="advanced" estimatedTime="120 min">

Build a complete action generation pipeline that:
1. Processes multimodal inputs (voice, vision, sensors)
2. Generates appropriate robot actions
3. Validates actions against safety constraints
4. Executes actions with monitoring and feedback

**Requirements:**
- Multimodal input processing
- Action planning and generation
- Safety validation pipeline
- Execution monitoring system

**Acceptance Criteria:**
- [ ] Multimodal inputs processed
- [ ] Actions generated correctly
- [ ] Safety constraints enforced
- [ ] Execution monitored properly

</Exercise>

## Exercise 4: Recovery System

<Exercise title="Intelligent Recovery System" difficulty="advanced" estimatedTime="150 min">

Create a sophisticated recovery system that:
1. Detects various types of action failures
2. Implements context-aware recovery strategies
3. Learns from recovery attempts to improve
4. Provides graceful degradation when recovery fails

**Requirements:**
- Failure detection mechanisms
- Multiple recovery strategies
- Learning from recovery attempts
- Fallback safety procedures

**Acceptance Criteria:**
- [ ] Failure types detected correctly
- [ ] Recovery strategies effective
- [ ] Learning system functional
- [ ] Fallback procedures safe

</Exercise>

## Exercise 5: Integration Challenge

<Exercise title="Complete VLA Safety System" difficulty="advanced" estimatedTime="240 min">

Build a complete end-to-end VLA safety system that:
1. Integrates all components: voice, vision, action, safety
2. Handles complex multi-modal commands safely
3. Implements comprehensive safety monitoring
4. Provides user feedback and system status

**System Architecture:**
```
┌─────────────────┐    Voice     ┌─────────────────┐
│  User Voice     │ ──────────▶ │  Whisper STT    │
│  Commands       │             │                 │
└─────────────────┘             └─────────────────┘
                                          │
                                          ▼
┌─────────────────┐    Vision    ┌─────────────────┐
│  Camera Input   │ ──────────▶ │  Vision Process │
│                 │             │                 │
└─────────────────┘             └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  VLA Action     │
                                │  Generation     │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  Safety         │
                                │  Validator      │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  Action         │
                                │  Executor       │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  Recovery &     │
                                │  Monitoring     │
                                └─────────────────┘
```

**Requirements:**
- Complete multimodal pipeline
- Comprehensive safety system
- Action execution with feedback
- Recovery and degradation handling

**Acceptance Criteria:**
- [ ] All components integrated
- [ ] Safe operation guaranteed
- [ ] Multi-modal commands processed
- [ ] Recovery systems functional

</Exercise>

## Self-Assessment

Rate your confidence (1-5) after completing these exercises:

| Skill | Target | Your Rating |
|-------|--------|-------------|
| Whisper integration with robotics | 4 | ___ |
| Multimodal input processing | 4 | ___ |
| Action generation from VLA outputs | 4 | ___ |
| Safety constraint validation | 4 | ___ |
| Real-time safety monitoring | 4 | ___ |
| Emergency response systems | 4 | ___ |
| Recovery system implementation | 3 | ___ |

If any rating is below target, review the corresponding lesson material.

## Submission Checklist

Before moving to Module 5, ensure:

- [ ] All exercises compile without warnings
- [ ] VLA system processes multimodal inputs safely
- [ ] Safety constraints are enforced
- [ ] Recovery systems are functional
- [ ] Code follows ROS 2 Python style guidelines

---

**Ready for Module 5?** Continue to [Week 12: System Integration](/module-5/week-12/system-architecture) to learn about integrating all components into a complete humanoid robotics system.