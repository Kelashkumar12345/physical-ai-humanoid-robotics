---
sidebar_position: 1
slug: /
title: Physical AI & Humanoid Robotics
description: A 13-Week Hands-On Course - From ROS 2 to VLA-Powered Humanoids
---

# Physical AI & Humanoid Robotics

**A 13-Week Hands-On Course: From ROS 2 to VLA-Powered Humanoids**

Welcome to the comprehensive hands-on course for building intelligent humanoid robots. This course takes you from ROS 2 fundamentals through advanced simulation with NVIDIA Isaac Sim to cutting-edge Vision-Language-Action (VLA) integration.

## Course Overview

<div className="learning-objectives">

### What You'll Learn

- **ROS 2 Humble**: Master the Robot Operating System, including nodes, topics, services, TF2, and Nav2
- **Gazebo Simulation**: Build and test robots in physics-accurate simulated environments
- **Isaac Sim**: Leverage NVIDIA's high-fidelity simulation for humanoid robotics
- **VLA Integration**: Connect vision-language models to robot action generation
- **End-to-End Pipeline**: Deploy complete AI-powered robotic systems

</div>

## Course Structure

| Module | Weeks | Topics |
|--------|-------|--------|
| **Module 1**: ROS 2 Fundamentals | 1-3 | Architecture, TF2, Navigation, Perception |
| **Module 2**: Gazebo Simulation | 4-6 | World Building, Sensors, TurtleBot3 |
| **Module 3**: Isaac Sim | 7-9 | Humanoid Assets, Physics, Domain Randomization |
| **Module 4**: VLA Integration | 10-11 | Ollama, Whisper, Action Generation |
| **Module 5**: Capstone | 12-13 | System Integration, Deployment |

## Prerequisites

<div className="prerequisites-checklist">

### Before You Begin

- [ ] Ubuntu 22.04 LTS (or WSL2 on Windows 11)
- [ ] NVIDIA GPU with RTX capabilities (for Isaac Sim modules)
- [ ] 16 GB RAM minimum (32 GB recommended)
- [ ] Basic Python programming experience
- [ ] Familiarity with Linux command line

</div>

## Quick Start

```bash
# 1. Install ROS 2 Humble
sudo apt update && sudo apt install ros-humble-desktop-full

# 2. Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# 3. Verify installation
ros2 doctor --report
```

## Getting Help

- **Issues**: Found a bug or have a suggestion? [Open an issue](https://github.com/kelashkumar12345/physical-ai-humanoid-robotics/issues)
- **Discussions**: Questions about concepts? Join our [Discussions](https://github.com/kelashkumar12345/physical-ai-humanoid-robotics/discussions)
- **Office Hours**: Weekly Q&A sessions (schedule in announcements)

## Reference Environment

All code examples are tested against the reference environment specifications:

- **ROS 2**: Humble Hawksbill
- **Gazebo**: Harmonic
- **Isaac Sim**: 2023.1.1
- **Python**: 3.10
- **Ollama**: 0.1.20+

---

**Ready to begin?** Start with [Prerequisites](/getting-started/prerequisites) to set up your development environment.
