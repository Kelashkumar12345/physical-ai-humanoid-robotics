---
sidebar_position: 1
title: Prerequisites
description: System requirements and prerequisite knowledge for the Physical AI & Humanoid Robotics course
---

# Prerequisites

Before diving into the course, ensure your system meets the requirements and you have the foundational knowledge needed for success.

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i5 / AMD Ryzen 5 (8 cores) | Intel i7 / AMD Ryzen 7 (12+ cores) |
| **RAM** | 16 GB | 32 GB |
| **GPU** | NVIDIA GTX 1660 (6 GB VRAM) | NVIDIA RTX 3080 (10 GB VRAM) |
| **Storage** | 100 GB SSD | 256 GB NVMe SSD |

:::warning GPU Requirements for Isaac Sim
Modules 3-5 (Isaac Sim and VLA integration) require an **NVIDIA GPU with RTX capabilities**. Apple Silicon (M1/M2/M3) is not supported for Isaac Sim sections.
:::

### Operating System

- **Primary**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Alternative**: Windows 11 with WSL2 running Ubuntu 22.04

:::tip WSL2 Setup
If using Windows, install WSL2 with Ubuntu 22.04:
```bash
wsl --install -d Ubuntu-22.04
```
See [Microsoft's WSL documentation](https://learn.microsoft.com/en-us/windows/wsl/install) for detailed setup instructions.
:::

## Software Prerequisites

The following software will be installed during the [Environment Setup](/getting-started/environment-setup):

| Software | Version | Purpose |
|----------|---------|---------|
| ROS 2 | Humble Hawksbill | Robot Operating System |
| Gazebo | Harmonic | Physics simulation (Weeks 4-6) |
| Isaac Sim | 2023.1.1 | Advanced simulation (Weeks 7-9) |
| Python | 3.10 | Programming language |
| Ollama | 0.1.20+ | Local LLM inference (Weeks 10-11) |

## Knowledge Prerequisites

### Required Knowledge

<Prerequisites
  items={[
    "Basic Python programming (functions, classes, loops, conditionals)",
    "Linux command line basics (cd, ls, mkdir, apt, nano/vim)",
    "Understanding of coordinate systems (x, y, z axes)",
    "Basic trigonometry (sine, cosine, radians vs degrees)"
  ]}
  optional={[
    "Git version control (clone, commit, push, pull)",
    "Docker basics (helpful for isolated environments)",
    "ROS 1 experience (we'll cover differences from ROS 2)"
  ]}
/>

### Self-Assessment

Rate your confidence (1-5) in the following areas:

| Topic | Description | Target Level |
|-------|-------------|--------------|
| **Python** | Write functions, classes, use pip | 3+ |
| **Linux CLI** | Navigate filesystem, run commands | 3+ |
| **Math** | Matrices, vectors, rotations | 2+ |
| **Robotics** | General understanding of robots | 1+ |

If you scored below the target in any area, we recommend these resources:

- **Python**: [Python for Everybody](https://www.py4e.com/) (free course)
- **Linux**: [Linux Journey](https://linuxjourney.com/) (interactive tutorial)
- **Math**: [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (video series)

## Development Environment

### Recommended IDE

We recommend **Visual Studio Code** with the following extensions:

```bash
# Install VS Code extensions from command line
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension ms-iot.vscode-ros
code --install-extension redhat.vscode-yaml
```

### Terminal Emulator

For ROS 2 development, a terminal that supports multiple panes is essential:

- **Terminator** (Linux): `sudo apt install terminator`
- **Windows Terminal** (WSL2): Built into Windows 11

## Verifying Prerequisites

Run this script to check your system:

```bash
#!/bin/bash
echo "=== System Check ==="

# Check Ubuntu version
echo -n "Ubuntu version: "
lsb_release -rs

# Check Python
echo -n "Python version: "
python3 --version

# Check available memory
echo -n "Available RAM: "
free -h | grep Mem | awk '{print $2}'

# Check GPU
echo -n "GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"

# Check disk space
echo -n "Available disk space: "
df -h / | tail -1 | awk '{print $4}'
```

Save this as `check-prerequisites.sh` and run:

```bash
chmod +x check-prerequisites.sh
./check-prerequisites.sh
```

---

**Ready to continue?** Proceed to [Environment Setup](/getting-started/environment-setup) to install the required software.
