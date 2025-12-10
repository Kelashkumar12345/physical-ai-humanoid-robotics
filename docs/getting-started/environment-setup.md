---
sidebar_position: 2
title: Environment Setup
description: Step-by-step installation guide for ROS 2, Gazebo, and development tools
---

# Environment Setup

This guide walks you through installing all software needed for the course. Follow each section in order.

<LearningObjectives
  objectives={[
    "Install ROS 2 Humble on Ubuntu 22.04",
    "Configure the ROS 2 environment",
    "Install Gazebo Harmonic simulation",
    "Set up TurtleBot3 packages",
    "Verify the complete installation"
  ]}
/>

## Step 1: System Update

Start by updating your system packages:

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Install ROS 2 Humble

### Add ROS 2 Repository

```bash
# Install required tools
sudo apt install -y software-properties-common curl

# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS 2 Desktop Full

```bash
sudo apt update
sudo apt install -y ros-humble-desktop-full
```

This installation includes:
- ROS 2 core libraries
- RViz2 visualization
- Demo nodes and tutorials
- Development tools

### Install Additional ROS 2 Packages

```bash
# Navigation and SLAM
sudo apt install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-robot-localization

# Development tools
sudo apt install -y \
  ros-humble-tf2-tools \
  ros-humble-rqt* \
  python3-colcon-common-extensions \
  python3-rosdep
```

### Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

## Step 3: Configure ROS 2 Environment

Add ROS 2 to your shell configuration:

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Apply changes
source ~/.bashrc
```

Verify the installation:

```bash
ros2 --version
```

Expected output:
```
ros2 0.10.x
```

## Step 4: Install Gazebo Harmonic

### Add Gazebo Repository

```bash
# Add Gazebo GPG key
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

# Add repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
```

### Install Gazebo

```bash
sudo apt update
sudo apt install -y gz-harmonic ros-humble-ros-gz
```

Verify installation:

```bash
gz sim --version
```

## Step 5: Install TurtleBot3 Packages

TurtleBot3 is the reference robot for Weeks 1-6:

```bash
# Install TurtleBot3 packages
sudo apt install -y \
  ros-humble-turtlebot3* \
  ros-humble-turtlebot3-simulations
```

### Configure TurtleBot3 Environment

```bash
# Add TurtleBot3 configuration to ~/.bashrc
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models' >> ~/.bashrc

source ~/.bashrc
```

## Step 6: Create Course Workspace

Create a dedicated workspace for course exercises:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build empty workspace to verify setup
colcon build
source install/setup.bash

# Add workspace to bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Step 7: Verify Installation

### Test ROS 2

Open two terminals and run:

**Terminal 1 - Talker:**
```bash
ros2 run demo_nodes_cpp talker
```

**Terminal 2 - Listener:**
```bash
ros2 run demo_nodes_cpp listener
```

You should see messages being sent and received.

### Test Gazebo with TurtleBot3

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

This should open Gazebo with TurtleBot3 in a test environment.

### Run Diagnostic

```bash
ros2 doctor --report
```

Review the output for any warnings or errors.

## Troubleshooting

### Common Issues

:::tip Package Not Found
If `ros2` command is not found:
```bash
source /opt/ros/humble/setup.bash
```
Add this to your `~/.bashrc` if not already present.
:::

:::tip Gazebo Crashes on Start
For GPU-related issues:
```bash
export LIBGL_ALWAYS_SOFTWARE=1
gz sim
```
:::

:::tip TurtleBot3 Model Error
Ensure the environment variable is set:
```bash
echo $TURTLEBOT3_MODEL
# Should output: burger
```
:::

## Environment Summary

Your environment is ready when you can:

- [ ] Run `ros2 --version` and see version output
- [ ] Run `gz sim --version` and see Gazebo version
- [ ] Launch TurtleBot3 in Gazebo without errors
- [ ] Run `ros2 doctor --report` with no critical warnings

## What's Next

With your environment configured, you're ready to start learning! Proceed to:

- [Quick Start Guide](/getting-started/quick-start) - Run your first ROS 2 nodes
- [Week 1: ROS 2 Architecture](/module-1/week-1/introduction) - Begin the course

---

:::info Need Help?
If you encounter issues during setup, check our [Troubleshooting Guide](/appendices/troubleshooting) or open an issue on the course repository.
:::
