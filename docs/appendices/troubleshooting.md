---
sidebar_position: 1
title: Troubleshooting
description: Common issues and solutions for ROS 2, Gazebo, and Isaac Sim
---

# Troubleshooting Guide

This guide addresses common issues encountered during the course. Search for your error message or browse by category.

## ROS 2 Issues

### Package Not Found

**Error:**
```
Package 'my_package' not found
```

**Solutions:**
1. Source your workspace:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. Rebuild the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_package
   ```

3. Check if package exists:
   ```bash
   ros2 pkg list | grep my_package
   ```

### Node Not Executable

**Error:**
```
[ros2run]: No executable found
```

**Solutions:**
1. Check setup.py entry_points are correct
2. Rebuild with `--symlink-install`:
   ```bash
   colcon build --symlink-install
   ```
3. Verify file permissions:
   ```bash
   chmod +x ~/ros2_ws/src/pkg/pkg/node.py
   ```

### Topic Not Found

**Error:**
```
[WARN] Topic '/my_topic' is not published yet
```

**Solutions:**
1. Check topic name spelling (case-sensitive)
2. Verify publisher node is running:
   ```bash
   ros2 node list
   ```
3. Check QoS compatibility between publisher and subscriber

### QoS Incompatibility

**Error:**
```
Requested incompatible QoS Policy
```

**Solutions:**
Match QoS policies between publisher and subscriber:
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.BEST_EFFORT
```

## Gazebo Issues

### Gazebo Won't Start

**Error:**
```
[Err] [Server.cc:xxx] Unable to start server
```

**Solutions:**
1. Kill existing Gazebo processes:
   ```bash
   killall gz sim gzserver gzclient
   ```

2. Check GPU drivers:
   ```bash
   nvidia-smi
   ```

3. Try software rendering:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   gz sim
   ```

### Model Not Found

**Error:**
```
Unable to find model [model_name]
```

**Solutions:**
1. Set model path:
   ```bash
   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models
   ```

2. Download models:
   ```bash
   gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/model_name
   ```

### Robot Falls Through Floor

**Solutions:**
1. Check collision geometry in URDF/SDF
2. Verify world has a ground plane
3. Reduce physics step size in world file

## TurtleBot3 Issues

### Model Not Set

**Error:**
```
TURTLEBOT3_MODEL is not defined
```

**Solution:**
```bash
export TURTLEBOT3_MODEL=burger
# Add to ~/.bashrc for persistence
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
```

### Simulation Slow

**Solutions:**
1. Reduce sensor rates in simulation config
2. Use simpler physics engine
3. Lower GPU usage by other applications

## Build Issues

### Colcon Build Fails

**Error:**
```
CMake Error at CMakeLists.txt
```

**Solutions:**
1. Install missing dependencies:
   ```bash
   rosdep install --from-paths src -y --ignore-src
   ```

2. Clean and rebuild:
   ```bash
   rm -rf build install log
   colcon build
   ```

### Python Import Error

**Error:**
```
ModuleNotFoundError: No module named 'my_package'
```

**Solutions:**
1. Source the workspace after building
2. Check package.xml has `<exec_depend>` for dependencies
3. Verify setup.py `packages` field includes your module

## Network Issues

### Nodes Not Discovering Each Other

**Solutions:**
1. Check `ROS_DOMAIN_ID` matches on all machines:
   ```bash
   echo $ROS_DOMAIN_ID
   export ROS_DOMAIN_ID=0
   ```

2. Disable firewall or open DDS ports
3. Use same DDS implementation on all machines

### High Latency

**Solutions:**
1. Use shared memory transport:
   ```bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   ```

2. Reduce message sizes
3. Check network bandwidth with `iperf3`

## Getting More Help

If your issue isn't listed:

1. Search ROS Answers: https://answers.ros.org
2. Check ROS 2 GitHub Issues: https://github.com/ros2/ros2/issues
3. Ask on ROS Discourse: https://discourse.ros.org
4. Open an issue on the course repository

When reporting issues, include:
- Full error message
- ROS 2 version (`ros2 --version`)
- Operating system (`lsb_release -a`)
- Steps to reproduce
