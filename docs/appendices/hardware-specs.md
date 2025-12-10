---
sidebar_position: 4
title: Hardware Specifications
description: Recommended hardware for the course
---

# Hardware Specifications

This guide details hardware requirements and recommendations for completing the course.

## Computer Requirements

### Minimum Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | Intel i5 / AMD Ryzen 5 (8 cores) | 2.5 GHz base clock |
| **RAM** | 16 GB DDR4 | 32 GB for Isaac Sim |
| **GPU** | NVIDIA GTX 1660 (6 GB) | Required for simulation |
| **Storage** | 100 GB SSD | NVMe preferred |
| **OS** | Ubuntu 22.04 LTS | WSL2 supported |

### Recommended Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | Intel i7 / AMD Ryzen 7 (12+ cores) | 3.0+ GHz base clock |
| **RAM** | 32 GB DDR4 | 64 GB for large VLA models |
| **GPU** | NVIDIA RTX 3080 (10 GB) | RTX 4080/4090 ideal |
| **Storage** | 256 GB NVMe SSD | Fast read/write for datasets |
| **OS** | Ubuntu 22.04 LTS | Native preferred |

### GPU Requirements by Module

| Module | GPU Requirement | VRAM |
|--------|-----------------|------|
| 1-3 (ROS 2 Basics) | Optional | N/A |
| 4-6 (Gazebo) | GTX 1060+ | 4 GB |
| 7-9 (Isaac Sim) | RTX 2070+ | 8 GB |
| 10-11 (VLA) | RTX 3080+ | 10 GB |
| 12-13 (Capstone) | RTX 3080+ | 10 GB |

:::warning Apple Silicon Not Supported
MacOS with Apple Silicon (M1/M2/M3) cannot run Isaac Sim modules. Use a cloud GPU instance or dual-boot Linux on Intel Mac.
:::

## TurtleBot3 Hardware (Optional)

For physical robot exercises, the TurtleBot3 Burger is recommended:

### TurtleBot3 Burger Specifications

| Component | Specification |
|-----------|---------------|
| **Dimensions** | 138mm × 178mm × 192mm |
| **Weight** | 1 kg (with battery) |
| **Max Speed** | 0.22 m/s |
| **Payload** | 5 kg |
| **LIDAR** | LDS-01 (360°, 5 Hz) |
| **SBC** | Raspberry Pi 4 |
| **MCU** | OpenCR1.0 (ARM Cortex-M7) |
| **Battery** | LiPo 11.1V 1800mAh |
| **Runtime** | ~2.5 hours |

### Purchasing Options

- ROBOTIS Official Store: https://www.robotis.us/turtlebot-3/
- Alternative: Build from open-source designs

### Additional Sensors (Optional)

| Sensor | Use Case | Integration |
|--------|----------|-------------|
| Intel RealSense D435 | RGB-D perception | USB 3.0 |
| ZED 2 | Stereo vision | USB 3.0 |
| Velodyne VLP-16 | 3D LIDAR | Ethernet |
| IMU (BNO055) | Orientation | I2C |

## Cloud Computing Alternatives

If local hardware is insufficient:

### AWS RoboMaker
- Pre-configured ROS 2 environments
- Gazebo simulation in cloud
- GPU instances available

### Google Cloud (GCP)
- NVIDIA T4/A100 instances
- Container-based development
- Vertex AI for ML workloads

### Lambda Labs
- Consumer GPU cloud (RTX 3090, A100)
- Hourly pricing
- Ubuntu environments

### Cost Estimates (2024)

| Provider | GPU | Hourly Cost |
|----------|-----|-------------|
| AWS g4dn.xlarge | T4 | ~$0.50 |
| GCP n1-standard-4 + T4 | T4 | ~$0.55 |
| Lambda | RTX 3090 | ~$0.75 |
| Lambda | A100 | ~$1.10 |

## Network Requirements

### For Simulation
- Standard broadband (10+ Mbps)
- Low latency not critical

### For Multi-Robot / Remote
- 50+ Mbps recommended
- Less than 50ms latency for real-time control
- Consider wired Ethernet for reliability

## Workspace Setup

### Physical Workspace
- Clear area: 2m × 2m minimum for TurtleBot3
- Flat, non-reflective flooring
- Good lighting (for camera-based exercises)
- Power outlets accessible

### Monitor Recommendations
- 1920×1080 minimum resolution
- Dual monitors helpful (code + visualization)
- 4K recommended for Isaac Sim

## Peripheral Recommendations

| Item | Purpose | Notes |
|------|---------|-------|
| USB Hub | Multiple devices | Powered, USB 3.0 |
| Gamepad | Teleop testing | Xbox/PS controller |
| Webcam | VLA testing | 720p minimum |
| Microphone | Speech input | For Whisper |

---

## Hardware Checklist

Before starting the course:

- [ ] Computer meets minimum specifications
- [ ] NVIDIA GPU with latest drivers
- [ ] Ubuntu 22.04 or WSL2 installed
- [ ] 100+ GB free disk space
- [ ] Stable internet connection
- [ ] (Optional) TurtleBot3 assembled and tested
