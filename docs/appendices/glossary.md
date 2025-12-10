---
sidebar_position: 2
title: Glossary
description: Key terms and definitions for robotics and ROS 2
---

# Glossary

## A

**Action**
A ROS 2 communication pattern for long-running tasks. Actions consist of a goal, feedback, and result, and support cancellation.

**Ament**
The build system used by ROS 2. Ament supports CMake and Python (setuptools) build types.

**AMCL (Adaptive Monte Carlo Localization)**
A probabilistic localization system that uses a particle filter to track a robot's pose in a known map.

## C

**Callback**
A function that executes when an event occurs, such as receiving a message on a subscribed topic.

**Colcon**
The command-line tool for building ROS 2 workspaces. Builds all packages in a workspace with proper dependency ordering.

**Costmap**
A grid-based representation of obstacles used by Nav2 for path planning and collision avoidance.

## D

**DDS (Data Distribution Service)**
The middleware standard used by ROS 2 for publish-subscribe communication. Provides QoS policies, discovery, and security.

**Domain ID**
A number (0-232) that isolates ROS 2 communication. Nodes with different domain IDs cannot discover each other.

## F

**Frame**
A coordinate system in TF2. Frames have positions and orientations relative to other frames, forming a tree structure.

## G

**Gazebo**
An open-source 3D robotics simulator that integrates with ROS 2 for physics-based simulation.

## I

**IMU (Inertial Measurement Unit)**
A sensor that measures acceleration and angular velocity. Used for robot orientation estimation.

**Isaac Sim**
NVIDIA's GPU-accelerated robotics simulator with high-fidelity physics and synthetic data generation.

## L

**Launch File**
A Python file that starts multiple ROS 2 nodes with specific configurations. Replaces ROS 1 XML launch files.

**LIDAR (Light Detection and Ranging)**
A sensor that measures distance using laser light. Commonly used for mapping and obstacle detection.

## M

**Message**
A data structure used for ROS 2 topic communication. Defined in `.msg` files.

## N

**Nav2 (Navigation2)**
The ROS 2 navigation stack. Provides autonomous navigation, path planning, and behavior trees.

**Node**
The fundamental unit of computation in ROS 2. Each node handles a specific task and communicates through topics, services, and actions.

## O

**Odometry**
Estimation of a robot's position and velocity based on wheel encoders or visual features.

**Ollama**
An open-source tool for running large language models locally. Used in VLA integration modules.

## P

**Package**
A unit of organization in ROS 2 containing nodes, libraries, configuration, and launch files.

**Publisher**
A node endpoint that sends messages to a topic.

## Q

**QoS (Quality of Service)**
DDS policies that control message delivery, including reliability, durability, history depth, and deadline.

**Quaternion**
A four-component representation of 3D rotation. Avoids gimbal lock issues associated with Euler angles.

## R

**RMW (ROS Middleware)**
The abstraction layer between ROS 2 and DDS implementations. Allows swapping DDS vendors.

**ROS 2 (Robot Operating System 2)**
Open-source middleware for robotics providing communication, hardware abstraction, and tools.

**RViz2**
The primary visualization tool for ROS 2. Displays sensor data, robot models, paths, and more.

## S

**Service**
A ROS 2 communication pattern for request-response interactions. Synchronous by nature.

**SLAM (Simultaneous Localization and Mapping)**
The problem of building a map while simultaneously tracking position within it.

**Subscriber**
A node endpoint that receives messages from a topic.

## T

**TF2**
The ROS 2 library for tracking coordinate frame transforms over time.

**Topic**
A named channel for publish-subscribe message passing in ROS 2.

**Transform**
A mathematical representation of position and orientation between two coordinate frames.

**TurtleBot3**
A low-cost, open-source robot platform commonly used for ROS 2 education and research.

## U

**URDF (Unified Robot Description Format)**
An XML format describing robot structure including links, joints, sensors, and visual properties.

## V

**VLA (Vision-Language-Action)**
A model architecture that combines visual perception, language understanding, and action generation for robot control.

## W

**Workspace**
A directory containing ROS 2 packages. Built with colcon and sourced to use packages.

**Whisper**
OpenAI's speech recognition model. Used for voice commands in VLA integration.
