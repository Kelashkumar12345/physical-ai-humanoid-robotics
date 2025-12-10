---
sidebar_position: 3
title: Quick Start
description: Run your first ROS 2 nodes and verify your installation
---

# Quick Start Guide

Let's verify your installation by running some basic ROS 2 examples. This will familiarize you with key concepts before diving into the course.

<LearningObjectives
  objectives={[
    "Run publisher and subscriber nodes",
    "Use ros2 CLI tools for introspection",
    "Visualize data with RViz2",
    "Control TurtleBot3 in simulation"
  ]}
/>

## Your First ROS 2 Nodes

### Understanding the Publisher-Subscriber Pattern

ROS 2 nodes communicate through **topics**. A publisher sends messages to a topic, and subscribers receive them.

```
┌──────────────┐     /chatter      ┌──────────────┐
│   Talker     │ ───────────────▶  │   Listener   │
│  (Publisher) │   std_msgs/String │  (Subscriber)│
└──────────────┘                   └──────────────┘
```

### Run the Demo

Open **three terminals** and run:

**Terminal 1 - Talker Node:**
```bash
ros2 run demo_nodes_cpp talker
```

**Terminal 2 - Listener Node:**
```bash
ros2 run demo_nodes_cpp listener
```

**Terminal 3 - Topic Inspection:**
```bash
# List all active topics
ros2 topic list

# See message data in real-time
ros2 topic echo /chatter

# Check publishing rate
ros2 topic hz /chatter
```

### What You Should See

**Talker output:**
```
[INFO] [talker]: Publishing: 'Hello World: 1'
[INFO] [talker]: Publishing: 'Hello World: 2'
[INFO] [talker]: Publishing: 'Hello World: 3'
```

**Listener output:**
```
[INFO] [listener]: I heard: [Hello World: 1]
[INFO] [listener]: I heard: [Hello World: 2]
[INFO] [listener]: I heard: [Hello World: 3]
```

## ROS 2 CLI Tools

ROS 2 provides powerful command-line tools for introspection:

### Node Information

```bash
# List running nodes
ros2 node list

# Get details about a node
ros2 node info /talker
```

### Topic Information

```bash
# List all topics
ros2 topic list

# Show topic type
ros2 topic info /chatter

# Display message structure
ros2 interface show std_msgs/msg/String
```

### Service and Action Discovery

```bash
# List services
ros2 service list

# List actions
ros2 action list
```

## TurtleBot3 in Simulation

Now let's interact with a simulated robot!

### Launch TurtleBot3 World

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Wait for Gazebo to fully load (may take 30-60 seconds on first run).

### Explore Robot Topics

In a new terminal:

```bash
# See what topics TurtleBot3 publishes/subscribes to
ros2 topic list
```

Key topics:

<TopicTable
  topics={[
    { name: "/cmd_vel", type: "geometry_msgs/Twist", direction: "subscribe", description: "Velocity commands to move the robot" },
    { name: "/odom", type: "nav_msgs/Odometry", direction: "publish", description: "Robot position and velocity" },
    { name: "/scan", type: "sensor_msgs/LaserScan", direction: "publish", description: "LIDAR scan data" },
    { name: "/imu", type: "sensor_msgs/Imu", direction: "publish", description: "Inertial measurement data" }
  ]}
/>

### Control the Robot

**Option 1: Keyboard Teleoperation**

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

Use these keys:
```
        w
   a    s    d
        x

w/x : increase/decrease linear velocity
a/d : increase/decrease angular velocity
s   : stop
```

**Option 2: Command Line**

Send a velocity command directly:

```bash
# Move forward for 2 seconds
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Stop
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Visualize with RViz2

RViz2 is the primary visualization tool for ROS 2.

### Launch RViz2 with TurtleBot3

```bash
ros2 launch turtlebot3_bringup rviz2.launch.py
```

Or launch RViz2 manually and add displays:

```bash
rviz2
```

### Essential Displays

Add these displays to visualize TurtleBot3 data:

1. **RobotModel** - Shows the robot URDF
2. **LaserScan** - Visualizes LIDAR data (topic: `/scan`)
3. **TF** - Shows coordinate frames
4. **Odometry** - Displays robot path (topic: `/odom`)

## Quick Exercises

<Exercise title="Explore ROS 2 Topics" difficulty="beginner" estimatedTime="10 min">

1. Launch TurtleBot3 in Gazebo
2. List all available topics
3. Echo the `/scan` topic to see LIDAR data
4. Use `ros2 topic hz /scan` to check the scan rate

<Hint>
The LIDAR scan rate should be approximately 5 Hz (every 200ms).
</Hint>

<Solution>
```bash
# Terminal 1: Launch simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: List topics
ros2 topic list

# Terminal 3: Echo scan data
ros2 topic echo /scan

# Terminal 4: Check rate
ros2 topic hz /scan
```
</Solution>

</Exercise>

<Exercise title="Robot Navigation" difficulty="beginner" estimatedTime="15 min">

Drive TurtleBot3 through the maze using keyboard teleoperation.

Goals:
- Navigate without hitting walls
- Complete one lap around an obstacle
- Return to the starting position

<Hint>
Use `w` to move forward, `a`/`d` to rotate, and `s` to stop. Start with low speeds!
</Hint>

</Exercise>

## Checkpoint

Before continuing to Week 1, verify you can:

- [ ] Run publisher/subscriber nodes
- [ ] Use `ros2 topic`, `ros2 node` commands
- [ ] Launch TurtleBot3 in Gazebo
- [ ] Control the robot with keyboard teleoperation
- [ ] View data in RViz2

## What's Next

You're ready to begin the course!

**Start with:** [Week 1: ROS 2 Architecture](/module-1/week-1/introduction)

In Week 1, you'll learn:
- ROS 2 architecture and design philosophy
- Creating custom nodes in Python
- Understanding DDS and Quality of Service
- Building your first ROS 2 package
