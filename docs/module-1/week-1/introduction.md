---
sidebar_position: 1
title: Introduction to ROS 2
description: Understanding ROS 2 architecture, design philosophy, and core concepts
---

# Week 1: Introduction to ROS 2

<WeekHeader
  week={1}
  title="ROS 2 Architecture"
  module={1}
  estimatedHours={8}
  skills={["ROS 2 CLI", "Node Concepts", "DDS Basics", "Package Structure"]}
/>

<LearningObjectives
  week={1}
  objectives={[
    "Explain the differences between ROS 1 and ROS 2",
    "Understand the DDS communication layer",
    "Create and run ROS 2 nodes",
    "Use ros2 CLI tools for introspection",
    "Build a ROS 2 workspace with colcon"
  ]}
/>

## What is ROS 2?

The **Robot Operating System 2 (ROS 2)** is a set of software libraries and tools for building robot applications. Despite its name, ROS 2 is not an operating system—it's middleware that provides:

- **Communication infrastructure**: Publish-subscribe messaging, services, actions
- **Hardware abstraction**: Standardized interfaces for sensors and actuators
- **Package management**: Reusable code organized into packages
- **Development tools**: Visualization, simulation, debugging

### Why ROS 2?

ROS 2 was developed to address limitations of ROS 1:

| Aspect | ROS 1 | ROS 2 |
|--------|-------|-------|
| **Communication** | Custom protocol (TCPROS) | Industry-standard DDS |
| **Real-time** | Not supported | Real-time capable |
| **Security** | No built-in security | DDS-Security integration |
| **Multi-robot** | Requires workarounds | Native support |
| **Platforms** | Linux only | Linux, Windows, macOS |

## ROS 2 Architecture

### The DDS Foundation

ROS 2 uses the **Data Distribution Service (DDS)** standard for communication:

<ArchitectureDiagram title="ROS 2 Communication Stack">
{`
┌─────────────────────────────────────────────────────┐
│                   Application Layer                  │
│              (Your Nodes and Code)                   │
├─────────────────────────────────────────────────────┤
│                    ROS 2 Client                      │
│                  Library (rclpy/rclcpp)              │
├─────────────────────────────────────────────────────┤
│                       RMW                            │
│            (ROS Middleware Interface)                │
├─────────────────────────────────────────────────────┤
│                       DDS                            │
│    (Fast DDS, Cyclone DDS, RTI Connext)              │
├─────────────────────────────────────────────────────┤
│                    Transport                         │
│              (UDP, Shared Memory)                    │
└─────────────────────────────────────────────────────┘
`}
</ArchitectureDiagram>

**Key benefits of DDS:**
- **Decentralized**: No central master node (unlike ROS 1's roscore)
- **Quality of Service (QoS)**: Configure reliability, durability, history
- **Discovery**: Automatic node discovery over the network
- **Security**: Optional encryption and authentication

### Core Concepts

#### Nodes

A **node** is a process that performs computation. Nodes communicate with each other through topics, services, and actions.

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Hello from MyNode!')

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Topics

**Topics** are named buses for streaming data. Publishers send messages to topics; subscribers receive them.

```
┌──────────┐   /camera/image    ┌──────────┐
│  Camera  │ ──────────────────▶│ Detector │
│  Driver  │  sensor_msgs/Image │   Node   │
└──────────┘                    └──────────┘
```

#### Services

**Services** provide request-response communication for synchronous operations:

```
┌──────────┐    Request    ┌──────────┐
│  Client  │ ────────────▶ │  Server  │
│          │ ◀──────────── │          │
└──────────┘    Response   └──────────┘
```

#### Actions

**Actions** handle long-running tasks with feedback:

```
┌──────────┐     Goal      ┌──────────┐
│  Client  │ ────────────▶ │  Server  │
│          │ ◀──────────── │          │
│          │    Feedback   │          │
│          │ ◀──────────── │          │
│          │    Result     │          │
└──────────┘               └──────────┘
```

## Setting Up Your Workspace

### Workspace Structure

A typical ROS 2 workspace:

```
ros2_ws/
├── src/                    # Source packages
│   ├── my_package/
│   │   ├── my_package/     # Python module
│   │   ├── resource/
│   │   ├── test/
│   │   ├── package.xml
│   │   └── setup.py
│   └── another_package/
├── build/                  # Build artifacts
├── install/                # Installed packages
└── log/                    # Build logs
```

### Creating a Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build (even if empty)
colcon build

# Source the workspace
source install/setup.bash
```

### Creating a Package

```bash
cd ~/ros2_ws/src

# Create a Python package
ros2 pkg create --build-type ament_python my_first_package \
  --dependencies rclpy std_msgs

# Or a C++ package
ros2 pkg create --build-type ament_cmake my_cpp_package \
  --dependencies rclcpp std_msgs
```

## ROS 2 CLI Tools

The `ros2` command provides comprehensive tools for introspection:

### Essential Commands

```bash
# Node operations
ros2 node list                    # List running nodes
ros2 node info /node_name         # Node details

# Topic operations
ros2 topic list                   # List topics
ros2 topic echo /topic_name       # Print messages
ros2 topic hz /topic_name         # Message rate
ros2 topic pub /topic msg_type    # Publish message

# Service operations
ros2 service list                 # List services
ros2 service call /srv type data  # Call service

# Interface inspection
ros2 interface show msg_type      # Show message structure
ros2 interface list               # All available interfaces
```

### Useful Aliases

Add these to your `~/.bashrc`:

```bash
alias r2='ros2'
alias r2tl='ros2 topic list'
alias r2te='ros2 topic echo'
alias r2nl='ros2 node list'
alias r2ni='ros2 node info'
alias cb='colcon build'
alias cbs='colcon build --symlink-install'
```

## Quality of Service (QoS)

QoS policies control how messages are delivered:

| Policy | Options | Use Case |
|--------|---------|----------|
| **Reliability** | RELIABLE, BEST_EFFORT | Sensor data vs commands |
| **Durability** | TRANSIENT_LOCAL, VOLATILE | Late-joining subscribers |
| **History** | KEEP_LAST(N), KEEP_ALL | Buffer size |
| **Depth** | Integer | Queue depth |

### Example: QoS Configuration

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For sensor data (allow drops)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=10
)

# For commands (ensure delivery)
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)
```

## Hands-On: Your First Node

Let's create a simple publisher node:

### Step 1: Create the Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week1_hello \
  --dependencies rclpy std_msgs
```

### Step 2: Write the Publisher

Create `~/ros2_ws/src/week1_hello/week1_hello/hello_publisher.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloPublisher(Node):
    def __init__(self):
        super().__init__('hello_publisher')
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0
        self.get_logger().info('HelloPublisher started')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello, ROS 2! Count: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = HelloPublisher()
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

### Step 3: Update setup.py

Edit `~/ros2_ws/src/week1_hello/setup.py` to add the entry point:

```python
entry_points={
    'console_scripts': [
        'hello_publisher = week1_hello.hello_publisher:main',
    ],
},
```

### Step 4: Build and Run

```bash
cd ~/ros2_ws
colcon build --packages-select week1_hello
source install/setup.bash
ros2 run week1_hello hello_publisher
```

In another terminal:

```bash
ros2 topic echo /hello_topic
```

## Summary

In this introduction, you learned:

- ✅ ROS 2 architecture and DDS foundation
- ✅ Core concepts: nodes, topics, services, actions
- ✅ Workspace and package structure
- ✅ Essential CLI tools
- ✅ Quality of Service basics
- ✅ Creating a simple publisher node

## Next Steps

Continue to [Nodes and Topics](/module-1/week-1/nodes-and-topics) to dive deeper into ROS 2 communication patterns and build more sophisticated nodes.

---

## References

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- DDS Specification: https://www.omg.org/spec/DDS/
- ROS 2 Design: https://design.ros2.org/
