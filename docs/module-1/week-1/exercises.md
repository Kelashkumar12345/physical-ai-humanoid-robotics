---
sidebar_position: 4
title: Week 1 Exercises
description: Hands-on exercises for ROS 2 fundamentals
---

# Week 1 Exercises

Complete these exercises to reinforce your understanding of ROS 2 fundamentals. Each exercise builds on previous concepts.

<Prerequisites
  items={[
    "Completed all Week 1 lessons",
    "ROS 2 Humble installed and sourced",
    "Workspace created at ~/ros2_ws",
    "TurtleBot3 packages installed"
  ]}
/>

## Exercise 1: Multi-Topic Sensor Simulation

<Exercise title="Sensor Suite Publisher" difficulty="beginner" estimatedTime="25 min">

Create a node that simulates a robot sensor suite by publishing to multiple topics:

**Requirements:**
1. Publish to these topics at specified rates:
   - `/temperature` (std_msgs/Float64) at 1 Hz
   - `/humidity` (std_msgs/Float64) at 1 Hz
   - `/battery_voltage` (std_msgs/Float64) at 0.5 Hz
   - `/imu/orientation` (geometry_msgs/Quaternion) at 10 Hz

2. Use realistic value ranges:
   - Temperature: 20-30°C
   - Humidity: 40-60%
   - Battery: 11.0-12.6V (slowly decreasing)
   - IMU: Small random variations around identity quaternion

**Acceptance Criteria:**
- [ ] All four topics publish at correct rates
- [ ] `ros2 topic hz` confirms rates within 10%
- [ ] Values stay within specified ranges

<Hint>
Use multiple timers with different periods:
```python
self.temp_timer = self.create_timer(1.0, self.publish_temp)
self.imu_timer = self.create_timer(0.1, self.publish_imu)
```
</Hint>

<Solution>
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Quaternion
import random
import math

class SensorSuite(Node):
    def __init__(self):
        super().__init__('sensor_suite')

        # Publishers
        self.temp_pub = self.create_publisher(Float64, 'temperature', 10)
        self.humidity_pub = self.create_publisher(Float64, 'humidity', 10)
        self.battery_pub = self.create_publisher(Float64, 'battery_voltage', 10)
        self.imu_pub = self.create_publisher(Quaternion, 'imu/orientation', 10)

        # Timers
        self.create_timer(1.0, self.publish_temp)
        self.create_timer(1.0, self.publish_humidity)
        self.create_timer(2.0, self.publish_battery)
        self.create_timer(0.1, self.publish_imu)

        self.battery_voltage = 12.6
        self.get_logger().info('Sensor suite started')

    def publish_temp(self):
        msg = Float64()
        msg.data = random.uniform(20.0, 30.0)
        self.temp_pub.publish(msg)

    def publish_humidity(self):
        msg = Float64()
        msg.data = random.uniform(40.0, 60.0)
        self.humidity_pub.publish(msg)

    def publish_battery(self):
        msg = Float64()
        self.battery_voltage = max(11.0, self.battery_voltage - 0.01)
        msg.data = self.battery_voltage
        self.battery_pub.publish(msg)

    def publish_imu(self):
        msg = Quaternion()
        # Small random rotation around identity
        angle = random.uniform(-0.01, 0.01)
        msg.x = 0.0
        msg.y = 0.0
        msg.z = math.sin(angle / 2)
        msg.w = math.cos(angle / 2)
        self.imu_pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(SensorSuite())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
</Solution>

</Exercise>

## Exercise 2: Data Aggregator Node

<Exercise title="Sensor Data Aggregator" difficulty="intermediate" estimatedTime="30 min">

Create a subscriber node that aggregates data from Exercise 1:

**Requirements:**
1. Subscribe to all four sensor topics
2. Maintain a rolling average of last 10 readings for temperature and humidity
3. Publish aggregated status to `/robot_status` (custom message) at 1 Hz

**Custom Message** (`RobotStatus.msg`):
```
float64 avg_temperature
float64 avg_humidity
float64 battery_voltage
bool battery_low    # True if < 11.5V
bool sensors_ok     # True if receiving all sensor data
```

**Acceptance Criteria:**
- [ ] Correctly computes rolling averages
- [ ] Battery warning triggers at correct threshold
- [ ] `sensors_ok` becomes False if any topic stops publishing

<Hint>
Use `collections.deque` with `maxlen=10` for rolling averages:
```python
from collections import deque
self.temp_history = deque(maxlen=10)
```
</Hint>

</Exercise>

## Exercise 3: Robot State Service

<Exercise title="State Query Service" difficulty="intermediate" estimatedTime="30 min">

Create a service that returns the current robot state:

**Service Definition** (`GetRobotState.srv`):
```
---
string state          # "idle", "moving", "charging", "error"
float64 uptime_seconds
int32 messages_received
float64 battery_level
```

**Requirements:**
1. Track node uptime since startup
2. Count total messages received from any topic
3. Simulate state changes based on battery level:
   - Battery > 50%: "idle" or "moving"
   - Battery 20-50%: "idle"
   - Battery < 20%: "charging"
   - No sensor data: "error"

**Acceptance Criteria:**
- [ ] Service responds within 100ms
- [ ] Uptime accurately tracks seconds since start
- [ ] State transitions follow battery rules

</Exercise>

## Exercise 4: TurtleBot3 Square Patrol

<Exercise title="Square Patrol Action" difficulty="advanced" estimatedTime="45 min">

Create an action server that commands TurtleBot3 to patrol in a square:

**Action Definition** (`SquarePatrol.action`):
```
# Goal
float64 side_length    # meters
int32 num_laps
---
# Result
bool success
float64 total_distance
float64 total_time
---
# Feedback
int32 current_lap
int32 current_side     # 1-4
float64 distance_traveled
```

**Requirements:**
1. Command TurtleBot3 to move in a square pattern
2. Publish feedback after completing each side
3. Support cancellation mid-patrol
4. Handle simulation vs real robot (check for `/odom` topic)

**Acceptance Criteria:**
- [ ] Robot completes square with < 10% position error
- [ ] Feedback updates 4 times per lap
- [ ] Cancellation stops robot safely
- [ ] Works in Gazebo simulation

<Hint>
Use a state machine approach:
```python
class PatrolState:
    MOVING_FORWARD = 0
    TURNING = 1
    COMPLETED_SIDE = 2
```

For turning, use odometry feedback to track rotation angle.
</Hint>

</Exercise>

## Exercise 5: Integration Challenge

<Exercise title="Complete Robot System" difficulty="advanced" estimatedTime="60 min">

Combine all previous exercises into a complete system:

**System Architecture:**
```
┌────────────────┐     Topics      ┌────────────────┐
│  Sensor Suite  │ ──────────────▶ │   Aggregator   │
│    (Ex. 1)     │                 │    (Ex. 2)     │
└────────────────┘                 └───────┬────────┘
                                           │
                                           ▼
┌────────────────┐    Service     ┌────────────────┐
│    Client      │ ◀────────────▶ │  State Server  │
│    (New)       │                │    (Ex. 3)     │
└────────────────┘                └────────────────┘
        │
        │ Action Goal
        ▼
┌────────────────┐
│ Square Patrol  │
│    (Ex. 4)     │
└────────────────┘
```

**Requirements:**
1. Create a supervisor node that:
   - Queries robot state before starting patrol
   - Only starts patrol if state is "idle" and battery > 30%
   - Monitors battery during patrol
   - Cancels patrol if battery drops below 15%
   - Logs all state transitions

2. Launch file that starts all nodes:
   - `sensor_suite_node`
   - `aggregator_node`
   - `state_server_node`
   - `patrol_server_node`
   - `supervisor_node`

**Acceptance Criteria:**
- [ ] All nodes start without errors
- [ ] Supervisor correctly gates patrol start
- [ ] Battery monitoring works during patrol
- [ ] Clean shutdown on Ctrl+C

</Exercise>

## Submission Checklist

Before moving to Week 2, ensure:

- [ ] All exercises compile without warnings
- [ ] Code follows ROS 2 Python style guidelines
- [ ] Each exercise has been tested with CLI tools
- [ ] Custom interfaces are properly defined
- [ ] Launch files work correctly

## Self-Assessment

Rate your confidence (1-5) after completing these exercises:

| Skill | Target | Your Rating |
|-------|--------|-------------|
| Creating publishers | 4 | ___ |
| Creating subscribers | 4 | ___ |
| Using standard messages | 4 | ___ |
| Creating custom messages | 3 | ___ |
| Implementing services | 4 | ___ |
| Implementing actions | 3 | ___ |
| Using ROS 2 CLI tools | 4 | ___ |
| Multi-node systems | 3 | ___ |

If any rating is below target, review the corresponding lesson material.

---

**Ready for Week 2?** Continue to [Week 2: TF2 & Navigation](/module-1/week-2/tf2-transforms) to learn about coordinate frames and robot transformations.
