---
sidebar_position: 3
title: Services and Actions
description: Request-response patterns and long-running tasks in ROS 2
---

# Services and Actions

While topics handle streaming data, services and actions provide structured communication for discrete operations. This lesson covers when and how to use each pattern.

<LearningObjectives
  objectives={[
    "Implement ROS 2 service servers and clients",
    "Understand synchronous vs asynchronous service calls",
    "Create action servers for long-running tasks",
    "Handle action feedback and cancellation",
    "Choose the right communication pattern for each use case"
  ]}
/>

## Communication Patterns Overview

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Topic** | Streaming data | Sensor readings, video |
| **Service** | Quick request-response | Get parameter, spawn object |
| **Action** | Long-running tasks | Navigate to goal, pick object |

## Services

Services provide **synchronous request-response** communication:

```
┌──────────┐    Request     ┌──────────┐
│  Client  │ ─────────────▶ │  Server  │
│          │                │          │
│          │ ◀───────────── │          │
└──────────┘    Response    └──────────┘
```

### When to Use Services

- ✅ Operations that complete quickly (< 1 second)
- ✅ Single request, single response
- ✅ No need for progress updates
- ✅ Examples: Get current pose, set parameter, trigger action

### Service Definition

Services are defined with request and response sections:

```bash
# View service definition
ros2 interface show std_srvs/srv/SetBool
```

Output:
```
bool data # Request
---
bool success # Response
string message
```

### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class LightController(Node):
    def __init__(self):
        super().__init__('light_controller')

        # Create service server
        self.srv = self.create_service(
            SetBool,
            'set_light',
            self.set_light_callback
        )

        self.light_on = False
        self.get_logger().info('Light controller ready')

    def set_light_callback(self, request, response):
        """Handle service request."""
        self.light_on = request.data

        response.success = True
        response.message = f'Light is now {"ON" if self.light_on else "OFF"}'

        self.get_logger().info(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LightController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Creating a Service Client

**Synchronous call** (blocks until response):

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class LightClient(Node):
    def __init__(self):
        super().__init__('light_client')
        self.client = self.create_client(SetBool, 'set_light')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for light service...')

    def turn_on_light(self):
        request = SetBool.Request()
        request.data = True

        # Synchronous call
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

def main():
    rclpy.init()
    client = LightClient()
    result = client.turn_on_light()
    print(f'Result: {result.message}')
    client.destroy_node()
    rclpy.shutdown()
```

**Asynchronous call** (non-blocking):

```python
def turn_on_light_async(self):
    request = SetBool.Request()
    request.data = True

    future = self.client.call_async(request)
    future.add_done_callback(self.light_response_callback)

def light_response_callback(self, future):
    result = future.result()
    self.get_logger().info(f'Light response: {result.message}')
```

### Calling Services from CLI

```bash
# List available services
ros2 service list

# Call a service
ros2 service call /set_light std_srvs/srv/SetBool "{data: true}"

# Check service type
ros2 service type /set_light
```

## Actions

Actions handle **long-running tasks** with feedback and cancellation:

```
┌──────────┐     Goal      ┌──────────┐
│  Client  │ ────────────▶ │  Server  │
│          │               │          │
│          │ ◀──────────── │          │
│          │   Accepted    │          │
│          │               │          │
│          │ ◀──────────── │          │
│          │   Feedback    │          │
│          │      ...      │          │
│          │ ◀──────────── │          │
│          │   Feedback    │          │
│          │               │          │
│          │ ◀──────────── │          │
│          │    Result     │          │
└──────────┘               └──────────┘
```

### When to Use Actions

- ✅ Tasks that take time to complete
- ✅ Need for progress feedback
- ✅ Possibility of cancellation
- ✅ Examples: Navigate to pose, execute trajectory, pick object

### Action Definition

Actions have goal, result, and feedback sections:

```bash
ros2 interface show action_tutorials_interfaces/action/Fibonacci
```

```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] partial_sequence
```

### Creating an Action Server

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from action_tutorials_interfaces.action import Fibonacci

class FibonacciServer(Node):
    def __init__(self):
        super().__init__('fibonacci_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )
        self.get_logger().info('Fibonacci server ready')

    def execute_callback(self, goal_handle):
        """Execute the action."""
        self.get_logger().info(f'Computing Fibonacci({goal_handle.request.order})')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # Check if canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return Fibonacci.Result()

            # Compute next number
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1]
            )

            # Send feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.partial_sequence}')

            # Simulate work
            import time
            time.sleep(1)

        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result

def main():
    rclpy.init()
    node = FibonacciServer()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Creating an Action Client

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_tutorials_interfaces.action import Fibonacci

class FibonacciClient(Node):
    def __init__(self):
        super().__init__('fibonacci_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(
            f'Progress: {feedback_msg.feedback.partial_sequence}'
        )

def main():
    rclpy.init()
    client = FibonacciClient()
    client.send_goal(10)
    rclpy.spin(client)
```

### Action CLI Commands

```bash
# List actions
ros2 action list

# Get action info
ros2 action info /fibonacci

# Send a goal
ros2 action send_goal /fibonacci action_tutorials_interfaces/action/Fibonacci "{order: 10}"

# Send goal with feedback
ros2 action send_goal /fibonacci action_tutorials_interfaces/action/Fibonacci "{order: 10}" --feedback
```

## Real-World Example: Navigation

Nav2 uses actions for navigation goals:

<ServiceTable
  services={[
    { name: "/navigate_to_pose", type: "nav2_msgs/action/NavigateToPose", description: "Navigate robot to target pose" },
    { name: "/follow_path", type: "nav2_msgs/action/FollowPath", description: "Follow a pre-computed path" },
    { name: "/spin", type: "nav2_msgs/action/Spin", description: "Rotate robot in place" }
  ]}
/>

Example: Sending a navigation goal:

```python
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

def navigate_to(self, x, y, theta):
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    goal.pose.pose.orientation.z = math.sin(theta / 2)
    goal.pose.pose.orientation.w = math.cos(theta / 2)

    self.nav_client.send_goal_async(goal)
```

## Choosing the Right Pattern

| Scenario | Pattern | Reason |
|----------|---------|--------|
| Stream camera images | Topic | Continuous data |
| Get robot battery level | Service | Quick query |
| Navigate to goal | Action | Long task with feedback |
| Emergency stop | Topic | Fast, fire-and-forget |
| Arm movement trajectory | Action | Needs progress/cancel |
| Get map from server | Service | Request-response |

## Exercises

<Exercise title="Calculator Service" difficulty="beginner" estimatedTime="20 min">

Create a calculator service that performs basic arithmetic:

Service definition (`Calculator.srv`):
```
float64 a
float64 b
string operation  # "add", "subtract", "multiply", "divide"
---
float64 result
bool success
string message
```

Requirements:
- Handle division by zero
- Return appropriate error messages
- Test with CLI

<Hint>
Use a dictionary to map operation strings to functions:
```python
operations = {
    'add': lambda a, b: a + b,
    'multiply': lambda a, b: a * b,
    ...
}
```
</Hint>

</Exercise>

<Exercise title="Progress Tracker Action" difficulty="intermediate" estimatedTime="30 min">

Create an action server that simulates a download:

- Goal: Number of "bytes" to download
- Feedback: Current progress (0-100%)
- Result: Total time taken, success status
- Support cancellation

</Exercise>

## Summary

Key concepts covered:

- ✅ Services for quick request-response operations
- ✅ Synchronous vs asynchronous service calls
- ✅ Actions for long-running tasks with feedback
- ✅ Goal, feedback, and result in action lifecycle
- ✅ Cancellation handling in actions
- ✅ Pattern selection guidelines

## Next Steps

Complete the [Week 1 Exercises](/module-1/week-1/exercises) to practice these communication patterns, then continue to [Week 2: TF2 & Navigation](/module-1/week-2/tf2-transforms).
