---
sidebar_position: 4
title: Week 12 Exercises
description: Hands-on exercises for system integration and testing
---

# Week 12 Exercises

Complete these exercises to build practical experience with system integration and testing for humanoid robotics systems.

<Prerequisites
  items={[
    "Completed all Week 12 lessons",
    "ROS 2 Humble with development tools",
    "Gazebo Harmonic or Isaac Sim installed",
    "Python testing frameworks (pytest, unittest)"
  ]}
/>

## Exercise 1: Component Integration Pipeline

<Exercise title="Modular Component Integration" difficulty="intermediate" estimatedTime="120 min">

Create a modular integration system that connects perception, planning, and control components using design patterns:

**Requirements:**
1. Implement Observer pattern for sensor data distribution
2. Use Strategy pattern for different control modes
3. Apply Command pattern for action execution
4. Include Circuit Breaker for component reliability
5. Add comprehensive error handling

**Acceptance Criteria:**
- [ ] All components communicate via defined interfaces
- [ ] Observer pattern properly distributes sensor data
- [ ] Strategy pattern selects appropriate control algorithm
- [ ] Command pattern manages action execution safely
- [ ] Circuit breaker prevents cascade failures
- [ ] Error handling is comprehensive and graceful

<Hint>
Structure your system with clear separation of concerns:
```python
class IntegrationSystem:
    def __init__(self):
        self.sensor_manager = SensorSubject()  # Observer pattern
        self.control_context = ControlContext()  # Strategy pattern
        self.command_invoker = CommandInvoker()  # Command pattern
        self.circuit_breaker = CircuitBreaker()  # Fault tolerance
```
</Hint>

<Solution>
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import threading
import time
import json

# Observer Pattern for Sensor Integration
class SensorObserver(ABC):
    @abstractmethod
    def update(self, sensor_type: str, data: Any):
        pass

class SensorSubject:
    def __init__(self):
        self._observers: List[SensorObserver] = []
        self._lock = threading.Lock()

    def attach(self, observer: SensorObserver):
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def detach(self, observer: SensorObserver):
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def notify(self, sensor_type: str, data: Any):
        with self._lock:
            for observer in self._observers:
                try:
                    observer.update(sensor_type, data)
                except Exception as e:
                    print(f"Observer error: {e}")

# Strategy Pattern for Control Algorithms
class ControlStrategy(ABC):
    @abstractmethod
    def compute_control(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        pass

class NavigationStrategy(ControlStrategy):
    def compute_control(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        # Simple navigation controller
        current_pos = state.get('position', {'x': 0, 'y': 0})
        target_pos = target.get('position', {'x': 0, 'y': 0})

        dx = target_pos['x'] - current_pos['x']
        dy = target_pos['y'] - current_pos['y']
        distance = (dx**2 + dy**2)**0.5

        if distance < 0.2:  # Close enough
            return {'linear_x': 0.0, 'angular_z': 0.0, 'status': 'reached'}

        linear_vel = min(0.5, distance * 0.5)
        angular_vel = 2.0 * math.atan2(dy, dx)

        return {
            'linear_x': linear_vel,
            'angular_z': angular_vel,
            'status': 'navigating'
        }

class ControlContext:
    def __init__(self):
        self.strategies = {
            'navigation': NavigationStrategy(),
            # Add more strategies as needed
        }
        self.current_strategy = 'navigation'

    def set_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name

    def execute(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        strategy = self.strategies[self.current_strategy]
        return strategy.compute_control(state, target)

# Command Pattern for Action Execution
class Command(ABC):
    @abstractmethod
    def execute(self) -> bool:
        pass

class NavigationCommand(Command):
    def __init__(self, target: Dict[str, Any], timeout: float = 30.0):
        self.target = target
        self.timeout = timeout
        self.executing = False

    def execute(self) -> bool:
        # This would interface with navigation stack
        print(f"Executing navigation to {self.target}")
        self.executing = True
        # Simulate execution
        time.sleep(0.1)
        self.executing = False
        return True

# Circuit Breaker for Fault Tolerance
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - (self.last_failure_time or 0) > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self._reset()
            elif self.state == 'CLOSED':
                self.failure_count = 0
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

    def _reset(self):
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None

# Main Integration Node
class IntegrationNode(Node):
    def __init__(self):
        super().__init__('integration_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'system_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, 'scan', self.laser_callback, 10)

        # Integration components
        self.sensor_subject = SensorSubject()
        self.control_context = ControlContext()
        self.circuit_breaker = CircuitBreaker()

        # Add observers
        self.perception_observer = PerceptionObserver(self)
        self.sensor_subject.attach(self.perception_observer)

        self.get_logger().info('Integration node initialized')

    def image_callback(self, msg):
        self.sensor_subject.notify('camera', msg)

    def laser_callback(self, msg):
        self.sensor_subject.notify('laser', msg)

class PerceptionObserver(SensorObserver):
    def __init__(self, node: IntegrationNode):
        self.node = node
        self.system_state = {
            'position': {'x': 0, 'y': 0},
            'obstacles': [],
            'targets': []
        }

    def update(self, sensor_type: str, data: Any):
        if sensor_type == 'laser':
            # Process laser data for obstacles
            obstacles = self.process_laser_data(data)
            self.system_state['obstacles'] = obstacles

            # Generate navigation command if target exists
            if self.system_state['targets']:
                target = self.system_state['targets'][0]

                # Use circuit breaker for safety
                try:
                    control_cmd = self.node.circuit_breaker.call(
                        self.node.control_context.execute,
                        self.system_state,
                        target
                    )

                    # Publish command
                    cmd_msg = Twist()
                    cmd_msg.linear.x = control_cmd.get('linear_x', 0.0)
                    cmd_msg.angular.z = control_cmd.get('angular_z', 0.0)
                    self.node.cmd_vel_pub.publish(cmd_msg)

                except Exception as e:
                    self.node.get_logger().error(f'Safety system triggered: {e}')
                    # Emergency stop
                    stop_msg = Twist()
                    self.node.cmd_vel_pub.publish(stop_msg)

    def process_laser_data(self, laser_msg):
        # Simple obstacle detection
        obstacles = []
        for i, range_val in enumerate(laser_msg.ranges):
            if 0 < range_val < 1.0:  # Obstacle within 1m
                angle = i * laser_msg.angle_increment + laser_msg.angle_min
                obstacles.append({
                    'distance': range_val,
                    'angle': angle,
                    'x': range_val * math.cos(angle),
                    'y': range_val * math.sin(angle)
                })
        return obstacles

def main():
    rclpy.init()
    node = IntegrationNode()

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
</Solution>

</Exercise>

## Exercise 2: Real-Time Performance Pipeline

<Exercise title="Real-Time Processing Pipeline" difficulty="advanced" estimatedTime="180 min">

Build a real-time processing pipeline that:
1. Implements bounded buffers for data flow management
2. Uses producer-consumer pattern for real-time processing
3. Maintains timing constraints for real-time performance
4. Handles buffer overflow scenarios gracefully
5. Includes performance monitoring and optimization

**Requirements:**
- Bounded buffer implementation with overflow handling
- Multi-threaded producer-consumer architecture
- Real-time timing constraint enforcement
- Performance monitoring and logging
- Dynamic optimization based on system load

**Acceptance Criteria:**
- [ ] Bounded buffers prevent memory overflow
- [ ] Real-time processing maintains target rates
- [ ] Buffer overflow is handled gracefully
- [ ] Performance metrics are collected and logged
- [ ] System adapts to varying loads

</Exercise>

## Exercise 3: Comprehensive Testing Framework

<Exercise title="Complete Testing Framework" difficulty="advanced" estimatedTime="240 min">

Create a comprehensive testing framework that:
1. Implements unit tests for all system components
2. Creates integration tests for component interactions
3. Builds simulation-based testing scenarios
4. Includes performance and stress testing
5. Generates automated test reports

**Requirements:**
- Complete unit test coverage (>80% for critical components)
- ROS 2 launch testing for integration scenarios
- Gazebo simulation tests for system behavior
- Performance benchmarking and stress testing
- Automated test reporting and CI/CD integration

**Acceptance Criteria:**
- [ ] Unit tests cover all major components
- [ ] Integration tests validate component interactions
- [ ] Simulation tests verify system behavior
- [ ] Performance tests measure real-time constraints
- [ ] Test reports are automatically generated

</Exercise>

## Exercise 4: Fault-Tolerant System

<Exercise title="Fault-Tolerant Integration" difficulty="advanced" estimatedTime="300 min">

Build a fault-tolerant system that:
1. Implements multiple fault-tolerance patterns
2. Handles component failures gracefully
3. Provides system recovery mechanisms
4. Includes health monitoring and alerting
5. Maintains operation during partial failures

**Requirements:**
- Circuit breaker pattern for service calls
- Retry mechanisms with exponential backoff
- Fallback strategies for component failures
- Health monitoring and self-diagnosis
- Graceful degradation capabilities

**Acceptance Criteria:**
- [ ] System continues operating during component failures
- [ ] Recovery mechanisms restore functionality
- [ ] Health monitoring detects issues proactively
- [ ] Performance degrades gracefully, not catastrophically
- [ ] Fallback strategies maintain core functionality

</Exercise>

## Exercise 5: Capstone Integration Challenge

<Exercise title="Complete Humanoid System Integration" difficulty="advanced" estimatedTime="480 min">

Build a complete humanoid robotics system that integrates all components from the course:
1. ROS 2 architecture with proper component organization
2. Perception system (vision, sensors, SLAM)
3. Planning system (navigation, manipulation)
4. Control system (motion, grasping)
5. VLA integration for high-level commands
6. Safety system with comprehensive constraints
7. Testing framework with all test types
8. Performance optimization and monitoring

**System Architecture:**
```
┌─────────────────┐    Voice    ┌─────────────────┐
│   User Input    │ ─────────▶ │   VLA Models    │
│                 │             │                 │
└─────────────────┘             └─────────────────┘
                                          │
                                          ▼
┌─────────────────┐              ┌─────────────────┐
│   Sensors       │ ───────────▶ │  Perception     │
│ (Cam, LiDAR,..) │   Data       │   System        │
└─────────────────┘              └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │   Planning      │
                                │   System        │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │   Control       │
                                │   System        │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │   Safety &      │
                                │   Monitoring    │
                                └─────────────────┘
```

**Requirements:**
- Complete system integration following best practices
- All components from previous modules integrated
- Comprehensive testing and validation
- Performance optimization for real-time operation
- Safety system ensuring safe operation
- Documentation and maintainability

**Acceptance Criteria:**
- [ ] All system components integrated and communicating
- [ ] System operates safely in simulation
- [ ] Performance meets real-time constraints
- [ ] Safety constraints are enforced
- [ ] Comprehensive tests pass
- [ ] System is maintainable and well-documented

</Exercise>

## Self-Assessment

Rate your confidence (1-5) after completing these exercises:

| Skill | Target | Your Rating |
|-------|--------|-------------|
| System architecture design | 4 | ___ |
| Component integration patterns | 4 | ___ |
| Real-time performance optimization | 4 | ___ |
| Comprehensive testing strategies | 4 | ___ |
| Fault-tolerant system design | 4 | ___ |
| Integration of multiple subsystems | 4 | ___ |
| Performance monitoring and optimization | 3 | ___ |

If any rating is below target, review the corresponding lesson material.

## Submission Checklist

Before moving to Week 13, ensure:

- [ ] All exercises compile without warnings
- [ ] Integration system operates correctly
- [ ] Testing framework provides adequate coverage
- [ ] Performance meets real-time requirements
- [ ] Safety systems are functional
- [ ] Code follows ROS 2 Python style guidelines

---

**Ready for Week 13?** Continue to [Week 13: Deployment](/module-5/week-13/deployment-checklist) to learn about deploying your humanoid robotics system to production.