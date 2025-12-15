---
sidebar_position: 2
title: Integration Patterns
description: Advanced integration patterns for humanoid robotics systems
---

# Integration Patterns

This lesson covers advanced integration patterns for connecting the various components of humanoid robotics systems, focusing on scalability, maintainability, and real-time performance.

<LearningObjectives
  objectives={[
    "Apply design patterns for robotics system integration",
    "Implement scalable component communication architectures",
    "Design fault-tolerant integration systems",
    "Optimize integration for real-time performance",
    "Create maintainable and testable integration code"
  ]}
/>

## Design Patterns for Robotics Integration

### Observer Pattern for Sensor Integration

```python
from abc import ABC, abstractmethod
from typing import List, Any, Dict
import threading

class SensorObserver(ABC):
    """Abstract base class for sensor observers."""

    @abstractmethod
    def update(self, sensor_data: Dict[str, Any]):
        """Called when sensor data is updated."""
        pass

class SensorSubject:
    """Subject that maintains list of observers and notifies them of state changes."""

    def __init__(self):
        self._observers: List[SensorObserver] = []
        self._lock = threading.Lock()
        self._sensor_data = {}

    def attach(self, observer: SensorObserver):
        """Attach an observer to the subject."""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def detach(self, observer: SensorObserver):
        """Detach an observer from the subject."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def notify(self, sensor_type: str, data: Any):
        """Notify all observers of sensor data update."""
        sensor_data = {
            'type': sensor_type,
            'data': data,
            'timestamp': time.time()
        }

        with self._lock:
            for observer in self._observers:
                try:
                    observer.update(sensor_data)
                except Exception as e:
                    print(f"Observer update failed: {e}")

class CameraObserver(SensorObserver):
    """Observer that processes camera data."""

    def update(self, sensor_data: Dict[str, Any]):
        if sensor_data['type'] == 'camera':
            # Process camera image
            image = sensor_data['data']
            processed_data = self.process_image(image)
            # Publish to vision processing pipeline
            self.publish_vision_data(processed_data)

class LaserObserver(SensorObserver):
    """Observer that processes laser scan data."""

    def update(self, sensor_data: Dict[str, Any]):
        if sensor_data['type'] == 'laser':
            # Process laser scan
            scan = sensor_data['data']
            obstacles = self.detect_obstacles(scan)
            # Update navigation system
            self.update_navigation(obstacles)

class SensorIntegrationManager:
    """Manages sensor integration using observer pattern."""

    def __init__(self):
        self.sensor_subject = SensorSubject()

        # Add observers
        self.camera_observer = CameraObserver()
        self.laser_observer = LaserObserver()
        self.odom_observer = OdometryObserver()

        self.sensor_subject.attach(self.camera_observer)
        self.sensor_subject.attach(self.laser_observer)
        self.sensor_subject.attach(self.odom_observer)

    def process_sensor_data(self, sensor_type: str, data: Any):
        """Process sensor data by notifying all observers."""
        self.sensor_subject.notify(sensor_type, data)
```

### Strategy Pattern for Control Algorithms

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class ControlStrategy(ABC):
    """Abstract base class for control strategies."""

    @abstractmethod
    def compute_control(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        """Compute control commands based on current state and target."""
        pass

    @abstractmethod
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this strategy is applicable for the given context."""
        pass

class NavigationControlStrategy(ControlStrategy):
    """Strategy for navigation control."""

    def compute_control(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        # Simple proportional controller for navigation
        current_pos = state['position']
        target_pos = target['position']

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:  # 10cm threshold
            return {'linear_x': 0.0, 'angular_z': 0.0, 'status': 'reached_target'}

        # Compute control
        linear_vel = min(0.5, distance * 0.5)  # Max 0.5 m/s
        angular_vel = np.arctan2(dy, dx) * 0.5  # Simple heading control

        return {
            'linear_x': linear_vel,
            'angular_z': angular_vel,
            'distance_to_target': distance,
            'status': 'navigating'
        }

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        return context.get('task_type') == 'navigation'

class ManipulationControlStrategy(ControlStrategy):
    """Strategy for manipulation control."""

    def compute_control(self, state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        # Compute joint angles for reaching target
        current_pose = state['end_effector_pose']
        target_pose = target['end_effector_target']

        # Simple inverse kinematics (in practice, use MoveIt2)
        joint_commands = self.compute_ik(current_pose, target_pose)

        return {
            'joint_commands': joint_commands,
            'status': 'manipulating'
        }

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        return context.get('task_type') == 'manipulation'

class ControlContext:
    """Context that uses different control strategies."""

    def __init__(self):
        self.strategies: List[ControlStrategy] = [
            NavigationControlStrategy(),
            ManipulationControlStrategy(),
            # Add more strategies as needed
        ]

    def execute_control(self, state: Dict[str, Any], target: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate control strategy based on context."""
        for strategy in self.strategies:
            if strategy.is_applicable(context):
                return strategy.compute_control(state, target)

        # Default strategy if none applicable
        return {'linear_x': 0.0, 'angular_z': 0.0, 'status': 'no_strategy'}

class AdaptiveControlManager:
    """Manager that adapts control strategies based on performance."""

    def __init__(self):
        self.control_context = ControlContext()
        self.performance_history = {}

    def select_strategy(self, state: Dict[str, Any], target: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Select control strategy based on context and performance history."""
        # Check performance history for this type of task
        task_key = context.get('task_type', 'default')

        if task_key in self.performance_history:
            # Use strategy that performed best historically
            best_strategy = self.performance_history[task_key]['best_strategy']
            return best_strategy.compute_control(state, target)

        # Use default strategy selection
        return self.control_context.execute_control(state, target, context)
```

### Command Pattern for Action Execution

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time

class Command(ABC):
    """Abstract command interface."""

    @abstractmethod
    def execute(self) -> bool:
        """Execute the command."""
        pass

    @abstractmethod
    def undo(self) -> bool:
        """Undo the command (if possible)."""
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if command execution is complete."""
        pass

class NavigationCommand(Command):
    """Command for navigation actions."""

    def __init__(self, target_pose: Dict[str, Any], timeout: float = 30.0):
        self.target_pose = target_pose
        self.timeout = timeout
        self.start_time = None
        self.is_executing = False

    def execute(self) -> bool:
        """Start navigation to target pose."""
        self.start_time = time.time()
        self.is_executing = True

        # Send navigation goal
        self.send_navigation_goal()
        return True

    def send_navigation_goal(self):
        """Send navigation goal to Nav2."""
        # Implementation would send goal to Nav2
        pass

    def undo(self) -> bool:
        """Stop navigation."""
        self.is_executing = False
        # Send stop command
        return True

    def is_complete(self) -> bool:
        """Check if navigation is complete."""
        if not self.is_executing:
            return True

        if time.time() - self.start_time > self.timeout:
            return True  # Timeout

        # Check if goal reached (would interface with Nav2)
        return self.check_navigation_complete()

    def check_navigation_complete(self) -> bool:
        """Check if navigation goal is reached."""
        # Implementation would check Nav2 status
        return False

class ManipulationCommand(Command):
    """Command for manipulation actions."""

    def __init__(self, action_type: str, parameters: Dict[str, Any], timeout: float = 15.0):
        self.action_type = action_type
        self.parameters = parameters
        self.timeout = timeout
        self.start_time = None
        self.is_executing = False

    def execute(self) -> bool:
        """Execute manipulation action."""
        self.start_time = time.time()
        self.is_executing = True

        if self.action_type == 'grasp':
            return self.execute_grasp()
        elif self.action_type == 'place':
            return self.execute_place()
        return False

    def execute_grasp(self) -> bool:
        """Execute grasp action."""
        # Implementation would use MoveIt2 or manipulation stack
        return True

    def execute_place(self) -> bool:
        """Execute place action."""
        # Implementation would use MoveIt2 or manipulation stack
        return True

    def undo(self) -> bool:
        """Undo manipulation action."""
        self.is_executing = False
        return True

    def is_complete(self) -> bool:
        """Check if manipulation is complete."""
        if not self.is_executing:
            return True

        if time.time() - self.start_time > self.timeout:
            return True

        # Check if action completed
        return self.check_action_complete()

    def check_action_complete(self) -> bool:
        """Check if manipulation action is complete."""
        # Implementation would check action status
        return False

class CommandInvoker:
    """Invoker that executes commands and manages command history."""

    def __init__(self):
        self.command_history: List[Command] = []
        self.active_commands: List[Command] = []

    def execute_command(self, command: Command) -> bool:
        """Execute a command."""
        success = command.execute()
        if success:
            self.active_commands.append(command)
            self.command_history.append(command)
        return success

    def execute_command_sequence(self, commands: List[Command]) -> bool:
        """Execute a sequence of commands."""
        for command in commands:
            if not self.execute_command(command):
                return False

            # Wait for command to complete
            while not command.is_complete():
                time.sleep(0.1)

        return True

    def cancel_active_commands(self):
        """Cancel all active commands."""
        for command in self.active_commands:
            command.undo()
        self.active_commands.clear()
```

## Communication Architecture Patterns

### Publisher-Subscriber with Quality of Service

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from typing import Dict, Callable

class QoSManager:
    """Manages Quality of Service settings for different data types."""

    def __init__(self):
        self.qos_profiles = {
            # Sensor data - best effort, small history
            'sensor': QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            ),
            # Control commands - reliable, small history
            'control': QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            ),
            # State information - reliable, larger history
            'state': QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            ),
            # Debug/monitoring - best effort, larger history
            'debug': QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=20
            )
        }

    def get_qos_profile(self, topic_type: str) -> QoSProfile:
        """Get appropriate QoS profile for topic type."""
        return self.qos_profiles.get(topic_type, self.qos_profiles['sensor'])

class CommunicationManager(Node):
    """Manages communication between system components."""

    def __init__(self):
        super().__init__('communication_manager')

        self.qos_manager = QoSManager()

        # Publishers
        self.control_pub = self.create_publisher(
            Twist, 'cmd_vel', self.qos_manager.get_qos_profile('control')
        )
        self.status_pub = self.create_publisher(
            String, 'system_status', self.qos_manager.get_qos_profile('state')
        )
        self.debug_pub = self.create_publisher(
            String, 'debug_info', self.qos_manager.get_qos_profile('debug')
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback,
            self.qos_manager.get_qos_profile('sensor')
        )
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback,
            self.qos_manager.get_qos_profile('sensor')
        )
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback,
            self.qos_manager.get_qos_profile('state')
        )

        # Message processors
        self.message_processors: Dict[str, Callable] = {}

        self.get_logger().info('Communication Manager initialized')

    def register_message_processor(self, topic_name: str, processor: Callable):
        """Register a processor for specific topic messages."""
        self.message_processors[topic_name] = processor

    def image_callback(self, msg):
        """Process camera image message."""
        if 'camera' in self.message_processors:
            self.message_processors['camera'](msg)

    def laser_callback(self, msg):
        """Process laser scan message."""
        if 'laser' in self.message_processors:
            self.message_processors['laser'](msg)

    def joint_callback(self, msg):
        """Process joint state message."""
        if 'joints' in self.message_processors:
            self.message_processors['joints'](msg)
```

### Service-Based Component Communication

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger, SetBool
from example_interfaces.srv import SetInt16, GetString

class ServiceManager(Node):
    """Manages service-based communication between components."""

    def __init__(self):
        super().__init__('service_manager')

        # Create callback group for reentrant services
        self.callback_group = ReentrantCallbackGroup()

        # System services
        self.start_system_srv = self.create_service(
            Trigger, 'system/start', self.start_system_callback,
            callback_group=self.callback_group
        )
        self.stop_system_srv = self.create_service(
            Trigger, 'system/stop', self.stop_system_callback,
            callback_group=self.callback_group
        )
        self.reset_system_srv = self.create_service(
            Trigger, 'system/reset', self.reset_system_callback,
            callback_group=self.callback_group
        )

        # Component control services
        self.enable_component_srv = self.create_service(
            SetBool, 'component/enable', self.enable_component_callback,
            callback_group=self.callback_group
        )
        self.set_component_priority_srv = self.create_service(
            SetInt16, 'component/priority', self.set_priority_callback,
            callback_group=self.callback_group
        )

        # Status services
        self.get_system_status_srv = self.create_service(
            Trigger, 'system/status', self.get_status_callback,
            callback_group=self.callback_group
        )

        self.system_status = {
            'is_running': False,
            'components': {},
            'last_error': None
        }

        self.get_logger().info('Service Manager initialized')

    def start_system_callback(self, request, response):
        """Start the entire system."""
        try:
            # Initialize all components
            self.initialize_components()
            self.system_status['is_running'] = True
            response.success = True
            response.message = "System started successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to start system: {str(e)}"

        return response

    def stop_system_callback(self, request, response):
        """Stop the entire system."""
        try:
            # Stop all components gracefully
            self.shutdown_components()
            self.system_status['is_running'] = False
            response.success = True
            response.message = "System stopped successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to stop system: {str(e)}"

        return response

    def reset_system_callback(self, request, response):
        """Reset the system to initial state."""
        try:
            # Stop and restart system
            self.shutdown_components()
            time.sleep(1.0)  # Brief pause
            self.initialize_components()
            response.success = True
            response.message = "System reset successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to reset system: {str(e)}"

        return response

    def enable_component_callback(self, request, response):
        """Enable or disable a specific component."""
        component_name = request.data  # Using data field to pass component name
        try:
            # Enable/disable component logic
            success = self.set_component_enabled(component_name, request.data)
            response.success = success
            response.message = f"Component {component_name} {'enabled' if request.data else 'disabled'}"
        except Exception as e:
            response.success = False
            response.message = f"Failed to control component: {str(e)}"

        return response

    def get_status_callback(self, request, response):
        """Get system status."""
        response.success = True
        response.message = json.dumps(self.system_status)
        return response

    def initialize_components(self):
        """Initialize all system components."""
        # Implementation would initialize each component
        pass

    def shutdown_components(self):
        """Shutdown all system components."""
        # Implementation would shutdown each component
        pass

    def set_component_enabled(self, component_name: str, enabled: bool) -> bool:
        """Enable or disable specific component."""
        # Implementation would control component state
        return True

class ComponentClient:
    """Client for interacting with system services."""

    def __init__(self, node: Node):
        self.node = node

        # Create clients for system services
        self.start_client = self.node.create_client(Trigger, 'system/start')
        self.stop_client = self.node.create_client(Trigger, 'system/stop')
        self.status_client = self.node.create_client(Trigger, 'system/status')
        self.enable_client = self.node.create_client(SetBool, 'component/enable')

        # Wait for services to be available
        timeout = time.time() + 10.0  # 10 second timeout
        while not self.start_client.wait_for_service(timeout_sec=1.0) and time.time() < timeout:
            self.node.get_logger().info('Waiting for system services...')

        if time.time() >= timeout:
            raise RuntimeError("System services not available")

    def start_system(self) -> bool:
        """Start the system."""
        request = Trigger.Request()
        future = self.start_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        response = future.result()
        return response.success if response else False

    def stop_system(self) -> bool:
        """Stop the system."""
        request = Trigger.Request()
        future = self.stop_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        response = future.result()
        return response.success if response else False

    def get_system_status(self) -> Dict:
        """Get system status."""
        request = Trigger.Request()
        future = self.status_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        response = future.result()
        if response and response.success:
            return json.loads(response.message)
        return {}
```

## Fault-Tolerance Patterns

### Circuit Breaker Pattern for Component Reliability

```python
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, no requests
    HALF_OPEN = "half_open" # Testing if fixed

class CircuitBreaker:
    """Circuit breaker pattern for component reliability."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.last_attempt_time = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            if self.state == CircuitState.HALF_OPEN:
                # Success in half-open state means circuit is fixed
                self._reset()
            elif self.state == CircuitState.CLOSED:
                # Success in closed state resets failure count
                self.failure_count = 0

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Record a failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None

class ComponentProxy:
    """Proxy for component with circuit breaker protection."""

    def __init__(self, component_name: str, failure_threshold: int = 3):
        self.component_name = component_name
        self.circuit_breaker = CircuitBreaker(failure_threshold=failure_threshold)
        self.component = None  # Will be set later

    def call_method(self, method_name: str, *args, **kwargs):
        """Call component method with circuit breaker protection."""
        def method_call():
            if not self.component:
                raise Exception(f"Component {self.component_name} not initialized")

            method = getattr(self.component, method_name, None)
            if not method:
                raise Exception(f"Method {method_name} not found in {self.component_name}")

            return method(*args, **kwargs)

        return self.circuit_breaker.call(method_call)
```

### Retry Pattern with Exponential Backoff

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed
                        break

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    total_delay = delay + jitter

                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s...")
                    time.sleep(total_delay)

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator

class RobustComponentInterface:
    """Component interface with robust communication."""

    def __init__(self):
        self.communication_timeout = 10.0
        self.max_retries = 3

    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def call_remote_service(self, service_name: str, request_data: Dict) -> Dict:
        """Call remote service with retry logic."""
        # This would interface with actual ROS 2 services
        # Implementation would handle timeouts and retries
        pass

    def execute_with_fallback(self, primary_func: Callable, fallback_func: Callable, *args, **kwargs):
        """Execute primary function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            print(f"Primary function failed: {e}. Using fallback...")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise e  # Raise original error
```

## Real-Time Performance Patterns

### Producer-Consumer with Bounded Buffers

```python
import queue
import threading
import time
from typing import Any, Callable

class BoundedBuffer:
    """Thread-safe bounded buffer for real-time data processing."""

    def __init__(self, size: int, drop_policy: str = 'oldest'):
        self.buffer = queue.Queue(maxsize=size)
        self.drop_policy = drop_policy  # 'oldest', 'newest', 'block'
        self.lock = threading.Lock()

    def put(self, item: Any, block: bool = True, timeout: float = None) -> bool:
        """Put item in buffer with specified policy."""
        if self.buffer.full():
            if self.drop_policy == 'oldest':
                try:
                    # Remove oldest item to make space
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            elif self.drop_policy == 'newest':
                # Don't add new item (drop it)
                return False
            elif self.drop_policy == 'block':
                # Wait for space (default behavior)
                pass

        try:
            self.buffer.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, block: bool = True, timeout: float = None) -> Any:
        """Get item from buffer."""
        return self.buffer.get(block=block, timeout=timeout)

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.buffer.full()

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()

class RealTimePipeline:
    """Real-time processing pipeline with bounded buffers."""

    def __init__(self, buffer_size: int = 10):
        # Bounded buffers for each processing stage
        self.sensor_buffer = BoundedBuffer(buffer_size, drop_policy='oldest')
        self.processing_buffer = BoundedBuffer(buffer_size, drop_policy='oldest')
        self.action_buffer = BoundedBuffer(buffer_size, drop_policy='oldest')

        # Processing threads
        self.sensor_thread = None
        self.processing_thread = None
        self.action_thread = None

        self.running = False

    def start_pipeline(self):
        """Start all processing threads."""
        self.running = True

        # Start sensor acquisition thread
        self.sensor_thread = threading.Thread(target=self.sensor_acquisition_loop, daemon=True)
        self.sensor_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Start action execution thread
        self.action_thread = threading.Thread(target=self.action_execution_loop, daemon=True)
        self.action_thread.start()

    def sensor_acquisition_loop(self):
        """Acquire sensor data and put in buffer."""
        while self.running:
            # Simulate sensor data acquisition
            sensor_data = self.acquire_sensor_data()

            if sensor_data:
                success = self.sensor_buffer.put(sensor_data, timeout=0.1)
                if not success:
                    print("Sensor buffer full, dropping data")

            # Maintain real-time rate (e.g., 30Hz for camera)
            time.sleep(1.0 / 30.0)

    def processing_loop(self):
        """Process sensor data and generate actions."""
        while self.running:
            try:
                # Get sensor data
                sensor_data = self.sensor_buffer.get(timeout=0.1)

                # Process data
                processed_data = self.process_sensor_data(sensor_data)

                # Put in action buffer
                success = self.action_buffer.put(processed_data, timeout=0.1)
                if not success:
                    print("Action buffer full, dropping processed data")

            except queue.Empty:
                continue  # No data available, continue loop

    def action_execution_loop(self):
        """Execute actions from processing."""
        while self.running:
            try:
                # Get action data
                action_data = self.action_buffer.get(timeout=0.1)

                # Execute action
                self.execute_action(action_data)

            except queue.Empty:
                continue  # No actions available, continue loop

    def acquire_sensor_data(self) -> Any:
        """Acquire sensor data (implementation specific)."""
        # This would interface with actual sensors
        pass

    def process_sensor_data(self, data: Any) -> Any:
        """Process sensor data (implementation specific)."""
        # This would run perception algorithms
        pass

    def execute_action(self, action: Any):
        """Execute action (implementation specific)."""
        # This would send commands to robot
        pass
```

## Exercises

<Exercise title="Integration Pattern Implementation" difficulty="advanced" estimatedTime="120 min">

Implement a complete integration system using multiple design patterns:
1. Use Observer pattern for sensor data distribution
2. Apply Strategy pattern for different control modes
3. Implement Command pattern for action execution
4. Add Circuit Breaker for component reliability

**Requirements:**
- Observer pattern for sensor integration
- Strategy pattern for control algorithms
- Command pattern for action execution
- Circuit breaker for fault tolerance
- Unit tests for each pattern

<Hint>
Structure your implementation with clear separation:
```python
class IntegrationSystem:
    def __init__(self):
        self.observer_manager = SensorSubject()
        self.strategy_context = ControlContext()
        self.command_invoker = CommandInvoker()
        self.circuit_breaker = CircuitBreaker()
```
</Hint>

</Exercise>

<Exercise title="Real-Time Pipeline" difficulty="advanced" estimatedTime="150 min">

Create a real-time processing pipeline that:
1. Uses bounded buffers to manage data flow
2. Implements producer-consumer pattern for real-time processing
3. Maintains timing constraints for real-time performance
4. Handles buffer overflow scenarios gracefully

**Requirements:**
- Bounded buffer implementation
- Real-time processing threads
- Timing constraint enforcement
- Overflow handling mechanisms

</Exercise>

## Summary

Key concepts covered:

- ✅ Observer pattern for sensor integration
- ✅ Strategy pattern for control algorithms
- ✅ Command pattern for action execution
- ✅ Service-based communication patterns
- ✅ Fault-tolerance patterns (circuit breaker, retry)
- ✅ Real-time performance patterns
- ✅ Producer-consumer with bounded buffers

## Next Steps

Continue to [Testing Strategies](/module-5/week-12/testing-strategies) to learn about comprehensive testing strategies for integrated humanoid robotics systems.