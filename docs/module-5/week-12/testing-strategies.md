---
sidebar_position: 3
title: Testing Strategies
description: Comprehensive testing strategies for integrated robotics systems
---

# Testing Strategies

This lesson covers comprehensive testing strategies for integrated humanoid robotics systems, including unit testing, integration testing, simulation testing, and real-world validation.

<LearningObjectives
  objectives={[
    "Design comprehensive testing strategies for robotics systems",
    "Implement unit tests for individual components",
    "Create integration tests for component interactions",
    "Develop simulation-based testing frameworks",
    "Validate system behavior in real-world scenarios"
  ]}
/>

## Testing Philosophy for Robotics

### Testing Pyramid for Robotics Systems

<ArchitectureDiagram title="Robotics Testing Pyramid">
{`
┌─────────────────────────────────────────┐
│           REAL-WORLD VALIDATION         │
│            (System Testing)             │
│  • Field testing                        │
│  • User studies                         │
│  • Performance validation               │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            SIMULATION TESTING           │
│           (Integration Testing)         │
│  • Gazebo/Isaac Sim integration        │
│  • Multi-robot scenarios               │
│  • Complex task validation             │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│           COMPONENT TESTING             │
│            (Integration Testing)        │
│  • ROS 2 node integration              │
│  • Service/client testing              │
│  • Action execution validation         │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            UNIT TESTING                 │
│          (Component Testing)            │
│  • Individual function testing         │
│  • Algorithm validation                │
│  • Data structure verification         │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│           MOCK TESTING                  │
│        (Isolation Testing)              │
│  • Interface mocking                   │
│  • Dependency injection                │
│  • Behavior verification               │
└─────────────────────────────────────────┘
`}
</ArchitectureDiagram>

### Testing Categories in Robotics

| Category | Focus | Tools | Frequency |
|----------|-------|-------|-----------|
| **Unit** | Individual functions/components | pytest, unittest | Continuous |
| **Integration** | Component interactions | launch_testing | Daily |
| **System** | Complete system behavior | Gazebo, Isaac Sim | Weekly |
| **Regression** | Preventing feature breakage | CI/CD | Continuous |
| **Performance** | Real-time constraints | ros2 bag, monitoring | As needed |

## Unit Testing for Robotics Components

### Basic Unit Testing with pytest

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from your_robot_package.perception.vision_processor import VisionProcessor
from your_robot_package.control.navigation_controller import NavigationController
from your_robot_package.utils.geometry import transform_pose

class TestVisionProcessor:
    """Unit tests for vision processing component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vision_processor = VisionProcessor()

    def test_object_detection_empty_scene(self):
        """Test object detection with empty scene."""
        # Create empty image
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process image
        detections = self.vision_processor.detect_objects(empty_image)

        # Verify results
        assert len(detections) == 0

    def test_object_detection_single_object(self):
        """Test object detection with single object."""
        # Create test image with single object
        test_image = self.create_test_image_with_object()

        # Process image
        detections = self.vision_processor.detect_objects(test_image)

        # Verify results
        assert len(detections) == 1
        assert detections[0]['confidence'] > 0.8

    def test_object_detection_multiple_objects(self):
        """Test object detection with multiple objects."""
        # Create test image with multiple objects
        test_image = self.create_test_image_with_multiple_objects()

        # Process image
        detections = self.vision_processor.detect_objects(test_image)

        # Verify results
        assert len(detections) >= 2
        for detection in detections:
            assert 0 <= detection['confidence'] <= 1.0

    def create_test_image_with_object(self):
        """Create test image with a colored rectangle."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a red rectangle in the center
        cv2.rectangle(image, (300, 200), (340, 240), (0, 0, 255), -1)
        return image

    def create_test_image_with_multiple_objects(self):
        """Create test image with multiple colored rectangles."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add red rectangle
        cv2.rectangle(image, (100, 100), (140, 140), (0, 0, 255), -1)
        # Add blue rectangle
        cv2.rectangle(image, (200, 200), (240, 240), (255, 0, 0), -1)
        return image

class TestNavigationController:
    """Unit tests for navigation controller."""

    def setup_method(self):
        """Set up test fixtures."""
        self.controller = NavigationController()

    def test_compute_velocity_stopped(self):
        """Test velocity computation when stopped."""
        current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        target_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

        cmd = self.controller.compute_velocity(current_pose, target_pose)

        assert cmd['linear'] == 0.0
        assert cmd['angular'] == 0.0

    def test_compute_velocity_forward(self):
        """Test velocity computation for forward movement."""
        current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        target_pose = {'x': 1.0, 'y': 0.0, 'theta': 0.0}

        cmd = self.controller.compute_velocity(current_pose, target_pose)

        assert cmd['linear'] > 0.0
        assert abs(cmd['angular']) < 0.1  # Should be mostly straight

    def test_compute_velocity_rotation(self):
        """Test velocity computation for rotation."""
        current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        target_pose = {'x': 0.0, 'y': 0.0, 'theta': np.pi/2}  # 90 degrees

        cmd = self.controller.compute_velocity(current_pose, target_pose)

        assert abs(cmd['angular']) > 0.1  # Should rotate
        assert abs(cmd['linear']) < 0.1   # Should not move forward

    def test_obstacle_avoidance(self):
        """Test obstacle avoidance behavior."""
        current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        target_pose = {'x': 2.0, 'y': 0.0, 'theta': 0.0}

        # Simulate obstacle directly in front
        obstacles = [{'distance': 0.5, 'angle': 0.0}]  # 50cm ahead

        cmd = self.controller.compute_velocity_with_obstacles(
            current_pose, target_pose, obstacles
        )

        assert cmd['linear'] < 0.1  # Should slow down near obstacle
```

### Mock-Based Testing

```python
import unittest
from unittest.mock import Mock, patch, MagicMock
from your_robot_package.communication.message_handler import MessageHandler
from your_robot_package.safety.safety_validator import SafetyValidator

class TestMessageHandlerWithMocks:
    """Test message handler using mocks for dependencies."""

    def setup_method(self):
        """Set up test fixtures with mocks."""
        self.mock_safety_validator = Mock(spec=SafetyValidator)
        self.mock_logger = Mock()

        self.message_handler = MessageHandler(
            safety_validator=self.mock_safety_validator,
            logger=self.mock_logger
        )

    def test_handle_safe_command(self):
        """Test handling of safe command."""
        # Setup mock return value
        self.mock_safety_validator.validate_command.return_value = True

        # Test command
        test_command = {
            'type': 'move',
            'linear': 0.5,
            'angular': 0.0
        }

        # Execute
        result = self.message_handler.handle_command(test_command)

        # Verify
        self.mock_safety_validator.validate_command.assert_called_once_with(test_command)
        assert result is True
        self.mock_logger.info.assert_called()

    def test_handle_unsafe_command(self):
        """Test handling of unsafe command."""
        # Setup mock return value
        self.mock_safety_validator.validate_command.return_value = False

        # Test command
        test_command = {
            'type': 'move',
            'linear': 2.0,  # Too fast
            'angular': 0.0
        }

        # Execute
        result = self.message_handler.handle_command(test_command)

        # Verify
        self.mock_safety_validator.validate_command.assert_called_once_with(test_command)
        assert result is False
        self.mock_logger.warning.assert_called()

    def test_handle_command_with_exception(self):
        """Test handling when safety validator raises exception."""
        # Setup mock to raise exception
        self.mock_safety_validator.validate_command.side_effect = Exception("Validation failed")

        # Test command
        test_command = {
            'type': 'move',
            'linear': 0.5,
            'angular': 0.0
        }

        # Execute and verify exception handling
        with pytest.raises(Exception):
            self.message_handler.handle_command(test_command)

        self.mock_logger.error.assert_called()

@patch('your_robot_package.perception.camera_interface.CameraInterface')
class TestPerceptionWithPatches:
    """Test perception component with patched dependencies."""

    def test_process_camera_image(self, mock_camera_class):
        """Test camera image processing."""
        # Setup mock camera
        mock_camera = Mock()
        mock_camera_class.return_value = mock_camera
        mock_camera.capture.return_value = np.zeros((480, 640, 3))

        # Create and test perception component
        from your_robot_package.perception.perception_node import PerceptionNode
        perception_node = PerceptionNode()

        # Process image
        result = perception_node.process_current_image()

        # Verify camera was used
        mock_camera.capture.assert_called_once()
        assert result is not None
```

## Integration Testing with ROS 2

### ROS 2 Launch Testing

```python
import unittest
import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
import pytest
from launch_testing_ros import WaitForTopics
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import rclpy
from rclpy.node import Node

@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate launch description for integration test."""
    # Launch the system under test
    system_node = launch_ros.actions.Node(
        package='your_robot_package',
        executable='system_integration_node',
        name='test_system',
        parameters=[{'use_sim_time': False}]
    )

    return launch.LaunchDescription([
        system_node,
        launch_testing.actions.ReadyToTest()
    ])

class TestSystemIntegration:
    """Integration tests for the complete system."""

    def test_system_startup(self, launch_service, proc_info, proc_output):
        """Test that system starts up correctly."""
        # Wait for system to start
        with WaitForTopics([('system_status', String)], timeout=10.0):
            pass  # System is ready when topic is available

        # Check that no errors occurred during startup
        proc_info.assertWaitForShutdown(process='test_system', timeout=5.0)

    def test_sensor_data_flow(self, launch_service, proc_info, proc_output):
        """Test that sensor data flows through the system."""
        rclpy.init()
        try:
            node = rclpy.create_node('test_sensor_flow')

            # Publisher to simulate sensor data
            laser_pub = node.create_publisher(LaserScan, 'scan', 10)

            # Subscriber to check processed data
            processed_data = []
            def data_callback(msg):
                processed_data.append(msg)

            result_sub = node.create_subscription(
                Twist, 'cmd_vel', data_callback, 10
            )

            # Publish test laser scan
            test_scan = LaserScan()
            test_scan.ranges = [1.0] * 360  # 360 range readings
            laser_pub.publish(test_scan)

            # Wait for processing
            start_time = time.time()
            while len(processed_data) == 0 and time.time() - start_time < 5.0:
                rclpy.spin_once(node, timeout_sec=0.1)

            assert len(processed_data) > 0, "No processed data received"

        finally:
            rclpy.shutdown()

    def test_command_execution(self, launch_service, proc_info, proc_output):
        """Test that commands are executed correctly."""
        rclpy.init()
        try:
            node = rclpy.create_node('test_command_execution')

            # Publisher for commands
            cmd_pub = node.create_publisher(Twist, 'cmd_vel', 10)

            # Publisher for system commands
            system_cmd_pub = node.create_publisher(String, 'system_command', 10)

            # Send test command
            test_cmd = Twist()
            test_cmd.linear.x = 0.5
            test_cmd.angular.z = 0.0
            cmd_pub.publish(test_cmd)

            # Verify system response
            # This would check for expected system behavior
            # Implementation depends on specific system behavior

        finally:
            rclpy.shutdown()

def test_service_integration():
    """Test ROS 2 service integration."""
    rclpy.init()
    try:
        node = rclpy.create_node('test_service_client')

        # Create client for system service
        client = node.create_client(String, 'system_status')

        # Wait for service
        if client.wait_for_service(timeout_sec=5.0):
            # Call service
            request = String.Request()
            future = client.call_async(request)

            # Wait for response
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

            response = future.result()
            assert response is not None
        else:
            assert False, "Service not available"

    finally:
        rclpy.shutdown()
```

### Component Interaction Testing

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from your_robot_package.components.perception_component import PerceptionComponent
from your_robot_package.components.planning_component import PlanningComponent
from your_robot_package.components.control_component import ControlComponent

class ComponentIntegrationTestNode(Node):
    """Node for testing component interactions."""

    def __init__(self):
        super().__init__('component_integration_test')

        # Create components
        self.perception = PerceptionComponent(self)
        self.planning = PlanningComponent(self)
        self.control = ControlComponent(self)

        # Test publishers and subscribers
        self.test_publisher = self.create_publisher(LaserScan, 'test_scan', 10)
        self.result_subscriber = self.create_subscription(
            Twist, 'cmd_vel', self.result_callback, 10
        )

        self.test_results = []
        self.test_active = False

    def result_callback(self, msg):
        """Handle result from component chain."""
        if self.test_active:
            self.test_results.append(msg)

    def test_perception_planning_control_chain(self):
        """Test the complete chain: perception -> planning -> control."""
        self.test_active = True
        self.test_results.clear()

        # Create test sensor data
        test_scan = LaserScan()
        test_scan.ranges = [2.0] * 360  # Clear path
        test_scan.angle_increment = 0.01745329251  # 1 degree
        test_scan.range_min = 0.1
        test_scan.range_max = 10.0

        # Publish test data (this should trigger the chain)
        self.test_publisher.publish(test_scan)

        # Wait for processing
        start_time = time.time()
        while len(self.test_results) == 0 and time.time() - start_time < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Verify results
        assert len(self.test_results) > 0, "No results from component chain"

        # Check that the result makes sense
        cmd = self.test_results[0]
        # Verify command is reasonable (not extreme values)
        assert abs(cmd.linear.x) <= 1.0, f"Invalid linear velocity: {cmd.linear.x}"
        assert abs(cmd.angular.z) <= 1.0, f"Invalid angular velocity: {cmd.angular.z}"

        self.test_active = False

    def test_component_error_propagation(self):
        """Test how errors propagate between components."""
        # This would test error handling in the chain
        pass

def run_component_integration_tests():
    """Run component integration tests."""
    rclpy.init()
    try:
        test_node = ComponentIntegrationTestNode()

        # Run tests
        test_node.test_perception_planning_control_chain()
        test_node.test_component_error_propagation()

        print("All component integration tests passed!")

    finally:
        rclpy.shutdown()
```

## Simulation-Based Testing

### Gazebo Integration Testing

```python
import unittest
import subprocess
import time
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros
from tf2_geometry_msgs import PointStamped

class GazeboIntegrationTest:
    """Tests for system behavior in Gazebo simulation."""

    def __init__(self):
        rospy.init_node('gazebo_integration_test')

        # Publishers for commanding the robot
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers for monitoring robot state
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # TF buffer for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_pose = None
        self.laser_data = None
        self.start_time = rospy.Time.now()

    def odom_callback(self, msg):
        """Update current pose from odometry."""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Update laser data."""
        self.laser_data = msg

    def test_navigation_to_waypoint(self):
        """Test navigation to a specific waypoint."""
        target_pose = Pose()
        target_pose.position.x = 2.0
        target_pose.position.y = 2.0

        # Simple navigation - move toward target
        start_pose = self.current_pose
        if start_pose is None:
            rospy.logerr("No initial pose available")
            return False

        # Calculate time limit (e.g., 30 seconds to reach target)
        timeout = rospy.Time.now() + rospy.Duration(30.0)

        while not rospy.is_shutdown():
            if self.current_pose is None:
                continue

            # Calculate distance to target
            dx = target_pose.position.x - self.current_pose.position.x
            dy = target_pose.position.y - self.current_pose.position.y
            distance = (dx**2 + dy**2)**0.5

            if distance < 0.5:  # Within 50cm of target
                rospy.loginfo(f"Reached target! Distance: {distance:.2f}")
                return True

            if rospy.Time.now() > timeout:
                rospy.logerr("Navigation timeout")
                return False

            # Send navigation command
            cmd = Twist()
            cmd.linear.x = min(0.5, distance * 0.5)  # Proportional control
            cmd.angular.z = 2.0 * math.atan2(dy, dx)  # Heading control

            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)

        return False

    def test_obstacle_avoidance(self):
        """Test obstacle avoidance behavior."""
        # Move forward until obstacle is detected
        timeout = rospy.Time.now() + rospy.Duration(10.0)

        while not rospy.is_shutdown() and rospy.Time.now() < timeout:
            if self.laser_data is None:
                continue

            # Check for obstacles in front (first 30 degrees)
            front_ranges = self.laser_data.ranges[:15] + self.laser_data.ranges[-15:]
            min_distance = min([r for r in front_ranges if 0 < r < float('inf')], default=float('inf'))

            if min_distance < 0.8:  # Obstacle within 80cm
                # Stop and turn
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn right
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(2.0)  # Turn for 2 seconds
                break

            # Move forward
            cmd = Twist()
            cmd.linear.x = 0.3
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)

        return True

    def test_sensor_fusion(self):
        """Test integration of multiple sensors."""
        # Wait for all sensor data
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while (self.current_pose is None or self.laser_data is None) and rospy.Time.now() < timeout:
            rospy.sleep(0.1)

        if self.current_pose is None or self.laser_data is None:
            rospy.logerr("Timeout waiting for sensor data")
            return False

        # Verify data is reasonable
        assert self.current_pose.position.x is not None
        assert len(self.laser_data.ranges) > 0
        assert min(self.laser_data.ranges) >= self.laser_data.range_min

        return True

def run_gazebo_tests():
    """Run all Gazebo integration tests."""
    test_suite = GazeboIntegrationTest()

    tests = [
        ("Sensor Fusion", test_suite.test_sensor_fusion),
        ("Obstacle Avoidance", test_suite.test_obstacle_avoidance),
        ("Navigation", test_suite.test_navigation_to_waypoint),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            print(f"{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: ERROR - {e}")

    # Print summary
    print("\nTest Results Summary:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    return all(results.values())
```

### Isaac Sim Testing Framework

```python
import omni
import carb
import omni.kit.test
from pxr import Gf, UsdGeom
import numpy as np

class IsaacSimIntegrationTest(omni.kit.test.AsyncTestCase):
    """Test case for Isaac Sim integration."""

    async def setUp(self):
        """Set up test environment."""
        await omni.usd.get_context().new_stage_async()
        self.stage = omni.usd.get_context().get_stage()

        # Load humanoid robot
        self._load_humanoid_robot()

        # Start simulation
        self._start_simulation()

    async def tearDown(self):
        """Clean up after tests."""
        self._stop_simulation()
        await omni.usd.get_context().new_stage_async()

    def _load_humanoid_robot(self):
        """Load humanoid robot into the scene."""
        # This would load your humanoid robot model
        # Implementation depends on your specific robot asset
        pass

    def _start_simulation(self):
        """Start Isaac Sim physics simulation."""
        omni.kit.commands.execute("ChangeStagePlayback", value=True)

    def _stop_simulation(self):
        """Stop Isaac Sim physics simulation."""
        omni.kit.commands.execute("ChangeStagePlayback", value=False)

    async def test_humanoid_walking(self):
        """Test humanoid walking behavior."""
        # Set initial joint positions for walking
        initial_joints = self._get_joint_positions()

        # Apply walking gait (simplified)
        walking_commands = self._generate_walking_commands()

        # Run simulation for walking test
        for command in walking_commands:
            self._apply_joint_commands(command)
            await self._simulate_frame()

        # Verify robot moved forward
        final_joints = self._get_joint_positions()
        displacement = self._calculate_displacement(initial_joints, final_joints)

        self.assertGreater(displacement, 0.1, "Robot should have moved forward")

    async def test_sensor_integration(self):
        """Test sensor integration in Isaac Sim."""
        # Verify sensors are publishing data
        sensor_data = await self._get_sensor_data()

        # Check that sensor data is being generated
        self.assertIsNotNone(sensor_data['camera'])
        self.assertIsNotNone(sensor_data['lidar'])
        self.assertIsNotNone(sensor_data['imu'])

    def _get_joint_positions(self):
        """Get current joint positions."""
        # Implementation would interface with Isaac Sim physics
        pass

    def _apply_joint_commands(self, commands):
        """Apply joint commands to robot."""
        # Implementation would send commands to Isaac Sim articulation
        pass

    async def _simulate_frame(self):
        """Simulate one physics frame."""
        await omni.kit.app.get_app().next_update_async()

    def _generate_walking_commands(self):
        """Generate walking gait commands."""
        # Generate simple walking pattern
        commands = []
        for i in range(100):  # 100 simulation steps
            # Simple oscillating pattern for walking
            t = i * 0.01  # Time step
            left_leg = np.sin(t * 2) * 0.1
            right_leg = np.sin(t * 2 + np.pi) * 0.1
            commands.append({
                'left_hip': left_leg,
                'right_hip': right_leg,
                # Add more joints as needed
            })
        return commands

    async def _get_sensor_data(self):
        """Get sensor data from Isaac Sim."""
        # Implementation would interface with Isaac Sim sensors
        return {
            'camera': np.random.rand(480, 640, 3),  # Simulated image
            'lidar': np.random.rand(1080),          # Simulated LIDAR
            'imu': {'linear_acc': [0, 0, 9.81], 'angular_vel': [0, 0, 0]}  # Simulated IMU
        }
```

## Performance and Stress Testing

### Real-Time Performance Testing

```python
import time
import threading
import statistics
from collections import deque
import psutil
import GPUtil

class PerformanceTester:
    """Test real-time performance of robotics system."""

    def __init__(self):
        self.metrics = {
            'latency': deque(maxlen=1000),
            'throughput': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000)
        }
        self.test_results = {}
        self.is_testing = False

    def measure_latency(self, func, *args, **kwargs):
        """Measure function execution latency."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.metrics['latency'].append(latency)

        return result, latency

    def test_perception_pipeline(self, test_duration=30.0):
        """Test perception pipeline performance."""
        start_time = time.time()
        frame_count = 0
        latencies = []

        while time.time() - start_time < test_duration:
            # Simulate perception processing
            _, latency = self.measure_latency(self.process_frame)
            latencies.append(latency)
            frame_count += 1

            # Monitor system resources
            self._monitor_resources()

        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        fps = frame_count / test_duration

        self.test_results['perception'] = {
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'fps': fps,
            'frame_count': frame_count
        }

        print(f"Perception Performance: {fps:.2f} FPS, "
              f"Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")

    def process_frame(self):
        """Simulate frame processing."""
        # Simulate processing time
        time.sleep(0.01)  # 10ms processing time
        return "processed_data"

    def test_navigation_performance(self, test_duration=60.0):
        """Test navigation system performance."""
        start_time = time.time()
        path_count = 0
        latencies = []

        while time.time() - start_time < test_duration:
            # Simulate path planning
            _, latency = self.measure_latency(self.plan_path)
            latencies.append(latency)
            path_count += 1

            # Monitor resources
            self._monitor_resources()

            time.sleep(0.1)  # Plan path every 100ms

        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        planning_rate = path_count / test_duration

        self.test_results['navigation'] = {
            'avg_planning_time_ms': avg_latency,
            'planning_rate_hz': planning_rate,
            'paths_planned': path_count
        }

        print(f"Navigation Performance: {planning_rate:.2f} Hz, "
              f"Avg: {avg_latency:.2f}ms")

    def plan_path(self):
        """Simulate path planning."""
        # Simulate planning time
        time.sleep(0.05)  # 50ms planning time
        return [(0, 0), (1, 1), (2, 2)]  # Simple path

    def _monitor_resources(self):
        """Monitor system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.metrics['cpu_usage'].append(cpu_percent)

        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        self.metrics['memory_usage'].append(memory_percent)

        # GPU usage (if available)
        try:
            gpu_list = GPUtil.getGPUs()
            if gpu_list:
                gpu_percent = gpu_list[0].load * 100
                self.metrics['gpu_usage'].append(gpu_percent)
            else:
                self.metrics['gpu_usage'].append(0)
        except:
            self.metrics['gpu_usage'].append(0)

    def run_stress_test(self, test_duration=120.0):
        """Run comprehensive stress test."""
        print(f"Starting stress test for {test_duration} seconds...")

        # Run multiple tests in parallel
        threads = []

        # Perception stress test
        perception_thread = threading.Thread(
            target=self.test_perception_pipeline,
            args=(test_duration,)
        )
        threads.append(perception_thread)

        # Navigation stress test
        navigation_thread = threading.Thread(
            target=self.test_navigation_performance,
            args=(test_duration,)
        )
        threads.append(navigation_thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Generate performance report
        self._generate_performance_report()

    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'tests': self.test_results,
            'system_metrics': {
                'avg_cpu': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'avg_memory': statistics.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'avg_gpu': statistics.mean(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0,
                'max_latency': max(self.metrics['latency']) if self.metrics['latency'] else 0,
                'avg_latency': statistics.mean(self.metrics['latency']) if self.metrics['latency'] else 0
            }
        }

        # Save report
        with open('/reports/performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("Performance report generated: /reports/performance_report.json")
        return report
```

### Load Testing Framework

```python
import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Callable
import time

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    duration: float  # Test duration in seconds
    concurrency: int  # Number of concurrent operations
    target_rate: float  # Target operations per second
    timeout: float  # Operation timeout
    ramp_up_time: float = 5.0  # Time to reach full load

class LoadTester:
    """Framework for load testing robotics components."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.start_time = None

    async def run_load_test(self, operation: Callable, *args, **kwargs):
        """Run load test with specified operation."""
        self.start_time = time.time()
        self.results = []

        # Calculate rate per worker
        rate_per_worker = self.config.target_rate / self.config.concurrency
        interval = 1.0 / rate_per_worker

        # Create tasks for each worker
        tasks = []
        for i in range(self.config.concurrency):
            task = asyncio.create_task(
                self._worker_operation(operation, interval, *args, **kwargs)
            )
            tasks.append(task)

        # Wait for duration
        await asyncio.sleep(self.config.duration)

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        return self._analyze_results()

    async def _worker_operation(self, operation: Callable, interval: float, *args, **kwargs):
        """Worker that executes operations at specified rate."""
        start_time = time.time()

        while time.time() - start_time < self.config.duration:
            operation_start = time.time()

            try:
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    self._execute_operation(operation, *args, **kwargs),
                    timeout=self.config.timeout
                )

                operation_time = time.time() - operation_start
                self.results.append({
                    'success': True,
                    'latency': operation_time,
                    'timestamp': time.time(),
                    'result': result
                })
            except asyncio.TimeoutError:
                self.results.append({
                    'success': False,
                    'latency': self.config.timeout,
                    'timestamp': time.time(),
                    'error': 'timeout'
                })
            except Exception as e:
                self.results.append({
                    'success': False,
                    'latency': time.time() - operation_start,
                    'timestamp': time.time(),
                    'error': str(e)
                })

            # Wait for next operation
            await asyncio.sleep(interval)

    async def _execute_operation(self, operation: Callable, *args, **kwargs):
        """Execute a single operation."""
        # This would typically call a ROS 2 service or action
        return operation(*args, **kwargs)

    def _analyze_results(self) -> Dict:
        """Analyze load test results."""
        if not self.results:
            return {'error': 'No results collected'}

        successful_ops = [r for r in self.results if r['success']]
        failed_ops = [r for r in self.results if not r['success']]

        total_ops = len(self.results)
        successful_ops_count = len(successful_ops)

        analysis = {
            'total_operations': total_ops,
            'successful_operations': successful_ops_count,
            'failed_operations': len(failed_ops),
            'success_rate': successful_ops_count / total_ops if total_ops > 0 else 0,
            'duration': time.time() - self.start_time,
            'throughput': total_ops / (time.time() - self.start_time) if self.start_time else 0
        }

        if successful_ops:
            latencies = [op['latency'] for op in successful_ops]
            analysis.update({
                'avg_latency': statistics.mean(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'p95_latency': self._percentile(latencies, 95),
                'p99_latency': self._percentile(latencies, 99)
            })

        return analysis

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

def run_robotics_load_tests():
    """Run comprehensive load tests for robotics system."""
    configs = [
        LoadTestConfig(
            duration=60.0,
            concurrency=10,
            target_rate=5.0,  # 5 operations per second
            timeout=2.0
        ),
        LoadTestConfig(
            duration=30.0,
            concurrency=20,
            target_rate=10.0,  # Higher load test
            timeout=3.0
        )
    ]

    load_tester = LoadTester(configs[0])

    # Example: Test service calls
    async def test_service_call():
        # This would be a real ROS 2 service call
        await asyncio.sleep(0.1)  # Simulate service call
        return "success"

    results = asyncio.run(load_tester.run_load_test(test_service_call))
    print("Load test results:", results)
```

## Exercises

<Exercise title="Comprehensive Unit Testing" difficulty="intermediate" estimatedTime="90 min">

Create comprehensive unit tests for your robotics components:
1. Write unit tests for perception algorithms
2. Test control algorithms with various inputs
3. Validate data structures and transformations
4. Implement mock-based testing for external dependencies

**Requirements:**
- Test coverage > 80% for critical components
- Mock external dependencies
- Parameterized tests for different scenarios
- Performance benchmarks included

<Hint>
Use pytest fixtures for common test setups:
```python
@pytest.fixture
def sample_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.mark.parametrize("input_angle,expected_output", [
    (0, 0), (np.pi/2, 1), (np.pi, 0)
])
def test_angle_conversion(input_angle, expected_output):
    # Test implementation
```
</Hint>

</Exercise>

<Exercise title="Integration Test Framework" difficulty="advanced" estimatedTime="180 min">

Build a complete integration test framework that:
1. Tests component interactions in simulation
2. Validates end-to-end system behavior
3. Includes performance and stress testing
4. Generates comprehensive test reports

**Requirements:**
- ROS 2 launch testing integration
- Gazebo/Isaac Sim test scenarios
- Performance benchmarking
- Automated test reporting
- CI/CD integration

</Exercise>

## Summary

Key concepts covered:

- ✅ Testing pyramid for robotics systems
- ✅ Unit testing with mocks and fixtures
- ✅ ROS 2 integration testing
- ✅ Simulation-based testing
- ✅ Performance and stress testing
- ✅ Load testing frameworks
- ✅ Test result analysis and reporting

## Next Steps

Complete the [Week 12 Exercises](/module-5/week-12/exercises) to practice implementing comprehensive testing strategies for humanoid robotics systems.