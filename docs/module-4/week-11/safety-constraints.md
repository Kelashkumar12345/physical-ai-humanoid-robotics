---
sidebar_position: 3
title: Safety Constraints
description: Advanced safety systems for VLA-powered robots
---

# Safety Constraints

This lesson covers advanced safety systems and constraint validation for Vision-Language-Action (VLA) powered robots, ensuring safe operation in dynamic environments.

<LearningObjectives
  objectives={[
    "Implement comprehensive safety constraint systems",
    "Validate VLA outputs against safety requirements",
    "Design fail-safe mechanisms for robot control",
    "Create safety monitoring and alert systems",
    "Implement graceful degradation strategies"
  ]}
/>

## Safety Architecture for VLA Systems

### Multi-Layer Safety Framework

<ArchitectureDiagram title="VLA Safety Framework">
{`
┌─────────────────┐    VLA Output    ┌─────────────────┐
│  VLA Model      │ ──────────────▶ │  Pre-Execution  │
│  (Vision+Lang)  │                 │  Validation     │
└─────────────────┘                 │                 │
                                    │ 1. Syntax check │
                                    │ 2. Feasibility  │
                                    │ 3. Safety check │
                                    └─────────────────┘
                                            │
                                            ▼
┌─────────────────┐              ┌─────────────────┐
│  Robot State    │ ───────────▶ │  Execution      │
│  Monitoring     │   Real-time  │  Safety Layer   │
│  (Sensors)      │              │                 │
└─────────────────┘              │ 1. Collision    │
                                 │ 2. Limit check  │
                                 │ 3. Emergency    │
                                 └─────────────────┘
                                         │
                                         ▼
┌─────────────────┐              ┌─────────────────┐
│  Emergency      │ ◀─────────── │  Post-Execution │
│  System         │   Alerts     │  Verification   │
└─────────────────┘              │                 │
                                 │ 1. Outcome      │
                                 │ 2. Log events   │
                                 │ 3. Update model │
                                 └─────────────────┘
`}
</ArchitectureDiagram>

### Safety Constraint Categories

| Category | Examples | Criticality |
|----------|----------|-------------|
| **Kinematic** | Joint limits, workspace bounds | High |
| **Dynamic** | Collision avoidance, speed limits | High |
| **Environmental** | Obstacle detection, terrain | Medium |
| **Operational** | Battery, temperature | Medium |
| **Social** | Human safety, privacy | High |

## Pre-Execution Validation

### VLA Output Validation

```python
import re
import json
from typing import Dict, List, Tuple, Any

class VLAOutputValidator:
    def __init__(self):
        self.action_patterns = {
            'navigation': [
                r'move_forward',
                r'move_backward',
                r'turn_left',
                r'turn_right',
                r'go_to_.*'
            ],
            'manipulation': [
                r'grasp_object',
                r'release_object',
                r'pick_up_.*'
            ],
            'communication': [
                r'speak_.*',
                r'listen',
                r'wait'
            ]
        }

    def validate_output(self, vla_output: str, robot_state: Dict) -> Tuple[bool, List[str], Dict]:
        """Validate VLA output against safety constraints."""
        errors = []
        warnings = []
        parsed_output = None

        # 1. Syntax validation
        is_syntax_valid, syntax_errors = self.validate_syntax(vla_output)
        if not is_syntax_valid:
            errors.extend(syntax_errors)
            return False, errors, {}

        # 2. Parse output
        try:
            parsed_output = self.parse_output(vla_output)
        except Exception as e:
            errors.append(f"Output parsing failed: {e}")
            return False, errors, {}

        # 3. Feasibility validation
        is_feasible, feasibility_errors = self.validate_feasibility(parsed_output, robot_state)
        if not is_feasible:
            errors.extend(feasibility_errors)

        # 4. Safety validation
        is_safe, safety_errors, safety_warnings = self.validate_safety(parsed_output, robot_state)
        if not is_safe:
            errors.extend(safety_errors)
        if safety_warnings:
            warnings.extend(safety_warnings)

        return len(errors) == 0, errors, {
            'parsed_output': parsed_output,
            'warnings': warnings,
            'is_safe': len(errors) == 0
        }

    def validate_syntax(self, output: str) -> Tuple[bool, List[str]]:
        """Validate output syntax."""
        errors = []

        # Check if it's valid JSON
        try:
            json.loads(output)
        except json.JSONDecodeError:
            # Check if it's a valid action sequence
            if not re.match(r'^\s*\[.*\]\s*$', output) and not self.is_valid_action_sequence(output):
                errors.append("Output is not valid JSON or action sequence")
                return False, errors

        return True, []

    def is_valid_action_sequence(self, output: str) -> bool:
        """Check if output is a valid action sequence."""
        # Simple validation - in practice, use more sophisticated parsing
        return 'move_' in output or 'turn_' in output or 'grasp' in output or 'speak' in output

    def parse_output(self, output: str) -> Dict:
        """Parse VLA output into structured format."""
        try:
            # Try JSON first
            return json.loads(output)
        except json.JSONDecodeError:
            # Try to parse as action sequence
            if output.strip().startswith('[') and output.strip().endswith(']'):
                # Remove brackets and split
                actions_str = output.strip()[1:-1]
                actions = [a.strip().strip('"\'') for a in actions_str.split(',')]
                return {'actions': [a for a in actions if a]}
            else:
                # Treat as single action
                return {'actions': [output.strip()]}

    def validate_feasibility(self, parsed_output: Dict, robot_state: Dict) -> Tuple[bool, List[str]]:
        """Validate if actions are feasible given robot capabilities."""
        errors = []

        if 'actions' not in parsed_output:
            errors.append("No actions found in output")
            return False, errors

        for action in parsed_output['actions']:
            # Check if action is supported by robot
            if not self.is_action_supported(action, robot_state):
                errors.append(f"Action not supported: {action}")

        return len(errors) == 0, errors

    def validate_safety(self, parsed_output: Dict, robot_state: Dict) -> Tuple[bool, List[str], List[str]]:
        """Validate safety constraints."""
        errors = []
        warnings = []

        for action in parsed_output.get('actions', []):
            action_errors, action_warnings = self.check_action_safety(action, robot_state)
            errors.extend(action_errors)
            warnings.extend(action_warnings)

        return len(errors) == 0, errors, warnings

    def is_action_supported(self, action: str, robot_state: Dict) -> bool:
        """Check if action is supported by robot."""
        # Check robot capabilities
        capabilities = robot_state.get('capabilities', [])

        if 'navigation' in action:
            return 'navigation' in capabilities
        elif 'manipulation' in action:
            return 'manipulation' in capabilities
        elif 'communication' in action:
            return 'communication' in capabilities

        return True  # Assume other actions are supported

    def check_action_safety(self, action: str, robot_state: Dict) -> Tuple[List[str], List[str]]:
        """Check safety for individual action."""
        errors = []
        warnings = []

        # Check battery for navigation actions
        if any(nav_word in action for nav_word in ['move', 'go', 'navigate']):
            battery = robot_state.get('battery_level', 100)
            if battery < 10:  # 10% threshold
                errors.append(f"Insufficient battery for navigation: {battery}%")

        # Check gripper state for manipulation
        if 'grasp' in action or 'pick' in action:
            gripper_state = robot_state.get('gripper_state', 'open')
            if gripper_state == 'closed':
                warnings.append("Gripper already closed, attempting grasp")

        return errors, warnings
```

### Constraint-Based Validation

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class ConstraintType(Enum):
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"
    ENVIRONMENTAL = "environmental"
    OPERATIONAL = "operational"

@dataclass
class SafetyConstraint:
    """Definition of a safety constraint."""
    name: str
    constraint_type: ConstraintType
    validator: Callable[[Any, Dict], Tuple[bool, str]]
    severity: str  # "error", "warning", "info"
    enabled: bool = True

class ConstraintValidator:
    def __init__(self):
        self.constraints = [
            SafetyConstraint(
                name="joint_limit",
                constraint_type=ConstraintType.KINEMATIC,
                validator=self.validate_joint_limits,
                severity="error"
            ),
            SafetyConstraint(
                name="collision_avoidance",
                constraint_type=ConstraintType.DYNAMIC,
                validator=self.validate_collision_avoidance,
                severity="error"
            ),
            SafetyConstraint(
                name="workspace_bounds",
                constraint_type=ConstraintType.ENVIRONMENTAL,
                validator=self.validate_workspace_bounds,
                severity="error"
            ),
            SafetyConstraint(
                name="battery_level",
                constraint_type=ConstraintType.OPERATIONAL,
                validator=self.validate_battery_level,
                severity="warning"
            )
        ]

    def validate_all(self, action: Dict, robot_state: Dict) -> Tuple[bool, List[Dict]]:
        """Validate action against all constraints."""
        results = []
        has_errors = False

        for constraint in self.constraints:
            if not constraint.enabled:
                continue

            is_valid, message = constraint.validator(action, robot_state)

            result = {
                'constraint': constraint.name,
                'type': constraint.constraint_type.value,
                'valid': is_valid,
                'message': message,
                'severity': constraint.severity
            }

            results.append(result)

            if not is_valid and constraint.severity == "error":
                has_errors = True

        return not has_errors, results

    def validate_joint_limits(self, action: Dict, robot_state: Dict) -> Tuple[bool, str]:
        """Validate joint position constraints."""
        current_joints = robot_state.get('joint_positions', {})
        target_joints = action.get('target_joints', {})

        for joint_name, target_pos in target_joints.items():
            limits = robot_state.get('joint_limits', {}).get(joint_name, {})
            current_pos = current_joints.get(joint_name, 0)

            # Check if target is within limits
            min_limit = limits.get('min', -float('inf'))
            max_limit = limits.get('max', float('inf'))

            if target_pos < min_limit or target_pos > max_limit:
                return False, f"Joint {joint_name} target {target_pos} exceeds limits [{min_limit}, {max_limit}]"

            # Check if movement is within velocity limits
            max_velocity = limits.get('max_velocity', float('inf'))
            if abs(target_pos - current_pos) > max_velocity:
                return False, f"Joint {joint_name} movement exceeds velocity limit"

        return True, "Joint limits satisfied"

    def validate_collision_avoidance(self, action: Dict, robot_state: Dict) -> Tuple[bool, str]:
        """Validate collision avoidance."""
        # Check if navigation path is collision-free
        if 'navigation' in action.get('type', ''):
            target = action.get('target_position')
            if target:
                path = robot_state.get('costmap', {}).get('path', [])
                for point in path:
                    if self.is_collision_at_point(point):
                        return False, f"Collision detected at path point {point}"

        return True, "Collision avoidance satisfied"

    def validate_workspace_bounds(self, action: Dict, robot_state: Dict) -> Tuple[bool, str]:
        """Validate workspace boundary constraints."""
        workspace = robot_state.get('workspace_bounds', {})
        if not workspace:
            return True, "No workspace bounds defined"

        target_pos = action.get('target_position')
        if target_pos:
            x, y = target_pos[0], target_pos[1]
            if (x < workspace.get('min_x', -float('inf')) or
                x > workspace.get('max_x', float('inf')) or
                y < workspace.get('min_y', -float('inf')) or
                y > workspace.get('max_y', float('inf'))):
                return False, f"Target position ({x}, {y}) outside workspace bounds"

        return True, "Workspace bounds satisfied"

    def validate_battery_level(self, action: Dict, robot_state: Dict) -> Tuple[bool, str]:
        """Validate battery level constraints."""
        battery_level = robot_state.get('battery_level', 100)

        # Different actions have different battery requirements
        action_type = action.get('type', '')
        required_battery = 5  # Default minimum

        if 'navigation' in action_type:
            required_battery = 15
        elif 'manipulation' in action_type:
            required_battery = 10

        if battery_level < required_battery:
            return False, f"Battery level {battery_level}% below required {required_battery}% for {action_type}"

        return True, "Battery level sufficient"
```

## Real-Time Safety Monitoring

### Execution-Time Safety Checks

```python
import threading
import time
from typing import Dict, Callable
import numpy as np

class RealTimeSafetyMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_thread = None
        self.safety_callbacks = []
        self.emergency_stop = False
        self.robot_state = {}
        self.last_safe_time = time.time()

    def start_monitoring(self, robot_state: Dict):
        """Start real-time safety monitoring."""
        self.robot_state = robot_state
        self.is_monitoring = True
        self.emergency_stop = False
        self.last_safe_time = time.time()

        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def monitor_loop(self):
        """Main monitoring loop running at 10Hz."""
        while self.is_monitoring:
            if self.emergency_stop:
                break

            current_state = self.get_robot_state()

            # Check all safety conditions
            is_safe, violations = self.check_safety_conditions(current_state)

            if not is_safe:
                self.handle_safety_violation(violations, current_state)

            time.sleep(0.1)  # 10Hz monitoring

    def check_safety_conditions(self, robot_state: Dict) -> Tuple[bool, List[str]]:
        """Check all real-time safety conditions."""
        violations = []

        # Check collision proximity
        if not self.check_collision_proximity(robot_state):
            violations.append("Obstacle too close")

        # Check joint limits
        if not self.check_joint_limits(robot_state):
            violations.append("Joint limit violation")

        # Check velocity limits
        if not self.check_velocity_limits(robot_state):
            violations.append("Velocity limit exceeded")

        # Check for human presence in danger zone
        if not self.check_human_safety(robot_state):
            violations.append("Human in danger zone")

        # Check battery level
        if not self.check_battery_level(robot_state):
            violations.append("Low battery")

        return len(violations) == 0, violations

    def check_collision_proximity(self, robot_state: Dict) -> bool:
        """Check if robot is too close to obstacles."""
        laser_data = robot_state.get('laser_scan', [])
        if not laser_data:
            return True

        # Check front 60 degrees for obstacles
        front_ranges = laser_data[:30] + laser_data[-30:]
        min_distance = min([r for r in front_ranges if 0 < r < float('inf')], default=float('inf'))

        # 0.5m safety threshold
        return min_distance > 0.5

    def check_joint_limits(self, robot_state: Dict) -> bool:
        """Check if joints are within limits."""
        current_positions = robot_state.get('joint_positions', {})
        limits = robot_state.get('joint_limits', {})

        for joint_name, pos in current_positions.items():
            joint_limits = limits.get(joint_name, {})
            min_limit = joint_limits.get('min', -float('inf'))
            max_limit = joint_limits.get('max', float('inf'))

            if pos < min_limit or pos > max_limit:
                return False

        return True

    def check_velocity_limits(self, robot_state: Dict) -> bool:
        """Check if velocities are within limits."""
        current_velocities = robot_state.get('joint_velocities', {})
        limits = robot_state.get('joint_limits', {})

        for joint_name, vel in current_velocities.items():
            max_vel = limits.get(joint_name, {}).get('max_velocity', float('inf'))
            if abs(vel) > max_vel:
                return False

        return True

    def check_human_safety(self, robot_state: Dict) -> bool:
        """Check for humans in danger zone."""
        # This would interface with human detection system
        human_detections = robot_state.get('human_detections', [])

        for detection in human_detections:
            distance = detection.get('distance', float('inf'))
            # 2m safety zone around robot
            if distance < 2.0:
                return False

        return True

    def check_battery_level(self, robot_state: Dict) -> bool:
        """Check battery level."""
        battery_level = robot_state.get('battery_level', 100)
        return battery_level > 5  # 5% minimum

    def handle_safety_violation(self, violations: List[str], robot_state: Dict):
        """Handle safety violations."""
        self.get_logger().error(f"Safety violations detected: {violations}")

        # Trigger emergency stop
        self.emergency_stop = True
        self.execute_emergency_stop()

        # Log violation
        self.log_safety_violation(violations, robot_state)

        # Notify monitoring system
        self.notify_violation(violations)

    def execute_emergency_stop(self):
        """Execute emergency stop procedure."""
        # This would send emergency stop command to robot
        self.get_logger().warn("EMERGENCY STOP ACTIVATED")
        # Implementation would send stop command to robot controller

    def add_safety_callback(self, callback: Callable):
        """Add callback for safety events."""
        self.safety_callbacks.append(callback)

    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
```

## Emergency Response Systems

### Emergency Stop Architecture

```python
import signal
import sys
from enum import Enum

class EmergencyLevel(Enum):
    WARNING = 1      # Potential issue, continue with caution
    CAUTION = 2      # Moderate risk, slow down/reassess
    EMERGENCY = 3    # Immediate danger, stop immediately
    CRITICAL = 4     # System failure, full shutdown

class EmergencyResponseSystem:
    def __init__(self):
        self.emergency_level = EmergencyLevel.WARNING
        self.response_actions = {
            EmergencyLevel.WARNING: self.handle_warning,
            EmergencyLevel.CAUTION: self.handle_caution,
            EmergencyLevel.EMERGENCY: self.handle_emergency,
            EmergencyLevel.CRITICAL: self.handle_critical
        }
        self.active_responses = []
        self.system_shutdown = False

    def trigger_emergency(self, level: EmergencyLevel, reason: str, source: str = "system"):
        """Trigger emergency response."""
        self.emergency_level = level
        self.get_logger().error(f"EMERGENCY {level.name}: {reason} (source: {source})")

        # Execute response for this level
        if level in self.response_actions:
            response = self.response_actions[level](reason, source)
            self.active_responses.append(response)

    def handle_warning(self, reason: str, source: str) -> Dict:
        """Handle warning level emergency."""
        response = {
            'action': 'reduce_speed',
            'reason': reason,
            'source': source,
            'timestamp': time.time()
        }

        # Reduce robot speed
        self.reduce_robot_speed(0.5)  # 50% speed
        self.log_event("WARNING", reason)

        return response

    def handle_caution(self, reason: str, source: str) -> Dict:
        """Handle caution level emergency."""
        response = {
            'action': 'pause_and_reassess',
            'reason': reason,
            'source': source,
            'timestamp': time.time()
        }

        # Stop robot temporarily
        self.stop_robot()
        self.log_event("CAUTION", reason)

        return response

    def handle_emergency(self, reason: str, source: str) -> Dict:
        """Handle emergency level situation."""
        response = {
            'action': 'immediate_stop',
            'reason': reason,
            'source': source,
            'timestamp': time.time()
        }

        # Immediate stop
        self.emergency_stop_robot()
        self.log_event("EMERGENCY", reason)

        return response

    def handle_critical(self, reason: str, source: str) -> Dict:
        """Handle critical system failure."""
        response = {
            'action': 'system_shutdown',
            'reason': reason,
            'source': source,
            'timestamp': time.time()
        }

        # Full system shutdown
        self.shutdown_robot_systems()
        self.system_shutdown = True
        self.log_event("CRITICAL", reason)

        return response

    def reduce_robot_speed(self, factor: float):
        """Reduce robot speed by factor."""
        # Implementation would reduce robot velocity commands
        pass

    def stop_robot(self):
        """Stop robot movement."""
        # Send stop command to robot
        pass

    def emergency_stop_robot(self):
        """Immediate emergency stop."""
        # Send emergency stop command
        pass

    def shutdown_robot_systems(self):
        """Shutdown all robot systems safely."""
        # Stop all motion
        self.emergency_stop_robot()

        # Disable actuators
        # Save current state
        # Log shutdown
        pass

    def log_event(self, level: str, message: str):
        """Log safety event."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        # Write to safety log file
        with open("/logs/safety_events.log", "a") as f:
            f.write(log_entry + "\n")
```

### Human Safety Systems

```python
class HumanSafetySystem:
    def __init__(self):
        self.human_detection_callback = None
        self.safe_zones = {}  # Defined safe areas
        self.emergency_stop_zones = {}  # Areas that trigger emergency stop
        self.personal_safety_bubble = 2.0  # meters

    def setup_safe_zones(self, zones_config: Dict):
        """Setup safe zones around the robot."""
        self.safe_zones = zones_config.get('safe_zones', {})
        self.emergency_stop_zones = zones_config.get('emergency_stop_zones', {})

    def monitor_human_proximity(self, detection_data: List[Dict]):
        """Monitor human proximity and trigger safety responses."""
        for detection in detection_data:
            person_id = detection.get('id')
            position = detection.get('position')  # (x, y, z)
            distance = detection.get('distance')

            if distance < self.personal_safety_bubble:
                # Too close - trigger emergency response
                self.trigger_human_safety_response(
                    person_id,
                    position,
                    distance,
                    "TOO_CLOSE"
                )
            elif self.is_in_emergency_zone(position):
                # In emergency zone - stop immediately
                self.trigger_human_safety_response(
                    person_id,
                    position,
                    distance,
                    "EMERGENCY_ZONE"
                )
            elif self.is_in_safe_zone(position):
                # In safe zone - continue with caution
                self.trigger_human_safety_response(
                    person_id,
                    position,
                    distance,
                    "SAFE_ZONE"
                )

    def is_in_emergency_zone(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is in emergency stop zone."""
        for zone in self.emergency_stop_zones.values():
            if self.is_point_in_zone(position, zone):
                return True
        return False

    def is_in_safe_zone(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is in safe zone."""
        for zone in self.safe_zones.values():
            if self.is_point_in_zone(position, zone):
                return True
        return False

    def is_point_in_zone(self, point: Tuple[float, float, float], zone: Dict) -> bool:
        """Check if point is within zone boundaries."""
        # For circular zone
        if zone.get('type') == 'circular':
            center = zone['center']  # (x, y)
            radius = zone['radius']
            distance = ((point[0] - center[0])**2 + (point[1] - center[1])**2)**0.5
            return distance <= radius

        # For rectangular zone
        elif zone.get('type') == 'rectangular':
            min_x, min_y = zone['min_corner']
            max_x, max_y = zone['max_corner']
            return (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)

        return False

    def trigger_human_safety_response(self, person_id: str, position: Tuple, distance: float, reason: str):
        """Trigger appropriate safety response for human detection."""
        if reason == "TOO_CLOSE":
            # Immediate stop and warning
            self.emergency_response_system.trigger_emergency(
                EmergencyLevel.EMERGENCY,
                f"Human too close: {distance:.2f}m at {position}",
                "human_safety"
            )
        elif reason == "EMERGENCY_ZONE":
            # Stop and wait
            self.emergency_response_system.trigger_emergency(
                EmergencyLevel.CAUTION,
                f"Human in emergency zone at {position}",
                "human_safety"
            )
        elif reason == "SAFE_ZONE":
            # Continue with awareness
            self.emergency_response_system.trigger_emergency(
                EmergencyLevel.WARNING,
                f"Human detected in safe zone at {position}",
                "human_safety"
            )
```

## ROS 2 Safety Integration

### Safety Monitor Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Publishers
        self.safety_status_pub = self.create_publisher(Bool, 'safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.safety_violation_pub = self.create_publisher(String, 'safety_violations', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )

        # Safety components
        self.constraint_validator = ConstraintValidator()
        self.real_time_monitor = RealTimeSafetyMonitor()
        self.emergency_system = EmergencyResponseSystem()
        self.human_safety = HumanSafetySystem()

        # Robot state
        self.robot_state = {
            'laser_scan': [],
            'joint_positions': {},
            'joint_velocities': {},
            'position': (0, 0, 0),
            'velocity': (0, 0, 0),
            'battery_level': 100,
            'safety_status': True
        }

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.1, self.check_safety)  # 10Hz

        self.get_logger().info('Safety Monitor Node initialized')

    def laser_callback(self, msg):
        """Handle laser scan data."""
        self.robot_state['laser_scan'] = list(msg.ranges)

    def joint_state_callback(self, msg):
        """Handle joint state data."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.robot_state['joint_positions'][name] = msg.position[i]
            if i < len(msg.velocity):
                self.robot_state['joint_velocities'][name] = msg.velocity[i]

    def odom_callback(self, msg):
        """Handle odometry data."""
        pos = msg.pose.pose.position
        self.robot_state['position'] = (pos.x, pos.y, pos.z)

        vel = msg.twist.twist
        self.robot_state['velocity'] = (vel.linear.x, vel.linear.y, vel.linear.z)

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands."""
        # Store last commanded velocity for safety checks
        self.robot_state['last_cmd_vel'] = (msg.linear.x, msg.angular.z)

    def check_safety(self):
        """Main safety checking function."""
        try:
            # Validate current state
            is_safe, violations = self.real_time_monitor.check_safety_conditions(self.robot_state)

            # Update safety status
            safety_msg = Bool()
            safety_msg.data = is_safe
            self.safety_status_pub.publish(safety_msg)

            if not is_safe:
                # Log violations
                violation_msg = String()
                violation_msg.data = "; ".join(violations)
                self.safety_violation_pub.publish(violation_msg)

                # Trigger emergency response
                self.emergency_system.trigger_emergency(
                    EmergencyLevel.EMERGENCY,
                    f"Safety violations: {', '.join(violations)}",
                    "safety_monitor"
                )

                # Send emergency stop
                stop_msg = Bool()
                stop_msg.data = True
                self.emergency_stop_pub.publish(stop_msg)

        except Exception as e:
            self.get_logger().error(f'Safety check error: {e}')

    def validate_action_safety(self, action: Dict) -> Tuple[bool, List[str]]:
        """Validate an action against safety constraints."""
        return self.constraint_validator.validate_all(action, self.robot_state)
```

## Exercises

<Exercise title="Comprehensive Safety Validator" difficulty="advanced" estimatedTime="90 min">

Create a complete safety validation system that:
1. Validates VLA outputs against multiple constraint types
2. Implements real-time monitoring of robot state
3. Provides different levels of safety responses
4. Includes human safety detection and response

**Requirements:**
- Pre-execution validation pipeline
- Real-time safety monitoring
- Multi-level emergency responses
- Human safety system integration
- Unit tests for safety logic

<Hint>
Structure your validation in layers:
1. Syntax validation
2. Feasibility validation
3. Safety validation
4. Runtime monitoring
</Hint>

</Exercise>

<Exercise title="Safety-Critical Action Execution" difficulty="advanced" estimatedTime="120 min">

Build a safety-critical action execution system that:
1. Intercepts all robot commands for safety validation
2. Implements safety-aware command queuing
3. Provides graceful degradation when safety issues occur
4. Maintains safety logs for analysis

**Requirements:**
- Command interception and validation
- Safety-aware queuing system
- Degradation strategies
- Comprehensive logging
- Recovery from safety violations

</Exercise>

## Summary

Key concepts covered:

- ✅ Multi-layer safety framework for VLA systems
- ✅ Pre-execution validation of VLA outputs
- ✅ Real-time safety monitoring
- ✅ Emergency response systems
- ✅ Human safety protocols
- ✅ ROS 2 safety integration
- ✅ Constraint-based validation

## Next Steps

Complete the [Week 11 Exercises](/module-4/week-11/exercises) to practice implementing comprehensive safety systems for VLA-powered robots.