---
sidebar_position: 2
title: Action Generation
description: Converting multimodal inputs to robot actions
---

# Action Generation

This lesson covers converting multimodal inputs (vision, language, speech) into executable robot actions with safety constraints and validation.

<LearningObjectives
  objectives={[
    "Convert multimodal inputs to structured robot actions",
    "Implement action planning with temporal and spatial reasoning",
    "Validate actions against safety and feasibility constraints",
    "Handle action failures and provide recovery mechanisms",
    "Design action execution pipelines with feedback"
  ]}
/>

## Multimodal Action Generation Pipeline

### Input Processing Architecture

<ArchitectureDiagram title="Action Generation Pipeline">
{`
┌─────────────────┐    Voice     ┌─────────────────┐
│  Voice Input    │ ──────────▶ │  Speech-to-Text │
│                 │             │                 │
└─────────────────┘             │ 1. Transcribe   │
                                │ 2. Clean        │
                                │ 3. Parse        │
┌─────────────────┐    Vision   └─────────────────┘
│  Vision Input   │ ──────────▶ │                 │
│                 │             │  Vision         │
└─────────────────┘             │  Processing     │
                                │                 │
┌─────────────────┐    Context  │ 1. Object det  │
│ Context Input   │ ──────────▶ │ 2. Scene desc  │
│ (Robot State)   │             │ 3. Environment │
└─────────────────┘             └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  Action         │
                                │  Planner        │
                                │                 │
                                │ 1. Parse intent │
                                │ 2. Plan steps   │
                                │ 3. Validate     │
                                │ 4. Optimize     │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │  Action         │
                                │  Executor       │
                                │                 │
                                │ 1. Execute      │
                                │ 2. Monitor      │
                                │ 3. Recover      │
                                └─────────────────┘
`}
</ArchitectureDiagram>

### Action Space Definition

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class ActionType(Enum):
    # Navigation
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    GOTO_WAYPOINT = "go_to_waypoint"

    # Manipulation
    GRASP_OBJECT = "grasp_object"
    RELEASE_OBJECT = "release_object"
    ROTATE_GRIPPER = "rotate_gripper"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"

    # Inspection
    LOOK_AT = "look_at"
    APPROACH_OBJECT = "approach_object"
    SCAN_AREA = "scan_area"

    # Communication
    SPEAK = "speak"
    SIGNAL = "signal"

class ActionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RobotAction:
    """Structured representation of a robot action."""
    action_type: ActionType
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 10.0
    constraints: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.metadata is None:
            self.metadata = {}
```

## Action Planning from Multimodal Inputs

### Intent Parsing

```python
import re
from typing import Tuple
import numpy as np

class IntentParser:
    def __init__(self):
        self.navigation_patterns = {
            'go_to_location': [
                r'go to (?:the )?(.+)',
                r'move to (?:the )?(.+)',
                r'navigate to (?:the )?(.+)',
                r'go over to (?:the )?(.+)',
                r'head to (?:the )?(.+)'
            ],
            'move_direction': [
                r'go (forward|backward|back)',
                r'move (forward|backward|back)',
                r'go (left|right)',
                r'turn (left|right)'
            ]
        }

        self.manipulation_patterns = {
            'grasp_object': [
                r'pick up (?:the )?(.+)',
                r'grab (?:the )?(.+)',
                r'get (?:the )?(.+)',
                r'pick (?:the )?(.+)'
            ],
            'place_object': [
                r'put (?:the )?(.+) (?:on|at) (.+)',
                r'place (?:the )?(.+) (?:on|at) (.+)'
            ]
        }

    def parse_intent(self, text: str, scene_description: str = "") -> RobotAction:
        """Parse natural language into structured action."""
        text_lower = text.lower().strip()

        # Check navigation patterns
        for intent, patterns in self.navigation_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if intent == 'go_to_location':
                        location = match.group(1)
                        return self.create_navigation_action(location, scene_description)
                    elif intent == 'move_direction':
                        direction = match.group(1)
                        return self.create_direction_action(direction)

        # Check manipulation patterns
        for intent, patterns in self.manipulation_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if intent == 'grasp_object':
                        object_name = match.group(1)
                        return self.create_grasp_action(object_name, scene_description)
                    elif intent == 'place_object':
                        object_name = match.group(1)
                        location = match.group(2)
                        return self.create_place_action(object_name, location, scene_description)

        # Default: speak action
        return RobotAction(
            action_type=ActionType.SPEAK,
            parameters={'text': text},
            priority=2
        )

    def create_navigation_action(self, location: str, scene_desc: str) -> RobotAction:
        """Create navigation action based on location."""
        # Extract location from scene if available
        if scene_desc and location in scene_desc.lower():
            # Could implement more sophisticated location resolution
            pass

        return RobotAction(
            action_type=ActionType.GOTO_WAYPOINT,
            parameters={'target_location': location},
            constraints=['no_collision', 'battery_above_20_percent'],
            timeout=30.0
        )

    def create_grasp_action(self, object_name: str, scene_desc: str) -> RobotAction:
        """Create grasp action based on object."""
        return RobotAction(
            action_type=ActionType.GRASP_OBJECT,
            parameters={'object_name': object_name, 'scene_context': scene_desc},
            constraints=['object_in_reach', 'gripper_free', 'object_graspable'],
            timeout=15.0
        )
```

### Spatial and Temporal Reasoning

```python
import math
from typing import Tuple

class SpatialReasoner:
    def __init__(self):
        self.robot_pose = (0, 0, 0)  # x, y, theta
        self.costmap = None  # Navigation costmap

    def calculate_path_to_object(self, object_position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Calculate safe path to object using costmap."""
        robot_pos = self.robot_pose[:2]

        # Simple A* or Dijkstra's algorithm implementation
        # In practice, use Nav2's path planner
        path = self.plan_path(robot_pos, object_position)

        return path

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan path from start to goal avoiding obstacles."""
        # This would interface with Nav2's global planner
        # For now, return direct path (not safe for real robots)
        return [start, goal]

    def check_reachability(self, target_position: Tuple[float, float], robot_state: Dict) -> bool:
        """Check if target is reachable given robot kinematics."""
        robot_pos = self.robot_pose[:2]
        distance = math.sqrt(
            (target_position[0] - robot_pos[0])**2 +
            (target_position[1] - robot_pos[1])**2
        )

        # Check if within manipulator reach (for manipulator robots)
        max_reach = robot_state.get('max_manipulator_reach', 1.0)
        return distance <= max_reach

    def validate_action_feasibility(self, action: RobotAction, robot_state: Dict) -> Tuple[bool, List[str]]:
        """Validate if action is feasible given current state."""
        errors = []

        # Check battery level for navigation actions
        if action.action_type in [ActionType.GOTO_WAYPOINT, ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD]:
            battery = robot_state.get('battery_level', 100)
            if battery < 15:  # 15% threshold
                errors.append("Battery level too low for navigation")

        # Check gripper state for manipulation
        if action.action_type in [ActionType.GRASP_OBJECT, ActionType.GRASP_OBJECT]:
            gripper_state = robot_state.get('gripper_state', 'open')
            if gripper_state == 'closed' and 'grasp' in action.action_type.value:
                errors.append("Gripper already closed")

        # Check for collisions
        if action.action_type in [ActionType.GOTO_WAYPOINT]:
            if self.would_collide(action.parameters.get('target_location')):
                errors.append("Collision detected in path")

        return len(errors) == 0, errors
```

## Safety and Constraint Validation

### Safety Constraint System

```python
class SafetyConstraints:
    def __init__(self):
        self.constraints = {
            'collision_avoidance': self.check_collision,
            'joint_limits': self.check_joint_limits,
            'workspace_bounds': self.check_workspace_bounds,
            'battery_level': self.check_battery_level,
            'gripper_state': self.check_gripper_state,
            'obstacle_distance': self.check_obstacle_distance
        }

    def validate_action(self, action: RobotAction, robot_state: Dict) -> Tuple[bool, List[str]]:
        """Validate action against all safety constraints."""
        violations = []

        for constraint_name in action.constraints:
            if constraint_name in self.constraints:
                is_valid, error_msg = self.constraints[constraint_name](action, robot_state)
                if not is_valid:
                    violations.append(f"{constraint_name}: {error_msg}")

        return len(violations) == 0, violations

    def check_collision(self, action: RobotAction, robot_state: Dict) -> Tuple[bool, str]:
        """Check if action would cause collision."""
        if action.action_type == ActionType.GOTO_WAYPOINT:
            path = self.calculate_path_to_object(action.parameters.get('target_location'))
            for point in path:
                if self.is_occupied(point):
                    return False, f"Collision path to {action.parameters.get('target_location')}"
        return True, ""

    def check_joint_limits(self, action: RobotAction, robot_state: Dict) -> Tuple[bool, str]:
        """Check if action would exceed joint limits."""
        # Implementation depends on robot type
        current_joints = robot_state.get('joint_positions', {})
        target_joints = action.parameters.get('target_joints', {})

        for joint_name, target_pos in target_joints.items():
            joint_limits = robot_state.get('joint_limits', {}).get(joint_name, {})
            if target_pos < joint_limits.get('min', -float('inf')):
                return False, f"Joint {joint_name} below minimum limit"
            if target_pos > joint_limits.get('max', float('inf')):
                return False, f"Joint {joint_name} above maximum limit"

        return True, ""

    def check_battery_level(self, action: RobotAction, robot_state: Dict) -> Tuple[bool, str]:
        """Check if battery is sufficient for action."""
        battery_level = robot_state.get('battery_level', 100)

        if action.action_type in [ActionType.GOTO_WAYPOINT]:
            estimated_battery_usage = self.estimate_battery_usage(action)
            if battery_level - estimated_battery_usage < 10:  # 10% safety margin
                return False, f"Insufficient battery for navigation (need {estimated_battery_usage:.1f}%)"

        return True, ""
```

### Action Execution with Monitoring

```python
import time
from threading import Thread, Event

class ActionExecutor:
    def __init__(self):
        self.active_action = None
        self.execution_thread = None
        self.stop_execution = Event()
        self.safety_constraints = SafetyConstraints()

    def execute_action(self, action: RobotAction, robot_state: Dict) -> ActionStatus:
        """Execute action with safety monitoring."""
        # Validate action first
        is_valid, violations = self.safety_constraints.validate_action(action, robot_state)
        if not is_valid:
            print(f"Action validation failed: {violations}")
            return ActionStatus.FAILED

        self.active_action = action
        self.stop_execution.clear()

        # Execute based on action type
        if action.action_type in [ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD, ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
            status = self.execute_navigation_action(action)
        elif action.action_type in [ActionType.GRASP_OBJECT, ActionType.RELEASE_OBJECT]:
            status = self.execute_manipulation_action(action)
        elif action.action_type == ActionType.SPEAK:
            status = self.execute_speech_action(action)
        else:
            status = self.execute_generic_action(action)

        self.active_action = None
        return status

    def execute_navigation_action(self, action: RobotAction) -> ActionStatus:
        """Execute navigation action with monitoring."""
        start_time = time.time()

        # Start navigation (this would interface with Nav2)
        self.start_navigation(action)

        # Monitor execution
        while not self.is_navigation_complete() and not self.stop_execution.is_set():
            if time.time() - start_time > action.timeout:
                self.cancel_navigation()
                return ActionStatus.FAILED

            # Check for safety violations
            if self.has_safety_violation():
                self.emergency_stop()
                return ActionStatus.FAILED

            time.sleep(0.1)  # Check every 100ms

        if self.is_navigation_complete():
            return ActionStatus.SUCCESS
        else:
            return ActionStatus.FAILED

    def execute_manipulation_action(self, action: RobotAction) -> ActionStatus:
        """Execute manipulation action with monitoring."""
        # This would interface with manipulation stack
        # For now, simulate execution
        time.sleep(2.0)  # Simulate action time
        return ActionStatus.SUCCESS

    def start_navigation(self, action: RobotAction):
        """Start navigation action."""
        # Interface with Nav2
        pass

    def is_navigation_complete(self) -> bool:
        """Check if navigation is complete."""
        # Check with Nav2
        return True

    def has_safety_violation(self) -> bool:
        """Check for safety violations during execution."""
        # Monitor sensors and robot state
        return False

    def cancel_action(self):
        """Cancel current action."""
        self.stop_execution.set()
```

## Feedback and Recovery Systems

### Action Monitoring and Feedback

```python
class ActionMonitor:
    def __init__(self):
        self.action_history = []
        self.feedback_callbacks = []

    def start_monitoring(self, action: RobotAction):
        """Start monitoring an action."""
        monitoring_data = {
            'action_id': len(self.action_history),
            'action': action,
            'start_time': time.time(),
            'status': ActionStatus.EXECUTING,
            'progress': 0.0
        }
        self.action_history.append(monitoring_data)
        return monitoring_data['action_id']

    def update_progress(self, action_id: int, progress: float, status: ActionStatus = None):
        """Update action progress and status."""
        if 0 <= action_id < len(self.action_history):
            action_data = self.action_history[action_id]
            action_data['progress'] = progress
            if status:
                action_data['status'] = status

            # Trigger feedback callbacks
            for callback in self.feedback_callbacks:
                callback(action_data)

    def get_action_result(self, action_id: int) -> Dict:
        """Get result of completed action."""
        if 0 <= action_id < len(self.action_history):
            return self.action_history[action_id]
        return None

    def add_feedback_callback(self, callback):
        """Add callback for action feedback."""
        self.feedback_callbacks.append(callback)
```

### Recovery Strategies

```python
class RecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'navigation_failure': self.recovery_navigation,
            'manipulation_failure': self.recovery_manipulation,
            'collision_detected': self.recovery_collision,
            'timeout': self.recovery_timeout
        }

    def handle_failure(self, action: RobotAction, failure_type: str, robot_state: Dict) -> List[RobotAction]:
        """Generate recovery actions for failure."""
        if failure_type in self.recovery_strategies:
            return self.recovery_strategies[failure_type](action, robot_state)
        else:
            # Default recovery: stop and report
            return [RobotAction(ActionType.STOP, {}, priority=5)]

    def recovery_navigation(self, action: RobotAction, robot_state: Dict) -> List[RobotAction]:
        """Recovery strategy for navigation failure."""
        recovery_actions = []

        # Try alternative route
        recovery_actions.append(RobotAction(
            action_type=ActionType.GOTO_WAYPOINT,
            parameters={'target_location': self.find_alternative_waypoint(action.parameters['target_location'])},
            priority=3
        ))

        return recovery_actions

    def recovery_collision(self, action: RobotAction, robot_state: Dict) -> List[RobotAction]:
        """Recovery strategy for collision detection."""
        recovery_actions = []

        # Stop immediately
        recovery_actions.append(RobotAction(ActionType.STOP, {}, priority=5))

        # Back up slightly
        recovery_actions.append(RobotAction(
            action_type=ActionType.MOVE_BACKWARD,
            parameters={'distance': 0.2},  # 20cm
            priority=4
        ))

        # Reassess situation
        recovery_actions.append(RobotAction(
            action_type=ActionType.SCAN_AREA,
            parameters={},
            priority=2
        ))

        return recovery_actions

    def recovery_timeout(self, action: RobotAction, robot_state: Dict) -> List[RobotAction]:
        """Recovery strategy for action timeout."""
        recovery_actions = []

        # Stop current action
        recovery_actions.append(RobotAction(ActionType.STOP, {}, priority=5))

        # Report timeout
        recovery_actions.append(RobotAction(
            action_type=ActionType.SPEAK,
            parameters={'text': f"Action timed out: {action.action_type.value}"},
            priority=3
        ))

        return recovery_actions
```

## Integration with ROS 2

### Action Execution Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from robot_action_interfaces.action import ExecuteAction  # Custom action interface

class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution_node')

        # Action server
        self._action_server = ActionServer(
            self,
            ExecuteAction,
            'execute_action',
            self.execute_action_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers
        self.status_pub = self.create_publisher(String, 'action_status', 10)
        self.feedback_pub = self.create_publisher(String, 'action_feedback', 10)

        # Components
        self.intent_parser = IntentParser()
        self.spatial_reasoner = SpatialReasoner()
        self.action_executor = ActionExecutor()
        self.action_monitor = ActionMonitor()
        self.recovery_manager = RecoveryManager()

        self.get_logger().info('Action Execution Node initialized')

    def goal_callback(self, goal_request):
        """Accept or reject action goal."""
        self.get_logger().info(f'Received action goal: {goal_request.command}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject action cancel request."""
        self.get_logger().info('Received action cancel request')
        return CancelResponse.ACCEPT

    def execute_action_callback(self, goal_handle):
        """Execute the action."""
        feedback_msg = ExecuteAction.Feedback()
        result_msg = ExecuteAction.Result()

        command = goal_handle.request.command
        self.get_logger().info(f'Executing command: {command}')

        try:
            # Parse intent from command
            action = self.intent_parser.parse_intent(command)

            # Validate action
            is_valid, errors = self.spatial_reasoner.validate_action_feasibility(action, self.get_robot_state())
            if not is_valid:
                result_msg.success = False
                result_msg.message = f"Action validation failed: {', '.join(errors)}"
                goal_handle.succeed()
                return result_msg

            # Execute action
            status = self.action_executor.execute_action(action, self.get_robot_state())

            if status == ActionStatus.SUCCESS:
                result_msg.success = True
                result_msg.message = "Action completed successfully"
            else:
                result_msg.success = False
                result_msg.message = f"Action failed with status: {status.value}"

        except Exception as e:
            self.get_logger().error(f'Action execution error: {e}')
            result_msg.success = False
            result_msg.message = f"Execution error: {str(e)}"

        goal_handle.succeed()
        return result_msg
```

## Exercises

<Exercise title="Action Validation System" difficulty="intermediate" estimatedTime="45 min">

Create a comprehensive action validation system that:
1. Validates robot actions against multiple safety constraints
2. Checks joint limits, workspace bounds, and battery level
3. Provides detailed error messages for invalid actions
4. Includes unit tests for validation logic

**Requirements:**
- Implement at least 5 different safety constraints
- Create validation pipeline
- Add detailed error reporting
- Include unit tests

<Hint>
Structure your validation as a pipeline:
```python
validators = [
    BatteryValidator(),
    CollisionValidator(),
    JointLimitValidator(),
    WorkspaceValidator(),
    KinematicValidator()
]
```
</Hint>

</Exercise>

<Exercise title="Recovery System" difficulty="advanced" estimatedTime="75 min">

Build a sophisticated recovery system that:
1. Detects different types of action failures
2. Implements appropriate recovery strategies
3. Learns from recovery attempts to improve
4. Provides graceful degradation when recovery fails

**Requirements:**
- Failure detection mechanisms
- Multiple recovery strategies
- Learning from recovery attempts
- Fallback safety procedures

</Exercise>

## Summary

Key concepts covered:

- ✅ Multimodal action generation pipeline
- ✅ Intent parsing from natural language
- ✅ Spatial and temporal reasoning
- ✅ Safety constraint validation
- ✅ Action execution with monitoring
- ✅ Feedback and recovery systems
- ✅ ROS 2 integration patterns

## Next Steps

Continue to [Safety Constraints](/module-4/week-11/safety-constraints) to learn about advanced safety systems and constraint validation for VLA-powered robots.