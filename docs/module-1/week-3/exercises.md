---
sidebar_position: 4
title: Week 3 Exercises
description: Hands-on exercises for perception and sensor integration
---

# Week 3: Perception Pipeline Exercises

These exercises will help you practice sensor integration and perception techniques covered this week.

## Exercise 1: Camera Stream Processing

**Objective**: Create a ROS 2 node that processes camera images in real-time.

### Requirements

1. Subscribe to a camera image topic
2. Apply edge detection (Canny)
3. Detect circles using Hough Transform
4. Publish the processed image with detected circles highlighted

### Starter Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()

        # TODO: Create subscriber for '/camera/image_raw'

        # TODO: Create publisher for '/camera/circles'

    def image_callback(self, msg):
        # TODO: Convert ROS Image to OpenCV
        # TODO: Convert to grayscale
        # TODO: Apply Gaussian blur
        # TODO: Detect circles using cv2.HoughCircles()
        # TODO: Draw circles on image
        # TODO: Publish result

        pass

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Testing

```bash
# Terminal 1: Start a simulated camera (or use USB camera)
ros2 run v4l2_camera v4l2_camera_node

# Terminal 2: Run your node
ros2 run my_perception circle_detector

# Terminal 3: View results
ros2 run rqt_image_view rqt_image_view
```

<details className="solution-block">
<summary>Solution</summary>
<div className="solution-content">

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/camera/circles',
            10
        )

        self.get_logger().info('Circle detector initialized')

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=200
        )

        # Draw circles
        output = cv_image.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            self.get_logger().info(f'Detected {len(circles)} circles')

        # Publish result
        result_msg = self.bridge.cv2_to_imgmsg(output, 'bgr8')
        result_msg.header = msg.header
        self.publisher.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

</div>
</details>

---

## Exercise 2: Color-Based Object Tracking

**Objective**: Track a colored object and publish its 3D position.

### Requirements

1. Detect an object by color (e.g., a red ball)
2. Use depth information to get 3D position
3. Publish position as `geometry_msgs/PointStamped`
4. Make color thresholds configurable via parameters

### Starter Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorTracker(Node):
    def __init__(self):
        super().__init__('color_tracker')
        self.bridge = CvBridge()

        # Declare parameters for HSV thresholds
        self.declare_parameter('h_min', 0)
        self.declare_parameter('h_max', 10)
        self.declare_parameter('s_min', 100)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 100)
        self.declare_parameter('v_max', 255)

        # TODO: Subscribe to color and depth images
        # TODO: Subscribe to camera info
        # TODO: Create publisher for 3D position

        self.camera_matrix = None
        self.depth_image = None

    def camera_info_callback(self, msg):
        # Extract camera intrinsics
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def depth_callback(self, msg):
        # Store depth image for later use
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def color_callback(self, msg):
        # TODO: Detect colored object
        # TODO: Get depth at object centroid
        # TODO: Convert to 3D position using camera intrinsics
        # TODO: Publish PointStamped
        pass

    def pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates to 3D point."""
        if self.camera_matrix is None:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return (x, y, z)

def main(args=None):
    rclpy.init(args=args)
    node = ColorTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<details className="hint-block">
<summary>Hint</summary>
<div className="hint-content">

Use `message_filters` to synchronize color and depth images:

```python
from message_filters import Subscriber, ApproximateTimeSynchronizer

color_sub = Subscriber(self, Image, '/camera/color/image_raw')
depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

sync = ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
sync.registerCallback(self.sync_callback)
```

</div>
</details>

---

## Exercise 3: LiDAR Obstacle Detection

**Objective**: Detect obstacles from LiDAR data and classify by distance.

### Requirements

1. Subscribe to `/scan` topic (LaserScan)
2. Identify obstacle clusters in different directions
3. Publish warning messages for close obstacles
4. Visualize obstacles as markers in RViz

### Starter Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Distance thresholds
        self.declare_parameter('danger_distance', 0.5)
        self.declare_parameter('warning_distance', 1.0)

        # TODO: Create subscriber for LaserScan
        # TODO: Create publishers for markers and warnings

    def scan_callback(self, msg):
        # TODO: Process scan data
        # TODO: Find minimum distance in each sector (front, left, right, back)
        # TODO: Classify obstacles by danger level
        # TODO: Create visualization markers
        # TODO: Publish warnings for dangerous obstacles
        pass

    def get_sector_min_distance(self, ranges, angle_min, angle_max, scan_angle_min, angle_increment):
        """Get minimum distance within an angular sector."""
        # TODO: Implement sector analysis
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<details className="solution-block">
<summary>Solution</summary>
<div className="solution-content">

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import numpy as np
import math

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        self.declare_parameter('danger_distance', 0.5)
        self.declare_parameter('warning_distance', 1.0)

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/obstacles/markers',
            10
        )

        self.warning_pub = self.create_publisher(
            String,
            '/obstacles/warnings',
            10
        )

        # Define sectors (in radians)
        self.sectors = {
            'front': (-math.pi/4, math.pi/4),
            'left': (math.pi/4, 3*math.pi/4),
            'back': (3*math.pi/4, -3*math.pi/4),
            'right': (-3*math.pi/4, -math.pi/4)
        }

    def scan_callback(self, msg):
        danger_dist = self.get_parameter('danger_distance').value
        warning_dist = self.get_parameter('warning_distance').value

        ranges = np.array(msg.ranges)
        angles = np.arange(
            msg.angle_min,
            msg.angle_max + msg.angle_increment,
            msg.angle_increment
        )[:len(ranges)]

        # Replace inf/nan with max range
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

        markers = MarkerArray()
        warnings = []

        for i, (sector_name, (angle_min, angle_max)) in enumerate(self.sectors.items()):
            # Handle wrap-around for back sector
            if angle_min > angle_max:
                mask = (angles >= angle_min) | (angles <= angle_max)
            else:
                mask = (angles >= angle_min) & (angles <= angle_max)

            sector_ranges = ranges[mask]
            if len(sector_ranges) == 0:
                continue

            min_dist = np.min(sector_ranges)
            min_idx = np.argmin(sector_ranges)
            sector_angles = angles[mask]
            min_angle = sector_angles[min_idx]

            # Determine danger level
            if min_dist < danger_dist:
                color = (1.0, 0.0, 0.0)  # Red
                warnings.append(f'DANGER: {sector_name} obstacle at {min_dist:.2f}m')
            elif min_dist < warning_dist:
                color = (1.0, 1.0, 0.0)  # Yellow
                warnings.append(f'WARNING: {sector_name} obstacle at {min_dist:.2f}m')
            else:
                color = (0.0, 1.0, 0.0)  # Green

            # Create marker
            marker = Marker()
            marker.header = msg.header
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = min_dist * math.cos(min_angle)
            marker.pose.position.y = min_dist * math.sin(min_angle)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8

            markers.markers.append(marker)

        self.marker_pub.publish(markers)

        if warnings:
            warning_msg = String()
            warning_msg.data = '; '.join(warnings)
            self.warning_pub.publish(warning_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

</div>
</details>

---

## Exercise 4: Point Cloud Object Segmentation

**Objective**: Build a complete point cloud segmentation pipeline.

### Requirements

1. Subscribe to depth camera point cloud
2. Downsample the point cloud
3. Remove ground plane
4. Cluster remaining points into objects
5. Publish bounding boxes for each object

### Starter Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
import open3d as o3d
import numpy as np

class PointCloudSegmenter(Node):
    def __init__(self):
        super().__init__('pointcloud_segmenter')

        # Parameters
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('ground_threshold', 0.02)
        self.declare_parameter('cluster_tolerance', 0.02)
        self.declare_parameter('min_cluster_size', 100)

        # TODO: Create subscriber and publisher

    def callback(self, msg):
        # Step 1: Convert to Open3D
        pcd = self.ros_to_open3d(msg)

        # TODO: Step 2: Downsample
        # TODO: Step 3: Remove outliers
        # TODO: Step 4: Remove ground plane
        # TODO: Step 5: Cluster objects
        # TODO: Step 6: Create and publish bounding box markers

    def ros_to_open3d(self, msg):
        """Convert ROS PointCloud2 to Open3D point cloud."""
        # Implementation provided
        pass

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSegmenter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Exercise 5: Multi-Sensor Fusion

**Objective**: Combine camera and LiDAR data for enhanced object detection.

### Requirements

1. Synchronize camera images and LiDAR scans
2. Project LiDAR points onto camera image
3. Combine visual detection with distance information
4. Publish fused detections with position and class

### Approach

```
Camera Image + LiDAR Scan
         ↓
   Time Synchronization
         ↓
   Extrinsic Calibration (provided)
         ↓
   Project LiDAR → Image
         ↓
   Combine Detections
         ↓
   Fused Object List
```

<details className="hint-block">
<summary>Hint: LiDAR to Camera Projection</summary>
<div className="hint-content">

```python
def project_lidar_to_camera(lidar_points, extrinsic_matrix, camera_matrix):
    """
    Project LiDAR points onto camera image plane.

    Args:
        lidar_points: Nx3 array of 3D points
        extrinsic_matrix: 4x4 transformation from LiDAR to camera
        camera_matrix: 3x3 camera intrinsic matrix

    Returns:
        uv_points: Nx2 array of pixel coordinates
        depths: N array of depths
    """
    # Transform to camera frame
    points_homogeneous = np.hstack([
        lidar_points,
        np.ones((len(lidar_points), 1))
    ])
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T[:, :3]

    # Filter points behind camera
    valid_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_mask]

    # Project to image plane
    points_normalized = points_camera[:, :2] / points_camera[:, 2:3]
    uv_homogeneous = np.hstack([
        points_normalized,
        np.ones((len(points_normalized), 1))
    ])
    uv_points = (camera_matrix @ uv_homogeneous.T).T[:, :2]

    return uv_points, points_camera[:, 2], valid_mask
```

</div>
</details>

---

## Challenge Exercise: Real-Time Object Recognition

**Objective**: Build a complete perception system that detects, classifies, and tracks objects.

### Requirements

1. Detect objects using both color and depth
2. Classify objects by shape and size
3. Track objects across frames (assign consistent IDs)
4. Handle occlusions gracefully
5. Publish detection results at minimum 10 Hz

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Correct detection (position accuracy less than 5cm) | 25 |
| Classification accuracy greater than 80% | 25 |
| Tracking consistency (same ID for same object) | 25 |
| Real-time performance (10+ Hz) | 15 |
| Handles occlusions | 10 |

### Tips

- Use Kalman filtering for smooth tracking
- Implement a Hungarian algorithm for ID assignment
- Consider using multiple detection methods and fusing results
- Profile your code to find bottlenecks

---

## Submission Checklist

Before submitting your exercises, verify:

- [ ] All nodes compile without errors
- [ ] Code follows ROS 2 Python style guidelines
- [ ] Each node has proper logging
- [ ] Parameters are declared and used correctly
- [ ] Publishers and subscribers use appropriate QoS settings
- [ ] Code handles edge cases (empty data, disconnections)
- [ ] Visualization works in RViz2

## Additional Resources

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [ROS 2 Perception Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [Point Cloud Library](https://pointclouds.org/)
