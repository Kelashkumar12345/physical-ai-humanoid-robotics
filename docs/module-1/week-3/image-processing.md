---
sidebar_position: 2
title: Image Processing
description: Computer vision with OpenCV and ROS 2
---

# Image Processing with OpenCV

This lesson covers computer vision techniques using OpenCV integrated with ROS 2 for robotic perception.

## Learning Objectives

By the end of this lesson, you will:

- Use cv_bridge to convert between ROS and OpenCV images
- Apply image filtering and preprocessing
- Implement object detection using color and shape
- Perform feature detection and matching

## OpenCV and ROS 2 Integration

### Installing Dependencies

```bash
# Install OpenCV for ROS 2
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-vision-opencv

# Python OpenCV
pip3 install opencv-python numpy
```

### The cv_bridge Package

cv_bridge converts between ROS Image messages and OpenCV's cv::Mat format:

```python
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

bridge = CvBridge()

# ROS Image to OpenCV
cv_image = bridge.imgmsg_to_cv2(ros_image_msg, 'bgr8')

# OpenCV to ROS Image
ros_image_msg = bridge.cv2_to_imgmsg(cv_image, 'bgr8')
```

### Supported Encodings

| Encoding | Description | Channels |
|----------|-------------|----------|
| `mono8` | Grayscale 8-bit | 1 |
| `mono16` | Grayscale 16-bit | 1 |
| `bgr8` | BGR color 8-bit | 3 |
| `rgb8` | RGB color 8-bit | 3 |
| `bgra8` | BGR with alpha | 4 |
| `32FC1` | Float 32-bit single channel | 1 |

## Basic Image Processing Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.bridge = CvBridge()

        # Subscriber for raw images
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for processed images
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_processed',
            10
        )

        self.get_logger().info('Image processor initialized')

    def image_callback(self, msg):
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Process image
        processed = self.process_image(cv_image)

        # Convert back to ROS and publish
        processed_msg = self.bridge.cv2_to_imgmsg(processed, 'bgr8')
        processed_msg.header = msg.header
        self.publisher.publish(processed_msg)

    def process_image(self, image):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to BGR for visualization
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return result

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Image Filtering Techniques

### Smoothing Filters

```python
import cv2
import numpy as np

def apply_filters(image):
    # Gaussian Blur - reduces noise
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    # Median Blur - good for salt-and-pepper noise
    median = cv2.medianBlur(image, 5)

    # Bilateral Filter - preserves edges while smoothing
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)

    return gaussian, median, bilateral
```

### Morphological Operations

```python
def morphological_operations(binary_image):
    # Define kernel
    kernel = np.ones((5, 5), np.uint8)

    # Erosion - shrinks white regions
    eroded = cv2.erode(binary_image, kernel, iterations=1)

    # Dilation - expands white regions
    dilated = cv2.dilate(binary_image, kernel, iterations=1)

    # Opening - erosion followed by dilation (removes noise)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Closing - dilation followed by erosion (fills holes)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return eroded, dilated, opened, closed
```

## Color-Based Object Detection

### HSV Color Space Detection

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        self.bridge = CvBridge()

        # HSV range for red color
        self.declare_parameter('hue_low', 0)
        self.declare_parameter('hue_high', 10)
        self.declare_parameter('saturation_low', 100)
        self.declare_parameter('saturation_high', 255)
        self.declare_parameter('value_low', 100)
        self.declare_parameter('value_high', 255)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.mask_publisher = self.create_publisher(
            Image, '/detection/mask', 10
        )

        self.centroid_publisher = self.create_publisher(
            Point, '/detection/centroid', 10
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Get parameters
        h_low = self.get_parameter('hue_low').value
        h_high = self.get_parameter('hue_high').value
        s_low = self.get_parameter('saturation_low').value
        s_high = self.get_parameter('saturation_high').value
        v_low = self.get_parameter('value_low').value
        v_high = self.get_parameter('value_high').value

        # Create mask
        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find centroid
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Publish centroid
            centroid_msg = Point()
            centroid_msg.x = float(cx)
            centroid_msg.y = float(cy)
            centroid_msg.z = 0.0
            self.centroid_publisher.publish(centroid_msg)

            self.get_logger().info(f'Object detected at ({cx}, {cy})')

        # Publish mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, 'mono8')
        self.mask_publisher.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Contour Detection and Shape Analysis

### Finding and Drawing Contours

```python
#!/usr/bin/env python3
import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass

    def detect_shapes(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            # Approximate polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Identify shape
            shape = self.identify_shape(approx)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            results.append({
                'shape': shape,
                'contour': contour,
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'area': area,
                'vertices': len(approx)
            })

        return results

    def identify_shape(self, approx):
        vertices = len(approx)

        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            # Check if square or rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                return 'square'
            else:
                return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        else:
            return 'circle'

    def draw_results(self, image, results):
        output = image.copy()

        for result in results:
            # Draw contour
            cv2.drawContours(output, [result['contour']], -1, (0, 255, 0), 2)

            # Draw bounding box
            x, y, w, h = result['bbox']
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw centroid
            cx, cy = result['centroid']
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

            # Label shape
            cv2.putText(
                output,
                result['shape'],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return output
```

## Feature Detection

### ORB Feature Detector

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector')

        self.bridge = CvBridge()

        # Create ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.publisher = self.create_publisher(
            Image, '/features/image', 10
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Draw keypoints
        output = cv2.drawKeypoints(
            cv_image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        self.get_logger().info(f'Detected {len(keypoints)} features')

        # Publish result
        result_msg = self.bridge.cv2_to_imgmsg(output, 'bgr8')
        self.publisher.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Feature Matching

```python
import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_images(self, img1, img2):
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect and compute
        kp1, desc1 = self.orb.detectAndCompute(gray1, None)
        kp2, desc2 = self.orb.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            return None, []

        # Match descriptors
        matches = self.bf.match(desc1, desc2)

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep top matches
        good_matches = matches[:50]

        # Draw matches
        result = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return result, good_matches
```

## Object Tracking

### Simple Centroid Tracker

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracker')

        self.bridge = CvBridge()
        self.tracker = CentroidTracker(max_disappeared=30)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.publisher = self.create_publisher(
            Image, '/tracking/image', 10
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Detect objects (using simple color detection)
        centroids = self.detect_objects(cv_image)

        # Update tracker
        objects = self.tracker.update(centroids)

        # Draw tracking results
        for (object_id, centroid) in objects.items():
            text = f'ID {object_id}'
            cv2.putText(cv_image, text,
                       (int(centroid[0]) - 10, int(centroid[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(cv_image, (int(centroid[0]), int(centroid[1])),
                      4, (0, 255, 0), -1)

        # Publish
        result_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        self.publisher.publish(result_msg)

    def detect_objects(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centroids.append([cx, cy])

        return np.array(centroids) if centroids else np.empty((0, 2))

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this lesson, you learned:

- How to convert between ROS Image messages and OpenCV using cv_bridge
- Image filtering and preprocessing techniques
- Color-based and shape-based object detection
- Feature detection with ORB and feature matching
- Simple object tracking using centroid tracking

## Next Steps

Continue to [Point Cloud Processing](/module-1/week-3/point-cloud-processing) to learn about 3D perception with point clouds.
