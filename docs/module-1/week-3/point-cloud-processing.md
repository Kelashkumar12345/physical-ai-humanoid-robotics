---
sidebar_position: 3
title: Point Cloud Processing
description: 3D perception with PCL and Open3D in ROS 2
---

# Point Cloud Processing

This lesson covers 3D point cloud processing for robotic perception using PCL (Point Cloud Library) and Open3D.

## Learning Objectives

By the end of this lesson, you will:

- Understand point cloud data structures
- Filter and downsample point clouds
- Detect planes and segment objects
- Perform 3D object recognition

## Understanding Point Clouds

A point cloud is a collection of 3D points representing the surfaces of objects in the environment. Each point typically contains:

- **Position**: (x, y, z) coordinates
- **Color**: RGB values (optional)
- **Normals**: Surface normal vectors (optional)
- **Intensity**: Reflection intensity (from LiDAR)

### PointCloud2 Message Structure

```python
from sensor_msgs.msg import PointCloud2, PointField
import struct

# PointCloud2 structure:
# - header: timestamp and frame_id
# - height: organized cloud height (1 for unorganized)
# - width: number of points per row
# - fields: description of point data
# - is_bigendian: byte order
# - point_step: bytes per point
# - row_step: bytes per row
# - data: actual point data
# - is_dense: true if no invalid points
```

## Point Cloud Processing with Open3D

### Installing Open3D

```bash
pip3 install open3d numpy
```

### Basic Point Cloud Operations

```python
#!/usr/bin/env python3
import open3d as o3d
import numpy as np

class PointCloudProcessor:
    def __init__(self):
        pass

    def create_point_cloud(self, points, colors=None):
        """Create Open3D point cloud from numpy array."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def visualize(self, pcd):
        """Visualize point cloud."""
        o3d.visualization.draw_geometries([pcd])

    def save_point_cloud(self, pcd, filename):
        """Save point cloud to file."""
        o3d.io.write_point_cloud(filename, pcd)

    def load_point_cloud(self, filename):
        """Load point cloud from file."""
        return o3d.io.read_point_cloud(filename)
```

### Converting ROS PointCloud2 to Open3D

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

class PointCloudConverter(Node):
    def __init__(self):
        super().__init__('pointcloud_converter')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback,
            10
        )

    def callback(self, msg):
        # Convert to Open3D format
        pcd = self.ros_to_open3d(msg)

        # Process point cloud
        processed = self.process_point_cloud(pcd)

        self.get_logger().info(f'Processed {len(processed.points)} points')

    def ros_to_open3d(self, msg):
        """Convert ROS PointCloud2 to Open3D point cloud."""
        points = []
        colors = []

        for point in pc2.read_points(msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])

            # Extract RGB if available
            if len(point) > 3:
                # RGB is packed as a float
                rgb = point[3]
                # Unpack RGB
                s = struct.pack('>f', rgb)
                i = struct.unpack('>l', s)[0]
                r = (i >> 16) & 0xFF
                g = (i >> 8) & 0xFF
                b = i & 0xFF
                colors.append([r/255.0, g/255.0, b/255.0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))

        if colors:
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def process_point_cloud(self, pcd):
        """Basic point cloud processing pipeline."""
        # Remove statistical outliers
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        return pcd_clean

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Filtering and Downsampling

### Voxel Grid Downsampling

```python
def voxel_downsample(pcd, voxel_size=0.01):
    """
    Downsample point cloud using voxel grid.

    Args:
        pcd: Open3D point cloud
        voxel_size: Size of voxel cube in meters

    Returns:
        Downsampled point cloud
    """
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled
```

### Statistical Outlier Removal

```python
def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from point cloud.

    Args:
        pcd: Open3D point cloud
        nb_neighbors: Number of neighbors for mean distance estimation
        std_ratio: Standard deviation ratio threshold

    Returns:
        Filtered point cloud and indices
    """
    pcd_clean, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean, indices
```

### Radius Outlier Removal

```python
def remove_radius_outliers(pcd, nb_points=16, radius=0.05):
    """
    Remove points with fewer than nb_points neighbors within radius.

    Args:
        pcd: Open3D point cloud
        nb_points: Minimum number of neighbors
        radius: Search radius in meters

    Returns:
        Filtered point cloud and indices
    """
    pcd_clean, indices = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius
    )
    return pcd_clean, indices
```

### Pass-Through Filter

```python
def passthrough_filter(pcd, axis='z', min_val=0.0, max_val=1.0):
    """
    Filter points outside specified range on given axis.

    Args:
        pcd: Open3D point cloud
        axis: 'x', 'y', or 'z'
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Filtered point cloud
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]

    # Create mask
    mask = (points[:, axis_idx] >= min_val) & (points[:, axis_idx] <= max_val)

    # Filter points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])

    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered_pcd
```

## Plane Detection (RANSAC)

### RANSAC Plane Segmentation

```python
def detect_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Detect dominant plane using RANSAC.

    Args:
        pcd: Open3D point cloud
        distance_threshold: Max distance to plane for inliers
        ransac_n: Number of points to estimate plane
        num_iterations: Number of RANSAC iterations

    Returns:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0
        inliers: Indices of inlier points
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    [a, b, c, d] = plane_model
    print(f'Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')

    return plane_model, inliers
```

### Ground Plane Removal

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

class GroundRemovalNode(Node):
    def __init__(self):
        super().__init__('ground_removal')

        self.declare_parameter('distance_threshold', 0.02)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 2.0)

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback,
            10
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            '/points/no_ground',
            10
        )

    def callback(self, msg):
        # Convert to Open3D
        pcd = self.ros_to_open3d(msg)

        if len(pcd.points) < 100:
            return

        # Filter by height
        z_min = self.get_parameter('z_min').value
        z_max = self.get_parameter('z_max').value
        pcd = self.passthrough_filter(pcd, 'z', z_min, z_max)

        # Detect and remove ground plane
        distance_threshold = self.get_parameter('distance_threshold').value
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        # Keep non-ground points
        objects_pcd = pcd.select_by_index(inliers, invert=True)

        self.get_logger().info(
            f'Removed {len(inliers)} ground points, '
            f'{len(objects_pcd.points)} remaining'
        )

        # Convert back to ROS and publish
        output_msg = self.open3d_to_ros(objects_pcd, msg.header)
        self.publisher.publish(output_msg)

    # ... (include ros_to_open3d, passthrough_filter, open3d_to_ros methods)
```

## Clustering and Segmentation

### DBSCAN Clustering

```python
def cluster_points(pcd, eps=0.02, min_points=10):
    """
    Cluster point cloud using DBSCAN.

    Args:
        pcd: Open3D point cloud
        eps: Maximum distance between neighbors
        min_points: Minimum points per cluster

    Returns:
        labels: Cluster label for each point (-1 for noise)
        num_clusters: Number of clusters found
    """
    labels = np.array(pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=False
    ))

    num_clusters = labels.max() + 1
    print(f'Found {num_clusters} clusters')

    return labels, num_clusters


def extract_clusters(pcd, labels):
    """
    Extract individual cluster point clouds.

    Args:
        pcd: Open3D point cloud
        labels: Cluster labels

    Returns:
        List of point clouds, one per cluster
    """
    clusters = []
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    for label in range(labels.max() + 1):
        mask = labels == label

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(points[mask])

        if colors is not None:
            cluster_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        clusters.append(cluster_pcd)

    return clusters
```

### Euclidean Clustering Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import open3d as o3d
import numpy as np

class ClusteringNode(Node):
    def __init__(self):
        super().__init__('clustering_node')

        self.declare_parameter('cluster_tolerance', 0.02)
        self.declare_parameter('min_cluster_size', 100)
        self.declare_parameter('max_cluster_size', 25000)

        self.subscription = self.create_subscription(
            PointCloud2,
            '/points/no_ground',
            self.callback,
            10
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/clusters/markers',
            10
        )

    def callback(self, msg):
        pcd = self.ros_to_open3d(msg)

        if len(pcd.points) < 10:
            return

        # Perform clustering
        eps = self.get_parameter('cluster_tolerance').value
        min_size = self.get_parameter('min_cluster_size').value

        labels = np.array(pcd.cluster_dbscan(
            eps=eps,
            min_points=min_size
        ))

        # Create markers for visualization
        markers = MarkerArray()

        for cluster_id in range(labels.max() + 1):
            mask = labels == cluster_id
            cluster_points = np.asarray(pcd.points)[mask]

            if len(cluster_points) < min_size:
                continue

            # Compute bounding box
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            center = (min_bound + max_bound) / 2
            size = max_bound - min_bound

            # Create marker
            marker = Marker()
            marker.header = msg.header
            marker.ns = 'clusters'
            marker.id = cluster_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = float(size[2])

            # Random color per cluster
            np.random.seed(cluster_id)
            marker.color.r = np.random.random()
            marker.color.g = np.random.random()
            marker.color.b = np.random.random()
            marker.color.a = 0.5

            markers.markers.append(marker)

        self.marker_publisher.publish(markers)
        self.get_logger().info(f'Published {len(markers.markers)} cluster markers')
```

## Normal Estimation

### Computing Surface Normals

```python
def estimate_normals(pcd, search_radius=0.1, max_nn=30):
    """
    Estimate surface normals for point cloud.

    Args:
        pcd: Open3D point cloud
        search_radius: Radius for neighbor search
        max_nn: Maximum number of neighbors

    Returns:
        Point cloud with normals
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=max_nn
        )
    )

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd


def visualize_normals(pcd, length=0.05):
    """Visualize point cloud with normals."""
    o3d.visualization.draw_geometries(
        [pcd],
        point_show_normal=True
    )
```

## 3D Object Recognition

### Bounding Box Fitting

```python
def fit_bounding_box(pcd, oriented=True):
    """
    Fit bounding box to point cloud.

    Args:
        pcd: Open3D point cloud
        oriented: If True, fit oriented bounding box

    Returns:
        Bounding box object
    """
    if oriented:
        bbox = pcd.get_oriented_bounding_box()
    else:
        bbox = pcd.get_axis_aligned_bounding_box()

    bbox.color = (1, 0, 0)
    return bbox


def get_object_dimensions(pcd):
    """
    Get object dimensions from point cloud.

    Returns:
        Dict with center, dimensions, and volume
    """
    bbox = pcd.get_axis_aligned_bounding_box()

    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    center = bbox.get_center()
    extent = bbox.get_extent()

    return {
        'center': center,
        'dimensions': extent,
        'volume': extent[0] * extent[1] * extent[2],
        'min_bound': min_bound,
        'max_bound': max_bound
    }
```

### Simple Object Classifier

```python
class SimpleObjectClassifier:
    """Classify objects based on geometric properties."""

    def __init__(self):
        # Define object templates (dimensions in meters)
        self.templates = {
            'cup': {'height': (0.08, 0.15), 'width': (0.06, 0.10)},
            'bottle': {'height': (0.15, 0.30), 'width': (0.05, 0.08)},
            'box': {'height': (0.05, 0.20), 'width': (0.10, 0.30)},
            'ball': {'height': (0.05, 0.15), 'width': (0.05, 0.15)},
        }

    def classify(self, pcd):
        """
        Classify object based on bounding box dimensions.

        Returns:
            object_class: Predicted class name
            confidence: Classification confidence
        """
        dims = get_object_dimensions(pcd)
        height = dims['dimensions'][2]
        width = max(dims['dimensions'][0], dims['dimensions'][1])

        # Check aspect ratio for ball detection
        aspect_ratio = height / width if width > 0 else 0

        best_match = 'unknown'
        best_score = 0

        for obj_class, template in self.templates.items():
            h_min, h_max = template['height']
            w_min, w_max = template['width']

            # Score based on dimension match
            h_score = 1.0 if h_min <= height <= h_max else 0.0
            w_score = 1.0 if w_min <= width <= w_max else 0.0

            # Special handling for ball (aspect ratio close to 1)
            if obj_class == 'ball':
                ar_score = 1.0 if 0.8 <= aspect_ratio <= 1.2 else 0.0
                score = (h_score + w_score + ar_score) / 3
            else:
                score = (h_score + w_score) / 2

            if score > best_score:
                best_score = score
                best_match = obj_class

        return best_match, best_score
```

## Complete Perception Pipeline

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
import open3d as o3d
import numpy as np
import json

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')

        self.classifier = SimpleObjectClassifier()

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, '/perception/markers', 10
        )

        self.detection_pub = self.create_publisher(
            String, '/perception/detections', 10
        )

    def callback(self, msg):
        # Convert to Open3D
        pcd = self.ros_to_open3d(msg)

        if len(pcd.points) < 100:
            return

        # Pipeline steps
        # 1. Downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        # 2. Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(20, 2.0)

        # 3. Filter by height (keep table-top objects)
        pcd = self.passthrough_filter(pcd, 'z', 0.5, 1.5)

        # 4. Remove ground/table plane
        if len(pcd.points) > 100:
            plane_model, inliers = pcd.segment_plane(0.01, 3, 1000)
            pcd = pcd.select_by_index(inliers, invert=True)

        # 5. Cluster objects
        if len(pcd.points) < 50:
            return

        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50))

        # 6. Process each cluster
        detections = []
        markers = MarkerArray()

        for cluster_id in range(labels.max() + 1):
            mask = labels == cluster_id
            cluster_pcd = pcd.select_by_index(np.where(mask)[0])

            if len(cluster_pcd.points) < 50:
                continue

            # Classify object
            obj_class, confidence = self.classifier.classify(cluster_pcd)

            # Get properties
            dims = get_object_dimensions(cluster_pcd)

            detection = {
                'id': cluster_id,
                'class': obj_class,
                'confidence': confidence,
                'center': dims['center'].tolist(),
                'dimensions': dims['dimensions'].tolist()
            }
            detections.append(detection)

            # Create visualization marker
            marker = self.create_marker(
                msg.header, cluster_id, dims, obj_class
            )
            markers.markers.append(marker)

        # Publish results
        self.marker_pub.publish(markers)

        detection_msg = String()
        detection_msg.data = json.dumps(detections)
        self.detection_pub.publish(detection_msg)

        self.get_logger().info(f'Detected {len(detections)} objects')

    def create_marker(self, header, id, dims, obj_class):
        marker = Marker()
        marker.header = header
        marker.ns = 'objects'
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = dims['center'][0]
        marker.pose.position.y = dims['center'][1]
        marker.pose.position.z = dims['center'][2]
        marker.pose.orientation.w = 1.0

        marker.scale.x = dims['dimensions'][0]
        marker.scale.y = dims['dimensions'][1]
        marker.scale.z = dims['dimensions'][2]

        # Color by class
        colors = {
            'cup': (0.0, 1.0, 0.0),
            'bottle': (0.0, 0.0, 1.0),
            'box': (1.0, 1.0, 0.0),
            'ball': (1.0, 0.0, 1.0),
            'unknown': (0.5, 0.5, 0.5)
        }
        r, g, b = colors.get(obj_class, (0.5, 0.5, 0.5))
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.7

        return marker

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this lesson, you learned:

- Point cloud data structures and ROS 2 integration
- Filtering and downsampling techniques
- Plane detection using RANSAC
- Object clustering with DBSCAN
- Normal estimation for surface analysis
- Simple 3D object recognition based on geometry

## Next Steps

Continue to [Week 3 Exercises](/module-1/week-3/exercises) to practice perception pipeline implementation.
