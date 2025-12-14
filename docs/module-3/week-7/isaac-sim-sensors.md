---
sidebar_position: 2
title: Isaac Sim Sensors
description: Simulating cameras, LiDAR, and IMU in Isaac Sim
---

# Sensor Simulation in Isaac Sim

Learn to add and configure high-fidelity sensors in Isaac Sim with RTX-accelerated rendering.

## Learning Objectives

By the end of this lesson, you will:

- Add cameras with RTX ray tracing
- Configure LiDAR and depth sensors
- Set up IMU and contact sensors
- Access sensor data programmatically

## Camera Sensors

### RGB Camera

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np

# Create world
world = World()
world.scene.add_default_ground_plane()

# Create camera
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([3.0, 0.0, 1.0]),
    frequency=30,
    resolution=(640, 480),
    orientation=np.array([0.5, -0.5, 0.5, -0.5])  # Looking at origin
)

world.scene.add(camera)
world.reset()

# Initialize camera
camera.initialize()

# Get camera data
for i in range(100):
    world.step(render=True)

    # Get RGB image
    rgb_image = camera.get_rgba()[:, :, :3]
    print(f"RGB shape: {rgb_image.shape}")

simulation_app.close()
```

### Depth Camera

```python
from omni.isaac.sensor import Camera

# Create depth camera
depth_camera = Camera(
    prim_path="/World/DepthCamera",
    position=np.array([2.0, 0.0, 1.0]),
    frequency=30,
    resolution=(640, 480)
)

world.scene.add(depth_camera)
depth_camera.initialize()

# After stepping the simulation
depth_image = depth_camera.get_depth()
print(f"Depth shape: {depth_image.shape}")
print(f"Depth range: [{depth_image.min():.2f}, {depth_image.max():.2f}]")
```

### Camera Parameters

```python
# Get camera parameters
intrinsics = camera.get_intrinsics_matrix()
print(f"Intrinsics:\n{intrinsics}")

# Focal length
focal_length = camera.get_focal_length()
print(f"Focal length: {focal_length}mm")

# Field of view
fov_h = camera.get_horizontal_fov()
fov_v = camera.get_vertical_fov()
print(f"FOV: {fov_h:.1f}° x {fov_v:.1f}°")
```

### RTX Features

```python
from omni.isaac.sensor import Camera
from pxr import UsdGeom

# Create camera with RTX rendering
camera = Camera(
    prim_path="/World/RTXCamera",
    position=np.array([3.0, 0.0, 1.5]),
    frequency=30,
    resolution=(1920, 1080)
)

# Enable RTX features via USD
camera_prim = camera.prim
camera_api = UsdGeom.Camera(camera_prim)

# Enable depth of field
# Set aperture and focus distance for bokeh effect
```

### Multiple Render Products

```python
from omni.isaac.sensor import Camera

# Camera with multiple outputs
camera = Camera(
    prim_path="/World/MultiCamera",
    position=np.array([2.0, 0.0, 1.0]),
    frequency=30,
    resolution=(640, 480)
)

# After initialization and stepping
rgb = camera.get_rgba()
depth = camera.get_depth()
normals = camera.get_normals()
semantic = camera.get_semantic_segmentation()
instance = camera.get_instance_segmentation()
motion_vectors = camera.get_motion_vectors()

print(f"Available render products:")
print(f"  RGB: {rgb.shape}")
print(f"  Depth: {depth.shape}")
print(f"  Normals: {normals.shape}")
```

## LiDAR Sensors

### Rotating LiDAR

```python
from omni.isaac.sensor import RotatingLidarPhysX
import numpy as np

# Create rotating LiDAR
lidar = RotatingLidarPhysX(
    prim_path="/World/Lidar",
    name="lidar",
    position=np.array([0, 0, 0.5]),
    rotation=np.array([0, 0, 0, 1]),
    translation=np.array([0, 0, 0])
)

# Configure LiDAR parameters
lidar.set_fov([360.0, 30.0])  # Horizontal, Vertical FOV
lidar.set_resolution([0.4, 0.4])  # Angular resolution
lidar.set_valid_range([0.1, 100.0])  # Min, Max range
lidar.set_rotation_frequency(10)  # 10 Hz rotation

world.scene.add(lidar)
world.reset()
lidar.initialize()

# Get LiDAR data
for i in range(100):
    world.step(render=True)

    # Get point cloud
    point_cloud = lidar.get_current_frame()

    if point_cloud is not None:
        print(f"Points: {len(point_cloud['points'])}")
        print(f"Intensities: {point_cloud['intensities'].shape}")
```

### RTX LiDAR (Ray-Traced)

```python
from omni.isaac.range_sensor import _range_sensor
import omni.isaac.core.utils.prims as prim_utils

# Create RTX LiDAR for higher fidelity
lidar_config = {
    "prim_path": "/World/RTXLidar",
    "translation": [0, 0, 0.5],
    "orientation": [1, 0, 0, 0],
}

# Create the LiDAR prim
lidar_prim = prim_utils.create_prim(
    lidar_config["prim_path"],
    prim_type="Lidar"
)

# Configure via RTX range sensor interface
range_sensor = _range_sensor.acquire_lidar_sensor_interface()

# Supported LiDAR profiles:
# - Velodyne VLP-16
# - Velodyne VLP-32C
# - Ouster OS0-128
# - Custom configurations
```

### LiDAR Point Cloud Processing

```python
import numpy as np

def process_lidar_data(lidar_data):
    """Process LiDAR point cloud data."""
    points = np.array(lidar_data['points'])
    intensities = np.array(lidar_data['intensities'])

    # Filter by range
    distances = np.linalg.norm(points, axis=1)
    valid_mask = (distances > 0.1) & (distances < 50.0)

    filtered_points = points[valid_mask]
    filtered_intensities = intensities[valid_mask]

    # Compute statistics
    stats = {
        'num_points': len(filtered_points),
        'min_range': distances[valid_mask].min() if len(filtered_points) > 0 else 0,
        'max_range': distances[valid_mask].max() if len(filtered_points) > 0 else 0,
        'mean_intensity': filtered_intensities.mean() if len(filtered_points) > 0 else 0
    }

    return filtered_points, filtered_intensities, stats
```

## IMU Sensor

### Contact-Based IMU

```python
from omni.isaac.sensor import IMUSensor
import numpy as np

# Create IMU
imu = IMUSensor(
    prim_path="/World/Robot/base_link/IMU",
    name="imu",
    frequency=100,  # 100 Hz
    translation=np.array([0, 0, 0]),
    orientation=np.array([1, 0, 0, 0])
)

world.scene.add(imu)
world.reset()
imu.initialize()

# Get IMU readings
for i in range(100):
    world.step(render=True)

    imu_data = imu.get_current_frame()

    if imu_data is not None:
        linear_acc = imu_data['lin_acc']
        angular_vel = imu_data['ang_vel']
        orientation = imu_data['orientation']

        print(f"Linear Acc: {linear_acc}")
        print(f"Angular Vel: {angular_vel}")
        print(f"Orientation: {orientation}")
```

### IMU Noise Model

```python
from omni.isaac.sensor import IMUSensor

# IMU with realistic noise
imu = IMUSensor(
    prim_path="/World/Robot/IMU",
    name="imu_noisy",
    frequency=100
)

# Configure noise parameters
imu.set_noise_parameters(
    linear_acceleration_stddev=0.02,  # m/s²
    angular_velocity_stddev=0.001,     # rad/s
    orientation_stddev=0.001           # rad
)
```

## Contact Sensor

```python
from omni.isaac.sensor import ContactSensor

# Create contact sensor on robot foot
contact_sensor = ContactSensor(
    prim_path="/World/Robot/foot_link/ContactSensor",
    name="foot_contact",
    min_threshold=0.1,   # Minimum force to register contact
    max_threshold=1000,  # Maximum force
    radius=0.02,         # Sensing radius
    translation=np.array([0, 0, -0.02])
)

world.scene.add(contact_sensor)
world.reset()
contact_sensor.initialize()

# Check for contacts
for i in range(100):
    world.step(render=True)

    contact_data = contact_sensor.get_current_frame()

    if contact_data is not None:
        is_in_contact = contact_data['in_contact']
        force = contact_data['force']
        normal = contact_data['normal']

        if is_in_contact:
            print(f"Contact detected! Force: {force}, Normal: {normal}")
```

## Force/Torque Sensor

```python
from omni.isaac.sensor import ForceSensor

# Create force/torque sensor at joint
force_sensor = ForceSensor(
    prim_path="/World/Robot/wrist_joint/ForceSensor",
    name="wrist_force",
    frequency=100
)

world.scene.add(force_sensor)
world.reset()
force_sensor.initialize()

# Read forces and torques
for i in range(100):
    world.step(render=True)

    data = force_sensor.get_current_frame()

    if data is not None:
        force = data['force']    # [Fx, Fy, Fz]
        torque = data['torque']  # [Tx, Ty, Tz]

        print(f"Force: {force}, Torque: {torque}")
```

## Complete Robot with Sensors

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera, RotatingLidarPhysX, IMUSensor
import numpy as np

class SensorRobot:
    def __init__(self, world, robot_prim_path):
        self.world = world
        self.robot_path = robot_prim_path

        # Create sensors
        self._setup_sensors()

    def _setup_sensors(self):
        # Front camera
        self.camera = Camera(
            prim_path=f"{self.robot_path}/camera_link/Camera",
            position=np.array([0.1, 0, 0.1]),
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # LiDAR
        self.lidar = RotatingLidarPhysX(
            prim_path=f"{self.robot_path}/lidar_link/Lidar",
            name="lidar",
            position=np.array([0, 0, 0.3])
        )
        self.lidar.set_fov([360.0, 30.0])
        self.lidar.set_resolution([0.4, 1.0])
        self.world.scene.add(self.lidar)

        # IMU
        self.imu = IMUSensor(
            prim_path=f"{self.robot_path}/base_link/IMU",
            name="imu",
            frequency=100
        )
        self.world.scene.add(self.imu)

    def initialize(self):
        self.camera.initialize()
        self.lidar.initialize()
        self.imu.initialize()

    def get_observations(self):
        return {
            'rgb': self.camera.get_rgba()[:, :, :3],
            'depth': self.camera.get_depth(),
            'lidar': self.lidar.get_current_frame(),
            'imu': self.imu.get_current_frame()
        }

# Usage
world = World()
world.scene.add_default_ground_plane()

# Add robot (assume URDF imported)
# sensor_robot = SensorRobot(world, "/World/Robot")

world.reset()
# sensor_robot.initialize()

for i in range(100):
    world.step(render=True)
    # obs = sensor_robot.get_observations()

simulation_app.close()
```

## Synthetic Data Generation

### Semantic Segmentation

```python
from omni.isaac.sensor import Camera
import omni.replicator.core as rep

# Create camera with annotations
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([3.0, 0.0, 1.5]),
    frequency=30,
    resolution=(1280, 720)
)

# Get semantic segmentation
semantic = camera.get_semantic_segmentation()

# Class ID mapping
class_mapping = {
    0: "background",
    1: "robot",
    2: "obstacle",
    3: "floor",
    4: "wall"
}
```

### Using Replicator

```python
import omni.replicator.core as rep

# Define camera
camera = rep.create.camera(position=(3, 0, 1.5))

# Define render product
render_product = rep.create.render_product(camera, (1280, 720))

# Define writers for different annotations
rgb_writer = rep.WriterRegistry.get("BasicWriter")
rgb_writer.initialize(
    output_dir="./output/rgb",
    rgb=True
)

# Semantic segmentation writer
semantic_writer = rep.WriterRegistry.get("BasicWriter")
semantic_writer.initialize(
    output_dir="./output/semantic",
    semantic_segmentation=True
)

# Attach writers
rgb_writer.attach([render_product])
semantic_writer.attach([render_product])

# Generate frames
for i in range(100):
    rep.orchestrator.step()
```

## Summary

In this lesson, you learned:

- Creating and configuring cameras with RTX features
- Setting up rotating and RTX LiDAR sensors
- Configuring IMU sensors with noise models
- Using contact and force/torque sensors
- Generating synthetic data for ML training

## Next Steps

Continue to [Isaac ROS Bridge](/module-3/week-7/isaac-ros-bridge) to integrate with ROS 2.
