"""
Calibration Manager Package for GACRND Dataset

This package provides classes for managing sensor calibrations:
- CameraIntrinsic: Camera intrinsic parameters (K, distortion, image size)
- SensorExtrinsic: Sensor extrinsic parameters (sensor-to-car transformation)
- CameraCalibration: Complete camera calibration (intrinsic + extrinsic)
- LidarCalibration: Lidar calibration (extrinsic only)
- CalibrationManager: Manager for all sensor calibrations
"""

from .camera_intrinsic import CameraIntrinsic
from .sensor_extrinsic import SensorExtrinsic
from .camera_calibration import CameraCalibration
from .lidar_calibration import LidarCalibration
from .calibration_manager import CalibrationManager, load_calibration_manager
# Import unified device names (includes backward compatibility aliases)
from .device_names import (
    DeviceNames,
    DeviceType,
    CameraNames,  # Backward compatibility alias
    LidarNames,   # Backward compatibility alias
    get_calib_name,
    get_standard_name,
    normalize_device_name,
    is_valid_device_name,
    get_device_type,
    is_camera,
    is_lidar,
    # Backward compatibility functions
    normalize_camera_name,
    is_valid_camera_name,
    get_lidar_calib_name,
    get_lidar_standard_name,
    normalize_lidar_name,
    is_valid_lidar_name,
)

__all__ = [
    'CameraIntrinsic',
    'SensorExtrinsic',
    'CameraCalibration',
    'LidarCalibration',
    'CalibrationManager',
    'load_calibration_manager',
    # Unified device names
    'DeviceNames',
    'DeviceType',
    # Backward compatibility
    'CameraNames',
    'LidarNames',
    # Unified functions
    'get_calib_name',
    'get_standard_name',
    'normalize_device_name',
    'is_valid_device_name',
    'get_device_type',
    'is_camera',
    'is_lidar',
    # Backward compatibility functions
    'normalize_camera_name',
    'is_valid_camera_name',
    'get_lidar_calib_name',
    'get_lidar_standard_name',
    'normalize_lidar_name',
    'is_valid_lidar_name',
]

