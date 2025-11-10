"""
Device Names Mapping

This module provides unified device name constants and mapping utilities 
for all sensors (cameras and LiDARs) in the GACRND dataset.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


class DeviceType(Enum):
    """Device type enumeration."""
    CAMERA = "camera"
    LIDAR = "lidar"


class DeviceNames:
    """
    Unified device name constants and mappings for GACRND dataset.
    
    This class provides standardized device names for all sensors (cameras and LiDARs)
    used in the dataset, along with mappings to calibration file names.
    """
    
    # ==================== Camera Names ====================
    CAMERA_BACK = "back"
    CAMERA_FISHEYE_BACK = "fisheye_back"
    CAMERA_FISHEYE_FRONT = "fisheye_front"
    CAMERA_FISHEYE_LEFT = "fisheye_left"
    CAMERA_FISHEYE_RIGHT = "fisheye_right"
    CAMERA_FRONT_NARROW = "front_narrow"
    CAMERA_FRONT_WIDE = "front_wide"
    CAMERA_LEFT_BACK = "left_back"
    CAMERA_LEFT_FRONT = "left_front"
    CAMERA_RIGHT_BACK = "right_back"
    CAMERA_RIGHT_FRONT = "right_front"
    
    # All camera names
    ALL_CAMERAS = [
        CAMERA_BACK,
        CAMERA_FISHEYE_BACK,
        CAMERA_FISHEYE_FRONT,
        CAMERA_FISHEYE_LEFT,
        CAMERA_FISHEYE_RIGHT,
        CAMERA_FRONT_NARROW,
        CAMERA_FRONT_WIDE,
        CAMERA_LEFT_BACK,
        CAMERA_LEFT_FRONT,
        CAMERA_RIGHT_BACK,
        CAMERA_RIGHT_FRONT,
    ]
    
    # ==================== LiDAR Names ====================
    LIDAR_FRONT = "lidar_front"
    LIDAR_TOP = "lidar_top"
    
    # All LiDAR names
    ALL_LIDARS = [
        LIDAR_FRONT,
        LIDAR_TOP,
    ]
    
    # ==================== All Devices ====================
    ALL_DEVICES = ALL_CAMERAS + ALL_LIDARS
    
    # ==================== Name to Calibration Mapping ====================
    # Camera mappings
    CAMERA_NAME_TO_CALIB: Dict[str, str] = {
        CAMERA_BACK: "camera-back",
        CAMERA_FISHEYE_BACK: "camera-fisheye-back",
        CAMERA_FISHEYE_FRONT: "camera-fisheye-front",
        CAMERA_FISHEYE_LEFT: "camera-fisheye-left",
        CAMERA_FISHEYE_RIGHT: "camera-fisheye-right",
        CAMERA_FRONT_NARROW: "camera-front-narrow",
        CAMERA_FRONT_WIDE: "camera-front-wide",
        CAMERA_LEFT_BACK: "camera-left-back",
        CAMERA_LEFT_FRONT: "camera-left-front",
        CAMERA_RIGHT_BACK: "camera-right-back",
        CAMERA_RIGHT_FRONT: "camera-right-front",
    }
    
    # LiDAR mappings
    LIDAR_NAME_TO_CALIB: Dict[str, str] = {
        LIDAR_FRONT: "lidar-front",
        LIDAR_TOP: "lidar-top",
    }
    
    # Combined mapping
    NAME_TO_CALIB: Dict[str, str] = {
        **CAMERA_NAME_TO_CALIB,
        **LIDAR_NAME_TO_CALIB,
    }
    
    # Reverse mapping: calibration name to standard name
    CALIB_TO_NAME: Dict[str, str] = {v: k for k, v in NAME_TO_CALIB.items()}
    
    # Device type mapping: standard name -> device type
    NAME_TO_TYPE: Dict[str, DeviceType] = {
        **{name: DeviceType.CAMERA for name in ALL_CAMERAS},
        **{name: DeviceType.LIDAR for name in ALL_LIDARS},
    }
    
    @classmethod
    def get_device_type(cls, device_name: str) -> Optional[DeviceType]:
        """
        Get device type (camera or lidar) from device name.
        
        Args:
            device_name: Standard device name or calibration name
        
        Returns:
            DeviceType or None if not found
        """
        # Try to normalize first
        normalized = cls.normalize_name(device_name)
        if normalized:
            return cls.NAME_TO_TYPE.get(normalized)
        
        # Check calibration name
        standard_name = cls.get_standard_name(device_name)
        if standard_name:
            return cls.NAME_TO_TYPE.get(standard_name)
        
        return None
    
    @classmethod
    def is_camera(cls, device_name: str) -> bool:
        """
        Check if device name is a camera.
        
        Args:
            device_name: Device name to check
        
        Returns:
            True if device is a camera
        """
        device_type = cls.get_device_type(device_name)
        return device_type == DeviceType.CAMERA
    
    @classmethod
    def is_lidar(cls, device_name: str) -> bool:
        """
        Check if device name is a LiDAR.
        
        Args:
            device_name: Device name to check
        
        Returns:
            True if device is a LiDAR
        """
        device_type = cls.get_device_type(device_name)
        return device_type == DeviceType.LIDAR
    
    @classmethod
    def get_calib_name(cls, standard_name: str) -> Optional[str]:
        """
        Get calibration file name from standard device name.
        
        Args:
            standard_name: Standard device name (e.g., "front_wide" or "lidar_front")
        
        Returns:
            Calibration file name (e.g., "camera-front-wide" or "lidar-front") or None if not found
        """
        return cls.NAME_TO_CALIB.get(standard_name)
    
    @classmethod
    def get_standard_name(cls, calib_name: str) -> Optional[str]:
        """
        Get standard device name from calibration file name.
        
        Args:
            calib_name: Calibration file name (e.g., "camera-front-wide" or "lidar-front")
        
        Returns:
            Standard device name (e.g., "front_wide" or "lidar_front") or None if not found
        """
        return cls.CALIB_TO_NAME.get(calib_name)
    
    @classmethod
    def is_valid(cls, device_name: str) -> bool:
        """
        Check if a device name is valid.
        
        Args:
            device_name: Device name to check
        
        Returns:
            True if the name is valid
        """
        return device_name in cls.ALL_DEVICES or device_name in cls.NAME_TO_CALIB.values()
    
    @classmethod
    def normalize_name(cls, device_name: str) -> Optional[str]:
        """
        Normalize device name to standard format.
        
        If input is a calibration name, convert to standard name.
        If input is already standard name, return as is.
        
        Args:
            device_name: Device name in any format
        
        Returns:
            Standard device name or None if not found
        """
        # Check if it's already a standard name
        if device_name in cls.ALL_DEVICES:
            return device_name
        
        # Check if it's a calibration name
        standard_name = cls.get_standard_name(device_name)
        if standard_name:
            return standard_name
        
        # Try to extract standard name from calibration name
        if device_name.startswith("camera-"):
            potential_name = device_name.replace("camera-", "").replace("-", "_")
            if potential_name in cls.ALL_CAMERAS:
                return potential_name
        elif device_name.startswith("lidar-"):
            potential_name = device_name.replace("lidar-", "lidar_")
            if potential_name in cls.ALL_LIDARS:
                return potential_name
        
        return None
    
    @classmethod
    def get_all_camera_names(cls) -> List[str]:
        """
        Get list of all standard camera names.
        
        Returns:
            List of standard camera names
        """
        return cls.ALL_CAMERAS.copy()
    
    @classmethod
    def get_all_lidar_names(cls) -> List[str]:
        """
        Get list of all standard LiDAR names.
        
        Returns:
            List of standard LiDAR names
        """
        return cls.ALL_LIDARS.copy()
    
    @classmethod
    def get_all_device_names(cls) -> List[str]:
        """
        Get list of all standard device names (cameras + LiDARs).
        
        Returns:
            List of standard device names
        """
        return cls.ALL_DEVICES.copy()
    
    @classmethod
    def get_all_calib_names(cls) -> List[str]:
        """
        Get list of all calibration file names.
        
        Returns:
            List of calibration file names
        """
        return list(cls.NAME_TO_CALIB.values())
    
    @classmethod
    def get_camera_calib_names(cls) -> List[str]:
        """
        Get list of all camera calibration file names.
        
        Returns:
            List of camera calibration file names
        """
        return list(cls.CAMERA_NAME_TO_CALIB.values())
    
    @classmethod
    def get_lidar_calib_names(cls) -> List[str]:
        """
        Get list of all LiDAR calibration file names.
        
        Returns:
            List of LiDAR calibration file names
        """
        return list(cls.LIDAR_NAME_TO_CALIB.values())


# Convenience functions
def get_calib_name(device_name: str) -> Optional[str]:
    """Get calibration file name from standard device name."""
    return DeviceNames.get_calib_name(device_name)


def get_standard_name(calib_name: str) -> Optional[str]:
    """Get standard device name from calibration file name."""
    return DeviceNames.get_standard_name(calib_name)


def normalize_device_name(device_name: str) -> Optional[str]:
    """Normalize device name to standard format."""
    return DeviceNames.normalize_name(device_name)


def is_valid_device_name(device_name: str) -> bool:
    """Check if a device name is valid."""
    return DeviceNames.is_valid(device_name)


def get_device_type(device_name: str) -> Optional[DeviceType]:
    """Get device type (camera or lidar) from device name."""
    return DeviceNames.get_device_type(device_name)


def is_camera(device_name: str) -> bool:
    """Check if device name is a camera."""
    return DeviceNames.is_camera(device_name)


def is_lidar(device_name: str) -> bool:
    """Check if device name is a LiDAR."""
    return DeviceNames.is_lidar(device_name)


# Backward compatibility: create aliases for CameraNames and LidarNames
class CameraNames:
    """Backward compatibility alias for camera-related constants."""
    BACK = DeviceNames.CAMERA_BACK
    FISHEYE_BACK = DeviceNames.CAMERA_FISHEYE_BACK
    FISHEYE_FRONT = DeviceNames.CAMERA_FISHEYE_FRONT
    FISHEYE_LEFT = DeviceNames.CAMERA_FISHEYE_LEFT
    FISHEYE_RIGHT = DeviceNames.CAMERA_FISHEYE_RIGHT
    FRONT_NARROW = DeviceNames.CAMERA_FRONT_NARROW
    FRONT_WIDE = DeviceNames.CAMERA_FRONT_WIDE
    LEFT_BACK = DeviceNames.CAMERA_LEFT_BACK
    LEFT_FRONT = DeviceNames.CAMERA_LEFT_FRONT
    RIGHT_BACK = DeviceNames.CAMERA_RIGHT_BACK
    RIGHT_FRONT = DeviceNames.CAMERA_RIGHT_FRONT
    ALL_CAMERAS = DeviceNames.ALL_CAMERAS
    NAME_TO_CALIB = DeviceNames.CAMERA_NAME_TO_CALIB
    
    @classmethod
    def get_calib_name(cls, standard_name: str) -> Optional[str]:
        return DeviceNames.CAMERA_NAME_TO_CALIB.get(standard_name)
    
    @classmethod
    def get_standard_name(cls, calib_name: str) -> Optional[str]:
        return DeviceNames.get_standard_name(calib_name)
    
    @classmethod
    def normalize_name(cls, camera_name: str) -> Optional[str]:
        return DeviceNames.normalize_name(camera_name)
    
    @classmethod
    def is_valid(cls, camera_name: str) -> bool:
        return camera_name in DeviceNames.ALL_CAMERAS or camera_name in DeviceNames.CAMERA_NAME_TO_CALIB.values()
    
    @classmethod
    def get_all_standard_names(cls) -> List[str]:
        return DeviceNames.get_all_camera_names()
    
    @classmethod
    def get_all_calib_names(cls) -> List[str]:
        return DeviceNames.get_camera_calib_names()


class LidarNames:
    """Backward compatibility alias for LiDAR-related constants."""
    FRONT = DeviceNames.LIDAR_FRONT
    TOP = DeviceNames.LIDAR_TOP
    ALL_LIDARS = DeviceNames.ALL_LIDARS
    NAME_TO_CALIB = DeviceNames.LIDAR_NAME_TO_CALIB
    
    @classmethod
    def get_calib_name(cls, standard_name: str) -> Optional[str]:
        return DeviceNames.LIDAR_NAME_TO_CALIB.get(standard_name)
    
    @classmethod
    def get_standard_name(cls, calib_name: str) -> Optional[str]:
        return DeviceNames.get_standard_name(calib_name)
    
    @classmethod
    def normalize_name(cls, lidar_name: str) -> Optional[str]:
        return DeviceNames.normalize_name(lidar_name)
    
    @classmethod
    def is_valid(cls, lidar_name: str) -> bool:
        return lidar_name in DeviceNames.ALL_LIDARS or lidar_name in DeviceNames.LIDAR_NAME_TO_CALIB.values()
    
    @classmethod
    def get_all_standard_names(cls) -> List[str]:
        return DeviceNames.get_all_lidar_names()
    
    @classmethod
    def get_all_calib_names(cls) -> List[str]:
        return DeviceNames.get_lidar_calib_names()


# Backward compatibility functions
def normalize_camera_name(camera_name: str) -> Optional[str]:
    """Normalize camera name to standard format (backward compatibility)."""
    return DeviceNames.normalize_name(camera_name)


def is_valid_camera_name(camera_name: str) -> bool:
    """Check if a camera name is valid (backward compatibility)."""
    return DeviceNames.is_camera(camera_name) and DeviceNames.is_valid(camera_name)


def get_lidar_calib_name(standard_name: str) -> Optional[str]:
    """Get calibration file name from standard LiDAR name (backward compatibility)."""
    return DeviceNames.LIDAR_NAME_TO_CALIB.get(standard_name)


def get_lidar_standard_name(calib_name: str) -> Optional[str]:
    """Get standard LiDAR name from calibration file name (backward compatibility)."""
    return DeviceNames.get_standard_name(calib_name)


def normalize_lidar_name(lidar_name: str) -> Optional[str]:
    """Normalize LiDAR name to standard format (backward compatibility)."""
    return DeviceNames.normalize_name(lidar_name)


def is_valid_lidar_name(lidar_name: str) -> bool:
    """Check if a LiDAR name is valid (backward compatibility)."""
    return DeviceNames.is_lidar(lidar_name) and DeviceNames.is_valid(lidar_name)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Device Names Summary")
    print("=" * 60)
    
    print("\nCameras:")
    for name in DeviceNames.get_all_camera_names():
        calib_name = DeviceNames.get_calib_name(name)
        print(f"  {name:20} -> {calib_name}")
    
    print("\nLiDARs:")
    for name in DeviceNames.get_all_lidar_names():
        calib_name = DeviceNames.get_calib_name(name)
        print(f"  {name:20} -> {calib_name}")
    
    print("\nTest device type detection:")
    test_devices = ["front_wide", "lidar_front", "camera-back", "lidar-top"]
    for device in test_devices:
        device_type = DeviceNames.get_device_type(device)
        normalized = DeviceNames.normalize_name(device)
        print(f"  {device:20} -> type: {device_type.value if device_type else None}, normalized: {normalized}")

