"""
Camera Calibration

This module defines the CameraCalibration class for complete camera calibration,
combining intrinsic and extrinsic parameters.
"""

from typing import Optional, Tuple
import numpy as np
from .camera_intrinsic import CameraIntrinsic
from .sensor_extrinsic import SensorExtrinsic


class CameraCalibration:
    """
    Complete camera calibration including intrinsic and extrinsic parameters.
    
    Attributes:
        name: Camera name
        intrinsic: CameraIntrinsic instance
        extrinsic: SensorExtrinsic instance (camera-to-car)
    """
    
    def __init__(self, 
                 name: str,
                 intrinsic: Optional[CameraIntrinsic] = None,
                 extrinsic: Optional[SensorExtrinsic] = None):
        """
        Initialize camera calibration.
        
        Args:
            name: Camera name
            intrinsic: CameraIntrinsic instance
            extrinsic: SensorExtrinsic instance (camera-to-car)
        """
        self.name = name
        self.intrinsic = intrinsic if intrinsic is not None else CameraIntrinsic()
        self.extrinsic = extrinsic if extrinsic is not None else SensorExtrinsic()
    
    def get_intrinsic(self) -> Optional[np.ndarray]:
        """
        Get camera intrinsic matrix (K).
        
        Returns:
            3x3 intrinsic matrix or None
        """
        return self.intrinsic.K if self.intrinsic else None
    
    def get_distortion(self) -> Optional[np.ndarray]:
        """
        Get camera distortion coefficients.
        
        Returns:
            Distortion coefficients array or None
        """
        return self.intrinsic.distortion if self.intrinsic else None
    
    def get_extrinsic(self) -> Optional[np.ndarray]:
        """
        Get camera-to-car transformation matrix.
        
        Returns:
            4x4 transformation matrix (camera-to-car) or None
        """
        return self.extrinsic.sensor2car if self.extrinsic else None
    
    def get_camera2car(self) -> Optional[np.ndarray]:
        """
        Alias for get_extrinsic. Get camera-to-car transformation.
        
        Returns:
            4x4 transformation matrix (camera-to-car) or None
        """
        return self.get_extrinsic()
    
    def get_car2camera(self) -> Optional[np.ndarray]:
        """
        Get car-to-camera transformation (inverse of camera-to-car).
        
        Returns:
            4x4 transformation matrix (car-to-camera) or None
        """
        return self.extrinsic.get_car2sensor() if self.extrinsic else None
    
    def get_image_size(self) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions.
        
        Returns:
            Tuple of (width, height) or None
        """
        return self.intrinsic.get_image_size() if self.intrinsic else None
    
    def is_valid(self) -> bool:
        """
        Check if camera calibration is valid (has both intrinsic and extrinsic).
        
        Returns:
            True if both intrinsic and extrinsic are valid
        """
        intrinsic_valid = self.intrinsic.is_valid() if self.intrinsic else False
        extrinsic_valid = self.extrinsic.is_valid() if self.extrinsic else False
        return intrinsic_valid and extrinsic_valid
    
    def update_intrinsic(self, intrinsic: CameraIntrinsic):
        """Update camera intrinsic parameters."""
        self.intrinsic = intrinsic
    
    def update_extrinsic(self, extrinsic: SensorExtrinsic):
        """Update camera extrinsic parameters."""
        self.extrinsic = extrinsic
    
    def __repr__(self) -> str:
        """String representation of CameraCalibration."""
        return f"CameraCalibration(name={self.name}, intrinsic={self.intrinsic.is_valid()}, extrinsic={self.extrinsic.is_valid()})"

