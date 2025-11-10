"""
Lidar Calibration

This module defines the LidarCalibration class for lidar calibration 
(extrinsic parameters only).
"""

from typing import Optional
import numpy as np
from .sensor_extrinsic import SensorExtrinsic


class LidarCalibration:
    """
    Lidar calibration (extrinsic parameters only).
    
    Attributes:
        name: Lidar name
        extrinsic: SensorExtrinsic instance (lidar-to-car)
    """
    
    def __init__(self, 
                 name: str,
                 extrinsic: Optional[SensorExtrinsic] = None):
        """
        Initialize lidar calibration.
        
        Args:
            name: Lidar name
            extrinsic: SensorExtrinsic instance (lidar-to-car)
        """
        self.name = name
        self.extrinsic = extrinsic if extrinsic is not None else SensorExtrinsic()
    
    def get_extrinsic(self) -> Optional[np.ndarray]:
        """
        Get lidar-to-car transformation matrix.
        
        Returns:
            4x4 transformation matrix (lidar-to-car) or None
        """
        return self.extrinsic.sensor2car if self.extrinsic else None
    
    def get_lidar2car(self) -> Optional[np.ndarray]:
        """
        Alias for get_extrinsic. Get lidar-to-car transformation.
        
        Returns:
            4x4 transformation matrix (lidar-to-car) or None
        """
        return self.get_extrinsic()
    
    def get_car2lidar(self) -> Optional[np.ndarray]:
        """
        Get car-to-lidar transformation (inverse of lidar-to-car).
        
        Returns:
            4x4 transformation matrix (car-to-lidar) or None
        """
        return self.extrinsic.get_car2sensor() if self.extrinsic else None
    
    def is_valid(self) -> bool:
        """
        Check if lidar calibration is valid.
        
        Returns:
            True if extrinsic is valid
        """
        return self.extrinsic.is_valid() if self.extrinsic else False
    
    def update_extrinsic(self, extrinsic: SensorExtrinsic):
        """Update lidar extrinsic parameters."""
        self.extrinsic = extrinsic
    
    def __repr__(self) -> str:
        """String representation of LidarCalibration."""
        return f"LidarCalibration(name={self.name}, extrinsic={self.extrinsic.is_valid()})"

