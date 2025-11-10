"""
Sensor Extrinsic Parameters

This module defines the SensorExtrinsic class for handling sensor extrinsic 
parameters (sensor-to-car transformation matrices).
"""

from typing import Dict, Optional
import numpy as np


class SensorExtrinsic:
    """
    Sensor extrinsic parameters (sensor-to-car transformation).
    
    Attributes:
        sensor2car: 4x4 transformation matrix from sensor to car frame
    """
    
    def __init__(self, sensor2car: Optional[np.ndarray] = None):
        """
        Initialize sensor extrinsic parameters.
        
        Args:
            sensor2car: 4x4 transformation matrix (sensor-to-car)
        """
        self.sensor2car = sensor2car
    
    @classmethod
    def from_json(cls, config: Dict) -> 'SensorExtrinsic':
        """
        Create SensorExtrinsic from JSON configuration.
        
        Args:
            config: Configuration dictionary with 'param' key containing:
                - sensor_calib: 4x4 transformation matrix data
        
        Returns:
            SensorExtrinsic instance
        """
        param = config.get("param", {})
        
        sensor2car = None
        if "sensor_calib" in param:
            calib_data = param["sensor_calib"]["data"]
            sensor2car = np.array(calib_data, dtype=np.float32)
        
        return cls(sensor2car=sensor2car)
    
    def get_car2sensor(self) -> Optional[np.ndarray]:
        """
        Get car-to-sensor transformation (inverse of sensor-to-car).
        
        Returns:
            4x4 transformation matrix (car-to-sensor) or None if not available
        """
        if self.sensor2car is not None:
            return np.linalg.inv(self.sensor2car)
        return None
    
    def is_valid(self) -> bool:
        """
        Check if extrinsic parameters are valid.
        
        Returns:
            True if sensor2car matrix is available
        """
        return self.sensor2car is not None
    
    def __repr__(self) -> str:
        """String representation of SensorExtrinsic."""
        return f"SensorExtrinsic(sensor2car={self.sensor2car is not None})"

