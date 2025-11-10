"""
Camera Intrinsic Parameters

This module defines the CameraIntrinsic class for handling camera intrinsic 
parameters including K matrix, distortion coefficients, and image dimensions.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class CameraIntrinsic:
    """
    Camera intrinsic parameters.
    
    Attributes:
        K: 3x3 camera intrinsic matrix
        distortion: Distortion coefficients array
        width: Image width in pixels
        height: Image height in pixels
    """
    
    def __init__(self, 
                 K: Optional[np.ndarray] = None,
                 distortion: Optional[np.ndarray] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None):
        """
        Initialize camera intrinsic parameters.
        
        Args:
            K: 3x3 camera intrinsic matrix
            distortion: Distortion coefficients array
            width: Image width in pixels
            height: Image height in pixels
        """
        self.K = K
        self.distortion = distortion
        self.width = width
        self.height = height
    
    @classmethod
    def from_json(cls, config: Dict, camera_name: Optional[str] = None) -> 'CameraIntrinsic':
        """
        Create CameraIntrinsic from JSON configuration.
        
        Args:
            config: Configuration dictionary with 'param' key containing:
                - cam_matrix: Camera matrix data
                - cam_dist: Distortion coefficients
                - width: Image width
                - height: Image height
            camera_name: Optional camera name to determine distortion model type
        
        Returns:
            CameraIntrinsic instance
        """
        param = config.get("param", {})
        
        # Load camera matrix (K)
        K = None
        if "cam_matrix" in param:
            cam_matrix_data = param["cam_matrix"]["data"]
            K = np.array(cam_matrix_data, dtype=np.float32)
        
        # Load distortion coefficients
        distortion = None
        if "cam_dist" in param:
            cam_dist_data = param["cam_dist"]["data"]
            
            # Determine distortion model based on camera name
            is_fisheye = False
            if camera_name:
                is_fisheye = "fisheye" in camera_name.lower()
            
            if is_fisheye:
                # Fisheye model: take first 4 coefficients
                if len(cam_dist_data) >= 4:
                    distortion = np.array(cam_dist_data[:4], dtype=np.float32)
                else:
                    # Pad with zeros if less than 4
                    dist_coeffs = list(cam_dist_data) + [0.0] * (4 - len(cam_dist_data))
                    distortion = np.array(dist_coeffs, dtype=np.float32)
            else:
                # Other cameras: take first 8 coefficients
                if len(cam_dist_data) >= 8:
                    distortion = np.array(cam_dist_data[:8], dtype=np.float32)
                else:
                    # Pad with zeros if less than 8
                    dist_coeffs = list(cam_dist_data) + [0.0] * (8 - len(cam_dist_data))
                    distortion = np.array(dist_coeffs, dtype=np.float32)
        
        # Load image dimensions
        width = param.get("width")
        height = param.get("height")
        
        return cls(K=K, distortion=distortion, width=width, height=height)
    
    def get_focal_length(self) -> Optional[Tuple[float, float]]:
        """
        Get focal length (fx, fy) from intrinsic matrix.
        
        Returns:
            Tuple of (fx, fy) or None if K is not available
        """
        if self.K is not None:
            return (self.K[0, 0], self.K[1, 1])
        return None
    
    def get_principal_point(self) -> Optional[Tuple[float, float]]:
        """
        Get principal point (cx, cy) from intrinsic matrix.
        
        Returns:
            Tuple of (cx, cy) or None if K is not available
        """
        if self.K is not None:
            return (self.K[0, 2], self.K[1, 2])
        return None
    
    def get_image_size(self) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions.
        
        Returns:
            Tuple of (width, height) or None if not available
        """
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None
    
    def is_valid(self) -> bool:
        """
        Check if intrinsic parameters are valid.
        
        Returns:
            True if K matrix is available
        """
        return self.K is not None
    
    def __repr__(self) -> str:
        """String representation of CameraIntrinsic."""
        size_str = f"{self.width}x{self.height}" if self.width and self.height else "Unknown"
        return f"CameraIntrinsic(K={self.K is not None}, distortion={self.distortion is not None}, size={size_str})"

