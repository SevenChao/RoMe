"""
Calibration Manager

This module defines the CalibrationManager class for managing all sensor 
calibrations in the GACRND dataset.
"""

import json
import os
from os.path import join, exists
from typing import Dict, Optional, Tuple
import numpy as np

# Handle both relative and absolute imports
try:
    from .camera_intrinsic import CameraIntrinsic
    from .sensor_extrinsic import SensorExtrinsic
    from .camera_calibration import CameraCalibration
    from .lidar_calibration import LidarCalibration
except ImportError:
    # Fallback to absolute imports for direct execution
    import sys
    from pathlib import Path
    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from calibration_manager.camera_intrinsic import CameraIntrinsic
    from calibration_manager.sensor_extrinsic import SensorExtrinsic
    from calibration_manager.camera_calibration import CameraCalibration
    from calibration_manager.lidar_calibration import LidarCalibration


class CalibrationManager:
    """
    Manages calibration parameters for all sensors in the GACRND dataset.
    
    This class loads and manages all camera and lidar calibrations from JSON files.
    It provides convenient access methods to retrieve calibration parameters.
    
    Attributes:
        calib_dir: Path to the calibration directory
        cameras: Dictionary mapping camera names to CameraCalibration instances
        lidars: Dictionary mapping lidar names to LidarCalibration instances
    """
    
    def __init__(self, calib_dir: str):
        """
        Initialize the CalibrationManager.
        
        Args:
            calib_dir: Path to the directory containing calibration JSON files
        """
        self.calib_dir = calib_dir
        self.cameras: Dict[str, CameraCalibration] = {}
        self.lidars: Dict[str, LidarCalibration] = {}
        # Sensor-to-sensor transformations (e.g., gnss-to-lidar-top, imu-to-lidar-top)
        self.sensor_to_sensor: Dict[str, SensorExtrinsic] = {}
        # IMU bias parameters (if available)
        self.imu_bias: Optional[Dict[str, Dict[str, float]]] = None
        
        # Load all calibration files
        self._load_all_calibrations()
    
    def _load_all_calibrations(self):
        """Load all calibration files from the calibration directory."""
        if not exists(self.calib_dir):
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")
        
        # Get all JSON files in the calibration directory
        calib_files = [f for f in os.listdir(self.calib_dir) if f.endswith('.json')]
        
        for calib_file in calib_files:
            file_path = join(self.calib_dir, calib_file)
            self._load_calibration_file(file_path)
    
    def _load_calibration_file(self, file_path: str):
        """
        Load a single calibration JSON file.
        
        Args:
            file_path: Path to the calibration JSON file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get filename for parsing sensor names if needed
        filename = os.path.basename(file_path)
        
        # Process each sensor configuration in the file
        for key, config in data.items():
            param_type = config.get("param_type", "")
            sensor_name = config.get("sensor_name", "")
            target_sensor_name = config.get("target_sensor_name", "")
            
            # Extract sensor names from key if not in config (for ENU files)
            if not sensor_name and "-to-" in key:
                parts = key.split("-to-")
                if len(parts) == 2:
                    sensor_name = parts[0]
                    if not target_sensor_name:
                        target_sensor_name = parts[1].replace("-enu", "").replace("_enu", "")
            
            if param_type == "intrinsic":
                # Camera intrinsic parameters
                # Skip undistort versions - we only want original intrinsic parameters
                # Check both key and sensor_name to skip undistort versions
                if "camera" in sensor_name.lower() and "-undistort" not in key.lower() and "-undistort" not in sensor_name.lower():
                    self._load_camera_intrinsic(sensor_name, config)
            
            elif param_type == "extrinsic":
                # Extrinsic parameters
                
                # Check if it's a sensor-to-car transformation
                if target_sensor_name == "car":
                    # Skip undistort versions - we only want original extrinsic parameters
                    if "-undistort" in key.lower():
                        continue  # Skip undistort versions
                    
                    # Extract sensor name from key if sensor_name is "car" (for some files)
                    # Key format: "camera-xxx-to-car" or "lidar-xxx-to-car"
                    actual_sensor_name = sensor_name
                    if sensor_name.lower() == "car" and "-to-car" in key.lower():
                        # Extract camera/lidar name from key
                        parts = key.split("-to-car")[0]
                        if "camera" in parts.lower() or "lidar" in parts.lower():
                            actual_sensor_name = parts
                    
                    # Load camera or lidar extrinsic
                    if "camera" in actual_sensor_name.lower():
                        self._load_camera_extrinsic(actual_sensor_name, config)
                    elif "lidar" in actual_sensor_name.lower():
                        self._load_lidar_extrinsic(actual_sensor_name, config)
                # Check if it's a sensor-to-sensor transformation (e.g., gnss-to-lidar-top)
                elif target_sensor_name and target_sensor_name != "car":
                    # Handle ENU variant
                    if "enu" in key.lower() or "enu" in filename.lower():
                        transform_key = f"{sensor_name}-to-{target_sensor_name}-enu"
                    else:
                        transform_key = f"{sensor_name}-to-{target_sensor_name}"
                    self._load_sensor_to_sensor_extrinsic_with_key(transform_key, config)
            
            # Handle IMU bias parameters (special case in imu-to-lidar-top.json)
            # IMU bias is stored in a separate key "imu" in the same file
            if key.lower() == "imu" and isinstance(config, dict):
                if "accelerometer_bias" in config or "gyroscope_bias" in config:
                    self._load_imu_bias({"imu": config})
    
    def _load_camera_intrinsic(self, sensor_name: str, config: Dict):
        """
        Load camera intrinsic parameters.
        
        Args:
            sensor_name: Name of the camera sensor (should not contain "-undistort")
            config: Configuration dictionary containing intrinsic parameters
        """
        # Use sensor_name directly as base_name (undistort versions are already filtered out)
        base_name = sensor_name
        
        # Create or get existing camera calibration
        if base_name not in self.cameras:
            self.cameras[base_name] = CameraCalibration(name=base_name)
        
        # Create intrinsic from config, passing camera name for distortion model detection
        intrinsic = CameraIntrinsic.from_json(config, camera_name=base_name)
        
        # Update intrinsic parameters
        self.cameras[base_name].update_intrinsic(intrinsic)
    
    def _load_camera_extrinsic(self, sensor_name: str, config: Dict):
        """
        Load camera extrinsic parameters (sensor-to-car transformation).
        
        Args:
            sensor_name: Name of the camera sensor
            config: Configuration dictionary containing extrinsic parameters
        """
        # Extract base sensor name (remove "-to-car" and "-undistort" suffixes)
        base_name = sensor_name.replace("-to-car", "").replace("-undistort", "")
        
        # Create or get existing camera calibration
        if base_name not in self.cameras:
            self.cameras[base_name] = CameraCalibration(name=base_name)
        
        # Create extrinsic from config
        extrinsic = SensorExtrinsic.from_json(config)
        self.cameras[base_name].update_extrinsic(extrinsic)
    
    def _load_lidar_extrinsic(self, sensor_name: str, config: Dict):
        """
        Load lidar extrinsic parameters (sensor-to-car transformation).
        
        Args:
            sensor_name: Name of the lidar sensor
            config: Configuration dictionary containing extrinsic parameters
        """
        # Extract base sensor name (remove "-to-car" suffix)
        base_name = sensor_name.replace("-to-car", "")
        
        # Create or get existing lidar calibration
        if base_name not in self.lidars:
            self.lidars[base_name] = LidarCalibration(name=base_name)
        
        # Create extrinsic from config
        extrinsic = SensorExtrinsic.from_json(config)
        self.lidars[base_name].update_extrinsic(extrinsic)
    
    def _load_sensor_to_sensor_extrinsic(self, sensor_name: str, target_sensor_name: str, config: Dict):
        """
        Load sensor-to-sensor extrinsic parameters (e.g., gnss-to-lidar-top, imu-to-lidar-top).
        
        Args:
            sensor_name: Name of the source sensor (e.g., "gnss", "imu")
            target_sensor_name: Name of the target sensor (e.g., "lidar-top")
            config: Configuration dictionary containing extrinsic parameters
        """
        # Create a key for sensor-to-sensor transformation
        transform_key = f"{sensor_name}-to-{target_sensor_name}"
        self._load_sensor_to_sensor_extrinsic_with_key(transform_key, config)
    
    def _load_sensor_to_sensor_extrinsic_with_key(self, transform_key: str, config: Dict):
        """
        Load sensor-to-sensor extrinsic parameters with a given transform key.
        
        Args:
            transform_key: Key for the transformation (e.g., "gnss-to-lidar-top", "gnss-to-lidar-top-enu")
            config: Configuration dictionary containing extrinsic parameters
        """
        # Create extrinsic from config
        extrinsic = SensorExtrinsic.from_json(config)
        self.sensor_to_sensor[transform_key] = extrinsic
    
    def _load_imu_bias(self, config: Dict):
        """
        Load IMU bias parameters.
        
        Args:
            config: Configuration dictionary containing IMU bias parameters
        """
        if "imu" in config:
            imu_config = config["imu"]
            self.imu_bias = {
                "accelerometer_bias": imu_config.get("accelerometer_bias", {}),
                "gyroscope_bias": imu_config.get("gyroscope_bias", {})
            }
    
    # ==================== Camera Access Methods ====================
    
    def get_camera(self, camera_name: str) -> Optional[CameraCalibration]:
        """
        Get CameraCalibration instance for a camera.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            CameraCalibration instance or None if not found
        """
        return self.cameras.get(camera_name)
    
    def get_camera_intrinsic(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get camera intrinsic matrix (K).
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            3x3 intrinsic matrix (K) or None if not found
        """
        camera = self.get_camera(camera_name)
        return camera.get_intrinsic() if camera else None
    
    def get_camera_distortion(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get camera distortion coefficients.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            Distortion coefficients array or None if not found
        """
        camera = self.get_camera(camera_name)
        return camera.get_distortion() if camera else None
    
    def get_camera_extrinsic(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get camera-to-car extrinsic transformation matrix.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            4x4 transformation matrix (camera-to-car) or None if not found
        """
        camera = self.get_camera(camera_name)
        return camera.get_extrinsic() if camera else None
    
    def get_camera2car(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Alias for get_camera_extrinsic. Get camera-to-car transformation.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            4x4 transformation matrix (camera-to-car) or None if not found
        """
        return self.get_camera_extrinsic(camera_name)
    
    def get_car2camera(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get car-to-camera transformation (inverse of camera-to-car).
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            4x4 transformation matrix (car-to-camera) or None if not found
        """
        camera = self.get_camera(camera_name)
        return camera.get_car2camera() if camera else None
    
    def get_camera_image_size(self, camera_name: str) -> Optional[Tuple[int, int]]:
        """
        Get camera image dimensions.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            Tuple of (width, height) or None if not found
        """
        camera = self.get_camera(camera_name)
        return camera.get_image_size() if camera else None
    
    # ==================== Lidar Access Methods ====================
    
    def get_lidar(self, lidar_name: str) -> Optional[LidarCalibration]:
        """
        Get LidarCalibration instance for a lidar.
        
        Args:
            lidar_name: Name of the lidar sensor
        
        Returns:
            LidarCalibration instance or None if not found
        """
        return self.lidars.get(lidar_name)
    
    def get_lidar_extrinsic(self, lidar_name: str) -> Optional[np.ndarray]:
        """
        Get lidar-to-car extrinsic transformation matrix.
        
        Args:
            lidar_name: Name of the lidar sensor
        
        Returns:
            4x4 transformation matrix (lidar-to-car) or None if not found
        """
        lidar = self.get_lidar(lidar_name)
        return lidar.get_extrinsic() if lidar else None
    
    def get_lidar2car(self, lidar_name: str) -> Optional[np.ndarray]:
        """
        Alias for get_lidar_extrinsic. Get lidar-to-car transformation.
        
        Args:
            lidar_name: Name of the lidar sensor
        
        Returns:
            4x4 transformation matrix (lidar-to-car) or None if not found
        """
        return self.get_lidar_extrinsic(lidar_name)
    
    def get_car2lidar(self, lidar_name: str) -> Optional[np.ndarray]:
        """
        Get car-to-lidar transformation (inverse of lidar-to-car).
        
        Args:
            lidar_name: Name of the lidar sensor
        
        Returns:
            4x4 transformation matrix (car-to-lidar) or None if not found
        """
        lidar = self.get_lidar(lidar_name)
        return lidar.get_car2lidar() if lidar else None
    
    # ==================== Sensor-to-Sensor Transformation Methods ====================
    
    def get_sensor_to_sensor(self, source_sensor: str, target_sensor: str) -> Optional[np.ndarray]:
        """
        Get sensor-to-sensor transformation matrix.
        
        Args:
            source_sensor: Name of the source sensor (e.g., "gnss", "imu")
            target_sensor: Name of the target sensor (e.g., "lidar-top")
        
        Returns:
            4x4 transformation matrix (source-to-target) or None if not found
        """
        transform_key = f"{source_sensor}-to-{target_sensor}"
        extrinsic = self.sensor_to_sensor.get(transform_key)
        return extrinsic.sensor2car if extrinsic and extrinsic.is_valid() else None
    
    def get_gnss_to_lidar_top(self) -> Optional[np.ndarray]:
        """
        Get GNSS-to-lidar-top transformation matrix.
        
        Returns:
            4x4 transformation matrix (GNSS-to-lidar-top) or None if not found
        """
        return self.get_sensor_to_sensor("gnss", "lidar-top")
    
    def get_imu_to_lidar_top(self) -> Optional[np.ndarray]:
        """
        Get IMU-to-lidar-top transformation matrix.
        
        Returns:
            4x4 transformation matrix (IMU-to-lidar-top) or None if not found
        """
        return self.get_sensor_to_sensor("imu", "lidar-top")
    
    def get_gnss_to_lidar_top_enu(self) -> Optional[np.ndarray]:
        """
        Get GNSS(ENU)-to-lidar-top transformation matrix.
        
        Returns:
            4x4 transformation matrix (GNSS-ENU-to-lidar-top) or None if not found
        """
        # Try exact key first
        transform_key = "gnss-to-lidar-top-enu"
        extrinsic = self.sensor_to_sensor.get(transform_key)
        if extrinsic and extrinsic.is_valid():
            return extrinsic.sensor2car
        
        # Fallback: check if there's a key containing "gnss", "enu", and "lidar"
        for key, extrinsic in self.sensor_to_sensor.items():
            key_lower = key.lower()
            if "gnss" in key_lower and "enu" in key_lower and "lidar" in key_lower:
                return extrinsic.sensor2car if extrinsic.is_valid() else None
        
        return None
    
    def get_imu_bias(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get IMU bias parameters.
        
        Returns:
            Dictionary containing accelerometer_bias and gyroscope_bias, or None if not available
        """
        return self.imu_bias
    
    def get_sensor_to_car_via_lidar_top(self, source_sensor: str) -> Optional[np.ndarray]:
        """
        Get sensor-to-car transformation via lidar-top.
        
        This is useful for sensors like GNSS and IMU that are calibrated relative to lidar-top.
        The transformation chain is: source_sensor -> lidar-top -> car
        
        Args:
            source_sensor: Name of the source sensor (e.g., "gnss", "imu")
        
        Returns:
            4x4 transformation matrix (source-to-car) or None if not found
        """
        # Get source-to-lidar-top transformation
        source_to_lidar = self.get_sensor_to_sensor(source_sensor, "lidar-top")
        if source_to_lidar is None:
            return None
        
        # Get lidar-top-to-car transformation
        lidar_top = self.get_lidar("lidar-top")
        if lidar_top is None:
            return None
        
        lidar_to_car = lidar_top.get_lidar2car()
        if lidar_to_car is None:
            return None
        
        # Combine transformations: source -> lidar-top -> car
        return lidar_to_car @ source_to_lidar
    
    # ==================== Utility Methods ====================
    
    def get_all_camera_names(self) -> list:
        """
        Get list of all available camera names.
        
        Returns:
            List of camera names
        """
        return list(self.cameras.keys())
    
    def get_all_lidar_names(self) -> list:
        """
        Get list of all available lidar names.
        
        Returns:
            List of lidar names
        """
        return list(self.lidars.keys())
    
    def print_summary(self):
        """Print a summary of all loaded calibration parameters."""
        print("=" * 60)
        print("Calibration Manager Summary")
        print("=" * 60)
        print(f"\nCalibration directory: {self.calib_dir}")
        
        print(f"\nCameras ({len(self.cameras)}):")
        for cam_name, cam_calib in self.cameras.items():
            print(f"  - {cam_name}")
            if cam_calib.intrinsic.is_valid():
                print(f"    Intrinsic: {cam_calib.intrinsic.K.shape}")
                size = cam_calib.get_image_size()
                if size:
                    print(f"    Image size: {size[0]}x{size[1]}")
            if cam_calib.extrinsic.is_valid():
                print(f"    Extrinsic: {cam_calib.extrinsic.sensor2car.shape}")
        
        print(f"\nLidars ({len(self.lidars)}):")
        for lidar_name, lidar_calib in self.lidars.items():
            print(f"  - {lidar_name}")
            if lidar_calib.extrinsic.is_valid():
                print(f"    Extrinsic: {lidar_calib.extrinsic.sensor2car.shape}")
        
        if self.sensor_to_sensor:
            print(f"\nSensor-to-Sensor Transformations ({len(self.sensor_to_sensor)}):")
            for transform_key, extrinsic in self.sensor_to_sensor.items():
                if extrinsic.is_valid():
                    print(f"  - {transform_key}: {extrinsic.sensor2car.shape}")
        
        if self.imu_bias:
            print(f"\nIMU Bias Parameters:")
            if "accelerometer_bias" in self.imu_bias:
                print(f"  - Accelerometer bias: {self.imu_bias['accelerometer_bias']}")
            if "gyroscope_bias" in self.imu_bias:
                print(f"  - Gyroscope bias: {self.imu_bias['gyroscope_bias']}")
        
        print("=" * 60)


def load_calibration_manager(calib_dir: str) -> CalibrationManager:
    """
    Convenience function to create and load a CalibrationManager.
    
    Args:
        calib_dir: Path to the calibration directory
    
    Returns:
        CalibrationManager instance
    """
    return CalibrationManager(calib_dir)


# Example usage
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for direct execution
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    import numpy as np
    
    # Default calibration directory
    default_calib_dir = "/home/stevenchao/CBZoom/Auto-Labeling/Dataset/GACRT026_1758521322/calib_extract"
    
    # Get calibration directory from command line or use default
    if len(sys.argv) > 1:
        calib_dir = sys.argv[1]
    else:
        calib_dir = default_calib_dir
    
    print("=" * 100)
    print("Calibration Manager - 完整参数打印与验证")
    print("=" * 100)
    print(f"\n标定文件目录: {calib_dir}\n")
    
    try:
        # Initialize CalibrationManager
        print("正在加载标定文件...")
        calib_manager = CalibrationManager(calib_dir)
        print("✓ 标定文件加载成功\n")
        
        # ==================== Print All Camera Parameters ====================
        print("=" * 100)
        print("相机参数详情")
        print("=" * 100)
        
        all_cameras = calib_manager.get_all_camera_names()
        print(f"\n总共 {len(all_cameras)} 个相机:\n")
        
        for idx, cam_name in enumerate(sorted(all_cameras), 1):
            camera = calib_manager.get_camera(cam_name)
            if not camera:
                print(f"{idx}. {cam_name}: ✗ 未找到")
                continue
            
            print(f"{idx}. {cam_name}")
            print("-" * 100)
            
            # Intrinsic parameters
            if camera.intrinsic.is_valid():
                K = camera.get_intrinsic()
                distortion = camera.get_distortion()
                image_size = camera.get_image_size()
                focal_length = camera.intrinsic.get_focal_length()
                principal_point = camera.intrinsic.get_principal_point()
                
                print(f"  内参矩阵 K:")
                print(f"    {K[0,0]:12.6f}  {K[0,1]:12.6f}  {K[0,2]:12.6f}")
                print(f"    {K[1,0]:12.6f}  {K[1,1]:12.6f}  {K[1,2]:12.6f}")
                print(f"    {K[2,0]:12.6f}  {K[2,1]:12.6f}  {K[2,2]:12.6f}")
                print(f"  焦距: fx={focal_length[0]:.2f}, fy={focal_length[1]:.2f}")
                print(f"  主点: cx={principal_point[0]:.2f}, cy={principal_point[1]:.2f}")
                print(f"  图像尺寸: {image_size[0]} x {image_size[1]}")
                
                if distortion is not None:
                    print(f"  畸变系数: {distortion}")
                
                # Validate intrinsic matrix
                valid_intrinsic = (
                    K[2, 2] == 1.0 and
                    K[0, 1] == 0.0 and K[1, 0] == 0.0 and
                    K[0, 0] > 0 and K[1, 1] > 0
                )
                print(f"  ✓ 内参矩阵验证: {'通过' if valid_intrinsic else '失败'}")
            else:
                print(f"  ✗ 内参: 缺失")
            
            # Extrinsic parameters
            if camera.extrinsic.is_valid():
                T_camera2car = camera.get_camera2car()
                T_car2camera = camera.get_car2camera()
                
                print(f"\n  外参矩阵 (Camera-to-Car):")
                print(f"    {T_camera2car[0,0]:12.6f}  {T_camera2car[0,1]:12.6f}  {T_camera2car[0,2]:12.6f}  {T_camera2car[0,3]:12.6f}")
                print(f"    {T_camera2car[1,0]:12.6f}  {T_camera2car[1,1]:12.6f}  {T_camera2car[1,2]:12.6f}  {T_camera2car[1,3]:12.6f}")
                print(f"    {T_camera2car[2,0]:12.6f}  {T_camera2car[2,1]:12.6f}  {T_camera2car[2,2]:12.6f}  {T_camera2car[2,3]:12.6f}")
                print(f"    {T_camera2car[3,0]:12.6f}  {T_camera2car[3,1]:12.6f}  {T_camera2car[3,2]:12.6f}  {T_camera2car[3,3]:12.6f}")
                
                # Extract rotation and translation
                R = T_camera2car[:3, :3]
                t = T_camera2car[:3, 3]
                print(f"  位置 (相对于车体): x={t[0]:.6f}, y={t[1]:.6f}, z={t[2]:.6f}")
                
                # Validate extrinsic matrix
                det_R = np.linalg.det(R)
                T_inv_check = T_camera2car @ T_car2camera
                identity = np.eye(4)
                inv_error = np.abs(T_inv_check - identity).max()
                
                valid_extrinsic = (
                    T_camera2car[3, 3] == 1.0 and
                    abs(det_R - 1.0) < 1e-6 and
                    inv_error < 1e-6
                )
                print(f"  旋转矩阵行列式: {det_R:.10f} (应为 1.0)")
                print(f"  逆矩阵误差: {inv_error:.2e} (应 < 1e-6)")
                print(f"  ✓ 外参矩阵验证: {'通过' if valid_extrinsic else '失败'}")
            else:
                print(f"\n  ✗ 外参: 缺失")
            
            print()
        
        # ==================== Print All LiDAR Parameters ====================
        print("=" * 100)
        print("LiDAR参数详情")
        print("=" * 100)
        
        all_lidars = calib_manager.get_all_lidar_names()
        print(f"\n总共 {len(all_lidars)} 个LiDAR:\n")
        
        for idx, lidar_name in enumerate(sorted(all_lidars), 1):
            lidar = calib_manager.get_lidar(lidar_name)
            if not lidar:
                print(f"{idx}. {lidar_name}: ✗ 未找到")
                continue
            
            print(f"{idx}. {lidar_name}")
            print("-" * 100)
            
            if lidar.extrinsic.is_valid():
                T_lidar2car = lidar.get_lidar2car()
                T_car2lidar = lidar.get_car2lidar()
                
                print(f"  外参矩阵 (LiDAR-to-Car):")
                print(f"    {T_lidar2car[0,0]:12.6f}  {T_lidar2car[0,1]:12.6f}  {T_lidar2car[0,2]:12.6f}  {T_lidar2car[0,3]:12.6f}")
                print(f"    {T_lidar2car[1,0]:12.6f}  {T_lidar2car[1,1]:12.6f}  {T_lidar2car[1,2]:12.6f}  {T_lidar2car[1,3]:12.6f}")
                print(f"    {T_lidar2car[2,0]:12.6f}  {T_lidar2car[2,1]:12.6f}  {T_lidar2car[2,2]:12.6f}  {T_lidar2car[2,3]:12.6f}")
                print(f"    {T_lidar2car[3,0]:12.6f}  {T_lidar2car[3,1]:12.6f}  {T_lidar2car[3,2]:12.6f}  {T_lidar2car[3,3]:12.6f}")
                
                # Extract rotation and translation
                R = T_lidar2car[:3, :3]
                t = T_lidar2car[:3, 3]
                print(f"  位置 (相对于车体): x={t[0]:.6f}, y={t[1]:.6f}, z={t[2]:.6f}")
                
                # Validate extrinsic matrix
                det_R = np.linalg.det(R)
                T_inv_check = T_lidar2car @ T_car2lidar
                identity = np.eye(4)
                inv_error = np.abs(T_inv_check - identity).max()
                
                valid_extrinsic = (
                    T_lidar2car[3, 3] == 1.0 and
                    abs(det_R - 1.0) < 1e-6 and
                    inv_error < 1e-6
                )
                print(f"  旋转矩阵行列式: {det_R:.10f} (应为 1.0)")
                print(f"  逆矩阵误差: {inv_error:.2e} (应 < 1e-6)")
                print(f"  ✓ 外参矩阵验证: {'通过' if valid_extrinsic else '失败'}")
            else:
                print(f"  ✗ 外参: 缺失")
            
            print()
        
        # ==================== Print Sensor-to-Sensor Transformations ====================
        if calib_manager.sensor_to_sensor:
            print("=" * 100)
            print("传感器间变换参数")
            print("=" * 100)
            print(f"\n总共 {len(calib_manager.sensor_to_sensor)} 个传感器间变换:\n")
            
            for idx, (transform_key, extrinsic) in enumerate(sorted(calib_manager.sensor_to_sensor.items()), 1):
                if not extrinsic.is_valid():
                    print(f"{idx}. {transform_key}: ✗ 无效")
                    continue
                
                T = extrinsic.sensor2car
                print(f"{idx}. {transform_key}")
                print("-" * 100)
                print(f"  变换矩阵:")
                print(f"    {T[0,0]:12.6f}  {T[0,1]:12.6f}  {T[0,2]:12.6f}  {T[0,3]:12.6f}")
                print(f"    {T[1,0]:12.6f}  {T[1,1]:12.6f}  {T[1,2]:12.6f}  {T[1,3]:12.6f}")
                print(f"    {T[2,0]:12.6f}  {T[2,1]:12.6f}  {T[2,2]:12.6f}  {T[2,3]:12.6f}")
                print(f"    {T[3,0]:12.6f}  {T[3,1]:12.6f}  {T[3,2]:12.6f}  {T[3,3]:12.6f}")
                
                # Validate matrix
                R = T[:3, :3]
                det_R = np.linalg.det(R)
                valid = (T[3, 3] == 1.0 and abs(det_R - 1.0) < 1e-6)
                print(f"  旋转矩阵行列式: {det_R:.10f} (应为 1.0)")
                print(f"  ✓ 矩阵验证: {'通过' if valid else '失败'}")
                print()
        
        # ==================== Print IMU Bias ====================
        imu_bias = calib_manager.get_imu_bias()
        if imu_bias:
            print("=" * 100)
            print("IMU偏置参数")
            print("=" * 100)
            print()
            if "accelerometer_bias" in imu_bias:
                acc_bias = imu_bias['accelerometer_bias']
                print(f"  加速度计偏置:")
                print(f"    x: {acc_bias.get('x', 0):.10f}")
                print(f"    y: {acc_bias.get('y', 0):.10f}")
                print(f"    z: {acc_bias.get('z', 0):.10f}")
            if "gyroscope_bias" in imu_bias:
                gyro_bias = imu_bias['gyroscope_bias']
                print(f"  陀螺仪偏置:")
                print(f"    x: {gyro_bias.get('x', 0):.10f}")
                print(f"    y: {gyro_bias.get('y', 0):.10f}")
                print(f"    z: {gyro_bias.get('z', 0):.10f}")
            print()
        
        # ==================== Summary ====================
        print("=" * 100)
        print("验证总结")
        print("=" * 100)
        
        total_cameras = len(all_cameras)
        cameras_with_intrinsic = sum(1 for cam_name in all_cameras 
                                     if calib_manager.get_camera(cam_name) and 
                                     calib_manager.get_camera(cam_name).intrinsic.is_valid())
        cameras_with_extrinsic = sum(1 for cam_name in all_cameras 
                                     if calib_manager.get_camera(cam_name) and 
                                     calib_manager.get_camera(cam_name).extrinsic.is_valid())
        
        total_lidars = len(all_lidars)
        lidars_with_extrinsic = sum(1 for lidar_name in all_lidars 
                                    if calib_manager.get_lidar(lidar_name) and 
                                    calib_manager.get_lidar(lidar_name).extrinsic.is_valid())
        
        print(f"\n相机:")
        print(f"  总数: {total_cameras}")
        print(f"  有内参: {cameras_with_intrinsic}/{total_cameras}")
        print(f"  有外参: {cameras_with_extrinsic}/{total_cameras}")
        
        print(f"\nLiDAR:")
        print(f"  总数: {total_lidars}")
        print(f"  有外参: {lidars_with_extrinsic}/{total_lidars}")
        
        print(f"\n传感器间变换: {len(calib_manager.sensor_to_sensor)}")
        print(f"IMU偏置: {'已加载' if imu_bias else '未找到'}")
        
        print("\n" + "=" * 100)
        
    except FileNotFoundError as e:
        print(f"✗ 错误: {e}")
        print(f"\n请提供有效的标定目录路径。")
        print(f"用法: python -m calibration_manager.calibration_manager [calib_dir]")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
