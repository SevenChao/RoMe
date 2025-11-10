import numpy as np
import cv2
import csv
from multiprocessing.pool import ThreadPool as Pool
import os
from os.path import join
from copy import deepcopy
from tqdm import tqdm
from typing import Optional
from pyquaternion import Quaternion
from datasets.base import BaseDataset
from utils.plane_fit import robust_estimate_flatplane
from calibration_manager import CameraCalibration, LidarCalibration, CalibrationManager, load_calibration_manager
from utils.pose import PoseBuffer, Pose6DOF

camera_names = [
    # "front_narrow",
    "front_wide",
    "left_front",
    "right_front",
    "back",
    "left_back",
    "right_back",
]

# Map camera names in code to calibration file names
CAMERA_NAME_MAPPING = {
    "front_wide": "camera-front-wide",
    "left_front": "camera-left-front",
    "right_front": "camera-right-front",
    "back": "camera-back",
    "left_back": "camera-left-back",
    "right_back": "camera-right-back",
    "front_narrow": "camera-front-narrow",
}


# CameraDataInfo type hint (for documentation only)
# CameraDataInfo = {
#     "camera_name": str,
#     "camera_intrinsic": np.ndarray,
#     "camera_distortion": np.ndarray,
#     "camera_width": int,
#     "camera_height": int,
#     "timestamp_ms": list,
#     "image_files": list,
#     "label_files": list,
#     "lane_files": list,
# }



class GACRNDDataset(BaseDataset):
    def __init__(self, configs):
        super().__init__()
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        self.label_dir = configs["label_dir"]
        self.lane_dir = configs["lane_dir"]
        self.ego_mask_dir = configs.get("ego_mask_dir", None)  # Optional: directory for ego vehicle masks
        # Camera-specific body occlusion masks (e.g., camera_mask/front_wide_mask.png)
        self.camera_mask_dir = configs.get("camera_mask_dir", "camera_mask")  # Default: "camera_mask"
        self.resized_image_size = (configs["image_width"], configs["image_height"])

        x_offset = -configs["center_point"]["x"] + configs["bev_x_length"]/2
        y_offset = -configs["center_point"]["y"] + configs["bev_y_length"]/2
        self.world2bev = np.asarray([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.min_distance = configs["min_distance"]


        self.calib_manager = load_calibration_manager(join(self.base_dir, "calib_extract"))
        self.camera_calibrations = self.calib_manager.cameras
        self.lidar_calibrations = self.calib_manager.lidars
        
        # Get LiDAR to car transformation for pose conversion
        # FramePose.txt contains poses in LiDAR coordinate system
        self.lidar_name = configs.get("lidar_name", "lidar-top")  # Default to lidar-top
        lidar_calib = self.lidar_calibrations.get(self.lidar_name)
        if lidar_calib is None:
            # Try alternative names
            for alt_name in ["lidar_top", "lidar-top", "top"]:
                if alt_name in self.lidar_calibrations:
                    self.lidar_name = alt_name
                    lidar_calib = self.lidar_calibrations[alt_name]
                    break
        
        if lidar_calib and lidar_calib.is_valid():
            # Get lidar2car transformation
            lidar2car = lidar_calib.get_lidar2car()
            # Get car2lidar (inverse) for converting lidar poses to car poses
            self.car2lidar = np.linalg.inv(lidar2car) if lidar2car is not None else None
        else:
            raise ValueError(f"LiDAR calibration '{self.lidar_name}' not found or invalid. "
                           f"Available LiDARs: {list(self.lidar_calibrations.keys())}")
        
        # Load pose buffer first (poses are in LiDAR coordinate system)
        # Try odom.csv first, fallback to FramePose.txt
        pose_file_type = configs.get("pose_file_type", "auto")  # "auto", "odom.csv", "FramePose.txt"
        if pose_file_type == "auto":
            # Auto-detect: check if odom.csv exists
            odom_file = join(self.base_dir, "odom.csv")
            frame_pose_file = join(self.base_dir, "FramePose.txt")
            if os.path.exists(odom_file):
                pose_file_type = "odom.csv"
            elif os.path.exists(frame_pose_file):
                pose_file_type = "FramePose.txt"
            else:
                raise FileNotFoundError(f"Neither odom.csv nor FramePose.txt found in {self.base_dir}")
        
        if pose_file_type == "odom.csv":
            self.pose_buffer = self.load_poses_from_odom_csv()
        else:
            self.pose_buffer = self.load_poses_from_frame_pose_file()
        
        # Set reference camera (similar to CAM_FRONT in nusc)
        self.ref_camera_name = camera_names[0]  # front_wide as reference
        
        # Store configs for later use
        self.configs = configs
        
        # Initialize lane_filenames_all (empty for now, can be populated later if needed)
        if not hasattr(self, 'lane_filenames_all'):
            self.lane_filenames_all = []
        
        # Load all camera data and organize by timestamp
        self._load_all_camera_data_by_timestamp()
        
        # Initialize lane_filenames_all to match image_filenames_all length if empty
        if len(self.lane_filenames_all) == 0:
            self.lane_filenames_all = [None] * len(self.image_filenames_all)
        
        # File check and filtering
        self._file_check()
        
        # Optional: Label valid check (can be enabled in configs)
        if configs.get("enable_label_valid_check", False):
            self.label_valid_check()
        
        # Initialize non-_all lists to match _all lists (for compatibility)
        self.image_filenames = self.image_filenames_all.copy()
        self.label_filenames = self.label_filenames_all.copy()
        self.ref_camera2world = self.ref_camera2world_all.copy()
        self.cameras_K = self.cameras_K_all.copy()
        self.cameras_idx = self.cameras_idx_all.copy()
        self.cameras_d = self.cameras_d_all.copy()
        if hasattr(self, 'ego_mask_filenames_all'):
            if not hasattr(self, 'ego_mask_filenames'):
                self.ego_mask_filenames = []
            self.ego_mask_filenames = self.ego_mask_filenames_all.copy()
        
        # Optional: Filter poses in BEV range (similar to nusc)
        # self._filter_poses_in_bev_range(configs)
    
    def _file_check(self):
        """Check if image and label files exist"""
        image_paths = [join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        label_paths = [join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        available_index = list(np.where(image_exists * label_exists)[0])
        print(f"Drop {len(image_paths) - len(available_index)} frames out of {len(image_paths)} by file exists check")
        if len(available_index) > 0:
            self.filter_by_index(available_index)
    
    def filter_by_index(self, index):
        """Filter data by index, including ego mask files"""
        super().filter_by_index(index)
        if hasattr(self, "ego_mask_filenames_all"):
            self.ego_mask_filenames_all = [self.ego_mask_filenames_all[i] for i in index]
            # Update ego_mask_filenames if it exists
            if hasattr(self, "ego_mask_filenames"):
                self.ego_mask_filenames = self.ego_mask_filenames_all.copy()
    
    def set_waypoint(self, center_xy, radius):
        """Set waypoint, including ego mask files"""
        super().set_waypoint(center_xy, radius)
        if hasattr(self, "ego_mask_filenames_all"):
            self.ego_mask_filenames = [self.ego_mask_filenames_all[i] for i in self.activated_idx]
    
    def label_valid_check(self):
        """Check label validity and filter invalid labels"""
        label_paths = [join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_valid = np.asarray(self.check_label_valid(label_paths))
        available_index = list(np.where(label_valid)[0])
        print(f"Drop {len(label_paths) - len(available_index)} frames out of {len(label_paths)} by label valid check")
        self.filter_by_index(available_index)
    
    def label_valid(self, label_name):
        """
        Check if a single label is valid for training
        
        Args:
            label_name: Path to label image
        
        Returns:
            bool: True if label is valid, False otherwise
        """
        label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
        if label is None:
            return False
        
        # Moving objects ratio (persons, riders, etc.)
        # Based on class_to_grayscale_mapping_panoptic.json
        # Person (75), Bicyclist (79), Motorcyclist (83), Other Rider (87)
        # Bird (0), Ground Animal (3)
        label_movable = np.isin(label, [0, 3, 75, 79, 83, 87])
        ratio_movable = label_movable.sum() / label_movable.size
        
        # Non-road areas ratio (should be filtered out)
        # Based on class_to_grayscale_mapping_panoptic.json
        # Sky (107), Mountain (99), Sand (103), Snow (111), Building (67), 
        # Tunnel (71), Bridge (63), Vegetation (119), Terrain (115)
        # Curb (7) - barrier, should be filtered
        label_off_road = np.isin(label, [
            7, 63, 67, 71, 99, 103, 107, 111, 115, 119
        ])
        ratio_static = label_off_road.sum() / label_off_road.size
        
        # Filter out labels with too many moving objects or non-road areas
        # These labels are not suitable for static scene reconstruction
        if ratio_movable > 0.3 or ratio_static > 0.9:
            return False
        else:
            return True
    
    def check_label_valid(self, filelist):
        """Check label validity for a list of files using multiprocessing"""
        with Pool(32) as p:
            exist_list = p.map(self.label_valid, filelist)
        return exist_list






    def _load_all_camera_data_by_timestamp(self):
        """
        Load all camera data organized by timestamp, similar to nusc.py
        For each timestamp, iterate through all cameras
        Multiple cameras can have images at the same timestamp (synchronized cameras)
        """
        # Collect all image data from all cameras
        camera_data_dict = {}
        for camera_name in camera_names:
            camera_data = self._load_camera_data_raw(camera_name)
            camera_data_dict[camera_name] = camera_data
        
        # Build timestamp to image index mapping for efficient lookup
        # This handles cases where multiple cameras share the same timestamp
        camera_timestamp_to_idx = {}
        for camera_name in camera_names:
            camera_timestamp_to_idx[camera_name] = {}
            timestamp_list = camera_data_dict[camera_name]["timestamp_ms"]
            for idx, ts in enumerate(timestamp_list):
                if ts not in camera_timestamp_to_idx[camera_name]:
                    camera_timestamp_to_idx[camera_name][ts] = idx
        
        # Get all unique timestamps across all cameras
        # Multiple cameras can have the same timestamp (synchronized capture)
        all_timestamps = set()
        for camera_name in camera_names:
            all_timestamps.update(camera_data_dict[camera_name]["timestamp_ms"])
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        # Reference camera extrinsic (identity)
        ref_calib_name = CAMERA_NAME_MAPPING.get(self.ref_camera_name, self.ref_camera_name)
        if ref_calib_name not in self.camera_calibrations:
            ref_calib_name = self.ref_camera_name  # Fallback
        ref_camera2car = self.camera_calibrations[ref_calib_name].get_camera2car()
        car2ref_camera = np.linalg.inv(ref_camera2car)
        
        # Pre-compute camera extrinsics for all cameras
        self.camera_extrinsics = []
        for camera_idx, camera_name in enumerate(camera_names):
            calib_camera_name = CAMERA_NAME_MAPPING.get(camera_name, camera_name)
            if calib_camera_name not in self.camera_calibrations:
                calib_camera_name = camera_name  # Fallback
            camera2car = self.camera_calibrations[calib_camera_name].get_camera2car()
            camera2ref_camera = car2ref_camera @ camera2car
            self.camera_extrinsics.append(camera2ref_camera.astype(np.float32))
        
        # Iterate through all timestamps
        # Note: Set max_time_diff_ms larger if image timestamps don't exactly match pose timestamps
        # Common scenarios: 
        # - Synchronized data: max_time_diff_ms = 100 (default)
        # - Unsynchronized data: max_time_diff_ms = 1000 or larger
        max_time_diff_ms = self.configs.get("max_pose_time_diff_ms", 100)
        
        for timestamp_ms in tqdm(all_timestamps, desc="Loading data by timestamp"):
            # Try to find pose at this timestamp
            car_pose = None
            if self.pose_buffer.has(timestamp_ms):
                # Exact match
                car_pose = self.pose_buffer.lookup(timestamp_ms)
            else:
                # Try to find closest pose within allowed time difference
                car_pose = self.pose_buffer.find_closest_pose(timestamp_ms, max_time_diff_ms)
                if car_pose is None:
                    # Skip this timestamp if no pose found within tolerance
                    continue
            
            # Get car pose in world coordinate
            car2world = car_pose.to_matrix().astype(np.float32)
            
            # Iterate through all cameras at this timestamp
            # Multiple cameras can have images at the same timestamp (synchronized capture)
            for camera_idx, camera_name in enumerate(camera_names):
                # Check if this camera has image at this timestamp using efficient dict lookup
                if timestamp_ms not in camera_timestamp_to_idx[camera_name]:
                    continue
                
                # Get the image index for this timestamp (O(1) lookup)
                img_idx = camera_timestamp_to_idx[camera_name][timestamp_ms]
                
                # Get image and label paths (already full paths)
                full_image_path = camera_data_dict[camera_name]["image_files"][img_idx]
                full_label_path = camera_data_dict[camera_name]["label_files"][img_idx]
                
                # Get ego mask path if available
                ego_mask_files = camera_data_dict[camera_name].get("ego_mask_files")
                rel_ego_mask_path = None
                if ego_mask_files and ego_mask_files[img_idx] is not None:
                    rel_ego_mask_path = os.path.relpath(ego_mask_files[img_idx], self.base_dir)
                
                # Make paths relative to base_dir
                rel_image_path = os.path.relpath(full_image_path, self.base_dir)
                rel_label_path = os.path.relpath(full_label_path, self.base_dir)
                
                # Get camera extrinsic
                calib_camera_name = CAMERA_NAME_MAPPING.get(camera_name, camera_name)
                if calib_camera_name not in self.camera_calibrations:
                    calib_camera_name = camera_name  # Fallback
                camera2car = self.camera_calibrations[calib_camera_name].get_camera2car()
                
                # Compute camera2world (car2world @ camera2car)
                camera2world = car2world @ camera2car
                
                # Get camera intrinsic
                camera_intrinsic = camera_data_dict[camera_name]["camera_intrinsic"]
                
                # Append to lists
                self.image_filenames_all.append(rel_image_path)
                self.label_filenames_all.append(rel_label_path)
                # Store ego mask path if available (use None as placeholder if not available)
                if not hasattr(self, 'ego_mask_filenames_all'):
                    self.ego_mask_filenames_all = []
                self.ego_mask_filenames_all.append(rel_ego_mask_path)
                self.ref_camera2world_all.append(camera2world.astype(np.float32))
                self.cameras_K_all.append(camera_intrinsic.astype(np.float32))
                self.cameras_idx_all.append(camera_idx)
                self.cameras_d_all.append(camera_data_dict[camera_name]["camera_distortion"])
        
        print(f"Loaded {len(self.image_filenames_all)} frames across all cameras")
    
    def _load_camera_data_raw(self, camera_name):
        # Map camera name to calibration file name
        calib_camera_name = CAMERA_NAME_MAPPING.get(camera_name, camera_name)
        if calib_camera_name not in self.camera_calibrations:
            # Try alternative names
            for alt_name in [f"camera-{camera_name}", f"camera_{camera_name}", camera_name]:
                if alt_name in self.camera_calibrations:
                    calib_camera_name = alt_name
                    break
            else:
                raise KeyError(f"Camera '{camera_name}' (mapped to '{calib_camera_name}') not found in calibrations. "
                             f"Available cameras: {list(self.camera_calibrations.keys())}")
        
        # get camera intrinsic 、 distortion and extrinsic
        camera_intrinsic = self.camera_calibrations[calib_camera_name].get_intrinsic()
        camera_distortion = self.camera_calibrations[calib_camera_name].get_distortion()
        camera_extrinsic = self.camera_calibrations[calib_camera_name].get_camera2car()
        camera_width = self.camera_calibrations[calib_camera_name].get_image_size()[0]
        camera_height = self.camera_calibrations[calib_camera_name].get_image_size()[1]

        image_folder = join(self.base_dir, "image_raw", camera_name)
        label_folder = join(self.base_dir, "raw_images_seg_mask2former", camera_name)

        ## get all image files and label files
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
        label_files = [f for f in os.listdir(label_folder) if f.endswith(".png")]

        ## sort image files and label files
        image_files.sort()
        label_files.sort()

        ## check if the image files have the corresponding label files
        for image_file in image_files:
            if image_file.replace(".jpg", ".png") not in label_files:
                raise ValueError(f"The image file {image_file} does not have a corresponding label file")
        
        # Extract timestamp from filename: format is "timestamp_ns_xxx_yyy.jpg"
        # timestamp is in nanoseconds, convert to milliseconds
        timestamp_ms = [int(image_file.split("_")[0]) // 1_000_000 for image_file in image_files]

        ## load image files and label files
        image_files = [join(self.base_dir, "image_raw", camera_name, f) for f in image_files]
        label_files = [join(self.base_dir, "raw_images_seg_mask2former", camera_name, f) for f in label_files]
        
        ## load ego mask files if available
        ego_mask_files = None
        if self.ego_mask_dir:
            ego_mask_folder = join(self.base_dir, self.ego_mask_dir, camera_name)
            if os.path.exists(ego_mask_folder):
                ego_mask_files_raw = [f for f in os.listdir(ego_mask_folder) if f.endswith((".png", ".jpg"))]
                ego_mask_files_raw.sort()
                # Match ego mask files with image files by timestamp
                ego_mask_files = []
                for image_file in image_files:
                    # Extract base name (timestamp) from image file
                    base_name = os.path.basename(image_file)
                    timestamp_prefix = base_name.split("_")[0]
                    # Find matching ego mask file
                    matching_mask = None
                    for mask_file in ego_mask_files_raw:
                        if mask_file.startswith(timestamp_prefix):
                            matching_mask = join(ego_mask_folder, mask_file)
                            break
                    ego_mask_files.append(matching_mask)  # None if not found
                if all(f is None for f in ego_mask_files):
                    ego_mask_files = None  # No valid mask files found
                    print(f"Warning: No ego mask files found in {ego_mask_folder}")
            else:
                print(f"Warning: Ego mask directory {ego_mask_folder} does not exist")

        return {
            "camera_name": camera_name,
            "camera_intrinsic": camera_intrinsic,
            "camera_distortion": camera_distortion,
            "camera_width": camera_width,
            "camera_height": camera_height,
            "timestamp_ms": timestamp_ms,
            "image_files": image_files,
            "label_files": label_files,
            "ego_mask_files": ego_mask_files,  # None if not available
        }
    
    def load_poses_from_frame_pose_file(self, 
                                         buffer_size_limit: Optional[int] = None) -> PoseBuffer:
        """
        Load pose data from FramePose.txt file
        
        Note: FramePose.txt contains poses in LiDAR coordinate system.
        These poses will be converted to car (chassis) coordinate system.
        
        Args:
            buffer_size_limit: Size limit for PoseBuffer, None means unlimited
        
        Returns:
            PoseBuffer object containing all poses (in car coordinate system)
        """
        buffer = PoseBuffer(buffer_size_limit=buffer_size_limit)
        frame_pose_file = join(self.base_dir, "FramePose.txt")
        
        with open(frame_pose_file, 'r') as f:
            for line in f:
                # Skip comment lines and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse line data
                parts = line.split(',')
                if len(parts) < 11:
                    continue
                
                try:
                    # Parse timestamp (nanoseconds) and convert to milliseconds
                    timestamp_ns = int(parts[0])
                    timestamp_ms = timestamp_ns // 1_000_000
                    
                    # Parse translation vector tx, ty, tz (indices 4, 5, 6)
                    tx = float(parts[4])
                    ty = float(parts[5])
                    tz = float(parts[6])
                    translation = np.array([tx, ty, tz], dtype=np.float64)
                    
                    # Parse quaternion qw, qx, qy, qz (indices 7, 8, 9, 10)
                    qw = float(parts[7])
                    qx = float(parts[8])
                    qy = float(parts[9])
                    qz = float(parts[10])
                    # Convert to [x, y, z, w] format (used by Pose6DOF)
                    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
                    
                    # Create pose in LiDAR coordinate system
                    lidar_pose = Pose6DOF(translation=translation, rotation=quaternion, rotation_type='quaternion')
                    
                    # Convert from LiDAR coordinate system to car coordinate system
                    # lidar2world -> car2world: car2world = lidar2world @ car2lidar
                    lidar2world = lidar_pose.to_matrix()
                    car2world = lidar2world @ self.car2lidar
                    car_pose = Pose6DOF.from_matrix(car2world)
                    
                    # Add pose (in car coordinate system) to buffer
                    buffer.push(timestamp_ms, car_pose)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line: {line[:50]}... Error: {e}")
                    continue
        
        return buffer
    
    def load_poses_from_odom_csv(self, 
                                  buffer_size_limit: Optional[int] = None) -> PoseBuffer:
        """
        Load pose data from odom.csv file
        
        Note: odom.csv contains poses in LiDAR coordinate system.
        These poses will be converted to car (chassis) coordinate system.
        
        File format:
            time_stamp, x, y, z, qx, qy, qz, qw, i2l_tx, i2l_ty, i2l_tz, i2l_qx, i2l_qy, i2l_qz, i2l_qw, ...
        
        Args:
            buffer_size_limit: Size limit for PoseBuffer, None means unlimited
        
        Returns:
            PoseBuffer object containing all poses (in car coordinate system)
        """
        buffer = PoseBuffer(buffer_size_limit=buffer_size_limit)
        odom_file = join(self.base_dir, "odom.csv")
        
        with open(odom_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse timestamp (already in milliseconds)
                    # Handle both 'time_stamp' and ' time_stamp' (with space)
                    timestamp_key = 'time_stamp' if 'time_stamp' in row else ' time_stamp'
                    timestamp_ms = int(row[timestamp_key].strip())
                    
                    # Parse translation vector x, y, z
                    # Handle column names with/without spaces
                    x_key = 'x' if 'x' in row else ' x'
                    y_key = 'y' if 'y' in row else ' y'
                    z_key = 'z' if 'z' in row else ' z'
                    tx = float(row[x_key].strip())
                    ty = float(row[y_key].strip())
                    tz = float(row[z_key].strip())
                    translation = np.array([tx, ty, tz], dtype=np.float64)
                    
                    # Parse quaternion qx, qy, qz, qw (already in [x, y, z, w] format)
                    qx_key = 'qx' if 'qx' in row else ' qx'
                    qy_key = 'qy' if 'qy' in row else ' qy'
                    qz_key = 'qz' if 'qz' in row else ' qz'
                    qw_key = 'qw' if 'qw' in row else ' qw'
                    qx = float(row[qx_key].strip())
                    qy = float(row[qy_key].strip())
                    qz = float(row[qz_key].strip())
                    qw = float(row[qw_key].strip())
                    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
                    
                    # Create pose in LiDAR coordinate system
                    lidar_pose = Pose6DOF(translation=translation, rotation=quaternion, rotation_type='quaternion')
                    
                    # Convert from LiDAR coordinate system to car coordinate system
                    # lidar2world -> car2world: car2world = lidar2world @ car2lidar
                    lidar2world = lidar_pose.to_matrix()
                    car2world = lidar2world @ self.car2lidar
                    car_pose = Pose6DOF.from_matrix(car2world)
                    
                    # Add pose (in car coordinate system) to buffer
                    buffer.push(timestamp_ms, car_pose)
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {row.get('time_stamp', 'unknown')}... Error: {e}")
                    continue
        
        return buffer
    
    def __getitem__(self, idx):
        """
        Get a data sample by index, similar to nusc.py
        """
        sample = dict()
        sample["idx"] = idx
        sample["camera_idx"] = self.cameras_idx[idx]
        sample["camera2ref"] = self.camera_extrinsics[sample["camera_idx"]]
        
        # Read image
        image_path = self.image_filenames[idx]
        sample["image_path"] = image_path
        input_image = cv2.imread(join(self.base_dir, image_path))
        if input_image is None:
            raise ValueError(f"Failed to load image: {join(self.base_dir, image_path)}")
        
        camera_name = os.path.basename(os.path.dirname(image_path))
        crop_cy = int(self.resized_image_size[1] / 2)
        K = self.cameras_K[idx]
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        sample["image"] = (np.asarray(resized_image) / 255.0).astype(np.float32)
        
        # Read label
        label_path = join(self.image_dir, self.label_filenames[idx])
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")
        
        resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
        
        # Load ego mask if available
        ego_mask = None
        if hasattr(self, 'ego_mask_filenames') and self.ego_mask_filenames[idx] is not None:
            ego_mask_path = join(self.base_dir, self.ego_mask_filenames[idx])
            if os.path.exists(ego_mask_path):
                ego_mask = cv2.imread(ego_mask_path, cv2.IMREAD_GRAYSCALE)
                if ego_mask is not None:
                    ego_mask = cv2.resize(ego_mask, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
                    # Convert to binary mask (non-zero pixels are ego vehicle)
                    ego_mask = (ego_mask > 0).astype(bool)
                else:
                    ego_mask = None
        
        mask, label = self.label2mask(resized_label, ego_mask=ego_mask, camera_name=camera_name)
        
        # Special handling for back camera (similar to nusc.py)
        # Note: This was removed as it's not needed for GACRND dataset
        # If you need to mask the bottom part of back camera images, uncomment the following:
        # if camera_name == "back":
        #     h = mask.shape[0]
        #     mask[int(0.83 * h):, :] = 0
        #     # Synchronize label with mask: set label to 64 (ignore) where mask is 0
        #     label[~(mask.astype(bool))] = 64
        
        # Remap semantic labels after masking
        label = self.remap_semantic(label).astype(np.int64)
        
        # Ensure mask and label are synchronized: set label to 0 where mask is 0
        # This ensures that masked areas (mask=0) have label=0 (background/ignore)
        label[~(mask.astype(bool))] = 0
        
        mask = mask[crop_cy:, :]  # crop the sky
        label = label[crop_cy:, :]
        sample["static_mask"] = mask
        sample["static_label"] = label
        
        # Compute camera2world transformation
        cv_camera2world = self.ref_camera2world[idx] @ sample["camera2ref"]
        camera2world = self.world2bev @ cv_camera2world
        sample["world2camera"] = np.linalg.inv(camera2world)
        
        # Resize camera intrinsic
        resized_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        resized_K[0, :] *= width_scale
        resized_K[1, :] *= height_scale
        resized_K[1, 2] -= crop_cy
        sample["camera_K"] = resized_K
        sample["image_shape"] = np.asarray(sample["image"].shape)[:2]
        
        sample = self.opencv_camera2pytorch3d_(sample)
        return sample
    
    def label2mask(self, label, ego_mask=None, camera_name=None):
        """
        Generate mask for label, similar to nusc.py
        Masks out non-road areas, moving objects, ego vehicle areas, and camera-specific body occlusion
        
        Args:
            label: Label image with gray values corresponding to class_to_grayscale_mapping_panoptic.json
            ego_mask: Optional binary mask for ego vehicle (True for ego vehicle, False otherwise)
                     If None, will try to identify ego vehicle from label using config
            camera_name: Name of the camera (e.g., "front_wide", "left_front") for loading camera-specific mask
        
        Returns:
            mask: Float32 mask (1.0 for valid static road areas, 0.0 for masked areas)
            label: Modified label with masked areas set to 64 (ignore label)
        """
        mask = np.ones_like(label, dtype=np.float32)
        
        # Define road areas (drivable areas)
        # Based on class_to_grayscale_mapping_panoptic.json
        road_gray_values = [
            27,  # Bike Lane
            35,  # Curb Cut
            39,  # Parking
            43,  # Pedestrian Area
            51,  # Road
            55,  # Service Lane
            59,  # Sidewalk
        ]
        road_mask = np.isin(label, road_gray_values)
        
        # Moving objects (persons, riders) - gray values in labels
        # Based on class_to_grayscale_mapping_panoptic.json
        # Person (75), Bicyclist (79), Motorcyclist (83), Other Rider (87)
        # Bird (0), Ground Animal (3) - also moving
        # Note: Motorcyclist (83) will be handled separately - only filtered in road areas
        label_movable = np.isin(label, [0, 3, 75, 79, 87])
        
        # Motorcyclist (83) - only filter in road areas
        motorcyclist_mask = (label == 83)
        motorcyclist_in_road = motorcyclist_mask & road_mask
        
        # Dilate moving objects (expand boundaries)
        kernel = np.ones((10, 10), dtype=np.uint8)
        label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, iterations=2).astype(bool)
        # Also dilate motorcyclist in road areas
        motorcyclist_in_road = cv2.dilate(motorcyclist_in_road.astype(np.uint8), kernel, iterations=2).astype(bool)
        
        # Non-road areas (sky, buildings, nature, etc.) - should be masked
        # Based on class_to_grayscale_mapping_panoptic.json
        # Sky (107), Mountain (99), Sand (103), Snow (111), Building (67), 
        # Tunnel (71), Bridge (63), Vegetation (119), Terrain (115)
        # Curb (7) - barrier, should be filtered
        label_off_road = np.isin(label, [
            7,   # Curb (barrier)
            63,  # Bridge
            67,  # Building
            71,  # Tunnel
            99,  # Mountain
            103, # Sand
            107, # Sky
            111, # Snow
            115, # Terrain
            119, # Vegetation
        ])
        
        # Vehicles - all vehicles should be masked (dynamic objects)
        # Based on class_to_grayscale_mapping_panoptic.json
        # Bicycle (207), Boat (211), Bus (215), Car (219), Caravan (223), 
        # Motorcycle (227), Other Vehicle (235), Trailer (239), Truck (243), Wheeled Slow (247)
        vehicle_gray_values = [207, 211, 215, 219, 223, 227, 235, 239, 243, 247]
        vehicle_mask = np.isin(label, vehicle_gray_values)
        
        # Handle ego vehicle mask
        # Ego Vehicle (255) - should always be masked
        ego_vehicle_mask = None
        if ego_mask is not None:
            # Use provided ego mask
            ego_vehicle_mask = ego_mask
        else:
            # Try to identify ego vehicle from label using config
            # Some datasets may have specific label values for ego vehicle
            ego_label_values = self.configs.get("ego_label_values", None)
            if ego_label_values is not None:
                ego_vehicle_mask = np.isin(label, ego_label_values)
            else:
                # Default: Ego Vehicle is 255 in panoptic mapping
                ego_vehicle_mask = (label == 255)
        
        # Load camera-specific body occlusion mask if available
        camera_body_mask = None
        if camera_name is not None:
            # Try to load camera-specific mask (e.g., camera_mask/front_wide_mask.png)
            # The mask file should be white (255) for areas to filter out
            # camera_mask_dir can be absolute path or relative to project root
            if os.path.isabs(self.camera_mask_dir):
                camera_mask_path = join(self.camera_mask_dir, f"{camera_name}_mask.png")
            else:
                # Relative path: assume it's relative to project root (where script is run from)
                # Or relative to base_dir - try both
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                camera_mask_path = join(project_root, self.camera_mask_dir, f"{camera_name}_mask.png")
                if not os.path.exists(camera_mask_path):
                    # Try relative to base_dir
                    camera_mask_path = join(self.base_dir, self.camera_mask_dir, f"{camera_name}_mask.png")
            if os.path.exists(camera_mask_path):
                camera_body_mask_img = cv2.imread(camera_mask_path, cv2.IMREAD_GRAYSCALE)
                if camera_body_mask_img is not None:
                    # Resize to match label size
                    camera_body_mask = cv2.resize(
                        camera_body_mask_img, 
                        dsize=(label.shape[1], label.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    # Convert to binary mask (white pixels = filter out)
                    camera_body_mask = (camera_body_mask > 0).astype(bool)
                else:
                    camera_body_mask = None
            # If mask file doesn't exist, this camera has no body occlusion (camera_body_mask remains None)
        
        # Combine all areas to mask
        # Motorcyclist is only filtered in road areas, other movables are filtered everywhere
        # Vehicles are always filtered regardless of location (road or non-road)
        label_to_mask = label_movable | label_off_road | motorcyclist_in_road | vehicle_mask
        if ego_vehicle_mask is not None:
            label_to_mask = label_to_mask | ego_vehicle_mask
        if camera_body_mask is not None:
            label_to_mask = label_to_mask | camera_body_mask
        
        # Set mask to 0 for masked areas
        mask[label_to_mask] = 0.0
        
        # Set label to 64 (ignore label) for masked areas
        label[~(mask.astype(bool))] = 64
        
        return mask, label
    
    @property
    def label_remaps(self):
        """
        Label remapping for semantic segmentation
        Maps 65 GACRND classes to 7 simplified classes
        Based on class_to_grayscale_mapping_panoptic.json
        
        Mapping:
            0: Background/Ignore
            1: Lane Marking (all types)
            2: Curb + Curb Cut
            3: Road + Parking + Service Lane + Bike Lane
            4: Sidewalk + Pedestrian Area
            5: Terrain
            6: Other background
        """
        remaps = np.ones((256, 1), dtype="uint8")
        remaps *= 0  # Default: background 
        
        # Lane Marking (class 1) - all lane marking types
        # Based on class_to_grayscale_mapping_panoptic.json
        lane_marking_gray_values = [
            91,  # Lane Marking - Crosswalk
            95,  # Lane Marking - General
        ]
        for gv in lane_marking_gray_values:
            remaps[gv] = 1
        
        # Curb (class 2) - Curb + Curb Cut
        # Based on class_to_grayscale_mapping_panoptic.json
        remaps[7] = 2   # Curb
        remaps[35] = 2  # Curb Cut
        
        # Road (class 3) - all road-related areas
        # Based on class_to_grayscale_mapping_panoptic.json
        road_gray_values = [
            27,  # Bike Lane
            39,  # Parking
            51,  # Road
            55,  # Service Lane
        ]
        for gv in road_gray_values:
            remaps[gv] = 3
        
        # Sidewalk (class 4) - Sidewalk + Pedestrian Area
        # Based on class_to_grayscale_mapping_panoptic.json
        remaps[59] = 4  # Sidewalk
        remaps[43] = 4  # Pedestrian Area
        
        # Terrain (class 5)
        # Based on class_to_grayscale_mapping_panoptic.json
        remaps[115] = 5  # Terrain
        
        # Other background (class 6) - keep default 0, but we can set some if needed
        
        return remaps
    
    @property
    def num_class(self):
        """Number of semantic classes after remapping"""
        return 7
    
    @property
    def origin_color_map(self):
        """
        Original color map for GACRND labels (for visualization)
        Based on class_to_grayscale_mapping_panoptic.json
        """
        colors = np.zeros((256, 1, 3), dtype='uint8')
        # Colors are based on color_rgb values from class_to_grayscale_mapping_panoptic.json
        
        # Moving objects
        colors[0, :, :] = [165, 42, 42]      # Bird
        colors[3, :, :] = [0, 192, 0]        # Ground Animal
        colors[75, :, :] = [220, 20, 60]    # Person
        colors[79, :, :] = [255, 0, 0]      # Bicyclist
        colors[83, :, :] = [255, 0, 100]    # Motorcyclist (only filtered in road areas)
        colors[87, :, :] = [255, 0, 200]    # Other Rider
        
        # Static road infrastructure
        colors[7, :, :] = [196, 196, 196]   # Curb
        colors[27, :, :] = [128, 64, 255]    # Bike Lane
        colors[35, :, :] = [170, 170, 170]  # Curb Cut
        colors[39, :, :] = [250, 170, 160]  # Parking
        colors[43, :, :] = [96, 96, 96]     # Pedestrian Area
        colors[51, :, :] = [128, 64, 128]   # Road
        colors[55, :, :] = [110, 110, 110]  # Service Lane
        colors[59, :, :] = [244, 35, 232]   # Sidewalk
        
        # Lane markings
        colors[91, :, :] = [200, 128, 128]  # Lane Marking - Crosswalk
        colors[95, :, :] = [255, 255, 255]  # Lane Marking - General
        
        # Non-road areas
        colors[63, :, :] = [150, 100, 100]  # Bridge
        colors[67, :, :] = [70, 70, 70]     # Building
        colors[71, :, :] = [150, 120, 90]   # Tunnel
        colors[99, :, :] = [64, 170, 64]    # Mountain
        colors[103, :, :] = [230, 160, 50]  # Sand
        colors[107, :, :] = [70, 130, 180] # Sky
        colors[111, :, :] = [190, 255, 255] # Snow
        colors[115, :, :] = [152, 251, 152] # Terrain
        colors[119, :, :] = [107, 142, 35]  # Vegetation
        
        # Vehicles (for visualization, though they are masked)
        colors[207, :, :] = [119, 11, 32]   # Bicycle
        colors[211, :, :] = [150, 0, 255]  # Boat
        colors[215, :, :] = [0, 60, 100]   # Bus
        colors[219, :, :] = [0, 0, 142]    # Car
        colors[223, :, :] = [0, 0, 90]     # Caravan
        colors[227, :, :] = [0, 0, 230]    # Motorcycle
        colors[235, :, :] = [128, 64, 64]  # Other Vehicle
        colors[239, :, :] = [0, 0, 110]    # Trailer
        colors[243, :, :] = [0, 0, 70]     # Truck
        colors[247, :, :] = [0, 0, 192]    # Wheeled Slow
        
        # Ego Vehicle
        colors[255, :, :] = [120, 10, 10]  # Ego Vehicle
        
        return colors
    
    @property
    def filted_color_map(self):
        """
        Filtered color map for remapped labels (for visualization)
        Maps 7 classes to colors
        """
        colors = np.zeros((256, 1, 3), dtype='uint8')
        colors[0, :, :] = [0, 0, 0]         # Background/Ignore (mask)
        colors[1, :, :] = [0, 0, 255]      # Lane Marking (blue)
        colors[2, :, :] = [255, 0, 0]      # Curb (red)
        colors[3, :, :] = [211, 211, 211]  # Road (light gray)
        colors[4, :, :] = [0, 191, 255]    # Sidewalk (sky blue)
        colors[5, :, :] = [152, 251, 152]  # Terrain (light green)
        colors[6, :, :] = [157, 234, 50]   # Other background (yellow-green)
        return colors


