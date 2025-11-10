"""
6-DOF pose definition and storage module

Reference to C++ implementation TransformInterpolationBuffer, provides Python version
of pose storage and interpolation functionality.
Used for storing a series of vehicle global poses.
"""

import numpy as np
from collections import deque
from typing import Optional, Union, Tuple
from scipy.spatial.transform import Rotation as RotLib


class Pose6DOF:
    """
    6-DOF pose class supporting multiple representation formats
    
    Uses translation vector + rotation (quaternion or Euler angles) to represent 6-DOF pose
    """
    
    def __init__(self, 
                 translation: np.ndarray = None,
                 rotation: Union[np.ndarray, RotLib] = None,
                 rotation_type: str = 'quaternion'):
        """
        Initialize 6-DOF pose
        
        Args:
            translation: Translation vector (3,) or (3, 1), unit: meters
            rotation: Rotation representation
                - If rotation_type='quaternion': quaternion (4,), order [x, y, z, w]
                - If rotation_type='euler': Euler angles (3,), order [roll, pitch, yaw], unit: radians
                - If rotation_type='matrix': rotation matrix (3, 3)
            rotation_type: Rotation representation type, 'quaternion', 'euler', or 'matrix'
        """
        if translation is None:
            self.translation = np.zeros(3, dtype=np.float64)
        else:
            self.translation = np.asarray(translation, dtype=np.float64).flatten()
            assert self.translation.shape == (3,), f"Translation must be shape (3,), got {self.translation.shape}"
        
        if rotation is None:
            self.quaternion = np.array([0., 0., 0., 1.], dtype=np.float64)  # Identity quaternion
        else:
            rotation = np.asarray(rotation, dtype=np.float64)
            if rotation_type == 'quaternion':
                self.quaternion = rotation.flatten()
                assert self.quaternion.shape == (4,), f"Quaternion must be shape (4,), got {self.quaternion.shape}"
                # Normalize quaternion
                norm = np.linalg.norm(self.quaternion)
                if norm > 1e-8:
                    self.quaternion = self.quaternion / norm
            elif rotation_type == 'euler':
                euler = rotation.flatten()
                assert euler.shape == (3,), f"Euler angles must be shape (3,), got {euler.shape}"
                rot = RotLib.from_euler('xyz', euler, degrees=False)
                self.quaternion = rot.as_quat()  # [x, y, z, w]
            elif rotation_type == 'matrix':
                rot_matrix = rotation.reshape(3, 3)
                rot = RotLib.from_matrix(rot_matrix)
                self.quaternion = rot.as_quat()  # [x, y, z, w]
            else:
                raise ValueError(f"Unknown rotation_type: {rotation_type}")
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """
        Create pose from 4x4 transformation matrix
        
        Args:
            matrix: 4x4 homogeneous transformation matrix
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        assert matrix.shape == (4, 4), f"Matrix must be shape (4, 4), got {matrix.shape}"
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        return cls(translation=translation, rotation=rotation_matrix, rotation_type='matrix')
    
    @classmethod
    def identity(cls):
        """Create identity pose (no translation, no rotation)"""
        return cls()
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert to 4x4 homogeneous transformation matrix
        
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4, dtype=np.float64)
        rot = RotLib.from_quat(self.quaternion)
        matrix[:3, :3] = rot.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix
    
    def to_euler(self) -> np.ndarray:
        """
        Convert to Euler angle representation
        
        Returns:
            Euler angles [roll, pitch, yaw], unit: radians
        """
        rot = RotLib.from_quat(self.quaternion)
        return rot.as_euler('xyz', degrees=False)
    
    def to_quaternion(self) -> np.ndarray:
        """
        Get quaternion representation
        
        Returns:
            Quaternion [x, y, z, w]
        """
        return self.quaternion.copy()
    
    def __repr__(self):
        euler = self.to_euler()
        return (f"Pose6DOF(translation=[{self.translation[0]:.3f}, "
                f"{self.translation[1]:.3f}, {self.translation[2]:.3f}], "
                f"euler=[{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}])")


class TimestampedPose:
    """
    Pose with timestamp
    """
    
    def __init__(self, timestamp_ms: int, pose: Pose6DOF):
        """
        Initialize timestamped pose
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            pose: 6-DOF pose
        """
        self.timestamp_ms = int(timestamp_ms)
        self.pose = pose
    
    def __lt__(self, other):
        """For sorting"""
        if isinstance(other, TimestampedPose):
            return self.timestamp_ms < other.timestamp_ms
        return self.timestamp_ms < other
    
    def __repr__(self):
        return f"TimestampedPose(timestamp={self.timestamp_ms}ms, pose={self.pose})"


class PoseBuffer:
    """
    Pose buffer for storing a series of vehicle global poses
    
    Supports temporal interpolation lookup, referencing TransformInterpolationBuffer implementation
    """
    
    def __init__(self, buffer_size_limit: Optional[int] = None):
        """
        Initialize pose buffer
        
        Args:
            buffer_size_limit: Buffer size limit, None means unlimited
        """
        self.timestamped_poses: deque = deque()
        self.buffer_size_limit = buffer_size_limit if buffer_size_limit is not None else float('inf')
    
    def set_size_limit(self, buffer_size_limit: int):
        """
        Set buffer size limit
        
        Args:
            buffer_size_limit: Buffer size limit
        """
        self.buffer_size_limit = buffer_size_limit
        self._remove_old_poses_if_needed()
    
    def push(self, timestamp_ms: int, pose: Union[Pose6DOF, np.ndarray]):
        """
        Add pose to buffer
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            pose: Pose, can be Pose6DOF object or 4x4 transformation matrix
        """
        if isinstance(pose, np.ndarray):
            pose = Pose6DOF.from_matrix(pose)
        
        if len(self.timestamped_poses) > 0:
            assert timestamp_ms >= self.timestamped_poses[-1].timestamp_ms, \
                f"New pose timestamp ({timestamp_ms}) is older than latest ({self.timestamped_poses[-1].timestamp_ms})"
        
        self.timestamped_poses.append(TimestampedPose(timestamp_ms, pose))
        self._remove_old_poses_if_needed()
    
    def push_euler(self, timestamp_ms: int, translation: np.ndarray, euler_angles: np.ndarray):
        """
        Add pose using Euler angles
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            translation: Translation vector (3,)
            euler_angles: Euler angles [roll, pitch, yaw] (3,), unit: radians
        """
        pose = Pose6DOF(translation=translation, rotation=euler_angles, rotation_type='euler')
        self.push(timestamp_ms, pose)
    
    def clear(self):
        """Clear buffer"""
        self.timestamped_poses.clear()
    
    def has(self, timestamp_ms: int) -> bool:
        """
        Check if pose exists for given timestamp (or can be obtained via interpolation)
        
        Args:
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            True if timestamp is within buffer range
        """
        if self.empty():
            return False
        return self.earliest_time() <= timestamp_ms <= self.latest_time()
    
    def lookup(self, timestamp_ms: int) -> Pose6DOF:
        """
        Lookup pose at given timestamp (with interpolation)
        
        Args:
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            6-DOF pose
        """
        assert self.has(timestamp_ms), f"Missing pose for timestamp: {timestamp_ms}"
        
        # Binary search
        poses_list = list(self.timestamped_poses)
        end_idx = self._binary_search(timestamp_ms)
        
        if poses_list[end_idx].timestamp_ms == timestamp_ms:
            return poses_list[end_idx].pose
        
        start_idx = end_idx - 1
        interpolated = self._interpolate(
            poses_list[start_idx],
            poses_list[end_idx],
            timestamp_ms
        )
        return interpolated.pose
    
    def find_closest_pose(self, timestamp_ms: int, max_time_diff_ms: int) -> Optional[Pose6DOF]:
        """
        Find closest pose (within allowed time difference)
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            max_time_diff_ms: Maximum allowed time difference in milliseconds
        
        Returns:
            6-DOF pose, returns None if not within range
        """
        if not self.has(timestamp_ms):
            return None
        
        poses_list = list(self.timestamped_poses)
        end_idx = self._binary_search(timestamp_ms)
        start_idx = end_idx - 1
        
        start_time_diff = abs(poses_list[start_idx].timestamp_ms - timestamp_ms)
        end_time_diff = abs(poses_list[end_idx].timestamp_ms - timestamp_ms)
        
        if start_time_diff > max_time_diff_ms and end_time_diff > max_time_diff_ms:
            return None
        
        if poses_list[end_idx].timestamp_ms == timestamp_ms:
            return poses_list[end_idx].pose
        
        interpolated = self._interpolate(
            poses_list[start_idx],
            poses_list[end_idx],
            timestamp_ms
        )
        return interpolated.pose
    
    def earliest_time(self) -> int:
        """
        Get earliest timestamp
        
        Returns:
            Earliest timestamp in milliseconds
        """
        assert not self.empty(), "Empty buffer"
        return self.timestamped_poses[0].timestamp_ms
    
    def latest_time(self) -> int:
        """
        Get latest timestamp
        
        Returns:
            Latest timestamp in milliseconds
        """
        assert not self.empty(), "Empty buffer"
        return self.timestamped_poses[-1].timestamp_ms
    
    def empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.timestamped_poses) == 0
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.timestamped_poses)
    
    def get_all_poses(self) -> list:
        """
        Get all timestamped poses
        
        Returns:
            List of all TimestampedPose objects
        """
        return list(self.timestamped_poses)
    
    def get_poses_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get array representation of all poses
        
        Returns:
            (timestamps, poses_matrix)
            - timestamps: (N,) timestamp array
            - poses_matrix: (N, 4, 4) transformation matrix array
        """
        timestamps = np.array([tp.timestamp_ms for tp in self.timestamped_poses], dtype=np.int64)
        poses_matrix = np.array([tp.pose.to_matrix() for tp in self.timestamped_poses], dtype=np.float64)
        return timestamps, poses_matrix
    
    def _binary_search(self, timestamp_ms: int) -> int:
        """
        Binary search for timestamp position
        
        Args:
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            Index position
        """
        poses_list = list(self.timestamped_poses)
        left, right = 0, len(poses_list) - 1
        
        while left < right:
            mid = (left + right) // 2
            if poses_list[mid].timestamp_ms < timestamp_ms:
                left = mid + 1
            else:
                right = mid
        
        return min(left, len(poses_list) - 1)
    
    def _interpolate(self, start: TimestampedPose, end: TimestampedPose, 
                     timestamp_ms: int) -> TimestampedPose:
        """
        Interpolate between two poses
        
        Args:
            start: Start pose
            end: End pose
            timestamp_ms: Target timestamp
        
        Returns:
            Interpolated pose
        """
        assert start.timestamp_ms <= timestamp_ms <= end.timestamp_ms
        
        duration_ms = end.timestamp_ms - start.timestamp_ms
        if duration_ms == 0:
            factor = 0.0
        else:
            factor = (timestamp_ms - start.timestamp_ms) / duration_ms
        
        # Linear interpolation for translation
        translation = start.pose.translation + (end.pose.translation - start.pose.translation) * factor
        
        # Spherical linear interpolation (SLERP) for quaternion
        q1 = start.pose.quaternion  # [x, y, z, w]
        q2 = end.pose.quaternion
        
        # Calculate dot product to determine direction
        dot = np.dot(q1, q2)
        # If dot product is negative, negate one quaternion to take shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation to avoid numerical issues
        if dot > 0.9995:
            quaternion = q1 + factor * (q2 - q1)
            quaternion = quaternion / np.linalg.norm(quaternion)
        else:
            # Standard SLERP
            theta = np.arccos(np.abs(dot))
            sin_theta = np.sin(theta)
            w1 = np.sin((1 - factor) * theta) / sin_theta
            w2 = np.sin(factor * theta) / sin_theta
            quaternion = w1 * q1 + w2 * q2
        
        interpolated_pose = Pose6DOF(translation=translation, rotation=quaternion, rotation_type='quaternion')
        return TimestampedPose(timestamp_ms, interpolated_pose)
    
    def _remove_old_poses_if_needed(self):
        """Remove oldest poses if buffer size limit is exceeded"""
        while len(self.timestamped_poses) > self.buffer_size_limit:
            self.timestamped_poses.popleft()
    
    def __repr__(self):
        return f"PoseBuffer(size={self.size()}, limit={self.buffer_size_limit}, " \
               f"time_range=[{self.earliest_time() if not self.empty() else 'N/A'}, " \
               f"{self.latest_time() if not self.empty() else 'N/A'}])"

