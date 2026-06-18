"""
sensor_manager.py

Unified sensor manager for PalmMapBot.

Provides a single interface to initialize, read from, and clean up all sensors:
- Camera
- GPS
- LiDAR
- MPU6050

This simplifies the mission code and provides consistent error handling.
"""

import logging
import threading
from typing import Dict, Optional, Any

from robot_control.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    USE_DUMMY_GPS, GPS_PORT,
    USE_DUMMY_LIDAR, LIDAR_PORT,
    USE_DUMMY_MPU, DANGEROUS_TILT_DEGREES, MAX_MPU_ERROR_COUNT
)

logger = logging.getLogger(__name__)


class SensorManager:
    """Unified sensor manager for PalmMapBot."""
    
    def __init__(self):
        self.camera = None
        self.gps = None
        self.lidar = None
        self.mpu = None
        self.mpu_error_count = 0
        self._initialized = False
        self._lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize all sensors. Returns True if successful."""
        try:
            # Import here to avoid circular imports
            from robot_control.gps_reader import GPSReader
            from robot_control.lidar_reader import LidarReader
            from robot_control.mpu6050_reader import MPU6050Reader
            import cv2
            
            # Initialize camera
            logger.info(f"Opening camera {CAMERA_INDEX}...")
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                self.cleanup()
                return False
            logger.info("Camera initialized")
            
            # Initialize GPS
            logger.info("Initializing GPS...")
            self.gps = GPSReader(GPS_PORT, use_dummy=USE_DUMMY_GPS)
            self.gps.start()
            logger.info("GPS initialized")
            
            # Initialize LiDAR
            logger.info("Initializing LiDAR...")
            self.lidar = LidarReader()
            self.lidar.start()
            logger.info("LiDAR initialized")
            
            # Initialize MPU6050
            logger.info("Initializing MPU6050...")
            self.mpu = MPU6050Reader()
            self.mpu.start()
            logger.info("MPU6050 initialized")
            
            self._initialized = True
            logger.info("All sensors initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sensors: {e}")
            self.cleanup()
            return False
    
    def get_camera_frame(self) -> tuple:
        """Get a frame from the camera. Returns (success, frame)."""
        if not self.camera or not self._initialized:
            return False, None
        ret, frame = self.camera.read()
        return ret, frame
    
    def get_gps_location(self) -> Dict[str, Any]:
        """Get GPS location. Returns dict with lat, lon, valid."""
        if not self.gps:
            return {"latitude": None, "longitude": None, "valid": False}
        return self.gps.get_location()
    
    def get_lidar_distance(self) -> Dict[str, Any]:
        """Get LiDAR distance. Returns dict with distance_m, valid."""
        if not self.lidar:
            return {"distance_m": None, "valid": False}
        return self.lidar.get_distance()
    
    def get_mpu_data(self) -> Dict[str, Any]:
        """Get MPU6050 data. Returns dict with accel, gyro, valid."""
        if not self.mpu:
            return {
                "accel_x": None, "accel_y": None, "accel_z": None,
                "gyro_x": None, "gyro_y": None, "gyro_z": None,
                "valid": False
            }
        data = self.mpu.get_data()
        if data.get("valid"):
            self.mpu_error_count = 0
        else:
            self.mpu_error_count += 1
        return data
    
    def is_tilted_dangerously(self) -> bool:
        """Check if the robot is tilted at a dangerous angle."""
        if not self.mpu:
            return False
        return self.mpu.is_tilted_dangerously()
    
    def has_too_many_mpu_errors(self) -> bool:
        """Check if MPU has exceeded error threshold."""
        return self.mpu_error_count >= MAX_MPU_ERROR_COUNT
    
    def get_all_sensor_data(self) -> Dict[str, Any]:
        """Get data from all sensors in a single call."""
        with self._lock:
            return {
                "gps": self.get_gps_location(),
                "lidar": self.get_lidar_distance(),
                "mpu": self.get_mpu_data(),
                "mpu_error_count": self.mpu_error_count,
                "tilted_dangerously": self.is_tilted_dangerously(),
                "mpu_errors_exceeded": self.has_too_many_mpu_errors()
            }
    
    def cleanup(self):
        """Clean up all sensors."""
        self._initialized = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        if self.gps:
            self.gps.stop()
            self.gps = None
            
        if self.lidar:
            self.lidar.stop()
            self.lidar = None
            
        if self.mpu:
            self.mpu.stop()
            self.mpu = None
            
        self.mpu_error_count = 0
        logger.info("All sensors cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


# Convenience function for quick sensor access
_sensor_manager = None

def get_sensor_manager() -> SensorManager:
    """Get or create global sensor manager instance."""
    global _sensor_manager
    if _sensor_manager is None:
        _sensor_manager = SensorManager()
    return _sensor_manager


def cleanup_sensor_manager():
    """Clean up global sensor manager."""
    global _sensor_manager
    if _sensor_manager is not None:
        _sensor_manager.cleanup()
        _sensor_manager = None