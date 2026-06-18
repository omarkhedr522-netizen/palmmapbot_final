"""
config.py

Centralized configuration for PalmMapBot robot control system.

All hardware pins, thresholds, and settings are defined here
for easy modification and consistency across modules.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CAPTURED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "captured_frames")

# Database
DB_PATH = os.path.join(DATA_DIR, "palmmapbot.db")

# =============================================================================
# GPIO Configuration (Raspberry Pi BCM numbering)
# =============================================================================

GPIO_FORWARD = 17
GPIO_BACKWARD = 27
GPIO_LEFT = 22
GPIO_RIGHT = 23
GPIO_ACTIVE_LOW = True  # Most relay modules are active low

# =============================================================================
# Camera Configuration
# =============================================================================

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# =============================================================================
# Binary Tree Classifier Configuration
# =============================================================================

BINARY_MODEL_PATH = os.path.join(MODELS_DIR, "tree_binary_classifier.pt")
BINARY_THRESHOLD = float(os.getenv("BINARY_THRESHOLD", "0.60"))
BINARY_CHECK_INTERVAL = float(os.getenv("BINARY_CHECK_INTERVAL", "0.1"))

# =============================================================================
# YOLO Configuration
# =============================================================================

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "palm_tree_detector.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
YOLO_AVAILABLE = True

# =============================================================================
# GPS Configuration
# =============================================================================

GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyUSB0")
GPS_BAUD = int(os.getenv("GPS_BAUD", "9600"))
USE_DUMMY_GPS = os.getenv("USE_DUMMY_GPS", "true").lower() == "true"
DUMMY_GPS_LAT = float(os.getenv("DUMMY_GPS_LAT", "29.203451"))
DUMMY_GPS_LON = float(os.getenv("DUMMY_GPS_LON", "25.519833"))

# =============================================================================
# LiDAR Configuration
# =============================================================================

LIDAR_PORT = os.getenv("LIDAR_PORT", "/dev/ttyUSB1")
LIDAR_BAUD = int(os.getenv("LIDAR_BAUD", "115200"))
USE_DUMMY_LIDAR = os.getenv("USE_DUMMY_LIDAR", "true").lower() == "true"
DUMMY_LIDAR_DISTANCE = float(os.getenv("DUMMY_LIDAR_DISTANCE", "2.0"))
TREE_STOP_DISTANCE_M = float(os.getenv("TREE_STOP_DISTANCE_M", "1.5"))

# =============================================================================
# MPU6050 Configuration
# =============================================================================

MPU_I2C_ADDRESS = 0x68
I2C_BUS = int(os.getenv("I2C_BUS", "1"))
USE_DUMMY_MPU = os.getenv("USE_DUMMY_MPU", "true").lower() == "true"
DANGEROUS_TILT_DEGREES = float(os.getenv("DANGEROUS_TILT_DEGREES", "30.0"))
MAX_MPU_ERROR_COUNT = int(os.getenv("MAX_MPU_ERROR_COUNT", "5"))

# =============================================================================
# Mission Configuration
# =============================================================================

STOP_SETTLE_TIME = float(os.getenv("STOP_SETTLE_TIME", "1.0"))
MISSION_CHECK_INTERVAL = float(os.getenv("MISSION_CHECK_INTERVAL", "0.01"))

# =============================================================================
# Dashboard Configuration
# =============================================================================

DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
DASHBOARD_REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5"))

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"