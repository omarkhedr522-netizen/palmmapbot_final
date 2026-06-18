"""
robot_control package

Raspberry Pi GPIO-based robot control system for PalmMapBot.

Main components:
- relay_gpio_controller: GPIO relay control (replaces Arduino)
- sensor_manager: Unified sensor management
- config: Centralized configuration
- robot_tree_stop_mission: Main mission runtime
- tree_binary_inference: Binary tree/no-tree classifier
- database: SQLite data logging

Deprecated (for reference only):
- arduino_serial: Old Arduino serial controller (no longer used)
"""

from robot_control.config import (
    GPIO_FORWARD, GPIO_BACKWARD, GPIO_LEFT, GPIO_RIGHT,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    BINARY_MODEL_PATH, BINARY_THRESHOLD,
    DB_PATH, CAPTURED_FRAMES_DIR
)

from robot_control.relay_gpio_controller import (
    RelayGPIOController, get_relay_controller,
    stop_all, cleanup_all
)

from robot_control.sensor_manager import (
    SensorManager, get_sensor_manager, cleanup_sensor_manager
)

__version__ = "2.0.0"
__all__ = [
    # Configuration
    "GPIO_FORWARD", "GPIO_BACKWARD", "GPIO_LEFT", "GPIO_RIGHT",
    "CAMERA_INDEX", "CAMERA_WIDTH", "CAMERA_HEIGHT",
    "BINARY_MODEL_PATH", "BINARY_THRESHOLD",
    "DB_PATH", "CAPTURED_FRAMES_DIR",
    
    # Relay control
    "RelayGPIOController", "get_relay_controller",
    "stop_all", "cleanup_all",
    
    # Sensor management
    "SensorManager", "get_sensor_manager", "cleanup_sensor_manager",
]