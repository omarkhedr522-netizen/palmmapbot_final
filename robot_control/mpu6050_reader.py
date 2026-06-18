"""
mpu6050_reader.py

MPU6050 IMU (accelerometer + gyroscope) reader for PalmMapBot robot.

This module provides accelerometer and gyroscope data from the MPU6050
sensor via I2C. It's used for:
- Tilt detection (safety: stop if robot is tilting dangerously)
- Motion logging (recording robot movement in tree records)

Hardware:
- MPU6050 IMU module
- I2C interface

Default Configuration:
- I2C Address: 0x68 (default MPU6050 address)
- I2C Bus: 1 (Raspberry Pi default)

Wiring:
- MPU6050 VCC -> Raspberry Pi 3.3V
- MPU6050 GND -> Raspberry Pi GND
- MPU6050 SDA -> Raspberry Pi GPIO2 (physical pin 3)
- MPU6050 SCL -> Raspberry Pi GPIO3 (physical pin 5)

Safety:
- Returns valid=False if reading fails
- Does not crash mission on sensor errors
- Configurable tilt threshold for safety stopping
"""

import os
import sys
import logging
import time
import threading
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_I2C_ADDRESS = 0x68  # MPU6050 default I2C address (AD0 pin low)
DEFAULT_I2C_BUS = 1  # Raspberry Pi I2C bus

# MPU6050 registers
MPU6050_PWR_MGMT_1 = 0x6B
MPU6050_ACCEL_CONFIG = 0x1C
MPU6050_GYRO_CONFIG = 0x1B
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_TEMP_OUT_H = 0x41
MPU6050_GYRO_XOUT_H = 0x43

# Accelerometer full scale range options
ACCEL_RANGE_2G = 0x00   # +/- 2g (sensitivity = 16384 LSB/g)
ACCEL_RANGE_4G = 0x08   # +/- 4g (sensitivity = 8192 LSB/g)
ACCEL_RANGE_8G = 0x10   # +/- 8g (sensitivity = 4096 LSB/g)
ACCEL_RANGE_16G = 0x18  # +/- 16g (sensitivity = 2048 LSB/g)

# Gyroscope full scale range options
GYRO_RANGE_250 = 0x00   # +/- 250 deg/s (sensitivity = 131 LSB/deg/s)
GYRO_RANGE_500 = 0x08   # +/- 500 deg/s (sensitivity = 65.5 LSB/deg/s)
GYRO_RANGE_1000 = 0x10  # +/- 1000 deg/s (sensitivity = 32.8 LSB/deg/s)
GYRO_RANGE_2000 = 0x18  # +/- 2000 deg/s (sensitivity = 16.4 LSB/deg/s)

# Safety thresholds
DANGEROUS_TILT_DEGREES = 30.0  # Stop if tilt exceeds this angle

# Try to import I2C library
try:
    from smbus2 import SMBus
    I2C_AVAILABLE = True
except ImportError as e:
    logger.warning(f"smbus2 not available: {e}")
    I2C_AVAILABLE = False

# Set to True to use dummy IMU data for testing
USE_DUMMY_MPU = False


class MPU6050Reader:
    """MPU6050 accelerometer and gyroscope reader."""
    
    def __init__(self, i2c_address=DEFAULT_I2C_ADDRESS, i2c_bus=DEFAULT_I2C_BUS,
                 accel_range=ACCEL_RANGE_2G, gyro_range=GYRO_RANGE_250,
                 poll_interval=0.05):
        """
        Initialize MPU6050 reader.
        
        Args:
            i2c_address: I2C device address (0x68 or 0x69)
            i2c_bus: I2C bus number
            accel_range: Accelerometer range (ACCEL_RANGE_2G, etc.)
            gyro_range: Gyroscope range (GYRO_RANGE_250, etc.)
            poll_interval: Time between readings (seconds)
        """
        self.i2c_address = i2c_address
        self.i2c_bus = i2c_bus
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.poll_interval = poll_interval
        
        self.bus = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest IMU data
        self.latest_data = {
            "connected": False,
            "accel_x": None,
            "accel_y": None,
            "accel_z": None,
            "gyro_x": None,
            "gyro_y": None,
            "gyro_z": None,
            "temperature_c": None,
            "tilt_angle": None,
            "valid": False,
            "timestamp": None,
            "error": "MPU6050 not started"
        }
        
        # Sensitivity values based on range
        self.accel_sensitivity = {
            ACCEL_RANGE_2G: 16384.0,
            ACCEL_RANGE_4G: 8192.0,
            ACCEL_RANGE_8G: 4096.0,
            ACCEL_RANGE_16G: 2048.0
        }
        self.gyro_sensitivity = {
            GYRO_RANGE_250: 131.0,
            GYRO_RANGE_500: 65.5,
            GYRO_RANGE_1000: 32.8,
            GYRO_RANGE_2000: 16.4
        }
        
    def _open_i2c(self):
        """Open I2C bus connection and initialize MPU6050."""
        if not I2C_AVAILABLE:
            self.latest_data["error"] = "smbus2 not installed"
            logger.error(self.latest_data["error"])
            return False
            
        try:
            self.bus = SMBus(self.i2c_bus)
            
            # Check if device is present
            self.bus.read_byte(self.i2c_address)
            
            # Wake up MPU6050 (clear sleep bit)
            self.bus.write_byte_data(self.i2c_address, MPU6050_PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Configure accelerometer range
            self.bus.write_byte_data(self.i2c_address, MPU6050_ACCEL_CONFIG, self.accel_range)
            time.sleep(0.1)
            
            # Configure gyroscope range
            self.bus.write_byte_data(self.i2c_address, MPU6050_GYRO_CONFIG, self.gyro_range)
            time.sleep(0.1)
            
            self.latest_data["connected"] = True
            self.latest_data["error"] = None
            logger.info(f"MPU6050 initialized at address 0x{self.i2c_address:02X}")
            return True
            
        except Exception as e:
            self.latest_data["error"] = f"I2C init error: {str(e)}"
            logger.error(self.latest_data["error"])
            return False
            
    def _close_i2c(self):
        """Close I2C bus connection."""
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None
            
    def _read_word(self, reg):
        """Read a 16-bit value from a register (high byte first)."""
        high = self.bus.read_byte_data(self.i2c_address, reg)
        low = self.bus.read_byte_data(self.i2c_address, reg + 1)
        return (high << 8) | low
        
    def _read_raw_data(self):
        """Read all raw sensor data."""
        # Read accelerometer (6 bytes)
        accel_x_raw = self._read_word(MPU6050_ACCEL_XOUT_H)
        accel_y_raw = self._read_word(MPU6050_ACCEL_XOUT_H + 2)
        accel_z_raw = self._read_word(MPU6050_ACCEL_XOUT_H + 4)
        
        # Read temperature
        temp_raw = self._read_word(MPU6050_TEMP_OUT_H)
        
        # Read gyroscope (6 bytes)
        gyro_x_raw = self._read_word(MPU6050_GYRO_XOUT_H)
        gyro_y_raw = self._read_word(MPU6050_GYRO_XOUT_H + 2)
        gyro_z_raw = self._read_word(MPU6050_GYRO_XOUT_H + 4)
        
        # Convert to signed 16-bit
        def to_signed(value):
            return value if value < 32768 else value - 65536
            
        return {
            "accel_x_raw": to_signed(accel_x_raw),
            "accel_y_raw": to_signed(accel_y_raw),
            "accel_z_raw": to_signed(accel_z_raw),
            "temp_raw": temp_raw,
            "gyro_x_raw": to_signed(gyro_x_raw),
            "gyro_y_raw": to_signed(gyro_y_raw),
            "gyro_z_raw": to_signed(gyro_z_raw)
        }
        
    def _read_once(self):
        """Read and process one set of sensor data."""
        try:
            if self.bus is None:
                self._open_i2c()
                
            raw = self._read_raw_data()
            
            # Get sensitivity values
            accel_sens = self.accel_sensitivity.get(self.accel_range, 16384.0)
            gyro_sens = self.gyro_sensitivity.get(self.gyro_range, 131.0)
            
            # Convert to physical units
            accel_x = raw["accel_x_raw"] / accel_sens
            accel_y = raw["accel_y_raw"] / accel_sens
            accel_z = raw["accel_z_raw"] / accel_sens
            
            gyro_x = raw["gyro_x_raw"] / gyro_sens
            gyro_y = raw["gyro_y_raw"] / gyro_sens
            gyro_z = raw["gyro_z_raw"] / gyro_sens
            
            # Temperature in Celsius
            temperature_c = (raw["temp_raw"] / 340.0) + 36.53
            
            # Calculate tilt angle from accelerometer
            # Using atan2 for better accuracy
            tilt_angle = math.degrees(
                math.atan2(
                    math.sqrt(accel_x**2 + accel_y**2),
                    accel_z
                )
            )
            
            return {
                "connected": True,
                "accel_x": round(accel_x, 6),
                "accel_y": round(accel_y, 6),
                "accel_z": round(accel_z, 6),
                "gyro_x": round(gyro_x, 3),
                "gyro_y": round(gyro_y, 3),
                "gyro_z": round(gyro_z, 3),
                "temperature_c": round(temperature_c, 2),
                "tilt_angle": round(tilt_angle, 2),
                "valid": True,
                "timestamp": time.time(),
                "error": None
            }
            
        except Exception as e:
            self._close_i2c()
            return {
                "connected": False,
                "accel_x": None,
                "accel_y": None,
                "accel_z": None,
                "gyro_x": None,
                "gyro_y": None,
                "gyro_z": None,
                "temperature_c": None,
                "tilt_angle": None,
                "valid": False,
                "timestamp": time.time(),
                "error": f"I2C read error: {str(e)}"
            }
            
    def _read_dummy(self):
        """Return dummy IMU data for testing."""
        import random
        # Simulate slight movement
        return {
            "connected": True,
            "accel_x": round(random.uniform(-0.1, 0.1), 6),
            "accel_y": round(random.uniform(-0.1, 0.1), 6),
            "accel_z": round(random.uniform(0.9, 1.1), 6),  # ~1g on Z axis
            "gyro_x": round(random.uniform(-1.0, 1.0), 3),
            "gyro_y": round(random.uniform(-1.0, 1.0), 3),
            "gyro_z": round(random.uniform(-1.0, 1.0), 3),
            "temperature_c": round(random.uniform(25.0, 35.0), 2),
            "tilt_angle": round(random.uniform(0.0, 5.0), 2),
            "valid": True,
            "timestamp": time.time(),
            "error": None
        }
        
    def _update_loop(self):
        """Background thread for continuous IMU reading."""
        while self.running:
            try:
                if USE_DUMMY_MPU:
                    data = self._read_dummy()
                else:
                    data = self._read_once()
                    
                with self.lock:
                    self.latest_data = data
                    
            except Exception as e:
                with self.lock:
                    self.latest_data["error"] = f"Update error: {str(e)}"
                    self.latest_data["valid"] = False
                    
            time.sleep(self.poll_interval)
            
    def start(self):
        """Start background IMU reading thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("MPU6050 reader started")
        
    def get_data(self):
        """
        Get current IMU data.
        
        Returns:
            dict: Complete IMU data
        """
        with self.lock:
            return dict(self.latest_data)
            
    def get_accel(self):
        """
        Get accelerometer data.
        
        Returns:
            tuple: (accel_x, accel_y, accel_z, valid)
        """
        with self.lock:
            data = dict(self.latest_data)
        return (data.get("accel_x"), data.get("accel_y"), data.get("accel_z"), data.get("valid", False))
        
    def get_gyro(self):
        """
        Get gyroscope data.
        
        Returns:
            tuple: (gyro_x, gyro_y, gyro_z, valid)
        """
        with self.lock:
            data = dict(self.latest_data)
        return (data.get("gyro_x"), data.get("gyro_y"), data.get("gyro_z"), data.get("valid", False))
        
    def get_tilt(self):
        """
        Get tilt angle.
        
        Returns:
            tuple: (tilt_angle_degrees, valid)
        """
        with self.lock:
            data = dict(self.latest_data)
        return (data.get("tilt_angle"), data.get("valid", False))
        
    def is_tilted_dangerously(self, threshold=DANGEROUS_TILT_DEGREES):
        """
        Check if robot is tilted at a dangerous angle.
        
        Args:
            threshold: Tilt angle threshold in degrees
            
        Returns:
            bool: True if tilt exceeds threshold
        """
        with self.lock:
            data = dict(self.latest_data)
            
        if not data.get("valid", False):
            return False
            
        tilt = data.get("tilt_angle", 0.0)
        return tilt is not None and abs(tilt) > threshold
        
    def stop(self):
        """Stop IMU reader and close connection."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._close_i2c()
        self.latest_data["connected"] = False


# Global MPU instance
_mpu = None


def get_mpu_reader(i2c_address=DEFAULT_I2C_ADDRESS):
    """
    Get or create global MPU6050 reader instance.
    
    Args:
        i2c_address: I2C address
        
    Returns:
        MPU6050Reader instance
    """
    global _mpu
    if _mpu is None:
        _mpu = MPU6050Reader(i2c_address=i2c_address)
    return _mpu


def get_mpu6050_data(i2c_address=DEFAULT_I2C_ADDRESS):
    """
    Get current MPU6050 data.
    
    Args:
        i2c_address: I2C address
        
    Returns:
        dict: Complete IMU data
    """
    reader = get_mpu_reader(i2c_address)
    if not reader.running:
        reader.start()
    return reader.get_data()


def stop_mpu():
    """Stop global MPU reader."""
    global _mpu
    if _mpu:
        _mpu.stop()


# Test function
if __name__ == "__main__":
    print("MPU6050 Reader Test")
    print("=" * 40)
    
    # Use dummy MPU for testing
    USE_DUMMY_MPU = True
    
    mpu = MPU6050Reader()
    mpu.start()
    
    print("Reading MPU6050 data (dummy mode)...")
    for i in range(10):
        data = mpu.get_data()
        print(f"  Accel: ({data['accel_x']}, {data['accel_y']}, {data['accel_z']})")
        print(f"  Gyro: ({data['gyro_x']}, {data['gyro_y']}, {data['gyro_z']})")
        print(f"  Tilt: {data['tilt_angle']}°, Temp: {data['temperature_c']}°C")
        print(f"  Dangerous tilt: {mpu.is_tilted_dangerously()}")
        print()
        time.sleep(0.5)
        
    mpu.stop()
    print("\nTest complete!")