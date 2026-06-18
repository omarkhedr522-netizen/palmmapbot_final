"""
lidar_reader.py

LiDAR distance reader for PalmMapBot robot.

This module provides distance reading from a Solid State LiDAR sensor
(8m range, UART/I2C capable). It supports both I2C and UART interfaces.

Hardware:
- Solid State LiDAR Sensor (8m range)
- TF-Luna or compatible
- I2C or UART interface

Default Configuration:
- Interface: I2C (configurable to UART)
- I2C Address: 0x10 (TF-Luna default)
- I2C Bus: 1 (Raspberry Pi default)
- UART Port: /dev/ttyUSB0 (if using UART)
- UART Baud: 115200

Safety:
- Returns valid=False if reading fails
- Does not crash mission on sensor errors
- Configurable distance threshold for safety stopping
"""

import os
import sys
import logging
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_INTERFACE = "i2c"  # or "uart"
DEFAULT_I2C_ADDRESS = 0x10  # TF-Luna I2C address
DEFAULT_I2C_BUS = 1  # Raspberry Pi I2C bus
DEFAULT_UART_PORT = "/dev/ttyUSB0"
DEFAULT_UART_BAUD = 115200

# Safety threshold
TREE_STOP_DISTANCE_METERS = 1.5  # Stop if object within this distance

# Try to import I2C library
try:
    from smbus2 import SMBus
    I2C_AVAILABLE = True
except ImportError as e:
    logger.warning(f"smbus2 not available: {e}")
    I2C_AVAILABLE = False

# Try to import serial for UART mode
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"pyserial not available: {e}")
    SERIAL_AVAILABLE = False

# Set to True to use dummy LiDAR data for testing
USE_DUMMY_LIDAR = False


class LidarReader:
    """LiDAR distance reader supporting I2C and UART interfaces."""
    
    def __init__(self, interface=DEFAULT_INTERFACE, i2c_address=DEFAULT_I2C_ADDRESS,
                 i2c_bus=DEFAULT_I2C_BUS, uart_port=DEFAULT_UART_PORT,
                 uart_baud=DEFAULT_UART_BAUD, poll_interval=0.1):
        """
        Initialize LiDAR reader.
        
        Args:
            interface: "i2c" or "uart"
            i2c_address: I2C device address
            i2c_bus: I2C bus number
            uart_port: Serial port for UART mode
            uart_baud: Baud rate for UART mode
            poll_interval: Time between readings (seconds)
        """
        self.interface = interface
        self.i2c_address = i2c_address
        self.i2c_bus = i2c_bus
        self.uart_port = uart_port
        self.uart_baud = uart_baud
        self.poll_interval = poll_interval
        
        self.bus = None  # I2C bus
        self.ser = None  # UART serial
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest LiDAR data
        self.latest_data = {
            "connected": False,
            "distance_m": None,
            "distance_cm": None,
            "strength": None,
            "temperature_c": None,
            "valid": False,
            "timestamp": None,
            "error": "LiDAR not started"
        }
        
    def _open_i2c(self):
        """Open I2C bus connection."""
        if not I2C_AVAILABLE:
            self.latest_data["error"] = "smbus2 not installed"
            logger.error(self.latest_data["error"])
            return False
            
        try:
            self.bus = SMBus(self.i2c_bus)
            self.latest_data["connected"] = True
            self.latest_data["error"] = None
            logger.info(f"LiDAR I2C opened on bus {self.i2c_bus}, address 0x{self.i2c_address:02X}")
            return True
        except Exception as e:
            self.latest_data["error"] = f"I2C error: {str(e)}"
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
            
    def _open_uart(self):
        """Open UART serial connection."""
        if not SERIAL_AVAILABLE:
            self.latest_data["error"] = "pyserial not installed"
            logger.error(self.latest_data["error"])
            return False
            
        try:
            self.ser = serial.Serial(
                self.uart_port,
                self.uart_baud,
                timeout=1.0
            )
            self.latest_data["connected"] = True
            self.latest_data["error"] = None
            logger.info(f"LiDAR UART opened on {self.uart_port}")
            return True
        except serial.SerialException as e:
            self.latest_data["error"] = f"UART error: {str(e)}"
            logger.error(self.latest_data["error"])
            return False
            
    def _close_uart(self):
        """Close UART serial connection."""
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            
    def _read_i2c_once(self):
        """
        Read distance from LiDAR via I2C.
        
        TF-Luna I2C register map:
        - 0x00-0x01: Distance (low byte first)
        - 0x02-0x03: Signal strength (low byte first)
        - 0x04-0x05: Temperature (low byte first, divide by 100)
        """
        try:
            if self.bus is None:
                self._open_i2c()
                
            # Read distance (2 bytes, low byte first)
            dist_low = self.bus.read_byte_data(self.i2c_address, 0x00)
            dist_high = self.bus.read_byte_data(self.i2c_address, 0x01)
            distance_cm = dist_low | (dist_high << 8)
            
            # Read signal strength
            strength_low = self.bus.read_byte_data(self.i2c_address, 0x02)
            strength_high = self.bus.read_byte_data(self.i2c_address, 0x03)
            strength = strength_low | (strength_high << 8)
            
            # Read temperature
            temp_low = self.bus.read_byte_data(self.i2c_address, 0x04)
            temp_high = self.bus.read_byte_data(self.i2c_address, 0x05)
            temperature_raw = temp_low | (temp_high << 8)
            temperature_c = temperature_raw / 100.0
            
            # Validate distance
            valid = True
            error = None
            
            if distance_cm == 0 or distance_cm > 1200:  # 12m max (sensor reports up to 12m)
                valid = False
                error = f"Unreliable reading: {distance_cm} cm"
                
            return {
                "connected": True,
                "distance_m": round(distance_cm / 100.0, 3),
                "distance_cm": distance_cm,
                "strength": strength,
                "temperature_c": round(temperature_c, 2),
                "valid": valid,
                "timestamp": time.time(),
                "error": error
            }
            
        except Exception as e:
            self._close_i2c()
            return {
                "connected": False,
                "distance_m": None,
                "distance_cm": None,
                "strength": None,
                "temperature_c": None,
                "valid": False,
                "timestamp": time.time(),
                "error": f"I2C read error: {str(e)}"
            }
            
    def _read_uart_once(self):
        """
        Read distance from LiDAR via UART.
        
        Note: UART protocol varies by sensor model.
        This is a placeholder that may need adjustment for your specific sensor.
        
        Typical TF-Luna UART frame (9600 baud, not 115200):
        0x59 0x59 [length] [distance_low] [distance_high] ... [checksum]
        """
        try:
            if self.ser is None:
                self._open_uart()
                
            # Read available data
            if self.ser.in_waiting >= 9:
                header = self.ser.read(2)
                if header == b'\x59\x59':
                    frame = header + self.ser.read(7)
                    
                    # Parse frame
                    if len(frame) == 9:
                        distance_cm = frame[3] | (frame[4] << 8)
                        strength = frame[5] | (frame[6] << 8)
                        
                        valid = 0 < distance_cm <= 1200
                        
                        return {
                            "connected": True,
                            "distance_m": round(distance_cm / 100.0, 3),
                            "distance_cm": distance_cm,
                            "strength": strength,
                            "temperature_c": None,
                            "valid": valid,
                            "timestamp": time.time(),
                            "error": None if valid else f"Invalid reading: {distance_cm} cm"
                        }
                        
            return {
                "connected": True,
                "distance_m": None,
                "distance_cm": None,
                "strength": None,
                "temperature_c": None,
                "valid": False,
                "timestamp": time.time(),
                "error": "No valid UART frame"
            }
            
        except Exception as e:
            self._close_uart()
            return {
                "connected": False,
                "distance_m": None,
                "distance_cm": None,
                "strength": None,
                "temperature_c": None,
                "valid": False,
                "timestamp": time.time(),
                "error": f"UART read error: {str(e)}"
            }
            
    def _read_dummy(self):
        """Return dummy LiDAR data for testing."""
        import random
        # Simulate distance varying between 1.0 and 5.0 meters
        distance = random.uniform(1.0, 5.0)
        return {
            "connected": True,
            "distance_m": round(distance, 3),
            "distance_cm": int(distance * 100),
            "strength": random.randint(100, 1000),
            "temperature_c": round(random.uniform(20.0, 30.0), 2),
            "valid": True,
            "timestamp": time.time(),
            "error": None
        }
        
    def _update_loop(self):
        """Background thread for continuous LiDAR reading."""
        while self.running:
            try:
                if USE_DUMMY_LIDAR:
                    data = self._read_dummy()
                elif self.interface == "i2c":
                    data = self._read_i2c_once()
                else:
                    data = self._read_uart_once()
                    
                with self.lock:
                    self.latest_data = data
                    
            except Exception as e:
                with self.lock:
                    self.latest_data["error"] = f"Update error: {str(e)}"
                    self.latest_data["valid"] = False
                    
            time.sleep(self.poll_interval)
            
    def start(self):
        """Start background LiDAR reading thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("LiDAR reader started")
        
    def get_distance(self):
        """
        Get current distance reading.
        
        Returns:
            tuple: (distance_meters, valid)
        """
        with self.lock:
            data = dict(self.latest_data)
        return (data.get("distance_m"), data.get("valid", False))
        
    def get_data(self):
        """
        Get full LiDAR data dictionary.
        
        Returns:
            dict: Complete LiDAR data
        """
        with self.lock:
            return dict(self.latest_data)
            
    def stop(self):
        """Stop LiDAR reader and close connections."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._close_i2c()
        self._close_uart()
        self.latest_data["connected"] = False


# Global LiDAR instance
_lidar = None


def get_lidar_reader(interface=DEFAULT_INTERFACE):
    """
    Get or create global LiDAR reader instance.
    
    Args:
        interface: "i2c" or "uart"
        
    Returns:
        LidarReader instance
    """
    global _lidar
    if _lidar is None:
        _lidar = LidarReader(interface=interface)
    return _lidar


def get_lidar_distance(interface=DEFAULT_INTERFACE):
    """
    Get current LiDAR distance.
    
    Args:
        interface: "i2c" or "uart"
        
    Returns:
        tuple: (distance_meters, valid)
    """
    reader = get_lidar_reader(interface)
    if not reader.running:
        reader.start()
    return reader.get_distance()


def stop_lidar():
    """Stop global LiDAR reader."""
    global _lidar
    if _lidar:
        _lidar.stop()


# Test function
if __name__ == "__main__":
    print("LiDAR Reader Test")
    print("=" * 40)
    
    # Use dummy LiDAR for testing
    USE_DUMMY_LIDAR = True
    
    lidar = LidarReader()
    lidar.start()
    
    print("Reading LiDAR data (dummy mode)...")
    for i in range(10):
        distance, valid = lidar.get_distance()
        data = lidar.get_data()
        print(f"  Distance: {distance}m, Valid: {valid}, "
              f"Strength: {data.get('strength')}, "
              f"Temp: {data.get('temperature_c')}°C")
        time.sleep(0.5)
        
    lidar.stop()
    print("\nTest complete!")