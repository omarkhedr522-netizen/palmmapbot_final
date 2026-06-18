"""
gps_reader.py

GPS reader module for PalmMapBot robot.

This module provides GPS location reading functionality.
It can work with real GPS hardware (NEO-6M, etc.) or provide
placeholder/dummy data for testing.

Hardware Support:
- NEO-6M GPS module (via serial/USB)
- Any NMEA-compatible GPS receiver

Default Configuration:
- Port: /dev/ttyUSB0 (configurable)
- Baud Rate: 9600
- Timeout: 1 second

For testing without hardware, set USE_DUMMY_GPS=True or
the module will return dummy coordinates.
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
DEFAULT_PORT = "/dev/ttyUSB0"  # Raspberry Pi default for USB GPS
DEFAULT_BAUD_RATE = 9600
DEFAULT_TIMEOUT = 1.0

# Try to import serial
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"pyserial not available: {e}")
    SERIAL_AVAILABLE = False

# Set to True to use dummy GPS data for testing
USE_DUMMY_GPS = False

# Dummy GPS coordinates (example: somewhere in a palm farm)
DUMMY_LATITUDE = 29.203451
DUMMY_LONGITUDE = 25.519833


class GPSReader:
    """GPS location reader with NMEA parsing."""
    
    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUD_RATE, 
                 timeout=DEFAULT_TIMEOUT, use_dummy=USE_DUMMY_GPS):
        """
        Initialize GPS reader.
        
        Args:
            port: Serial port (e.g., /dev/ttyUSB0)
            baudrate: Baud rate
            timeout: Serial timeout in seconds
            use_dummy: If True, return dummy coordinates for testing
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.use_dummy = use_dummy
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest GPS data
        self.latest_data = {
            "connected": False,
            "fix": False,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "satellites": None,
            "speed_kmh": None,
            "heading": None,
            "timestamp": None,
            "error": "GPS not started"
        }
        
    def connect(self):
        """
        Connect to GPS module.
        
        Returns:
            bool: True if connected successfully
        """
        if self.use_dummy:
            logger.info("Using dummy GPS mode")
            self.latest_data = {
                "connected": True,
                "fix": True,
                "latitude": DUMMY_LATITUDE,
                "longitude": DUMMY_LONGITUDE,
                "altitude_m": 100.0,
                "satellites": 8,
                "speed_kmh": 0.0,
                "heading": 0.0,
                "timestamp": time.time(),
                "error": None
            }
            return True
            
        if not SERIAL_AVAILABLE:
            self.latest_data["error"] = "pyserial not installed"
            logger.error(self.latest_data["error"])
            return False
            
        try:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            self.latest_data["connected"] = True
            self.latest_data["error"] = None
            logger.info(f"GPS connected on {self.port}")
            return True
            
        except serial.SerialException as e:
            self.latest_data["error"] = f"Serial error: {str(e)}"
            logger.error(self.latest_data["error"])
            return False
            
    def _parse_nmea_line(self, line):
        """
        Parse an NMEA sentence.
        
        Args:
            line: NMEA sentence string
            
        Returns:
            dict: Parsed GPS data
        """
        if not line.startswith("$"):
            return None
            
        parts = line.split(",")
        if len(parts) < 6:
            return None
            
        # Parse GGA sentence (most common for position)
        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
            return self._parse_gga(parts, line)
            
        # Parse RMC sentence
        if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
            return self._parse_rmc(parts, line)
            
        return None
        
    def _parse_gga(self, parts, raw_line):
        """Parse GGA NMEA sentence."""
        try:
            lat_raw = parts[2] if len(parts) > 2 else None
            lat_dir = parts[3] if len(parts) > 3 else None
            lon_raw = parts[4] if len(parts) > 4 else None
            lon_dir = parts[5] if len(parts) > 5 else None
            fix_quality = int(parts[6]) if len(parts) > 6 and parts[6] else None
            satellites = int(parts[7]) if len(parts) > 7 and parts[7] else None
            altitude = float(parts[9]) if len(parts) > 9 and parts[9] else None
            
            latitude = self._nmea_to_decimal(lat_raw, lat_dir, is_latitude=True)
            longitude = self._nmea_to_decimal(lon_raw, lon_dir, is_latitude=False)
            
            fix = fix_quality is not None and fix_quality > 0
            
            return {
                "connected": True,
                "fix": fix,
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": altitude,
                "satellites": satellites,
                "timestamp": time.time(),
                "error": None if fix else "No GPS fix"
            }
        except Exception as e:
            logger.error(f"GGA parse error: {e}")
            return None
            
    def _parse_rmc(self, parts, raw_line):
        """Parse RMC NMEA sentence."""
        try:
            status = parts[2] if len(parts) > 2 else None
            lat_raw = parts[3] if len(parts) > 3 else None
            lat_dir = parts[4] if len(parts) > 4 else None
            lon_raw = parts[5] if len(parts) > 5 else None
            lon_dir = parts[6] if len(parts) > 6 else None
            speed_knots = float(parts[7]) if len(parts) > 7 and parts[7] else None
            heading = float(parts[8]) if len(parts) > 8 and parts[8] else None
            
            latitude = self._nmea_to_decimal(lat_raw, lat_dir, is_latitude=True)
            longitude = self._nmea_to_decimal(lon_raw, lon_dir, is_latitude=False)
            
            fix = status == "A"
            speed_kmh = round(speed_knots * 1.852, 2) if speed_knots else None
            
            return {
                "connected": True,
                "fix": fix,
                "latitude": latitude,
                "longitude": longitude,
                "speed_kmh": speed_kmh,
                "heading": heading,
                "timestamp": time.time(),
                "error": None if fix else "No GPS fix"
            }
        except Exception as e:
            logger.error(f"RMC parse error: {e}")
            return None
            
    @staticmethod
    def _nmea_to_decimal(value, direction, is_latitude):
        """
        Convert NMEA coordinate format to decimal degrees.
        
        NMEA format: DDMM.MMMM or DDDMM.MMMM
        """
        if not value or not direction:
            return None
            
        try:
            if is_latitude:
                deg_len = 2
            else:
                deg_len = 3
                
            degrees = int(value[:deg_len])
            minutes = float(value[deg_len:])
            decimal = degrees + (minutes / 60.0)
            
            if direction in ("S", "W"):
                decimal *= -1
                
            return round(decimal, 7)
        except Exception:
            return None
            
    def _update_loop(self):
        """Background thread for continuous GPS reading."""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('ascii', errors='replace').strip()
                    if line:
                        parsed = self._parse_nmea_line(line)
                        if parsed:
                            with self.lock:
                                # Merge new data with existing
                                for key, value in parsed.items():
                                    if value is not None:
                                        self.latest_data[key] = value
            except Exception as e:
                with self.lock:
                    self.latest_data["error"] = f"Read error: {str(e)}"
                time.sleep(1)
                
    def start(self):
        """Start background GPS reading thread."""
        if self.running:
            return
            
        self.connect()
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
    def get_location(self):
        """
        Get current GPS location.
        
        Returns:
            tuple: (latitude, longitude, valid)
        """
        with self.lock:
            data = dict(self.latest_data)
            
        if self.use_dummy:
            # Update dummy timestamp
            data["timestamp"] = time.time()
            
        return (
            data.get("latitude"),
            data.get("longitude"),
            data.get("fix", False)
        )
        
    def get_data(self):
        """
        Get full GPS data dictionary.
        
        Returns:
            dict: Complete GPS data
        """
        with self.lock:
            return dict(self.latest_data)
            
    def stop(self):
        """Stop GPS reader and close connection."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.latest_data["connected"] = False


# Global GPS instance
_gps = None


def get_gps_reader(port=DEFAULT_PORT, use_dummy=USE_DUMMY_GPS):
    """
    Get or create global GPS reader instance.
    
    Args:
        port: Serial port
        use_dummy: Use dummy data
        
    Returns:
        GPSReader instance
    """
    global _gps
    if _gps is None:
        _gps = GPSReader(port, use_dummy=use_dummy)
    return _gps


def get_gps_location(port=DEFAULT_PORT, use_dummy=USE_DUMMY_GPS):
    """
    Get current GPS location.
    
    Args:
        port: Serial port
        use_dummy: Use dummy data
        
    Returns:
        tuple: (latitude, longitude, valid)
    """
    reader = get_gps_reader(port, use_dummy)
    if not reader.running:
        reader.start()
    return reader.get_location()


def stop_gps():
    """Stop global GPS reader."""
    global _gps
    if _gps:
        _gps.stop()


# Test function
if __name__ == "__main__":
    print("GPS Reader Test")
    print("=" * 40)
    
    # Use dummy GPS for testing
    gps = GPSReader(use_dummy=True)
    gps.start()
    
    print("Reading GPS data (dummy mode)...")
    for i in range(5):
        lat, lon, valid = gps.get_location()
        print(f"  Lat: {lat}, Lon: {lon}, Valid: {valid}")
        time.sleep(1)
        
    gps.stop()
    print("\nTest complete!")