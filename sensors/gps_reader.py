import serial
import threading
import time


class GPSReader:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_data = {
            "connected": False,
            "fix": False,
            "fix_quality": None,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "satellites": None,
            "speed_knots": None,
            "speed_kmh": None,
            "heading": None,
            "utc_time": None,
            "date": None,
            "raw_sentence": None,
            "timestamp": None,
            "error": "GPS not started"
        }

    def _open_serial(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )

    def _close_serial(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    @staticmethod
    def _nmea_to_decimal(value: str, direction: str, is_latitude: bool):
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

    @staticmethod
    def _safe_int(value):
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    def _parse_gga(self, parts, raw_line: str) -> dict:
        utc_time = parts[1] if len(parts) > 1 else None
        lat_raw = parts[2] if len(parts) > 2 else None
        lat_dir = parts[3] if len(parts) > 3 else None
        lon_raw = parts[4] if len(parts) > 4 else None
        lon_dir = parts[5] if len(parts) > 5 else None
        fix_quality = self._safe_int(parts[6]) if len(parts) > 6 else None
        satellites = self._safe_int(parts[7]) if len(parts) > 7 else None
        altitude_m = self._safe_float(parts[9]) if len(parts) > 9 else None

        latitude = self._nmea_to_decimal(lat_raw, lat_dir, is_latitude=True)
        longitude = self._nmea_to_decimal(lon_raw, lon_dir, is_latitude=False)

        fix = fix_quality is not None and fix_quality > 0

        return {
            "connected": True,
            "fix": fix,
            "fix_quality": fix_quality,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude_m,
            "satellites": satellites,
            "speed_knots": self.latest_data.get("speed_knots"),
            "speed_kmh": self.latest_data.get("speed_kmh"),
            "heading": self.latest_data.get("heading"),
            "utc_time": utc_time,
            "date": self.latest_data.get("date"),
            "raw_sentence": raw_line,
            "timestamp": time.time(),
            "error": None if fix else "No GPS fix"
        }

    def _parse_rmc(self, parts, raw_line: str) -> dict:
        utc_time = parts[1] if len(parts) > 1 else None
        status = parts[2] if len(parts) > 2 else None
        lat_raw = parts[3] if len(parts) > 3 else None
        lat_dir = parts[4] if len(parts) > 4 else None
        lon_raw = parts[5] if len(parts) > 5 else None
        lon_dir = parts[6] if len(parts) > 6 else None
        speed_knots = self._safe_float(parts[7]) if len(parts) > 7 else None
        heading = self._safe_float(parts[8]) if len(parts) > 8 else None
        date = parts[9] if len(parts) > 9 else None

        latitude = self._nmea_to_decimal(lat_raw, lat_dir, is_latitude=True)
        longitude = self._nmea_to_decimal(lon_raw, lon_dir, is_latitude=False)

        fix = status == "A"
        speed_kmh = round(speed_knots * 1.852, 2) if speed_knots is not None else None

        return {
            "connected": True,
            "fix": fix,
            "fix_quality": self.latest_data.get("fix_quality"),
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": self.latest_data.get("altitude_m"),
            "satellites": self.latest_data.get("satellites"),
            "speed_knots": speed_knots,
            "speed_kmh": speed_kmh,
            "heading": heading,
            "utc_time": utc_time,
            "date": date,
            "raw_sentence": raw_line,
            "timestamp": time.time(),
            "error": None if fix else "No GPS fix"
        }

    def _parse_line(self, line: str):
        if not line.startswith("$"):
            return None

        parts = line.split(",")

        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
            return self._parse_gga(parts, line)

        if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
            return self._parse_rmc(parts, line)

        return None

    def _update_loop(self):
        while self.running:
            try:
                self._open_serial()
                raw_line = self.ser.readline().decode("ascii", errors="replace").strip()

                if not raw_line:
                    continue

                parsed = self._parse_line(raw_line)
                if parsed is not None:
                    with self.lock:
                        previous = dict(self.latest_data)

                        merged = previous
                        for key, value in parsed.items():
                            if value is not None:
                                merged[key] = value

                        # Preserve fix=False if a new parsed sentence explicitly says no fix
                        merged["fix"] = parsed["fix"]
                        merged["connected"] = True
                        merged["raw_sentence"] = parsed["raw_sentence"]
                        merged["timestamp"] = parsed["timestamp"]
                        merged["error"] = parsed["error"]

                        self.latest_data = merged

            except Exception as e:
                with self.lock:
                    self.latest_data = {
                        "connected": False,
                        "fix": False,
                        "fix_quality": None,
                        "latitude": None,
                        "longitude": None,
                        "altitude_m": None,
                        "satellites": None,
                        "speed_knots": None,
                        "speed_kmh": None,
                        "heading": None,
                        "utc_time": None,
                        "date": None,
                        "raw_sentence": None,
                        "timestamp": time.time(),
                        "error": f"{type(e).__name__}: {e}"
                    }
                self._close_serial()
                time.sleep(1)

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def get_data(self) -> dict:
        with self.lock:
            return dict(self.latest_data)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self._close_serial()


gps = GPSReader()


def start_gps():
    gps.start()


def get_gps_data():
    return gps.get_data()


def stop_gps():
    gps.stop()
