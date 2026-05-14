from smbus2 import SMBus
import threading
import time


class LidarReader:
    TF_LUNA_ADDR = 0x10
    REG_DIST_LOW = 0x00
    REG_AMP_LOW = 0x02
    REG_TEMP_LOW = 0x04

    def __init__(self, bus_id: int = 1, poll_interval: float = 0.1):
        self.bus_id = bus_id
        self.poll_interval = poll_interval
        self.bus = None

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_data = {
            "connected": False,
            "distance_cm": None,
            "strength": None,
            "temperature_c": None,
            "timestamp": None,
            "error": "LiDAR not started"
        }

    def _open_bus(self):
        if self.bus is None:
            self.bus = SMBus(self.bus_id)

    def _close_bus(self):
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None

    def _read_word(self, reg_low: int) -> int:
        low = self.bus.read_byte_data(self.TF_LUNA_ADDR, reg_low)
        high = self.bus.read_byte_data(self.TF_LUNA_ADDR, reg_low + 1)
        return low | (high << 8)

    def _read_once(self) -> dict:
        self._open_bus()

        dist_cm = self._read_word(self.REG_DIST_LOW)
        strength = self._read_word(self.REG_AMP_LOW)
        temp_raw = self._read_word(self.REG_TEMP_LOW)

        temp_c = temp_raw / 100.0

        if dist_cm == 0 or dist_cm > 1200:
            # TF-Luna often gives invalid values when target is weak/out of range
            error = f"Unreliable distance reading: {dist_cm} cm"
        else:
            error = None

        return {
            "connected": True,
            "distance_cm": dist_cm,
            "strength": strength,
            "temperature_c": round(temp_c, 2),
            "timestamp": time.time(),
            "error": error
        }

    def _update_loop(self):
        while self.running:
            try:
                data = self._read_once()
            except Exception as e:
                data = {
                    "connected": False,
                    "distance_cm": None,
                    "strength": None,
                    "temperature_c": None,
                    "timestamp": time.time(),
                    "error": f"{type(e).__name__}: {e}"
                }
                self._close_bus()

            with self.lock:
                self.latest_data = data

            time.sleep(self.poll_interval)

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
        self._close_bus()


lidar = LidarReader()


def start_lidar():
    lidar.start()


def get_lidar_data():
    return lidar.get_data()


def stop_lidar():
    lidar.stop()
