"""
robot_tree_stop_mission.py

Main mission runtime for PalmMapBot robot.

This is the final mission file that orchestrates:
- Camera feed reading
- Binary tree/no-tree detection for immediate stopping
- YOLO palm tree detection after stopping
- LiDAR distance reading
- GPS location reading
- MPU6050 IMU reading
- SQLite database logging
- Raspberry Pi GPIO relay control for robot movement

Mission Logic:
==============
1. Initialize relay GPIO (STOP is default - all relays OFF)
2. Wait for dashboard Start Mission click
3. Send FWD via relay GPIO
4. Start reading camera frames
5. For each frame:
   a. Run binary tree detector
   b. If tree detected: STOP immediately
   c. After stopping: Read LiDAR, run YOLO, read GPS, read MPU6050
   d. If YOLO confirms palm: Save image, store in database
   e. Send FWD again (if mission still active)
6. Repeat until Stop Mission clicked or Ctrl+C

Safety:
=======
- Relays initialize OFF (car never moves on startup)
- Sends STOP on any exception
- Sends STOP on Ctrl+C
- Checks for dangerous tilt
- Validates all sensor readings
- Does not crash on sensor failures
- Dashboard-gated: only moves when Start Mission is clicked
"""

import os
import sys
import time
import logging
import signal
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import robot control modules
from robot_control.relay_gpio_controller import RelayGPIOController, stop_all, cleanup_all
from robot_control.tree_binary_inference import BinaryTreeClassifier, DEFAULT_MODEL_PATH as BINARY_MODEL_PATH
from robot_control.database import TreeDatabase, get_db_path
from robot_control.gps_reader import GPSReader, USE_DUMMY_GPS, DEFAULT_PORT as GPS_PORT
from robot_control.lidar_reader import LidarReader, USE_DUMMY_LIDAR
from robot_control.mpu6050_reader import MPU6050Reader, USE_DUMMY_MPU, DANGEROUS_TILT_DEGREES
from robot_control.map_builder import estimate_tree_position_simple

# Import YOLO from ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics YOLO not available")
    YOLO_AVAILABLE = False

# Configuration
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tree_binary_classifier.pt")
BINARY_THRESHOLD = float(os.getenv("BINARY_THRESHOLD", "0.60"))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "palmmapbot.db")
CAPTURED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "captured_frames")
BINARY_CHECK_INTERVAL = 0.1
STOP_SETTLE_TIME = 1.0
MAX_MPU_ERROR_COUNT = 5


class PalmMapBotMission:
    """Main mission controller for PalmMapBot."""
    
    def __init__(self):
        self.running = False
        self.mpu_error_count = 0
        self.mission_active = False
        self.emergency_stop = False
        self.mode = "IDLE"
        self.relay_controller = None
        self.camera = None
        self.binary_classifier = None
        self.yolo_model = None
        self.database = None
        self.gps = None
        self.lidar = None
        self.mpu = None
        self.state = "initializing"
        self.last_binary_check = 0
        self.tree_detected = False
        
    def setup(self):
        logger.info("Setting up PalmMapBot mission...")
        os.makedirs(CAPTURED_FRAMES_DIR, exist_ok=True)
        
        logger.info("Initializing Raspberry Pi GPIO relay controller...")
        try:
            self.relay_controller = RelayGPIOController()
            self.relay_controller.stop()
            logger.info("Relay GPIO controller initialized - all relays OFF")
        except Exception as e:
            logger.error(f"Failed to initialize relay controller: {e}")
            raise
            
        logger.info(f"Opening camera {CAMERA_INDEX}...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            self.cleanup()
            raise RuntimeError("Camera not available")
        logger.info("Camera opened")
        
        logger.info(f"Loading binary tree classifier from {BINARY_MODEL_PATH}...")
        if os.path.exists(BINARY_MODEL_PATH):
            self.binary_classifier = BinaryTreeClassifier(BINARY_MODEL_PATH, BINARY_THRESHOLD)
            if not self.binary_classifier.load_model():
                logger.error("Failed to load binary classifier")
                self.cleanup()
                raise RuntimeError("Binary classifier not available")
            logger.info("Binary classifier loaded")
        else:
            logger.warning(f"Binary classifier model not found: {BINARY_MODEL_PATH}")
            self.cleanup()
            raise FileNotFoundError("Binary classifier model not found")
            
        if YOLO_AVAILABLE:
            yolo_path = os.path.join(PROJECT_ROOT, "models", "palm_tree_detector.pt")
            if os.path.exists(yolo_path):
                logger.info(f"Loading YOLO model from {yolo_path}...")
                self.yolo_model = YOLO(yolo_path)
                logger.info("YOLO model loaded")
            else:
                logger.warning(f"YOLO model not found: {yolo_path}")
        else:
            logger.warning("YOLO not available")
            
        logger.info(f"Initializing database at {DB_PATH}...")
        self.database = TreeDatabase(DB_PATH)
        self.database.connect()
        logger.info("Database ready")
        
        logger.info("Initializing GPS...")
        self.gps = GPSReader(GPS_PORT, use_dummy=USE_DUMMY_GPS)
        self.gps.start()
        logger.info("GPS started")
        
        logger.info("Initializing LiDAR...")
        self.lidar = LidarReader()
        self.lidar.start()
        logger.info("LiDAR started")
        
        logger.info("Initializing MPU6050...")
        self.mpu = MPU6050Reader()
        self.mpu.start()
        logger.info("MPU6050 started")
        logger.info("All components initialized!")
        
    def cleanup(self):
        logger.info("Cleaning up...")
        if self.relay_controller:
            try:
                self.relay_controller.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up relay controller: {e}")
        else:
            stop_all()
            cleanup_all()
        if self.camera:
            self.camera.release()
        if self.gps:
            self.gps.stop()
        if self.lidar:
            self.lidar.stop()
        if self.mpu:
            self.mpu.stop()
        if self.database:
            self.database.close()
        logger.info("Cleanup complete")
        
    def save_frame(self, frame, prefix="tree"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(CAPTURED_FRAMES_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath
        
    def run_yolo_detection(self, frame):
        if self.yolo_model is None:
            return None, 0.0, None
        try:
            results = self.yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_idx = 0
                best_conf = 0
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_idx = i
                best_box = boxes[best_idx]
                class_name = self.yolo_model.names[int(best_box.cls[0])]
                bbox = best_box.xyxy[0].cpu().numpy().astype(int).tolist()
                return class_name, best_conf, bbox
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        return None, 0.0, None
        
    def process_tree_detection(self, frame):
        logger.info("TREE DETECTED - Processing...")
        time.sleep(STOP_SETTLE_TIME)
        lidar_dist, lidar_valid = self.lidar.get_distance()
        logger.info(f"LiDAR: {lidar_dist}m (valid: {lidar_valid})")
        class_name, yolo_conf, bbox = self.run_yolo_detection(frame)
        if class_name and yolo_conf > 0:
            logger.info(f"YOLO confirmed: {class_name} ({yolo_conf:.2f})")
            image_path = self.save_frame(frame)
            logger.info(f"Saved frame: {image_path}")
            lat, lon, gps_valid = self.gps.get_location()
            logger.info(f"GPS: ({lat}, {lon}) valid={gps_valid}")
            mpu_data = self.mpu.get_data()
            self.mpu_error_count = 0 if mpu_data.get("valid") else self.mpu_error_count + 1
            if self.mpu.is_tilted_dangerously():
                logger.warning("DANGEROUS TILT DETECTED - Stopping mission!")
                return "dangerous_tilt"
            est_lat, est_lon, mapping_mode = estimate_tree_position_simple(lat, lon, lidar_dist, lidar_valid)
            record_id = self.database.insert(
                latitude=lat, longitude=lon, gps_valid=gps_valid,
                estimated_latitude=est_lat, estimated_longitude=est_lon,
                tree_present_confidence=self.binary_classifier.threshold,
                lidar_distance_m=lidar_dist, lidar_valid=lidar_valid,
                accel_x=mpu_data.get("accel_x"), accel_y=mpu_data.get("accel_y"),
                accel_z=mpu_data.get("accel_z"), gyro_x=mpu_data.get("gyro_x"),
                gyro_y=mpu_data.get("gyro_y"), gyro_z=mpu_data.get("gyro_z"),
                mpu_valid=mpu_data.get("valid"), yolo_confidence=yolo_conf,
                class_name=class_name,
                bbox_x1=bbox[0] if bbox else None, bbox_y1=bbox[1] if bbox else None,
                bbox_x2=bbox[2] if bbox else None, bbox_y2=bbox[3] if bbox else None,
                image_path=image_path, robot_status="stopped", mapping_mode=mapping_mode
            )
            logger.info(f"TREE STORED - Database ID: {record_id}")
        else:
            logger.info("TREE YES/NO DETECTOR TRIGGERED BUT YOLO DID NOT CONFIRM")
        return "processed"
        
    def run_mission(self):
        logger.info("Starting mission loop (waiting for dashboard Start Mission)...")
        self.running = True
        self.state = "IDLE"
        self.mission_active = False
        self.relay_controller.stop()
        time.sleep(0.5)
        while self.running and not self.mission_active:
            time.sleep(0.1)
            if self.emergency_stop:
                logger.warning("Emergency stop active - waiting for reset")
                continue
        if not self.running:
            return
        self.relay_controller.forward()
        self.state = "moving_forward"
        self.mode = "AUTO"
        logger.info("Mission STARTED - Moving FORWARD")
        frame_count = 0
        detection_count = 0
        while self.running:
            try:
                if not self.mission_active or self.emergency_stop:
                    self.relay_controller.stop()
                    self.state = "STOPPED"
                    while self.running and (not self.mission_active or self.emergency_stop):
                        time.sleep(0.1)
                    if not self.running:
                        break
                    self.relay_controller.forward()
                    self.state = "moving_forward"
                    logger.info("Mission resumed - Moving FORWARD")
                    continue
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.error("Failed to read camera frame")
                    self.relay_controller.stop()
                    time.sleep(1)
                    continue
                frame_count += 1
                if self.mpu and self.mpu.is_tilted_dangerously():
                    logger.warning("Dangerous tilt detected - stopping!")
                    self.relay_controller.stop()
                    self.running = False
                    break
                if self.mpu_error_count >= MAX_MPU_ERROR_COUNT:
                    logger.error("Too many MPU errors - stopping for safety")
                    self.relay_controller.stop()
                    self.running = False
                    break
                current_time = time.time()
                if current_time - self.last_binary_check >= BINARY_CHECK_INTERVAL:
                    self.last_binary_check = current_time
                    tree_present, tree_confidence = self.binary_classifier.detect_tree_presence(frame)
                    if tree_present:
                        logger.info(f"TREE DETECTED - confidence: {tree_confidence:.4f}")
                        detection_count += 1
                        self.relay_controller.stop()
                        self.state = "stopped_for_tree"
                        result = self.process_tree_detection(frame)
                        if result == "dangerous_tilt":
                            logger.warning("Stopping mission due to dangerous tilt")
                            self.running = False
                            break
                        if self.mission_active and not self.emergency_stop:
                            self.relay_controller.forward()
                            self.state = "moving_forward"
                            logger.info("Resuming FORWARD movement")
                        else:
                            logger.info("Mission not active - staying stopped")
                time.sleep(0.01)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Mission loop error: {e}")
                self.relay_controller.stop()
                self.state = "ERROR"
                time.sleep(1)
        self.relay_controller.stop()
        self.state = "stopped"
        self.mission_active = False
        logger.info(f"Mission ended. Processed {frame_count} frames, detected {detection_count} trees.")
        
    # Dashboard control methods
    def start_mission(self):
        logger.info("Start Mission clicked from dashboard")
        self.emergency_stop = False
        self.mission_active = True
        self.mode = "AUTO"
        
    def stop_mission(self):
        logger.info("Stop Mission clicked from dashboard")
        self.mission_active = False
        self.mode = "STOPPED"
        self.relay_controller.stop()
        
    def emergency_stop_mission(self):
        logger.warning("EMERGENCY STOP clicked from dashboard!")
        self.emergency_stop = True
        self.mission_active = False
        self.mode = "EMERGENCY_STOP"
        self.relay_controller.stop()
        
    def switch_to_manual(self):
        logger.info("Switching to Manual Control")
        self.relay_controller.stop()
        self.mission_active = False
        self.mode = "MANUAL"
        
    def switch_to_auto(self):
        logger.info("Switching to Auto mode (ready)")
        self.relay_controller.stop()
        self.mission_active = False
        self.mode = "IDLE"
        
    def manual_forward(self):
        if self.mode == "MANUAL" and not self.emergency_stop:
            self.relay_controller.forward()
            return True
        return False
        
    def manual_backward(self):
        if self.mode == "MANUAL" and not self.emergency_stop:
            self.relay_controller.backward()
            return True
        return False
        
    def manual_left(self):
        if self.mode == "MANUAL" and not self.emergency_stop:
            self.relay_controller.left()
            return True
        return False
        
    def manual_right(self):
        if self.mode == "MANUAL" and not self.emergency_stop:
            self.relay_controller.right()
            return True
        return False
        
    def manual_stop(self):
        self.relay_controller.stop()
        
    def get_status(self):
        return {
            "mode": self.mode,
            "mission_active": self.mission_active,
            "emergency_stop": self.emergency_stop,
            "state": self.state,
            "running": self.running
        }
        
    def start(self):
        try:
            self.setup()
            self.run_mission()
        except Exception as e:
            logger.error(f"Mission failed: {e}")
        finally:
            self.cleanup()


# Global mission instance for dashboard access
_mission = None

def get_mission():
    global _mission
    if _mission is None:
        _mission = PalmMapBotMission()
    return _mission

def signal_handler(sig, frame):
    logger.info("Received interrupt signal")
    mission = get_mission()
    if mission:
        mission.running = False
        mission.emergency_stop = True

def main():
    print("=" * 60)
    print("PalmMapBot - Tree Detection Mission")
    print("=" * 60)
    print()
    print("Mission Logic:")
    print("  MOVE FORWARD")
    print("  -> BINARY TREE YES/NO DETECTION")
    print("  -> IF TREE YES: STOP IMMEDIATELY")
    print("  -> READ LiDAR DISTANCE")
    print("  -> RUN YOLO PALM DETECTION")
    print("  -> READ GPS + MPU6050")
    print("  -> STORE ALL DATA")
    print("  -> MOVE FORWARD AGAIN")
    print("  -> REPEAT")
    print()
    print("Safety:")
    print("  - Press Ctrl+C to stop")
    print("  - Robot will STOP on any error")
    print("  - Dangerous tilt will trigger emergency stop")
    print()
    print("=" * 60)
    print()
    mission = PalmMapBotMission()
    signal_handler.mission = mission
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    mission.start()
    print()
    print("Mission complete!")
    print(f"Database: {DB_PATH}")
    print(f"Captured frames: {CAPTURED_FRAMES_DIR}")

if __name__ == "__main__":
    main()