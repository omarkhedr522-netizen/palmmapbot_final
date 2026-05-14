import sys
import os
import sqlite3
import atexit
import time
import threading

import cv2
import numpy as np

# add project root to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from flask import Flask, jsonify, redirect, url_for, Response, render_template_string

from backend.controller_instance import controller

# --- Optional sensor imports: keep dashboard alive even if a sensor stack fails ---
LIDAR_IMPORT_ERROR = None
GPS_IMPORT_ERROR = None
CAMERA_IMPORT_ERROR = None

try:
    from sensors.lidar_reader import start_lidar, get_lidar_data, stop_lidar
except Exception as e:
    LIDAR_IMPORT_ERROR = str(e)

    def start_lidar():
        return None

    def stop_lidar():
        return None

    def get_lidar_data():
        return {
            "connected": False,
            "distance_cm": None,
            "strength": None,
            "temperature_c": None,
            "error": LIDAR_IMPORT_ERROR
        }

try:
    from sensors.gps_reader import start_gps, get_gps_data, stop_gps
except Exception as e:
    GPS_IMPORT_ERROR = str(e)

    def start_gps():
        return None

    def stop_gps():
        return None

    def get_gps_data():
        return {
            "connected": False,
            "fix": False,
            "latitude": None,
            "longitude": None,
            "satellites": None,
            "altitude_m": None,
            "speed_kmh": None,
            "heading": None,
            "error": GPS_IMPORT_ERROR
        }

try:
    from sensors.camera_reader import start_camera, get_camera_jpeg, stop_camera
except Exception as e:
    CAMERA_IMPORT_ERROR = str(e)

    def start_camera():
        return None

    def stop_camera():
        return None

    def get_camera_jpeg():
        return None

# --- Optional Ultralytics import ---
YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
try:
    from ultralytics import YOLO
except Exception as e:
    ULTRALYTICS_IMPORT_ERROR = str(e)

DB_PATH = os.path.join(BASE_DIR, "data", "palms.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "palm_tree_detector_ncnn_model")

PALM_CONF_TH = float(os.getenv("PALM_CONF_TH", "0.25"))
PALM_LOG_COOLDOWN = float(os.getenv("PALM_LOG_COOLDOWN", "4.0"))
AI_INTERVAL = float(os.getenv("AI_INTERVAL", "0.1"))   # seconds between inferences
MJPEG_FPS = float(os.getenv("MJPEG_FPS", "8.0"))       # low-rate MJPEG stream

app = Flask(__name__)

_yolo_model = None
_yolo_error = None
_last_logged_time = 0.0

latest_frame_lock = threading.Lock()
latest_display_jpeg = None
latest_ai = {
    "enabled": YOLO is not None,
    "import_error": ULTRALYTICS_IMPORT_ERROR,
    "model_loaded": False,
    "model_path": MODEL_PATH,
    "model_error": None,
    "status": "Starting",
    "has_palm": False,
    "confidence": 0.0,
    "last_analysis": None,
    "last_logged_at": None,
    "logging_state": "Not recording",
    "gps_state": "Unknown"
}

ai_thread = None
ai_running = False


def make_placeholder_jpeg(text: str):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    cv2.putText(img, text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        return buf.tobytes()
    return None


def get_connection():
    return sqlite3.connect(DB_PATH)


def load_summary():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM trees")
    total_trees = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM missions")
    total_missions = cursor.fetchone()[0]

    conn.close()

    return {
        "total_trees": total_trees,
        "total_detections": total_detections,
        "total_missions": total_missions
    }


def load_missions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT mission_id, mission_name, start_time, end_time, area_name
        FROM missions
        ORDER BY mission_id DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    missions = []
    for r in rows:
        missions.append({
            "mission_id": r[0],
            "mission_name": r[1],
            "start_time": r[2],
            "end_time": r[3],
            "area_name": r[4]
        })
    return missions


def load_trees(limit=100):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT tree_id, latitude, longitude, status, first_seen, last_seen
        FROM trees
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    trees = []
    for r in rows:
        trees.append({
            "tree_id": r[0],
            "lat": r[1],
            "lon": r[2],
            "status": r[3] if r[3] else "active",
            "first_seen": r[4],
            "last_seen": r[5]
        })
    return trees


def load_detections(limit=50):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT detection_id, tree_id, mission_id, latitude, longitude, confidence, detected_at
        FROM detections
        ORDER BY detection_id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    detections = []
    for r in rows:
        detections.append({
            "detection_id": r[0],
            "tree_id": r[1],
            "mission_id": r[2],
            "lat": r[3],
            "lon": r[4],
            "confidence": r[5],
            "detected_at": r[6]
        })
    return detections


def load_yolo_model():
    global _yolo_model, _yolo_error

    if _yolo_model is not None:
        return _yolo_model

    if YOLO is None:
        _yolo_error = ULTRALYTICS_IMPORT_ERROR or "Ultralytics import failed"
        latest_ai["model_error"] = _yolo_error
        latest_ai["status"] = "AI unavailable"
        return None

    if not os.path.exists(MODEL_PATH):
        _yolo_error = f"Model path not found: {MODEL_PATH}"
        latest_ai["model_error"] = _yolo_error
        latest_ai["status"] = "Model missing"
        return None

    try:
        _yolo_model = YOLO(MODEL_PATH, task="detect")
        latest_ai["model_loaded"] = True
        latest_ai["model_error"] = None
        latest_ai["status"] = "Watching"
        return _yolo_model
    except Exception as e:
        _yolo_error = str(e)
        latest_ai["model_loaded"] = False
        latest_ai["model_error"] = _yolo_error
        latest_ai["status"] = "AI error"
        return None


def maybe_log_detection(confidence: float):
    global _last_logged_time

    now = time.time()
    if now - _last_logged_time < PALM_LOG_COOLDOWN:
        latest_ai["logging_state"] = "Cooldown active"
        return

    state = controller.get_state()
    mission_id = state.get("current_mission_id")
    if mission_id is None:
        latest_ai["logging_state"] = "Mission not active"
        return

    gps = get_gps_data()
    lat = gps.get("latitude")
    lon = gps.get("longitude")
    gps_fix = gps.get("fix")

    if not gps_fix or lat is None or lon is None:
        latest_ai["logging_state"] = "GPS fix unavailable"
        return

    pose = state.get("current_pose", {})
    controller.record_tree_detection(
        robot_x=pose.get("x", 0.0),
        robot_y=pose.get("y", 0.0),
        robot_yaw_rad=pose.get("yaw", 0.0),
        gps_lat=lat,
        gps_lon=lon,
        confidence=confidence
    )

    _last_logged_time = now
    latest_ai["last_logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    latest_ai["logging_state"] = "Detection recorded"


def analyze_latest_frame():
    global latest_display_jpeg

    model = load_yolo_model()

    jpeg_bytes = get_camera_jpeg()
    if jpeg_bytes is None:
        latest_ai["status"] = "No camera frame"
        latest_ai["has_palm"] = False
        latest_ai["confidence"] = 0.0
        with latest_frame_lock:
            latest_display_jpeg = make_placeholder_jpeg("No camera frame")
        return

    if model is None:
        latest_ai["has_palm"] = False
        latest_ai["confidence"] = 0.0
        with latest_frame_lock:
            latest_display_jpeg = jpeg_bytes
        return

    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            latest_ai["status"] = "Frame decode failed"
            latest_ai["has_palm"] = False
            latest_ai["confidence"] = 0.0
            with latest_frame_lock:
                latest_display_jpeg = jpeg_bytes
            return

        results = model(frame, conf=PALM_CONF_TH, verbose=False)
        boxes = results[0].boxes

        has_palm = boxes is not None and len(boxes) > 0
        confidence = 0.0
        if has_palm:
            confidence = max(float(box.conf[0]) for box in boxes)

        latest_ai["has_palm"] = has_palm
        latest_ai["confidence"] = round(confidence, 4)
        latest_ai["last_analysis"] = time.strftime("%Y-%m-%d %H:%M:%S")

        gps = get_gps_data()
        latest_ai["gps_state"] = "Fix" if gps.get("fix") else "No fix"

        if has_palm:
            latest_ai["status"] = f"Palm detected ({confidence:.2f})"
            maybe_log_detection(confidence)
        else:
            latest_ai["status"] = "Watching"
            latest_ai["logging_state"] = "Not recording"

        annotated = results[0].plot()
        ok, buf = cv2.imencode(".jpg", annotated)
        with latest_frame_lock:
            latest_display_jpeg = buf.tobytes() if ok else jpeg_bytes

    except Exception as e:
        latest_ai["model_error"] = str(e)
        latest_ai["status"] = "AI error"
        latest_ai["has_palm"] = False
        latest_ai["confidence"] = 0.0
        with latest_frame_lock:
            latest_display_jpeg = jpeg_bytes


def ai_worker_loop():
    while ai_running:
        analyze_latest_frame()
        time.sleep(AI_INTERVAL)


@app.route("/start_mission", methods=["POST"])
def start_mission():
    controller.start_mission(
        mission_name="Dashboard Survey Mission",
        area_name="Web-Controlled Farm Sector",
        notes="Started from dashboard"
    )
    return redirect(url_for("index"))


@app.route("/complete_mission", methods=["POST"])
def complete_mission():
    try:
        controller.complete_mission()
    except Exception as e:
        print(f"Complete mission error: {e}")
    return redirect(url_for("index"))


@app.route("/abort_mission", methods=["POST"])
def abort_mission():
    controller.abort_mission()
    return redirect(url_for("index"))


@app.route("/return_home", methods=["POST"])
def return_home():
    controller.return_home()
    return redirect(url_for("index"))


@app.route("/api/state")
def api_state():
    return jsonify(controller.get_state())


@app.route("/api/summary")
def api_summary():
    return jsonify(load_summary())


@app.route("/api/trees")
def api_trees():
    return jsonify(load_trees())


@app.route("/api/detections")
def api_detections():
    return jsonify(load_detections())


@app.route("/api/missions")
def api_missions():
    return jsonify(load_missions())


@app.route("/api/sensors")
def api_sensors():
    return jsonify({
        "lidar": get_lidar_data(),
        "gps": get_gps_data()
    })


@app.route("/api/ai")
def api_ai():
    return jsonify(latest_ai)


def generate_frames():
    sleep_s = 1.0 / MJPEG_FPS if MJPEG_FPS > 0 else 0.33
    while True:
        try:
            with latest_frame_lock:
                frame = latest_display_jpeg

            if frame is None:
                frame = get_camera_jpeg()

            if frame is None:
                frame = make_placeholder_jpeg("Waiting for frame")

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(sleep_s)

        except Exception as e:
            print("generate_frames error:", e)
            time.sleep(0.3)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    summary = load_summary()
    missions = load_missions()
    trees = load_trees()
    detections = load_detections()
    robot_state = controller.get_state()
    lidar = get_lidar_data()
    gps = get_gps_data()
    ai_status = dict(latest_ai)

    map_center_lat = 29.203451
    map_center_lon = 25.519833

    if gps.get("fix") and gps.get("latitude") is not None and gps.get("longitude") is not None:
        map_center_lat = gps["latitude"]
        map_center_lon = gps["longitude"]
    elif trees:
        map_center_lat = trees[0]["lat"]
        map_center_lon = trees[0]["lon"]

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>PalmMapBot Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f7f9fc; color: #222; }
        .cards { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
        .card { background: white; border: 1px solid #ddd; border-radius: 12px; padding: 16px; min-width: 180px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .card h3 { margin: 0 0 8px 0; font-size: 16px; }
        .card p { margin: 0; font-size: 24px; font-weight: bold; }
        .section { background: white; padding: 16px; border-radius: 12px; border: 1px solid #ddd; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 24px; }
        .sensor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
        .sensor-item { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }
        .sensor-item h4 { margin: 0 0 8px 0; }
        .camera-box { display: flex; justify-content: center; align-items: center; background: #111; border-radius: 12px; padding: 12px; overflow: hidden; }
        .camera-box img { max-width: 100%; border-radius: 8px; display: block; }
        .camera-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }
        .meta-chip { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }
        .meta-chip b { display: block; margin-bottom: 4px; }
        #map { height: 520px; border-radius: 12px; overflow: hidden; border: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #e5e5e5; padding: 8px; text-align: left; font-size: 14px; }
        th { background: #f0f3f8; }
        form.inline { display: inline-block; margin-right: 10px; }
        button { padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; }
        .start { background: #198754; color: white; }
        .home { background: #0d6efd; color: white; }
        .abort { background: #dc3545; color: white; }
        .complete { background: #6f42c1; color: white; }
        .muted { color: #666; font-size: 13px; }
    </style>
</head>
<body>

    <h1>PalmMapBot Dashboard</h1>
    <p>Mission control + database + map view + live sensors + low-rate MJPEG palm detection.</p>

    <div class="cards">
        <div class="card"><h3>Total Trees</h3><p>{{ summary.total_trees }}</p></div>
        <div class="card"><h3>Total Detections</h3><p>{{ summary.total_detections }}</p></div>
        <div class="card"><h3>Total Missions</h3><p>{{ summary.total_missions }}</p></div>
        <div class="card"><h3>Robot Status</h3><p>{{ robot_state.status }}</p></div>
        <div class="card"><h3>AI Status</h3><p id="ai-status-card">{{ ai_status.status }}</p></div>
    </div>

    <div class="section">
        <h2>Mission Control</h2>
        <form class="inline" method="post" action="/start_mission">
            <button class="start" type="submit">Start Mission</button>
        </form>
        <form class="inline" method="post" action="/return_home">
            <button class="home" type="submit">Return Home</button>
        </form>
        <form class="inline" method="post" action="/complete_mission">
            <button class="complete" type="submit">Complete Mission</button>
        </form>
        <form class="inline" method="post" action="/abort_mission">
            <button class="abort" type="submit">Abort Mission</button>
        </form>

        <h3>Robot State</h3>
        <p><b>Status:</b> {{ robot_state.status }}</p>
        <p><b>Current Mission:</b> {{ robot_state.current_mission_id }}</p>
        <p><b>Current Pose:</b> {{ robot_state.current_pose }}</p>
        <p><b>Home Pose:</b> {{ robot_state.home_pose }}</p>
    </div>

    <div class="section">
        <h2>Live Camera + Palm AI</h2>
        <div class="camera-box">
            <img src="/video_feed" alt="Live camera feed">
        </div>
        <div class="camera-meta">
            <div class="meta-chip"><b>AI Status</b><span id="ai-status">{{ ai_status.status }}</span></div>
            <div class="meta-chip"><b>Palm Detected</b><span id="ai-has-palm">{{ ai_status.has_palm }}</span></div>
            <div class="meta-chip"><b>Confidence</b><span id="ai-confidence">{{ ai_status.confidence }}</span></div>
            <div class="meta-chip"><b>Model Loaded</b><span id="ai-model-loaded">{{ ai_status.model_loaded }}</span></div>
            <div class="meta-chip"><b>Mission Logging</b><span id="ai-logging">{{ ai_status.logging_state }}</span></div>
            <div class="meta-chip"><b>GPS State</b><span id="ai-gps">{{ ai_status.gps_state }}</span></div>
            <div class="meta-chip"><b>Last Analysis</b><span id="ai-last-analysis">{{ ai_status.last_analysis }}</span></div>
            <div class="meta-chip"><b>Last Detection Logged</b><span id="ai-last-logged">{{ ai_status.last_logged_at }}</span></div>
        </div>
        <p class="muted"><b>Model:</b> <span id="ai-model-path">{{ ai_status.model_path }}</span></p>
        <p class="muted"><b>Model error:</b> <span id="ai-model-error">{{ ai_status.model_error }}</span></p>
    </div>

    <div class="section">
        <h2>Live Sensor Readings</h2>
        <div class="sensor-grid">
            <div class="sensor-item">
                <h4>LiDAR</h4>
                <p><b>Connected:</b> {{ lidar.connected }}</p>
                <p><b>Distance:</b> {{ lidar.distance_cm }} cm</p>
                <p><b>Strength:</b> {{ lidar.strength }}</p>
                <p><b>Temperature:</b> {{ lidar.temperature_c }} °C</p>
                <p class="muted"><b>Error:</b> {{ lidar.error }}</p>
            </div>

            <div class="sensor-item">
                <h4>GPS</h4>
                <p><b>Connected:</b> {{ gps.connected }}</p>
                <p><b>Fix:</b> {{ gps.fix }}</p>
                <p><b>Latitude:</b> {{ gps.latitude }}</p>
                <p><b>Longitude:</b> {{ gps.longitude }}</p>
                <p><b>Satellites:</b> {{ gps.satellites }}</p>
                <p><b>Altitude:</b> {{ gps.altitude_m }} m</p>
                <p><b>Speed:</b> {{ gps.speed_kmh }} km/h</p>
                <p><b>Heading:</b> {{ gps.heading }}</p>
                <p class="muted"><b>Error:</b> {{ gps.error }}</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Tree Map</h2>
        <div id="map"></div>
    </div>

    <div class="section">
        <h2>Recent Missions</h2>
        <table>
            <thead>
                <tr>
                    <th>Mission ID</th>
                    <th>Mission Name</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Area Name</th>
                </tr>
            </thead>
            <tbody>
                {% for m in missions %}
                <tr>
                    <td>{{ m.mission_id }}</td>
                    <td>{{ m.mission_name }}</td>
                    <td>{{ m.start_time }}</td>
                    <td>{{ m.end_time }}</td>
                    <td>{{ m.area_name }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Recent Trees</h2>
        <table>
            <thead>
                <tr>
                    <th>Tree ID</th>
                    <th>Latitude</th>
                    <th>Longitude</th>
                    <th>Status</th>
                    <th>First Seen</th>
                    <th>Last Seen</th>
                </tr>
            </thead>
            <tbody>
                {% for tree in trees %}
                <tr>
                    <td>{{ tree.tree_id }}</td>
                    <td>{{ tree.lat }}</td>
                    <td>{{ tree.lon }}</td>
                    <td>{{ tree.status }}</td>
                    <td>{{ tree.first_seen }}</td>
                    <td>{{ tree.last_seen }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Recent Detections</h2>
        <table>
            <thead>
                <tr>
                    <th>Detection ID</th>
                    <th>Tree ID</th>
                    <th>Mission ID</th>
                    <th>Latitude</th>
                    <th>Longitude</th>
                    <th>Confidence</th>
                    <th>Detected At</th>
                </tr>
            </thead>
            <tbody>
                {% for d in detections %}
                <tr>
                    <td>{{ d.detection_id }}</td>
                    <td>{{ d.tree_id }}</td>
                    <td>{{ d.mission_id }}</td>
                    <td>{{ d.lat }}</td>
                    <td>{{ d.lon }}</td>
                    <td>{{ d.confidence }}</td>
                    <td>{{ d.detected_at }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

<script>
    var map = L.map('map').setView([{{ map_center_lat }}, {{ map_center_lon }}], 18);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 20
    }).addTo(map);

    fetch('/api/trees')
        .then(r => r.json())
        .then(data => {
            data.forEach(tree => {
                let color = "green";
                if (tree.status === "inactive") color = "gray";
                if (tree.status === "removed") color = "red";

                var marker = L.circleMarker([tree.lat, tree.lon], {
                    radius: 7,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.85
                }).addTo(map);

                marker.bindPopup(
                    "<b>" + tree.tree_id + "</b><br>" +
                    "Status: " + tree.status + "<br>" +
                    "Latitude: " + tree.lat + "<br>" +
                    "Longitude: " + tree.lon + "<br>" +
                    "First Seen: " + tree.first_seen + "<br>" +
                    "Last Seen: " + tree.last_seen
                );
            });
        });

    function refreshAIStatus() {
        fetch('/api/ai')
            .then(r => r.json())
            .then(ai => {
                document.getElementById('ai-status-card').textContent = ai.status;
                document.getElementById('ai-status').textContent = ai.status;
                document.getElementById('ai-has-palm').textContent = ai.has_palm;
                document.getElementById('ai-confidence').textContent = ai.confidence;
                document.getElementById('ai-model-loaded').textContent = ai.model_loaded;
                document.getElementById('ai-logging').textContent = ai.logging_state;
                document.getElementById('ai-gps').textContent = ai.gps_state;
                document.getElementById('ai-last-analysis').textContent = ai.last_analysis;
                document.getElementById('ai-last-logged').textContent = ai.last_logged_at;
                document.getElementById('ai-model-path').textContent = ai.model_path;
                document.getElementById('ai-model-error').textContent = ai.model_error;
            });
    }

    setInterval(refreshAIStatus, 1000);
</script>

</body>
</html>
        """,
        summary=summary,
        missions=missions,
        trees=trees,
        detections=detections,
        robot_state=robot_state,
        lidar=lidar,
        gps=gps,
        ai_status=ai_status,
        map_center_lat=map_center_lat,
        map_center_lon=map_center_lon
    )


def _safe_start(name, fn):
    try:
        fn()
        print(f"{name} started.")
    except Exception as e:
        print(f"{name} failed to start: {e}")


def _safe_stop(name, fn):
    try:
        fn()
        print(f"{name} stopped.")
    except Exception as e:
        print(f"{name} failed to stop cleanly: {e}")


def start_all_sensors():
    _safe_start("LiDAR", start_lidar)
    _safe_start("GPS", start_gps)
    _safe_start("Camera", start_camera)


def stop_all_sensors():
    global ai_running

    ai_running = False
    if ai_thread is not None and ai_thread.is_alive():
        ai_thread.join(timeout=1.0)

    _safe_stop("LiDAR", stop_lidar)
    _safe_stop("GPS", stop_gps)
    _safe_stop("Camera", stop_camera)


def start_ai_worker():
    global ai_thread, ai_running
    if ai_thread is not None and ai_thread.is_alive():
        return

    ai_running = True
    ai_thread = threading.Thread(target=ai_worker_loop, daemon=True)
    ai_thread.start()
    print("AI worker started.")


atexit.register(stop_all_sensors)


if __name__ == "__main__":
    start_all_sensors()
    start_ai_worker()
    app.run(host="0.0.0.0", port=5000, debug=False)
