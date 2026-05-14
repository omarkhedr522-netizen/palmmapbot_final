import cv2
import threading
import time

_camera = {
    "running": False,
    "cap": None,
    "frame": None,
    "jpeg": None,
    "lock": threading.Lock(),
    "thread": None,
    "error": None
}


def _camera_loop():
    global _camera

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    # Optional tuning
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        with _camera["lock"]:
            _camera["error"] = "Could not open /dev/video0"
            _camera["running"] = False
        return

    with _camera["lock"]:
        _camera["cap"] = cap
        _camera["error"] = None

    while _camera["running"]:
        ok, frame = cap.read()
        if not ok or frame is None:
            with _camera["lock"]:
                _camera["error"] = "Failed to read frame from /dev/video0"
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            with _camera["lock"]:
                _camera["error"] = "Failed to encode JPEG"
            time.sleep(0.02)
            continue

        with _camera["lock"]:
            _camera["frame"] = frame
            _camera["jpeg"] = buf.tobytes()
            _camera["error"] = None

        time.sleep(0.01)

    cap.release()

    with _camera["lock"]:
        _camera["cap"] = None


def start_camera():
    global _camera

    if _camera["running"]:
        return

    _camera["running"] = True
    _camera["thread"] = threading.Thread(target=_camera_loop, daemon=True)
    _camera["thread"].start()


def get_camera_jpeg():
    with _camera["lock"]:
        return _camera["jpeg"]


def get_camera_status():
    with _camera["lock"]:
        return {
            "running": _camera["running"],
            "connected": _camera["cap"] is not None,
            "error": _camera["error"]
        }


def stop_camera():
    global _camera

    _camera["running"] = False

    thread = _camera.get("thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)

    with _camera["lock"]:
        cap = _camera["cap"]
        if cap is not None:
            cap.release()
        _camera["cap"] = None
        _camera["frame"] = None
        _camera["jpeg"] = None
        _camera["thread"] = None
