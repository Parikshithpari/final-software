import os
import re
import cv2
import time
import uuid
import sqlite3
import logging
import threading
import requests
import easyocr
import torch

from collections import defaultdict, deque, Counter
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/opt/ptag-anpr"

CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "INPD_more_accuracy_n.pt"
)

CONFIG_API_URL = "http://127.0.0.1:8080/api/anpr/camera-configs"
NOTIFY_API_URL = "http://127.0.0.1:8080/api/anpr/notify"

# Set this in /opt/ptag-anpr/.env.
# This must be the dedicated ANPR API key, not a normal user JWT.
ANPR_API_KEY = os.environ.get("ANPR_API_KEY", "").strip()

DATABASE_PATH = os.path.join(BASE_DIR, "anpr_events.db")

CONFIG_REFRESH_SECONDS = 30
CAMERA_RECONNECT_SECONDS = 5
PROCESS_EVERY_N_FRAMES = 3

# Require the same plate in 3 separate processed frames.
REQUIRED_VOTES = 3
VOTE_WINDOW_SIZE = 5

# Per branch + plate, regardless of ENTRY/EXIT.
DUPLICATE_COOLDOWN_SECONDS = 120

# Reset voting state after a plate has not been read for this long.
PLATE_FORGET_SECONDS = 15

VEHICLE_CONFIDENCE = 0.15
PLATE_CONFIDENCE = 0.10
INFERENCE_IMAGE_SIZE = 1280

PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
)

# COCO class IDs: 2 = car, 3 = motorcycle.
VEHICLE_CLASSES = [2, 3]

API_TIMEOUT_SECONDS = 20


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("PTAG-ANPR")


# ============================================================
# STARTUP VALIDATION AND MODEL LOADING
# ============================================================

os.makedirs(BASE_DIR, exist_ok=True)

if not ANPR_API_KEY:
    raise RuntimeError(
        "ANPR_API_KEY is missing. Add the dedicated ANPR key "
        "to /opt/ptag-anpr/.env"
    )

if not os.path.isfile(CAR_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle model not found: {CAR_MODEL_PATH}")

if not os.path.isfile(PLATE_MODEL_PATH):
    raise FileNotFoundError(f"Plate model not found: {PLATE_MODEL_PATH}")

DEVICE = 0 if torch.cuda.is_available() else "cpu"

log.info("Loading YOLO models on device: %s", DEVICE)

car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

if DEVICE != "cpu":
    car_model.to(DEVICE)
    plate_model.to(DEVICE)

log.info("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# YOLO and EasyOCR are shared by camera threads. Lock inference so
# concurrent camera workers do not call the same model simultaneously.
MODEL_LOCK = threading.Lock()

# Protects the check -> backend request -> successful timestamp sequence.
# This prevents two camera workers from submitting the same branch/plate
# simultaneously.
SUBMISSION_LOCK = threading.Lock()

# SQLite access lock.
DB_LOCK = threading.Lock()

API_HEADERS = {
    "X-ANPR-Key": ANPR_API_KEY,
    "Content-Type": "application/json"
}


# ============================================================
# SQLITE COOLDOWN DATABASE
# ============================================================

def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submitted_plates (
                branch_id INTEGER NOT NULL,
                plate_number TEXT NOT NULL,
                submitted_at REAL NOT NULL,
                PRIMARY KEY (branch_id, plate_number)
            )
        """)
        conn.commit()


def already_submitted_recently(branch_id, plate):
    now = time.time()

    with DB_LOCK:
        with sqlite3.connect(DATABASE_PATH) as conn:
            row = conn.execute("""
                SELECT submitted_at
                FROM submitted_plates
                WHERE branch_id = ?
                  AND plate_number = ?
            """, (branch_id, plate)).fetchone()

    if row is None:
        return False

    elapsed = now - float(row[0])
    return 0 <= elapsed < DUPLICATE_COOLDOWN_SECONDS


def save_successful_submission(branch_id, plate):
    now = time.time()

    with DB_LOCK:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.execute("""
                INSERT INTO submitted_plates (
                    branch_id,
                    plate_number,
                    submitted_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(branch_id, plate_number)
                DO UPDATE SET submitted_at = excluded.submitted_at
            """, (branch_id, plate, now))
            conn.commit()


# ============================================================
# BACKEND: FETCH CAMERA CONFIGURATIONS
# ============================================================

def fetch_camera_configs():
    try:
        response = requests.get(
            CONFIG_API_URL,
            headers=API_HEADERS,
            timeout=15
        )
        response.raise_for_status()

        cameras = response.json()

        if not isinstance(cameras, list):
            log.error("Camera config API did not return a JSON list.")
            return None

        return cameras

    except Exception as exc:
        log.error("Camera configuration request failed: %s", exc)
        # None means API failure. An empty list means no cameras configured.
        return None


# ============================================================
# BACKEND: SUBMIT A CONFIRMED PLATE
# ============================================================

def notify_backend(plate, camera):
    branch_id = int(camera["branchId"])
    camera_id = str(camera["cameraId"])

    payload = {
        "eventId": str(uuid.uuid4()),
        "plate": plate,
        "branchId": branch_id,
        "cameraId": camera_id
    }

    # Keep this sequence atomic across camera threads.
    with SUBMISSION_LOCK:
        if already_submitted_recently(branch_id, plate):
            log.info(
                "COOLDOWN ACTIVE | Branch %s | Plate %s",
                branch_id,
                plate
            )
            return "duplicate"

        try:
            response = requests.post(
                NOTIFY_API_URL,
                json=payload,
                headers=API_HEADERS,
                timeout=API_TIMEOUT_SECONDS
            )

            if response.ok:
                save_successful_submission(branch_id, plate)

                log.info(
                    "BACKEND ACCEPTED | Branch %s | Camera %s | Plate %s "
                    "| HTTP %s | Response: %s",
                    branch_id,
                    camera_id,
                    plate,
                    response.status_code,
                    response.text[:300]
                )
                return "success"

            log.error(
                "BACKEND REJECTED | Plate %s | HTTP %s | %s",
                plate,
                response.status_code,
                response.text[:500]
            )
            return "failed"

        except Exception as exc:
            log.error(
                "BACKEND REQUEST FAILED | Branch %s | Plate %s | %s",
                branch_id,
                plate,
                exc
            )
            return "failed"


# ============================================================
# OCR HELPERS
# ============================================================

def clean_plate_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_number_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return None

    try:
        padded = cv2.copyMakeBorder(
            plate_crop,
            15, 15, 20, 20,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        enlarged = cv2.resize(
            padded,
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        with MODEL_LOCK:
            results = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

        candidates = []

        for result in results:
            if len(result) < 3:
                continue

            plate = clean_plate_text(result[1])
            confidence = float(result[2])

            if PLATE_PATTERN.fullmatch(plate):
                candidates.append((plate, confidence))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    except Exception:
        log.exception("OCR processing failed")
        return None


# ============================================================
# VEHICLE AND NUMBER-PLATE DETECTION
# ============================================================

def detect_vehicles_and_plates(frame):
    with MODEL_LOCK:
        vehicle_results = car_model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONFIDENCE,
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False,
            device=DEVICE
        )

        plate_results = plate_model.predict(
            frame,
            conf=PLATE_CONFIDENCE,
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False,
            device=DEVICE
        )

    vehicles = []
    plates = []

    for result in vehicle_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())

            vehicle_type = "CAR" if class_id == 2 else "MOTORCYCLE"

            vehicles.append({
                "box": (x1, y1, x2, y2),
                "type": vehicle_type
            })

    for result in plate_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plates.append((x1, y1, x2, y2))

    return vehicles, plates


def plate_belongs_to_vehicle(plate_box, vehicles):
    px1, py1, px2, py2 = plate_box
    center_x = (px1 + px2) // 2
    center_y = (py1 + py2) // 2

    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["box"]

        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)

        if (
            x1 - margin_x <= center_x <= x2 + margin_x
            and y1 - margin_y <= center_y <= y2 + margin_y
        ):
            return vehicle["type"]

    return None


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(threading.Thread):

    def __init__(self, camera):
        super().__init__(daemon=True)

        self.camera = camera
        self.stop_event = threading.Event()

        self.branch_id = int(camera["branchId"])
        self.camera_id = str(camera["cameraId"])
        self.stream_url = str(camera["streamUrl"])

        # For each plate, store up to the last 5 valid OCR readings.
        self.vote_windows = defaultdict(
            lambda: deque(maxlen=VOTE_WINDOW_SIZE)
        )

        self.last_seen = {}
        self.triggered = set()

    def stop(self):
        self.stop_event.set()

    def process_frame(self, frame):
        vehicles, plate_boxes = detect_vehicles_and_plates(frame)

        # One vote maximum per plate per processed frame.
        frame_plates = set()

        for plate_box in plate_boxes:
            vehicle_type = plate_belongs_to_vehicle(
                plate_box,
                vehicles
            )

            if vehicle_type is None:
                continue

            x1, y1, x2, y2 = plate_box
            height, width = frame.shape[:2]

            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            plate = read_number_plate(plate_crop)

            if not plate:
                continue

            frame_plates.add(plate)

            log.info(
                "DETECTED | Branch %s | Camera %s | %s | %s",
                self.branch_id,
                self.camera_id,
                vehicle_type,
                plate
            )

        now = time.time()

        for plate in frame_plates:
            self.last_seen[plate] = now

            # ------------------------------------------------
            # BRANCH-WIDE 120-SECOND COOLDOWN
            # ------------------------------------------------
            # Check the persistent cooldown BEFORE collecting votes.
            # This also works if the vehicle disappeared and reappeared
            # during the cooldown, or if another camera saw the same plate.
            if already_submitted_recently(self.branch_id, plate):
                self.triggered.add(plate)
                self.vote_windows[plate].clear()
                continue

            # If this plate was previously triggered but its cooldown has
            # now expired, start a completely fresh set of 3 votes.
            if plate in self.triggered:
                self.triggered.discard(plate)
                self.vote_windows[plate].clear()

            # Add one vote from this processed frame.
            self.vote_windows[plate].append(plate)

            vote_count = Counter(self.vote_windows[plate])[plate]

            log.info(
                "VOTE %s/%s | Branch %s | Camera %s | Plate %s",
                min(vote_count, REQUIRED_VOTES),
                REQUIRED_VOTES,
                self.branch_id,
                self.camera_id,
                plate
            )

            if vote_count >= REQUIRED_VOTES:
                self.triggered.add(plate)

                log.info(
                    "THREE MATCHING VOTES CONFIRMED | "
                    "Branch %s | Plate %s",
                    self.branch_id,
                    plate
                )

                result = notify_backend(plate, self.camera)

                if result == "failed":
                    # Backend did not accept the request.
                    # Remove the trigger and collect 3 fresh votes
                    # before trying again.
                    self.triggered.discard(plate)
                    self.vote_windows[plate].clear()

                # On success, SQLite starts the 120-second cooldown.
                # On duplicate, the existing cooldown remains in force.

        # Clear voting state when a plate has disappeared for 15 seconds.
        expired_plates = [
            plate
            for plate, last_time in self.last_seen.items()
            if now - last_time > PLATE_FORGET_SECONDS
        ]

        for plate in expired_plates:
            self.last_seen.pop(plate, None)
            self.vote_windows.pop(plate, None)
            self.triggered.discard(plate)

    def run(self):
        log.info(
            "CAMERA WORKER STARTING | Branch %s | Camera %s | Stream %s",
            self.branch_id,
            self.camera_id,
            self.stream_url
        )

        while not self.stop_event.is_set():

            cap = cv2.VideoCapture(
                self.stream_url,
                cv2.CAP_FFMPEG
            )
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                log.error(
                    "CAMERA CONNECTION FAILED | Camera %s",
                    self.camera_id
                )
                cap.release()
                self.stop_event.wait(CAMERA_RECONNECT_SECONDS)
                continue

            log.info(
                "CAMERA CONNECTED | Branch %s | Camera %s",
                self.branch_id,
                self.camera_id
            )

            frame_number = 0

            while not self.stop_event.is_set():
                success, frame = cap.read()

                if not success or frame is None:
                    log.warning(
                        "STREAM LOST | Camera %s",
                        self.camera_id
                    )
                    break

                frame_number += 1

                if frame_number % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                try:
                    self.process_frame(frame)
                except Exception:
                    log.exception(
                        "FRAME PROCESSING ERROR | Camera %s",
                        self.camera_id
                    )

            cap.release()

            if not self.stop_event.is_set():
                log.info(
                    "RECONNECTING CAMERA | %s",
                    self.camera_id
                )
                self.stop_event.wait(CAMERA_RECONNECT_SECONDS)

        log.info("CAMERA WORKER STOPPED | %s", self.camera_id)


# ============================================================
# DYNAMIC CAMERA MANAGER
# ============================================================

def camera_signature(camera):
    return (
        int(camera["branchId"]),
        str(camera["cameraId"]),
        str(camera["streamUrl"])
    )


def camera_manager():
    workers = {}

    while True:
        cameras = fetch_camera_configs()

        # API failure: keep existing workers running and retry later.
        if cameras is None:
            time.sleep(CONFIG_REFRESH_SECONDS)
            continue

        active_signatures = set()

        for camera in cameras:
            required_fields = ("branchId", "cameraId", "streamUrl")

            if not all(camera.get(field) for field in required_fields):
                log.warning("Skipping incomplete camera config: %s", camera)
                continue

            try:
                signature = camera_signature(camera)
            except (TypeError, ValueError):
                log.warning("Skipping invalid camera config: %s", camera)
                continue

            active_signatures.add(signature)

            if signature not in workers:
                worker = CameraWorker(camera)
                worker.start()
                workers[signature] = worker

                log.info(
                    "CAMERA ADDED | Branch %s | Camera %s",
                    camera["branchId"],
                    camera["cameraId"]
                )

        # Stop workers for cameras removed or changed in the backend.
        for signature in list(workers.keys()):
            if signature not in active_signatures:
                log.info("Stopping removed/changed camera: %s", signature)

                worker = workers.pop(signature)
                worker.stop()
                worker.join(timeout=10)

        if not cameras:
            log.warning("No cameras are currently configured in the backend.")

        time.sleep(CONFIG_REFRESH_SECONDS)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    initialize_database()

    log.info("==============================================")
    log.info("             PTAG MULTI-CAMERA ANPR")
    log.info("==============================================")
    log.info("Inference device: %s", DEVICE)
    log.info("Required matching votes: %s", REQUIRED_VOTES)
    log.info("Vote window: %s readings", VOTE_WINDOW_SIZE)
    log.info("Duplicate cooldown: %s seconds", DUPLICATE_COOLDOWN_SECONDS)
    log.info("Camera config refresh: %s seconds", CONFIG_REFRESH_SECONDS)
    log.info("Dynamic camera monitoring enabled")

    try:
        camera_manager()
    except KeyboardInterrupt:
        log.info("PTag ANPR service stopped by operator.")
