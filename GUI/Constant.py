import os
from cv2 import VideoCapture
from ultralytics import YOLO

HEIGHT = 850
WIDTH = 1750
VIDEO_WIDTH = 1340
VIDEO_HEIGHT = 730
FRAME_EXTRACT = False
FRAME_INTERVAL = 3 if FRAME_EXTRACT else 1
CSV_RECORD = False
MODEL_NAME = "yolov8m"
RESOLUTION = 720
GRID_SIZE = 200 * RESOLUTION / 720
DISTANCE_THRESHOLD = 0.5
RECORD_THRESHOLD = 30 * (1 - 1 / FRAME_INTERVAL) if FRAME_INTERVAL != 1 else 30
DIRECTIONS = [
    "In BusinessDepart",  # red
    "Out BusinessDepart",  # green
    "In School",  # blue
    "Out School",  # magenta
    "Undefined",  # white
]
DIRECTIONS_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 255),
]
AREAS_IGNORED = []
CAMERA = VideoCapture(0)
MODEL_FOLDER = "../ObjectDetectionModel/final_models"
model = YOLO(os.path.join(MODEL_FOLDER, MODEL_NAME, "weights", "best.pt"))
