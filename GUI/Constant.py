from ultralytics import YOLO

HEIGHT = 850
WIDTH = 1750
FRAME_EXTRACT = False
FRAME_INTERVAL = 3 if FRAME_EXTRACT else 1
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

model = YOLO(f"../ObjectDetectionModel/final_models/{MODEL_NAME}/weights/best.pt")
