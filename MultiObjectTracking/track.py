import os
import cv2
import hashlib
import colorsys
import time
import numpy as np

# from ultralytics import YOLO, RTDETR
from ultralytics import YOLO, RTDETR
# from ultralytics import YOLO
from utils.track_utils import *

model_names = ["yolov8m-nms-free", "yolov8m", "yolov8n", "yolov8l-nms-free", "yolov8l", "yolov8n-rt-detr", "rtdetr-r18"]
model_name = model_names[0]
draw_grid = False
reso = 720
reso = 1080
reso = "4k"
distance_thresh = 0.5
record_thresh = 35
# (width, height) = (1280, 720) if reso == 720 else (1920, 1080)
(width, height) = (3840 ,2160)
size = (width, height)

directions_class = [
    "InBusinessDepartment", # red
    "OutBusinessDepartment", # green
    "InSchool", # blue
    "OutSchool", # magenta
    "Undefined" # white
]

directions_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 255)
]

# Open the video file
video_path = f"./RealData/tku{reso}.MOV"
video_out_path = f"./tracking_result/{model_name}{reso}.mp4"
cap = cv2.VideoCapture(video_path)

# Load the YOLOv8 model
model = YOLO(f"./final_models/{model_name}/weights/best.pt")
# model = RTDETR(f"./final_models/{model_name}/weights/best.pt")
# model = YOLO("./pretrained_models/yolov8n.pt")
# model = RTDETR("./pretrained_models/rtdetr-l.pt")

track_history = dict()
direction_history = dict()
miss_track = dict()

# Read a frame from the video
cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    size
)

frame_count = 0
sum_person = 0
print("Start")
start_time = time.time()
prev_time = time.time()
# Loop through the video frames
while cap.isOpened():

    success, frame = cap.read()

    activate_id = []
    directions = [0 for _ in range(5)]
    if success:
        current_time = time.time()
        time_interval = (current_time - prev_time)
        prev_time = current_time

        # frame extraction
        """        
        if frame_count % 1 != 0:
            frame_count += 1
            prev_time = current_time
            continue
        """

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            device="cuda",
            verbose=False,
            classes=[0],
            tracker="./tracker_config.yaml",
            conf=0.2,
            iou=0.8
        )

        # Visualize the results on the frame
        results = results[0]
        annotated_frame = frame # or results.plot() with bboxes

        center = results.boxes.xywh.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        id_center = np.hstack((ids.reshape(-1, 1), center))
        person_num = id_center.shape[0]
        sum_person += person_num
        for i in range(person_num):
            obj = id_center[i]
            id, x, y = obj[0], obj[1], obj[2]

            # region ignored
            if y <= 200: continue
            if y >= 400 and x >= 800: continue

            # if already recorded
            if id in track_history.keys():
                track_history[id].append((x, y))
                #
                # dire = get_four_direction(track_history[id][0], track_history[id][1])[1]
                start_pos = next((value for value in track_history.get(id, []) if value != (-1, -1)), None)
                dire = get_school_direction(start_pos, (x, y))
                if dire != -1 and direction_history[id] != dire:
                    # if the direction is detected and different to past -> update the direction
                    direction_history[id] = dire
                else:
                    dire = direction_history[id]
                directions[dire] += 1
                miss_track[id] = 0
            # if not, initialize the track
            else:
                # first time detected
                track_history[id] = [(x, y)]
                # add to miss_track dict
                miss_track[id] = 0
                # initialize undefined direction
                direction_history[id] = 4
            # add to activated track id in this frame
            activate_id.append(id)

        for id in track_history.keys():
            for track in track_history[id]:
                if track[0] != -1:
                    cv2.circle(annotated_frame, track, 5, directions_colors[direction_history[id]], -1)
                    # cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    # update previous position
                    # print(f'{id}: {len(track_history[id])}')
                    # last_coord = track

            # add direction type for each track id
            """
            cv2.putText(
                annotated_frame,
                directions_class[direction_history[id]],
                last_coord,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                directions_colors[direction_history[id]]
            )
            """
            if id not in activate_id:
                track_history[id].append((-1, -1))
            # only plot the last 20 frame of a track
            if len(track_history[id]) >= record_thresh :
                track_history[id].pop(0)

        # it a track keep missing for more than 20 times, delete it
        miss_track = {k: v + 1 for k, v in miss_track.items()}
        miss_track = {k: v for k, v in miss_track.items() if v <= record_thresh}

        # add direction count and FPS every second (30 frames)
        
        if frame_count % 30 == 0:
            text = (
                f'{directions_class[0]}: {directions[0]}, '
                f'{directions_class[1]}: {directions[1]}, '
                f'{directions_class[2]}: {directions[2]}, '
                f'{directions_class[3]}: {directions[3]}  '
            )
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        """
        if draw_grid:
            cv2.line(annotated_frame, (0, 200), (width, 200), (0, 255, 255), 2)
            cv2.line(annotated_frame, (0, 400), (width, 400), (0, 255, 255), 2)
            cv2.line(annotated_frame, (0, 600), (width, 600), (0, 255, 255), 2)

            cv2.line(annotated_frame, (200, 0), (200, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (400, 0), (400, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (600, 0), (600, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (800, 0), (800, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (1000, 0), (1000, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (1200, 0), (1200, height), (0, 255, 255), 2)
        """
        # cv2.imshow("Tracking", annotated_frame)
        cap_out.write(annotated_frame)

        frame_count += 1
        
        if frame_count >= 1800: break 
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

person_mean = sum_person / 1800
spend = (time.time() - start_time)
print(f"Spend: {spend:.4f}")
fps = 1800 / spend
print(f'Avg Fps: {fps:.4f}')
with open("./track_result.txt", "a") as file:
    file.write(f'Model: {model_name}, Resolution {reso}, Spend: {spend:.4f}, Avg Fps: {fps:.4f}, Avg Person {person_mean:.4f}  \n')
# Release the video capture object and close the display window
cap.release()
cap_out.release()
cv2.destroyAllWindows()


