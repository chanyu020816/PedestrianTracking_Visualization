import os
import cv2
import hashlib
import colorsys
import time
import numpy as np

# from ultralytics import YOLO, RTDETR, YOLO
from ultralytics_local import YOLO, RTDETR
from ultralytics import YOLO
from utils.track_utils import *

# Load the YOLOv8 model
model = YOLO("../ObjectDetectionModel/final_models/yolov8m-nms-free/weights/best.pt")
# model = RTDETR("../ObjectDetectionModel/final_models/yolov8n-rt-detr/weights/best.pt")
# model = YOLO("./pretrained_models/yolov8n.pt")
# model = RTDETR("./pretrained_models/rtdetr-l.pt")
print(model.info())

# Open the video file
video_path = "./testing.mp4"
video_path = "../RealData/tku1080p.MOV"
video_out_path = "./output.mp4"
cap = cv2.VideoCapture(video_path)

track_history = dict()
direction_history = dict()
miss_track = dict()

# Read a frame from the video
cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (960, 540)
)

frame_count = 0

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

        if frame_count % 1 != 0:
            frame_count += 1
            prev_time = current_time
            continue

        print(int(1 / time_interval))
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            # device="mps",
            verbose=True,
            classes=[0],
            tracker="./tracker_config.yaml"
        )

        # print(results)
        # Visualize the results on the frame
        results = results[0]
        annotated_frame = frame # results.plot()

        center = results.boxes.xywh.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        id_center = np.hstack((ids.reshape(-1, 1), center))

        for i in range(id_center.shape[0]):
            obj = id_center[i]
            id, x, y = obj[0], obj[1], obj[2]
            if id in track_history.keys():
                track_history[id].append((x, y))
                dire = get_four_direction(track_history[id][0], track_history[id][1])[0]
                if dire != -1:
                    direction_history[id] = dire
                else:
                    dire = direction_history[id]
                # track_history[id][2] = dire
                directions[dire] += 1
                miss_track[id] -= 1
            else:
                # first time detected
                track_history[id] = [(x, y)]
                miss_track[id] = 1
                direction_history[id] = 4
            activate_id.append(id)

        for id in track_history.keys():
            for track in track_history[id]:
                if track[0] == -1: continue
                cv2.circle(annotated_frame, track, 4, id_to_color(id), -1)
                # cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                # update previous position
                # print(f'{id}: {len(track_history[id])}')
            if id not in activate_id:
                track_history[id].append((-1, -1))
            # only plot the last 20 frame of a track
            if len(track_history[id]) >= 15:
                track_history[id].pop(0)

        # it a track keep missing for more than 20 times, delete it
        miss_track = {k: v - 1 for k, v in miss_track.items()}
        miss_track = {k: v for k, v in miss_track.items() if v <= 15}

        if frame_count % 20 == 0:
            text = f'Dire1: {directions[0]} Dire2: {directions[1]} Dire3: {directions[2]}, Dire4: {directions[3]}'
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Display the annotated frame

        cv2.imshow("Tracking", annotated_frame)
        # cap_out.write(annotated_frame)

        frame_count += 1
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cap_out.release()
cv2.destroyAllWindows()


