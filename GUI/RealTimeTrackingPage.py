import datetime
import os
import tkinter as tk
from base64 import b64decode
from io import BytesIO
from tkinter import filedialog, ttk

import cv2
import numpy as np
from Base64Image import PedestrianTrackingPageBG
from Constant import *
from numpy import array
from PIL import Image, ImageTk
from tkmacosx import Button
from ultralytics import YOLO
from utils.track_utils import *
from utils.VideoTracking import *


class RealTimeTrackingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#040405")
        self.window = parent
        self.controller = controller

        bg_image_data = BytesIO(b64decode(PedestrianTrackingPageBG))
        self.bg_frame = Image.open(bg_image_data)
        resized_bg = self.bg_frame.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(resized_bg)
        self.bg_panel = tk.Label(self, image=photo)
        self.bg_panel.image = photo
        self.bg_panel.pack(fill="both", expand="yes")

        self.configure(width=WIDTH, height=HEIGHT)
        self.grid_propagate(False)
        self.grid()
        self.cap = None
        self.video_cfg = None

        self.display_background()
        self.create_realtime_tracking_page()

    def create_realtime_tracking_page(self):
        style = ttk.Style()
        style.theme_use("clam")

        self.canvas.place(x=40, y=80)

    def display_background(self):
        background = tk.Canvas(
            self,
            bg="#FFFFFF",
            bd=0,
            borderwidth=0,
            border=0,
            relief="solid",
            width=1710,
            height=810,
            highlightthickness=0,
        )
        background.place(x=20, y=20)
        self.canvas = tk.Canvas(
            self,
            bg="#F2F3F4",
            bd=0,
            borderwidth=0,
            border=0,
            relief="solid",
            width=1340,
            height=730,
            highlightthickness=0,
        )
        main_title = tk.Label(
            self,
            text="Pedestrian Tracking & Visualization",
            font=("Canva Sans", 35, "bold"),
            bg="#FFFFFF",
            fg="#4f4e4d",
        )
        main_title.place(x=400, y=25)

        self.summary_title = tk.Label(
            self,
            text="Summary",
            font=("Canva Sans", 35, "bold"),
            bg="#FFFFFF",
            fg="#4f4e4d",
        )
        self.summary_title.place(x=1490, y=25)

    def upload_video(self):
        video_path = filedialog.askopenfilename(
            filetypes=[("MOV Files", "*.mov"), ("MP4 Files", "*.mp4")]
        )
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            fourcc, size, fps = get_video_cfg(video_path)
            self.video_cfg = (fourcc, size, fps)

    def close_page(self):
        pass

    def show_video(self, detect=False, track=True):
        track_history = dict()
        direction_history = dict()
        miss_track = dict()

        if self.cap is not None:
            ret, frame = self.cap.read()
            video_width = 1340
            video_height = 730
            frame_count = 0

            activate_id = []
            directions = [0 for _ in range(5)]
            while ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((video_width, video_height), Image.ANTIALIAS)

                if FRAME_EXTRACT:
                    if frame_count % FRAME_INTERVAL == 0:
                        self.window.update_idletasks()  # Update the GUI
                        ret, frame = self.cap.read()
                        # print(f'{frame_count} not showed')
                        frame_count += 1
                        continue

                if detect:

                    if not track:
                        # only detect
                        results = model(
                            img, verbose=False, stream=True, device="mps", classes=[0]
                        )
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0]

                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                img_t = cv2.rectangle(
                                    array(img),
                                    (x1, y1),
                                    (x2, y2),
                                    color=(0, 0, 0),
                                    thickness=2,
                                )
                                img = Image.fromarray(img_t)
                                img = img.resize(
                                    (video_width, video_height), Image.ANTIALIAS
                                )
                    else:

                        results = model.track(
                            frame,
                            persist=True,
                            device="mps",
                            verbose=False,
                            tracker="./tracker_config.yaml",
                        )

                        # Visualize the results on the frame
                        results = results[0]
                        annotated_frame = frame  # or result.plot() with bboxes
                        """
                        center = results.boxes.xywh.cpu().numpy().astype(int)
                        ids = results.boxes.id.cpu().numpy().astype(int)
                        id_center = np.hstack((ids.reshape(-1, 1), center))

                        for i in range(id_center.shape[0]):
                            obj = id_center[i]
                            id, x, y = obj[0], obj[1], obj[2]
                            if id in track_history.keys():
                                track_history[id].append((x, y))
                                dire = get_direction(track_history[id][0], track_history[id][1])
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


                        """

                        img = Image.fromarray(annotated_frame)
                        img = img.resize((video_width, video_height), Image.ANTIALIAS)

                photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=photo)
                self.canvas.image = (
                    photo  # Keep reference to prevent garbage collection
                )

                self.window.update_idletasks()  # Update the GUI
                ret, frame = self.cap.read()

                frame_count += 1

            self.cap.release()
            self.cap = None
            self.canvas.image = None
            self.canvas.place(x=40, y=80)
