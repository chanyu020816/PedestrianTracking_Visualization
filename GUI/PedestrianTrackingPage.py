import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from base64 import b64decode
from io import BytesIO
import numpy as np
from tkmacosx import Button
import datetime
import cv2
import os
from ultralytics import YOLO
from Constant import *
from Base64Image import PedestrianTrackingPageBG
from numpy import array
from utils.VideoTracking import *
from utils.track_utils import *

model = YOLO("yolov8n.pt")

class PedestrianTrackingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#040405')
        self.window=parent
        self.controller = controller

        bg_image_data = BytesIO(b64decode(PedestrianTrackingPageBG))
        self.bg_frame = Image.open(bg_image_data)
        resized_bg = self.bg_frame.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(resized_bg)
        self.bg_panel = tk.Label(self, image=photo)
        self.bg_panel.image = photo
        self.bg_panel.pack(fill='both', expand='yes')

        self.configure(width=WIDTH, height=HEIGHT)
        self.grid_propagate(False)
        self.grid()
        self.cap = None
        self.video_cfg = None

        background = tk.Canvas(self, bg="#FFFFFF", bd=0, borderwidth=0, border=0,
            relief="solid", width=1710, height=810, highlightthickness=0)
        background.place(x=20, y=20)
        self.canvas = tk.Canvas(self, bg="#F2F3F4", bd=0, borderwidth=0,
            border=0, relief="solid", width=1340, height=730, highlightthickness=0)
        main_title = tk.Label(self, text="Pedestrian Tracking & Visualization",
            font=("Canva Sans", 35, "bold"), bg="#FFFFFF", fg="#4f4e4d")
        main_title.place(x=400, y=25)

        upload_button = Button(self, text="Upload Video", command=self.upload_video,
           padx=10, pady=8, bg="#3f97df", bd=0, font=("Open Sans", 23, "bold"),
           activebackground="#032f54", activeforeground="white",
           highlightthickness=0, borderwidth=0, highlightcolor="#FFF3F3",
           fg="white", highlightbackground="#FFF3F3", width=300, height=55)
        upload_button.place(x=1405, y=750)

        self.summary_title = tk.Label(self, text="Summary",
            font=("Canva Sans", 35, "bold"), bg="#FFFFFF", fg="#4f4e4d")
        self.summary_title.place(x=1490, y=25)
        self.create_pedestrian_tracking_page()

        self.display_summary()

    def create_pedestrian_tracking_page(self):
        style = ttk.Style()
        style.theme_use('clam')

        self.canvas.place(x=40, y=80)


    def upload_video(self):
        video_path = filedialog.askopenfilename(
            filetypes=[('MOV Files', '*.mov'), ('MP4 Files', '*.mp4')]
        )
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            fourcc, size, fps = get_video_cfg(video_path)
            self.video_cfg = (fourcc, size, fps)
            self.show_video(True)

    def close_page(self):
        pass

    def show_video(self, detect=False):
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

                if frame_count % 2 != 0:
                    self.window.update_idletasks()  # Update the GUI
                    ret, frame = self.cap.read()
                    # print(f'{frame_count} not showed')
                    frame_count += 1
                    continue

                if detect:
                    # print(f'{frame_count} showed')
                    """
                     results = model(img, verbose=False, stream=True, device="mps", classes=[0])
                     for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]

                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            img_t = cv2.rectangle(array(img), (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)
                            img = Image.fromarray(img_t)
                            img = img.resize((video_width, video_height), Image.ANTIALIAS)
                    """
                    results = model.track(
                        frame,
                        persist=True,
                        device="mps",
                        verbose=False,
                        classes=[0],
                        tracker="./tracker_config.yaml"
                    )

                    # Visualize the results on the frame
                    results = results[0]
                    annotated_frame = results.plot() # frame
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

                    if frame_count % 20 == 0:
                        text = f'Dire1: {directions[0]} Dire2: {directions[1]} Dire3: {directions[2]}, Dire4: {directions[3]}'
                    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    """

                    img = Image.fromarray(annotated_frame)
                    img = img.resize((video_width, video_height), Image.ANTIALIAS)

                photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=photo)
                self.canvas.image = photo  # Keep reference to prevent garbage collection

                self.window.update_idletasks()  # Update the GUI
                ret, frame = self.cap.read()

                frame_count += 1

            print(frame_count)
            self.cap.release()
            self.cap = None
            self.canvas.image = None
            self.canvas.place(x=40, y=80)

    def display_summary(self):
        total_pedestrian = 72
        dir711_count = 24
        bus_stop_count = 32
        school_count = 16
        fps = 15
        pedestrian_count_title = tk.Label(self, text=f"Total Pedestrian:{total_pedestrian:>11}",
            font=("Canva Sans", 30, "bold"), bg="#FFFFFF", fg="#4471e3")
        pedestrian_count_title.place(x=1390, y=85)
        dir711_count_title = tk.Label(self, text=f"Direction 711:{dir711_count:>22}",
            font=("Canva Sans", 28, "bold"), bg="#FFFFFF", fg="#4471e3")
        dir711_count_title.place(x=1390, y=125)
        bus_stop_count_title = tk.Label(self, text=f"Direction Bus Stop:{bus_stop_count:>10}",
            font=("Canva Sans", 28, "bold"), bg="#FFFFFF", fg="#4471e3")
        bus_stop_count_title.place(x=1390, y=165)
        school_count_title = tk.Label(self, text=f"Direction School:{school_count:15}",
            font=("Canva Sans", 28, "bold"), bg="#FFFFFF", fg="#4471e3")
        school_count_title.place(x=1390, y=205)

        fps_title = tk.Label(self, text=f"FPS: {fps}",
            font=("Canva Sans", 20, "bold"), bg="#FFFFFF", fg="#4f4e4d")
        fps_title.place(x=1520, y=715)

