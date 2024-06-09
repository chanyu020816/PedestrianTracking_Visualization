import tkinter as tk
from base64 import b64decode
from io import BytesIO
from time import time
from tkinter import filedialog, ttk

import cv2
import numpy as np
from Base64Image import PedestrianTrackingPageBG
from Constant import *
from numpy import array
from PIL import Image, ImageTk
from tkmacosx import Button
from utils.track_utils import *
from utils.VideoTracking import *


class PedestrianTrackingPage(tk.Frame):
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
        self.running_camera_frame = False

        self.summary_title = tk.Label(
            self,
            text="Summary",
            font=("Canva Sans", 35, "bold"),
            bg="#FFFFFF",
            fg="#4f4e4d",
        )
        self.summary_title.place(x=1490, y=25)
        self.inbusiness_label = None
        self.outbusiness_label = None
        self.inschool_label = None
        self.outschool_label = None
        self.total_label = None
        self.fps_label = None
        self.inbusiness_label_num = None
        self.outbusiness_label_num = None
        self.inschool_label_num = None
        self.outschool_label_num = None
        self.total_label_num = None
        self.fps_label_num = None

        self.display_background()
        self.create_pedestrian_tracking_page()

    def create_pedestrian_tracking_page(self):
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

        upload_button = Button(
            self,
            text="Upload Video",
            command=self.upload_video,
            padx=10,
            pady=8,
            bg="#3f97df",
            bd=0,
            font=("Open Sans", 23, "bold"),
            activebackground="#032f54",
            activeforeground="white",
            highlightthickness=0,
            borderwidth=0,
            highlightcolor="#FFF3F3",
            fg="white",
            highlightbackground="#FFF3F3",
            width=300,
            height=55,
        )
        upload_button.place(x=1405, y=750)

        to_rt_tracking_page_button = Button(
            self,
            text="Camera",
            command=self.start_camera_frame,
            padx=10,
            pady=8,
            bg="#3f97df",
            bd=0,
            font=("Open Sans", 23, "bold"),
            activebackground="#032f54",
            activeforeground="white",
            highlightthickness=0,
            borderwidth=0,
            highlightcolor="#FFF3F3",
            fg="white",
            highlightbackground="#FFF3F3",
            width=300,
            height=55,
        )
        to_rt_tracking_page_button.place(x=1405, y=670)

    def upload_video(self):
        self.stop_camera_frame()
        video_path = filedialog.askopenfilename(
            filetypes=[("MOV Files", "*.mov"), ("MP4 Files", "*.mp4")]
        )
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            fourcc, size, fps = get_video_cfg(video_path)
            self.video_cfg = (fourcc, size, fps)
            self.show_video(detect=True, track=True)

    def show_video(self, detect=False, track=False):
        track_history = dict()
        direction_history = dict()
        miss_track = dict()
        video_width = VIDEO_WIDTH
        video_height = VIDEO_HEIGHT
        with open("log.txt", "w") as f:
            f.write("")
        if CSV_RECORD:
            with open("log.csv", "w") as f:
                f.write(
                    f"frame,{DIRECTIONS[0]},{DIRECTIONS[1]},{DIRECTIONS[2]},{DIRECTIONS[3]}\n"
                )
        if self.cap is not None:
            ret, frame = self.cap.read()
            self.init_summary()

            frame_count = 0

            while ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                activate_id = []
                directions = [0, 0, 0, 0, 0]
                # print(frame_count)
                if FRAME_EXTRACT:
                    if frame_count % FRAME_INTERVAL != 0:
                        self.window.update_idletasks()  # Update the GUI
                        ret, frame = self.cap.read()
                        frame_count += 1
                        continue
                if detect:
                    if not track:
                        img = self.detection(frame)
                    else:
                        start_time = time()
                        results = MODEL.track(
                            frame,
                            persist=True,
                            device="mps",
                            verbose=False,
                            tracker="./tracker_config.yaml",
                        )[0]

                        # Visualize the results on the frame
                        annotated_frame = frame  # or result.plot() with bboxes

                        center = results.boxes.xywh.cpu().numpy().astype(int)
                        ids = results.boxes.id.cpu().numpy().astype(int)
                        id_center = np.hstack((ids.reshape(-1, 1), center))
                        del center, ids

                        for i in range(id_center.shape[0]):
                            obj = id_center[i]
                            id, x, y = obj[0], obj[1], obj[2]

                            # region ignored
                            if y <= 200:
                                continue
                            if y >= 400 and x >= 800:
                                continue

                            # if already recorded
                            if id in track_history.keys():
                                track_history[id].append((x, y))
                                start_pos = next(
                                    (
                                        value
                                        for value in track_history.get(id, [])
                                        if value != (-1, -1)
                                    ),
                                    None,
                                )
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
                                    cv2.circle(
                                        annotated_frame,
                                        track,
                                        5,
                                        DIRECTIONS_COLORS[direction_history[id]],
                                        -1,
                                    )
                                    # update previous position
                                    last_coord = track

                            if id not in activate_id:
                                track_history[id].append((-1, -1))
                            # only plot the last 20 frame of a track
                            if len(track_history[id]) >= RECORD_THRESHOLD:
                                track_history[id].pop(0)

                        # it a track keep missing for more than 20 times, delete it
                        miss_track = {k: v + 1 for k, v in miss_track.items()}

                        if frame_count % 30 == 0:
                            deactivate_id = [
                                k for k, v in miss_track.items() if v > RECORD_THRESHOLD
                            ]
                            miss_track = {
                                k: v
                                for k, v in miss_track.items()
                                if v <= RECORD_THRESHOLD
                            }
                            track_history = {
                                k: v
                                for k, v in track_history.items()
                                if k not in deactivate_id
                            }
                            direction_history = {
                                k: v
                                for k, v in direction_history.items()
                                if k not in deactivate_id
                            }
                            self.update_summary(directions)

                            with open("log.txt", "a") as f:
                                directions_class = DIRECTIONS
                                text = (
                                    f"frame {frame_count:3d} - ["
                                    f"{directions_class[0]}: {directions[0]:3}, "
                                    f"{directions_class[1]}: {directions[1]:3d}, "
                                    f"{directions_class[2]}: {directions[2]:3d}, "
                                    f"{directions_class[3]}: {directions[3]:3d}]\n"
                                )
                                f.write(text)

                        if CSV_RECORD:
                            # save to csv file
                            with open("log.csv", "a") as f:
                                directions_class = DIRECTIONS
                                text = f"{frame_count},{directions[0]},{directions[1]},{directions[2]},{directions[3]}\n"
                                f.write(text)

                        img = Image.fromarray(annotated_frame)
                        img = img.resize((video_width, video_height), Image.ANTIALIAS)
                else:
                    img = Image.fromarray(frame)
                    img = img.resize((video_width, video_height), Image.ANTIALIAS)

                photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=photo)
                # self.canvas.image = photo

                self.window.update_idletasks()  # Update the GUI
                ret, frame = self.cap.read()

                frame_count += 1
            self.stop_video()
            self.canvas.place(x=40, y=80)

    def init_summary(self):
        self.inbusiness_label = tk.Label(
            self,
            text=f"{DIRECTIONS[0]}:",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.inbusiness_label.place(x=1390, y=85)

        self.outbusiness_label = tk.Label(
            self,
            text=f"{DIRECTIONS[1]}:",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.outbusiness_label.place(x=1390, y=125)

        self.inschool_label = tk.Label(
            self,
            text=f"{DIRECTIONS[2]}:",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.inschool_label.place(x=1390, y=165)

        self.outschool_label = tk.Label(
            self,
            text=f"{DIRECTIONS[3]}:",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.outschool_label.place(x=1390, y=205)
        self.total_label = tk.Label(
            self,
            text=f"Total Pedestrian:",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )

        self.total_label.place(x=1390, y=245)

        self.fps_label = tk.Label(
            self,
            text=f"FPS:",
            font=("Canva Sans", 20, "bold"),
            bg="#FFFFFF",
            fg="#4f4e4d",
        )
        # self.fps_label.place(x=1515, y=715)

    def update_summary(self, directions, fps=30):
        if self.inbusiness_label_num is not None:
            self.inbusiness_label_num.place_forget()
            self.outbusiness_label_num.place_forget()
            self.inschool_label_num.place_forget()
            self.outschool_label_num.place_forget()
            self.total_label_num.place_forget()
            self.fps_label_num.place_forget()

        self.inbusiness_label_num = tk.Label(
            self,
            text=f"{directions[0]}",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.inbusiness_label_num.place(x=1615, y=85)

        self.outbusiness_label_num = tk.Label(
            self,
            text=f"{directions[1]}",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.outbusiness_label_num.place(x=1635, y=125)

        self.inschool_label_num = tk.Label(
            self,
            text=f"{directions[2]}",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.inschool_label_num.place(x=1510, y=165)

        self.outschool_label_num = tk.Label(
            self,
            text=f"{directions[3]}",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )
        self.outschool_label_num.place(x=1530, y=205)

        self.total_label_num = tk.Label(
            self,
            text=f"{np.sum(directions[:4])}",
            font=("Canva Sans", 25, "bold"),
            bg="#FFFFFF",
            fg="#4471e3",
        )

        self.total_label_num.place(x=1595, y=245)

        self.fps_label_num = tk.Label(
            self,
            text=f"{int(fps)}",
            font=("Canva Sans", 20, "bold"),
            bg="#FFFFFF",
            fg="#4f4e4d",
        )
        # self.fps_label_num.place(x=1560, y=715)

    def clean_summary(self):
        if self.inbusiness_label is not None:
            self.inbusiness_label.place_forget()
            self.outbusiness_label.place_forget()
            self.inschool_label.place_forget()
            self.outschool_label.place_forget()
            self.total_label.place_forget()
            self.fps_label.place_forget()
            self.inbusiness_label_num.place_forget()
            self.outbusiness_label_num.place_forget()
            self.inschool_label_num.place_forget()
            self.outschool_label_num.place_forget()
            self.total_label_num.place_forget()
            self.fps_label_num.place_forget()

    def to_rt_tracking_page(self):
        self.stop_video()
        self.controller.show_rt_tracking_page()

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.canvas.image = None
            self.clean_summary()

    def start_camera_frame(self):
        self.stop_video()
        self.running_camera_frame = True
        self.show_camera_frame()

    def show_camera_frame(self):
        if not self.running_camera_frame:
            return

        ret, frame = CAMERA.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo
        self.after(20, self.show_camera_frame)

    def stop_camera_frame(self):
        self.running_camera_frame = False

    @staticmethod
    def detection(frame):
        results = MODEL(frame, verbose=False, stream=True, device="mps", classes=[0])
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img_t = cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color=(0, 0, 0),
                    thickness=2,
                )
        img = Image.fromarray(img_t)
        img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.ANTIALIAS)
        return img
