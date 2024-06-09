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
        self.running_show_frame = False

        self.display_background()
        self.create_realtime_tracking_page()

    def create_realtime_tracking_page(self):
        style = ttk.Style()
        style.theme_use("clam")

        self.canvas.place(x=40, y=80)
        self.show_frame(self.canvas)

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

    def stop_camera(self):
        self.running_show_frame = False
        self.canvas.place_forget()

    def resume_show_frame(self):
        self.running_show_frame = True
        self.show_frame(self.canvas)
        self.canvas.place(x=40, y=80)

    def show_frame(self, canvas):
        if not self.running_show_frame:
            return
        ret, frame = CAMERA.read()
        video_width = 1340
        video_height = 730
        if ret:
            print("asdasdasdasd")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((video_width, video_height), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image=img)

            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo
        self.after(20, self.show_frame, canvas)