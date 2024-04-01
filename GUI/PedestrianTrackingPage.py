import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from base64 import b64decode
from io import BytesIO
from tkmacosx import Button
import datetime
import cv2
import os
from Constant import *
from Base64Image import PedestrianTrackingPageBG

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

        self.summary_title = tk.Label(self, text="Summary",
            font=("Canva Sans", 35, "bold"), bg="#FFF3F3", fg="#FF3A3A")
        self.canvas = tk.Canvas(self, bg="#FFFFFF", bd=0, borderwidth=0,
            border=0, relief="solid", width=350, highlightthickness=0)

        self.create_pedestrian_tracking_page()

    def create_pedestrian_tracking_page(self):
        style = ttk.Style()
        style.theme_use('clam')

        main_title = tk.Label(self, text="Pedestrian Tracking & Visualization",
            font=("Canva Sans", 25, "bold"), bg="#FFF3F3", fg="#4f4e4d")
        main_title.place(x=150, y=10)


        upload_button = Button(self, text="Upload Video", command=self.upload_video,
            padx=10, pady=8, bg="#DF3F3F", bd=0, font=("Open Sans", 23, "bold"),
            activebackground="#FF3A3A", activeforeground="white",
            highlightthickness=0, borderwidth=0, highlightcolor="#FFF3F3",
            fg="white", highlightbackground="#FFF3F3", width=400, height=55)
        upload_button.place(x=160, y=630)




    def upload_video(self):
        self.summary_title.place(x=930, y=80)