import tkinter as tk
from Constant import *
from PedestrianTrackingPage import PedestrianTrackingPage
class MainView(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Pedestrian Tracking Visualization")
        self.geometry(f"{WIDTH}x{HEIGHT}")
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")

        self.pages = {
            "PedestrianTrackingPage": PedestrianTrackingPage(self.container, self),
        }

        self.initial()

    def initial(self):
        self.pages["PedestrianTrackingPage"].grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = MainView()
    app.mainloop()
