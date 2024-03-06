import tkinter as tk


class MainView(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Pedestrian Tracking Visualization")
        self.geometry("1400x800")
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = MainView()
    app.mainloop()
