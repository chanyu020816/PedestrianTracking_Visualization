import cv2

class VideoTracking():
    def __init__(self, video_path, obj_model, tracking_algo):
        self.video_path = video_path