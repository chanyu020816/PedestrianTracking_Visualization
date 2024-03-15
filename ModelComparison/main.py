import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
import time
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tracker import Tracker
import datetime

class Detector:
    def __init__(self, model: str, input_name: str, save: bool = True, detection_threshold: float = 0.5):
        self.model_path = os.path.join('./ObjectDetectionModel', model)
        self.model = YOLO(self.model_path) if model.lower().startswith('yolo') else RTDETR(self.model_path)
        self.input_path = os.path.join('.', 'data', input_name)
        self.save = save
        self.detection_threshold = detection_threshold
        self.start_time = None
        self.end_time = None
        self.run_time = datetime.datetime.now().strftime("%m%d%H%M")

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        return self.end_time - self.start_time if self.start_time and self.end_time else None

class VideoTracker(Detector):
    def __init__(self, model: str, input_name: str, save: bool = True, detection_threshold: float = 0.5):
        super().__init__(model, input_name, save, detection_threshold)
        self.cap = cv2.VideoCapture(self.input_path)
        self.video_output_path: str = os.path.join('./Output', f'{model}_{self.run_time}_thold{self.detection_threshold}.mp4') if self.save else None
        self.tracker = Tracker()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self) -> None:
        print(f'Total frames: {self.total_frames}')
        ret, frame = self.cap.read()
        cap_out = cv2.VideoWriter(self.video_output_path, cv2.VideoWriter_fourcc(*'MP4V'), self.cap.get(cv2.CAP_PROP_FPS),
                                  (frame.shape[1], frame.shape[0]))
        self.start_timer()
        while ret:
            people_count = 0
            results = self.model(frame, classes=[0, 1], verbose = False)
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    if score > self.detection_threshold:
                        detections.append([x1, y1, x2, y2, score])
                self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    center = track.center
                    # Add the dot to the image
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    # cv2.rectangle(frame, (), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                    people_count += 1
                cv2.putText(frame, f'Total {people_count}', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)
                cv2.imshow('Tracking', frame)
                cap_out.write(frame)

                ret, frame = self.cap.read()
        self.stop_timer()
        elapsed_time = self.get_elapsed_time() / self.total_frames
        self.cap.release()
        cap_out.release()
        cv2.destroyAllWindows()
        print(f'Result saved as {self.video_output_path}')
        print(f"Average elapsed time: {elapsed_time:.4f} seconds")

    def get_model_summary(self) -> None:
        self.model.model.eval()
        self.model.info(detailed=False)

class ImageDetection(Detector):
    def  __init__(self, model: str, input_name: str, save: bool = True, detection_threshold: float = 0.5):
        super().__init__(model, input_name, save, detection_threshold)
        self.image = Image.open(self.input_path).convert("RGB")
        self.image_sizes = self.image.size[::-1]
        self.image_np = np.array(self.image)
        self.image_output_path: str = os.path.join('./Output', f'{model}_{self.run_time}_thold{self.detection_threshold}.png') if self.save else None
        self.plt = plt

    def _predict(self):
        results = self.model(self.image_np, classes=[0])
        detections = []
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > self.detection_threshold:
                detections.append([x1, y1, x2, y2, score])
        return detections

    def run(self) -> None:
        detections_result = self._predict()
        self.plt.figure(figsize=(16,10))
        self.plt.imshow(self.image_np)
        count = 1
        for d in detections_result:
            x1, y1, x2, y2, score = d
            self.plt.text(x1, y1, f'{count}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
            count += 1
        self.plt.text(13, 20, f'Total count: {count - 1}', fontsize=12)
        self.plt.axis('off')
        self.plt.savefig(self.image_output_path, bbox_inches='tight')
        print(f'Result saved as {self.image_output_path}')

    def get_model_summary(self) -> None:
        self.model.model.eval()
        self.model.info(detailed=False)
        

def main(model: str, name: str, save: bool, video: bool, detection_threshold: float) -> None:
    if video:
        detector = VideoTracker(model = model, input_name = name, save = save, detection_threshold = detection_threshold)
    else:
        detector = ImageDetection(model = model, input_name = name, save = save, detection_threshold = detection_threshold)
    detector.run()
    # detector.get_model_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify the model to use.")
    parser.add_argument("--model", '-M', type=str, default='yolov8n.pt')
    parser.add_argument("--type", '-T', type=str, default='video', choices=['image', 'video'])
    parser.add_argument("--file", type=str, default='people.mp4')
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--detection_threshold", '-DT', type=float, default=0.5)

    args = parser.parse_args()
    video = args.type == 'video'
    main(args.model, args.file, args.save, video, args.detection_threshold)

