import os
import argparse
import cv2
from ultralytics import YOLO
from ultralytics import RTDETR
import time

from tracker import Tracker

class MOT:
    def __init__(self, model: str, video_name: str, save_video: bool = True,
                 yolo: bool = False, detection_threshold: float = 0.5):
        self.model_path = os.path.join('./ObjectDetectionModel', f'{model}.pt')
        self.model = YOLO(self.model_path) if model.lower().startswith('yolo') else  RTDETR(self.model_path)

        self.video_path = os.path.join('.', 'data', f'{video_name}.mp4')
        self.cap = cv2.VideoCapture(self.video_path)
        self.save_video = save_video
        if self.save_video:
            self.video_output_path: str = os.path.join('./Output', f'{model}_output.mp4')
        else:
            self.video_output_path: str = ''
        self.tracker = Tracker()
        self.detection_threshold = detection_threshold
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        return self.end_time - self.start_time if self.start_time and self.end_time else None

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
        print(f"Average elapsed time: {elapsed_time:.4f} seconds")

    def get_model_summary(self) -> None:
        self.model.model.eval()
        self.model.info(detailed=False)

def main(model: str, video_name: str, save_video: bool = True,
         yolo: bool = False, detection_threshold: float = 0.5) -> None:
    mot = MOT(model = model, video_name = video_name, save_video = save_video,
              yolo = yolo, detection_threshold = detection_threshold)
    # mot.run()
    mot.get_model_summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify the model to use.")
    parser.add_argument("--model", type=str, default='yolov8n')
    parser.add_argument("--video_name", type=str, default='people')
    parser.add_argument("--save_video", type=bool, default=True)
    parser.add_argument("--yolo", type=bool, default=True)
    parser.add_argument("--detection_threshold", type=float, default=0.5)

    args = parser.parse_args()
    main(args.model, args.video_name, args.save_video, args.yolo, args.detection_threshold)

