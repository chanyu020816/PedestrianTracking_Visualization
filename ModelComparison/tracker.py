from TrackingAlgorithms.DeepSort.deep_sort.tracker import Tracker as DeepSortTracker
from TrackingAlgorithms.DeepSort.tools import generate_detections as gdet
from TrackingAlgorithms.DeepSort.deep_sort import nn_matching
from TrackingAlgorithms.DeepSort.deep_sort.detection import Detection
import numpy as np

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'EncodingModel/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            center = (int(x1 * 0.5 + x2 * 0.5), int(y1 * 0.5 + y2 * 0.5))
            tracks.append(Track(track_id, bbox, center))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    center = None

    def __init__(self, track_id, bbox, center):
        self.track_id = track_id
        self.bbox = bbox
        self.center = center
