import logging
from collections import deque

import numpy as np


class PoseEstimation:
    KEYPOINT_NAMES = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    MODEL_NAME = "yolo11n-pose.pt"
    PERSON_CONFIDENCE = 0.25
    MOVEMENT_THRESHOLD = 2.0
    ORIENTATION_THRESHOLD = 5.0
    HISTORY_SIZE = 4

    def __init__(self) -> None:
        self._model = None
        self._tracker = None
        self._reid_model = None
        self._history: dict[int, deque] = {}

    def start_video(self) -> None:
        self._history.clear()
        if self._tracker is None:
            return
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
            return
        self._tracker = None

    def execute(self, frame: np.ndarray, timestamp: float) -> list[list]:
        result = self._predict(frame)
        detections, keypoints = self._extract_detections(result)
        if len(detections) == 0:
            return []
        tracks = self._track(detections, frame)
        return [
            self._person_record(track, detections, keypoints, timestamp)
            for track in tracks
            if len(track) <= 7 or int(track[7]) >= 0
        ]

    def _predict(self, frame: np.ndarray):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.MODEL_NAME)
        results = self._model(frame, verbose=False, device="cpu")
        return results[0]

    def _extract_detections(self, result) -> tuple[np.ndarray, list[list[list[float]]]]:
        boxes = result.boxes
        xyxy = self._to_numpy(boxes.xyxy)
        confidence = self._to_numpy(boxes.conf).reshape(-1)
        classes = self._to_numpy(boxes.cls).reshape(-1)
        raw_keypoints = self._to_numpy(result.keypoints.xy)
        raw_keypoint_confidence = getattr(result.keypoints, "conf", None)
        if raw_keypoint_confidence is None:
            keypoint_confidence = np.ones(raw_keypoints.shape[:2], dtype=float)
        else:
            keypoint_confidence = self._to_numpy(raw_keypoint_confidence)
        detections = []
        keypoints = []
        for index, (box, score, class_id) in enumerate(zip(xyxy, confidence, classes)):
            if int(class_id) != 0 or float(score) < self.PERSON_CONFIDENCE:
                continue
            detections.append([*box.tolist(), float(score), 0.0])
            points = raw_keypoints[index]
            point_confidence = keypoint_confidence[index]
            keypoints.append(
                [
                    [round(float(point[0]), 4), round(float(point[1]), 4), round(float(point_confidence[point_index]), 4)]
                    for point_index, point in enumerate(points)
                ]
            )
        return np.asarray(detections, dtype=np.float32), keypoints

    def _track(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        tracker = self._get_tracker()
        tracks = tracker.update(detections, frame)
        if tracks is None:
            return np.empty((0, 8), dtype=float)
        return self._to_numpy(tracks).reshape(-1, 8)

    def _get_tracker(self):
        if self._tracker is not None:
            return self._tracker
        try:
            from boxmot.trackers.registry import create_tracker

            logging.getLogger("boxmot").setLevel(logging.ERROR)
            self._tracker = create_tracker(
                "strongsort",
                reid_weights="osnet_x0_25_msmt17.pt",
                device="cpu",
                half=False,
                tracker_kwargs={"n_init": 1},
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("Layer 6 requires ultralytics and boxmot. Run uv sync before analyzing videos.") from error
        return self._tracker

    def _person_record(
        self,
        track: np.ndarray,
        detections: np.ndarray,
        keypoints: list[list[list[float]]],
        timestamp: float,
    ) -> list:
        track_id = int(track[4])
        detection_index = int(track[7]) if len(track) > 7 else -1
        if detection_index < 0 or detection_index >= len(detections):
            detection_index = self._nearest_detection(track[:4], detections)
        points = keypoints[detection_index]
        bbox = [round(float(value), 4) for value in track[:4]]
        confidence = round(float(np.clip(track[5], 0, 1)), 4)
        rotation, orientation = self._body_orientation(points)
        movement = self._movement(track_id, bbox, timestamp)
        return [
            f"person_{track_id + 1:03d}",
            confidence,
            points,
            [points[9], points[10]],
            [rotation, orientation],
            movement,
        ]

    def _body_orientation(self, points: list[list[float]]) -> list:
        if len(points) < 13:
            return [0.0, "unknown"]
        shoulders = np.asarray([points[5][:2], points[6][:2]], dtype=float)
        hips = np.asarray([points[11][:2], points[12][:2]], dtype=float)
        if min(points[5][2], points[6][2], points[11][2], points[12][2]) <= 0:
            return [0.0, "unknown"]
        shoulder_center = shoulders.mean(axis=0)
        hip_center = hips.mean(axis=0)
        rotation = float(np.degrees(np.arctan2(hip_center[0] - shoulder_center[0], shoulder_center[1] - hip_center[1])))
        if abs(rotation) < self.ORIENTATION_THRESHOLD:
            direction = "center"
        elif rotation > 0:
            direction = "right"
        else:
            direction = "left"
        return [round(rotation, 4), direction]

    def _movement(self, track_id: int, bbox: list[float], timestamp: float) -> list:
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        history = self._history.setdefault(track_id, deque(maxlen=self.HISTORY_SIZE))
        previous = history[-1] if history else None
        history.append((timestamp, center))
        if previous is None:
            return [False, "stationary", 0.0]
        previous_timestamp, previous_center = previous
        elapsed = max(float(timestamp) - float(previous_timestamp), 1e-6)
        delta = np.asarray(center) - np.asarray(previous_center)
        speed = float(np.linalg.norm(delta) / elapsed)
        if speed <= self.MOVEMENT_THRESHOLD:
            direction = "stationary"
        elif abs(float(delta[0])) >= abs(float(delta[1])):
            direction = "right" if delta[0] > 0 else "left"
        else:
            direction = "down" if delta[1] > 0 else "up"
        return [speed > self.MOVEMENT_THRESHOLD, direction, round(speed, 4)]

    def _nearest_detection(self, bbox: np.ndarray, detections: np.ndarray) -> int:
        centers = (detections[:, :2] + detections[:, 2:4]) / 2
        target = (bbox[:2] + bbox[2:4]) / 2
        return int(np.argmin(np.linalg.norm(centers - target, axis=1)))

    def _to_numpy(self, value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
