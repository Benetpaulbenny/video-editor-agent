import logging

import numpy as np


class ObjectDetection:
    MODEL_NAME = "yolo11n.pt"
    YOLO_CONFIDENCE = 0.20
    STRONGSORT_CONFIDENCE = 0.70

    def __init__(self) -> None:
        self._model = None
        self._tracker = None
        self._tracker_object_ids: dict[int, int] = {}
        self._object_history: dict[int, tuple[int, list[float]]] = {}
        self._next_object_id = 1

    def start_video(self) -> None:
        self._tracker_object_ids.clear()
        self._object_history.clear()
        self._next_object_id = 1
        if self._tracker is None:
            return
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
            return
        self._tracker = None

    def execute(self, frame: np.ndarray, timestamp: float) -> list[list]:
        result = self._predict(frame)
        detections, class_ids = self._extract_detections(result)
        if len(detections) == 0:
            return []
        tracks = self._track(detections, frame)
        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        height, width = frame.shape[:2]
        records = []
        matched_detections = set()
        used_object_numbers = set()
        for track in tracks:
            detection_index = self._detection_index(track, len(detections))
            if detection_index is None:
                continue
            track_id = int(track[4])
            if track_id not in self._tracker_object_ids:
                self._tracker_object_ids[track_id] = self._allocate_object_id()
            object_number = self._tracker_object_ids[track_id]
            record = self._object_record(track, detections, class_ids, names, width, height, object_number)
            records.append(record)
            matched_detections.add(detection_index)
            used_object_numbers.add(object_number)
            self._remember_object(object_number, class_ids[detection_index], record[2])
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            object_number = self._fallback_object_id(class_ids[detection_index], detection[:4], used_object_numbers)
            fallback_track = np.array(
                [*detection[:4], 0, detection[4], detection[5], detection_index],
                dtype=np.float32,
            )
            records.append(self._object_record(fallback_track, detections, class_ids, names, width, height, object_number))
            used_object_numbers.add(object_number)
        return records

    def _predict(self, frame: np.ndarray):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.MODEL_NAME)
        results = self._model(frame, conf=self.YOLO_CONFIDENCE, verbose=False, device="cpu")
        return results[0]

    def _extract_detections(self, result) -> tuple[np.ndarray, list[int]]:
        boxes = result.boxes
        xyxy = self._to_numpy(boxes.xyxy)
        confidence = self._to_numpy(boxes.conf).reshape(-1)
        classes = self._to_numpy(boxes.cls).reshape(-1)
        detections = []
        class_ids = []
        for box, score, class_id in zip(xyxy, confidence, classes):
            if float(score) < self.YOLO_CONFIDENCE:
                continue
            detections.append([*box.tolist(), float(score), float(class_id)])
            class_ids.append(int(class_id))
        return np.asarray(detections, dtype=np.float32), class_ids

    def _track(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        tracks = self._get_tracker().update(detections, frame)
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
                tracker_kwargs={
                    "det_thresh": self.STRONGSORT_CONFIDENCE,
                    "min_conf": self.STRONGSORT_CONFIDENCE,
                    "n_init": 1,
                },
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("Layer 7 requires ultralytics and boxmot. Run uv sync before analyzing videos.") from error
        return self._tracker

    def _object_record(
        self,
        track: np.ndarray,
        detections: np.ndarray,
        class_ids: list[int],
        names,
        width: int,
        height: int,
        object_number: int,
    ) -> list:
        detection_index = self._detection_index(track, len(detections))
        if detection_index is None:
            detection_index = self._nearest_detection(track[:4], detections)
        bbox = [round(float(value), 4) for value in track[:4]]
        class_id = class_ids[detection_index]
        object_class = self._class_name(names, class_id)
        normalized = [
            round(float(np.clip(bbox[0] / width, 0, 1)), 4),
            round(float(np.clip(bbox[1] / height, 0, 1)), 4),
            round(float(np.clip(bbox[2] / width, 0, 1)), 4),
            round(float(np.clip(bbox[3] / height, 0, 1)), 4),
        ]
        box_width = round(max(bbox[2] - bbox[0], 0), 4)
        box_height = round(max(bbox[3] - bbox[1], 0), 4)
        area_ratio = round(float(box_width * box_height / (width * height)), 6)
        return [
            f"object_{object_number:03d}",
            object_class,
            bbox,
            normalized,
            round(float(np.clip(track[5], 0, 1)), 4),
            [box_width, box_height, area_ratio],
        ]

    def _detection_index(self, track: np.ndarray, detection_count: int) -> int | None:
        if len(track) <= 7:
            return None
        detection_index = int(track[7])
        if detection_index < 0 or detection_index >= detection_count:
            return None
        return detection_index

    def _allocate_object_id(self) -> int:
        object_number = self._next_object_id
        self._next_object_id += 1
        return object_number

    def _fallback_object_id(self, class_id: int, bbox: np.ndarray, used_object_numbers: set[int]) -> int:
        best_id = None
        best_iou = 0.05
        for object_number, (known_class_id, known_bbox) in self._object_history.items():
            if class_id != known_class_id or object_number in used_object_numbers:
                continue
            iou = self._iou(bbox, known_bbox)
            if iou > best_iou:
                best_id = object_number
                best_iou = iou
        if best_id is None:
            best_id = self._allocate_object_id()
        self._remember_object(best_id, class_id, bbox)
        return best_id

    def _remember_object(self, object_number: int, class_id: int, bbox: list[float] | np.ndarray) -> None:
        self._object_history[object_number] = (class_id, [float(value) for value in bbox])

    def _iou(self, first: np.ndarray | list[float], second: np.ndarray | list[float]) -> float:
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(right - left, 0) * max(bottom - top, 0)
        first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
        second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
        union = first_area + second_area - intersection
        return float(intersection / union) if union else 0.0

    def _class_name(self, names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and class_id < len(names):
            return str(names[class_id])
        return str(class_id)

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
