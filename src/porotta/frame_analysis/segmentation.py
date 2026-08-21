import os
from pathlib import Path

import numpy as np


class Segmentation:
    SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"
    SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
    SKY_LABELS = {"sky"}
    GROUND_LABELS = {"earth", "road", "floor", "grass", "field", "sand", "dirt", "ground"}

    def __init__(self) -> None:
        self._sam_predictor = None
        self._semantic_processor = None
        self._semantic_model = None

    def start_video(self) -> None:
        return None

    def execute(self, frame: np.ndarray, objects: list[list], person_tracks: list[list]) -> dict:
        height, width = frame.shape[:2]
        object_masks = self._object_masks(frame, objects)
        foreground = self._union(object_masks, height, width)
        background = np.logical_not(foreground)
        sky, ground = self._semantic_masks(frame)
        persons = self._person_records(objects, object_masks, person_tracks, height, width)
        return {
            "persons": persons,
            "foreground_mask": self._encode_mask(foreground),
            "background_mask": self._encode_mask(background),
            "sky_mask": self._encode_mask(sky),
            "ground_mask": self._encode_mask(ground),
        }

    def _person_records(
        self,
        objects: list[list],
        object_masks: list[np.ndarray],
        person_tracks: list[list],
        height: int,
        width: int,
    ) -> list[dict]:
        records = []
        used_track_indexes = set()
        for object_data, mask in zip(objects, object_masks):
            if object_data[1] != "person":
                continue
            track_index = self._match_person_track(object_data[2], person_tracks, used_track_indexes)
            if track_index is None:
                continue
            used_track_indexes.add(track_index)
            area_ratio = round(float(np.count_nonzero(mask) / (height * width)), 6)
            records.append(
                {
                    "person_id": person_tracks[track_index][0],
                    "mask_rle": self._encode_mask(mask),
                    "area_ratio": area_ratio,
                }
            )
        return records

    def _match_person_track(
        self,
        object_bbox: list[float],
        person_tracks: list[list],
        used_track_indexes: set[int],
    ) -> int | None:
        object_center = self._box_center(object_bbox)
        object_width = max(float(object_bbox[2]) - float(object_bbox[0]), 1.0)
        object_height = max(float(object_bbox[3]) - float(object_bbox[1]), 1.0)
        best_index = None
        best_distance = float("inf")
        for index, person_track in enumerate(person_tracks):
            if index in used_track_indexes or len(person_track) < 3:
                continue
            keypoints = np.asarray(person_track[2], dtype=float)
            if keypoints.ndim != 2 or keypoints.shape[1] < 2:
                continue
            confidence = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(len(keypoints))
            visible = keypoints[confidence > 0]
            if len(visible) == 0:
                continue
            track_center = visible[:, :2].mean(axis=0)
            distance = np.linalg.norm((track_center - object_center) / [object_width, object_height])
            if distance < best_distance:
                best_index = index
                best_distance = float(distance)
        if best_index is None or best_distance > 1.5:
            return None
        return best_index

    def _box_center(self, box: list[float]) -> np.ndarray:
        return np.asarray([(float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2])

    def _object_masks(self, frame: np.ndarray, objects: list[list]) -> list[np.ndarray]:
        if not objects:
            return []
        predictor = self._get_sam_predictor()
        rgb_frame = frame[:, :, ::-1].copy()
        predictor.set_image(rgb_frame)
        masks = []
        for object_data in objects:
            box = np.asarray(object_data[2], dtype=np.float32)
            predicted_masks, _, _ = predictor.predict(box=box, multimask_output=False)
            mask = np.asarray(predicted_masks)[0].astype(bool)
            masks.append(mask)
        return masks

    def _semantic_masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        if self._semantic_processor is None or self._semantic_model is None:
            self._semantic_processor = AutoImageProcessor.from_pretrained(self.SEGFORMER_MODEL_ID)
            self._semantic_model = AutoModelForSemanticSegmentation.from_pretrained(self.SEGFORMER_MODEL_ID)
            self._semantic_model.to("cpu").eval()
        height, width = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1].copy()
        inputs = self._semantic_processor(images=rgb_frame, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._semantic_model(**inputs)
        label_map = self._semantic_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(height, width)],
        )[0].cpu().numpy()
        labels = self._semantic_model.config.id2label
        sky_ids = self._label_ids(labels, self.SKY_LABELS)
        ground_ids = self._label_ids(labels, self.GROUND_LABELS)
        return np.isin(label_map, sky_ids), np.isin(label_map, ground_ids)

    def _get_sam_predictor(self):
        if self._sam_predictor is not None:
            return self._sam_predictor
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("Layer 8 requires SAM 2. Install the official sam2 package before analyzing videos.") from error
        checkpoint = Path(os.getenv("POROTTA_SAM2_CHECKPOINT", self.SAM2_CHECKPOINT)).expanduser()
        config = Path(os.getenv("POROTTA_SAM2_CONFIG", self.SAM2_CONFIG)).expanduser()
        try:
            if checkpoint.exists() and config.exists():
                from sam2.build_sam import build_sam2

                model = build_sam2(str(config), str(checkpoint), device="cpu")
                self._sam_predictor = SAM2ImagePredictor(model)
            else:
                model_id = os.getenv("POROTTA_SAM2_MODEL_ID", self.SAM2_MODEL_ID)
                self._sam_predictor = SAM2ImagePredictor.from_pretrained(model_id, device="cpu")
        except Exception as error:
            raise RuntimeError("Layer 8 could not load the SAM 2 checkpoint. Set POROTTA_SAM2_CHECKPOINT and POROTTA_SAM2_CONFIG or use a Hugging Face SAM 2 model.") from error
        return self._sam_predictor

    def _label_ids(self, labels, target_labels: set[str]) -> list[int]:
        return [
            int(label_id)
            for label_id, label in labels.items()
            if str(label).lower().replace("_", " ") in target_labels
        ]

    def _union(self, masks: list[np.ndarray], height: int, width: int) -> np.ndarray:
        if not masks:
            return np.zeros((height, width), dtype=bool)
        return np.logical_or.reduce(masks)

    def _encode_mask(self, mask: np.ndarray) -> dict:
        mask = np.asarray(mask, dtype=bool)
        flat = mask.flatten(order="F")
        padded = np.concatenate(([False], flat, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        counts = np.diff(np.concatenate(([0], changes, [flat.size]))).astype(int).tolist()
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": ",".join(map(str, counts))}
