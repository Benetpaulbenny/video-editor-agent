import json
import logging
import os
import warnings

import cv2
import numpy as np


class OCR:
    DETECTION_MODEL = "PP-OCRv5_server_det"
    RECOGNITION_MODEL = "PP-OCRv5_server_rec"

    def __init__(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self._ocr = None

    def execute(self, frame: np.ndarray) -> dict:
        ocr_frame, scale = self._resize_for_ocr(frame)
        result = self._predict(ocr_frame)
        data = self._result_data(result)
        polygons = self._values(data, "rec_polys") or self._values(data, "dt_polys")
        texts = self._values(data, "rec_texts")
        scores = self._values(data, "rec_scores")
        languages = self._values(data, "languages") or self._values(data, "language")
        height, width = frame.shape[:2]
        records = []
        for index, text in enumerate(texts):
            if index >= len(polygons) or index >= len(scores):
                continue
            polygon = self._polygon(polygons[index], scale)
            if len(polygon) < 3:
                continue
            bbox = self._bbox(polygon, width, height)
            record = {
                "text": str(text),
                "polygon": polygon,
                "bbox": bbox,
                "confidence": round(float(scores[index]), 4),
                "area_ratio": round(self._area_ratio(bbox, width, height), 6),
            }
            language = self._language(languages, index)
            if language is not None:
                record["language"] = language
            records.append(record)
        return {"ocr": {"texts": records}}

    def _predict(self, frame: np.ndarray):
        if self._ocr is None:
            warnings.filterwarnings("ignore", message="No ccache found.*", category=UserWarning)
            from paddleocr import PaddleOCR

            logging.getLogger("paddleocr").setLevel(logging.ERROR)
            logging.getLogger("paddlex").setLevel(logging.ERROR)
            self._ocr = PaddleOCR(
                text_detection_model_name=self.DETECTION_MODEL,
                text_recognition_model_name=self.RECOGNITION_MODEL,
                device="cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_det_limit_side_len=1280,
                text_det_limit_type="max",
                text_det_box_thresh=0.3,
                text_rec_score_thresh=0.5,
                enable_mkldnn=False,
            )
        return next(iter(self._ocr.predict(frame)), None)

    def _result_data(self, result) -> dict:
        if result is None:
            return {}
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            nested = data.get("res")
            return nested if isinstance(nested, dict) else data
        if hasattr(result, "__getitem__"):
            return result
        return {}

    def _values(self, data, key: str) -> list:
        try:
            values = data[key]
        except (KeyError, TypeError, IndexError):
            return []
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            return values.tolist()
        return list(values) if not isinstance(values, str) else [values]

    def _resize_for_ocr(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        longest_side = max(height, width)
        if longest_side <= 1280:
            return frame, 1.0
        scale = 1280 / longest_side
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _polygon(self, value, scale: float) -> list[list[float]]:
        points = np.asarray(value, dtype=float).reshape(-1, 2)
        return [[round(float(point[0] / scale), 2), round(float(point[1] / scale), 2)] for point in points]

    def _bbox(self, polygon: list[list[float]], width: int, height: int) -> dict:
        x_values = [point[0] for point in polygon]
        y_values = [point[1] for point in polygon]
        return {
            "x1": round(float(np.clip(min(x_values), 0, width)), 2),
            "y1": round(float(np.clip(min(y_values), 0, height)), 2),
            "x2": round(float(np.clip(max(x_values), 0, width)), 2),
            "y2": round(float(np.clip(max(y_values), 0, height)), 2),
        }

    def _area_ratio(self, bbox: dict, width: int, height: int) -> float:
        area = max(bbox["x2"] - bbox["x1"], 0) * max(bbox["y2"] - bbox["y1"], 0)
        return float(area / (width * height))

    def _language(self, languages: list, index: int) -> str | None:
        if index >= len(languages):
            return None
        value = languages[index]
        if isinstance(value, str) and value:
            return value
        return None
